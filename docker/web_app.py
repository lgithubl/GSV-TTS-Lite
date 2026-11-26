from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field


APP_PORT = int(os.environ.get("PORT", "8000"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))
REFS_DIR = Path(os.environ.get("REFS_DIR", "/refs"))
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "/outputs"))
DEFAULT_SPEAKER_AUDIO = os.environ.get("DEFAULT_SPEAKER_AUDIO", "/refs/speaker.wav")
DEFAULT_PROMPT_AUDIO = os.environ.get("DEFAULT_PROMPT_AUDIO", "/refs/prompt.wav")
DEFAULT_PROMPT_TEXT = os.environ.get("DEFAULT_PROMPT_TEXT", "")
DEFAULT_TEXT_LANGUAGE = os.environ.get("DEFAULT_TEXT_LANGUAGE", "auto")
DEFAULT_PROMPT_LANGUAGE = os.environ.get("DEFAULT_PROMPT_LANGUAGE", "auto")
GSVI_VERSION = os.environ.get("GSVI_VERSION", "v2ProPlus")
GSVI_MODEL_NAME = os.environ.get("GSVI_MODEL_NAME", "default-refs")

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GSV-TTS-Lite Web", version="0.1.0")
_tts = None
_tts_lock = threading.Lock()
# _tts_lock 只保护"第一次创建 _tts 这个单例对象"这一下，创建完就立刻释放了，
# 真正跑推理的 tts.infer()/tts.infer_batched() 完全没有加锁保护。
# asyncio.to_thread 会把每个请求的推理扔到线程池里不同的 OS 线程执行，两个
# 请求同时到就是两个线程同时在用同一个 tts 对象做推理——底层 GPT-SoVITS
# 引擎内部的共享状态（KV-cache、position 计数器之类）不是为并发重入设计的，
# 两个线程同时读写会相互踩，表现成"张量形状对不上"之类的报错（比如
# "index 512 is out of bounds for dimension 1 with size 512"）。
# 用这把独立的锁把真正的推理调用整个串行化，不跟 _tts_lock 混用——
# threading.Lock 不可重入，_get_tts() 内部已经拿了 _tts_lock，如果外层
# 推理调用也用同一把锁包住 _get_tts()，同一线程二次拿锁会直接死锁。
_infer_lock = threading.Lock()


def _env_flag_enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


logger = logging.getLogger("gsv_web")

# ---- 按请求选模型 ----
# voices.json: {"alice": {"gpt_model": "voices/alice.ckpt", "sovits_model": "voices/alice.pth",
#               "speaker_audio": "alice.wav", "prompt_audio": "alice.wav", "prompt_text": "...",
#               "prompt_language": "zh", "preload": true}}
# 模型相对路径基于 MODELS_DIR，参考音频相对路径基于 REFS_DIR。文件改动后下次请求自动重新读取。
VOICES_FILE = Path(os.environ.get("GSV_TTS_VOICES_FILE", str(MODELS_DIR / "voices.json")))
# 是否允许请求直接传 gpt_model / sovits_model 路径（只能指向 MODELS_DIR 内部）。
# .ckpt/.pth 用 pickle 加载，放开等于允许加载任意文件，默认关闭，只允许用 voice。
ALLOW_MODEL_PATHS = _env_flag_enabled("GSV_TTS_ALLOW_MODEL_PATHS", False)
# GPT、SoVITS 各自最多同时驻留几个模型，超出按最久未使用卸载。<=0 表示不限制。
MAX_LOADED_MODELS = int(os.environ.get("GSV_TTS_MAX_LOADED_MODELS", "3"))

_voices_cache: dict = {"mtime": None, "voices": {}}
_model_lru: dict[str, OrderedDict] = {"gpt": OrderedDict(), "sovits": OrderedDict()}


def _model_key(path: str | Path) -> str:
    return str(Path(path).resolve())


def _default_gpt_key() -> str:
    return _model_key(MODELS_DIR / "s1v3.ckpt")


def _default_sovits_key() -> str:
    return _model_key(MODELS_DIR / "s2Gv2ProPlus.pth")


def _load_voices() -> dict[str, dict]:
    try:
        mtime = VOICES_FILE.stat().st_mtime
    except FileNotFoundError:
        _voices_cache.update(mtime=None, voices={})
        return {}
    if _voices_cache["mtime"] != mtime:
        try:
            data = json.loads(VOICES_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"invalid voices file {VOICES_FILE}: {exc}") from exc
        if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
            raise HTTPException(status_code=500, detail=f"voices file {VOICES_FILE} must be an object of objects")
        _voices_cache.update(mtime=mtime, voices=data)
    return _voices_cache["voices"]


def _config_path(base: Path, value: str) -> str:
    """voices.json 里的路径：相对路径基于 base，绝对路径原样使用（配置文件是可信的）。"""
    path = Path(value)
    return str(path if path.is_absolute() else base / path)


def _request_model_path(value: str, field: str) -> str:
    """请求里直接传的模型路径：必须开启开关，且解析后必须在 MODELS_DIR 内。"""
    if not ALLOW_MODEL_PATHS:
        raise HTTPException(
            status_code=400,
            detail=f"{field} is disabled; use 'voice' or set GSV_TTS_ALLOW_MODEL_PATHS=1",
        )
    path = Path(value)
    if not path.is_absolute():
        path = MODELS_DIR / path
    path = path.resolve()
    if not path.is_relative_to(MODELS_DIR.resolve()):
        raise HTTPException(status_code=400, detail=f"{field} must be inside {MODELS_DIR}")
    return str(path)


def _resolve_synthesis(
    voice: str | None,
    gpt_model: str | None,
    sovits_model: str | None,
    speaker_audio: str | None,
    prompt_audio: str | None,
    prompt_text: str | None,
    text_language: str | None,
    prompt_language: str | None,
) -> dict:
    """优先级：请求显式字段 > voice 配置 > DEFAULT_* 环境变量 > 默认底模。"""
    cfg: dict = {}
    if voice and voice != GSVI_MODEL_NAME:
        voices = _load_voices()
        if voice not in voices:
            raise HTTPException(status_code=404, detail=f"voice not found: {voice}")
        cfg = voices[voice]

    if gpt_model:
        gpt_key = _request_model_path(gpt_model, "gpt_model")
    elif cfg.get("gpt_model"):
        gpt_key = _config_path(MODELS_DIR, cfg["gpt_model"])
    else:
        gpt_key = _default_gpt_key()
    if sovits_model:
        sovits_key = _request_model_path(sovits_model, "sovits_model")
    elif cfg.get("sovits_model"):
        sovits_key = _config_path(MODELS_DIR, cfg["sovits_model"])
    else:
        sovits_key = _default_sovits_key()
    gpt_key, sovits_key = _model_key(gpt_key), _model_key(sovits_key)

    cfg_speaker = _config_path(REFS_DIR, cfg["speaker_audio"]) if cfg.get("speaker_audio") else None
    cfg_prompt = _config_path(REFS_DIR, cfg["prompt_audio"]) if cfg.get("prompt_audio") else None
    # voice 只配了一个参考音频时，两者共用
    speaker_audio = speaker_audio or cfg_speaker or cfg_prompt or DEFAULT_SPEAKER_AUDIO
    prompt_audio = prompt_audio or cfg_prompt or cfg_speaker or DEFAULT_PROMPT_AUDIO
    prompt_text = prompt_text or cfg.get("prompt_text") or DEFAULT_PROMPT_TEXT
    text_language = _normalize_language(text_language or cfg.get("text_language"), DEFAULT_TEXT_LANGUAGE)
    prompt_language = _normalize_language(prompt_language or cfg.get("prompt_language"), DEFAULT_PROMPT_LANGUAGE)

    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text is required unless the voice or DEFAULT_PROMPT_TEXT sets it.")
    for name, path in (("gpt_model", gpt_key), ("sovits_model", sovits_key),
                       ("speaker_audio", speaker_audio), ("prompt_audio", prompt_audio)):
        if not Path(path).exists():
            raise HTTPException(status_code=400, detail=f"{name} not found: {path}")

    return {
        "voice": voice or None,
        "gpt_model": gpt_key,
        "sovits_model": sovits_key,
        "speaker_audio": speaker_audio,
        "prompt_audio": prompt_audio,
        "prompt_text": prompt_text,
        "text_language": text_language,
        "prompt_language": prompt_language,
    }


def _ensure_model(tts, kind: str, key: str):
    """必须在 _infer_lock 内调用：保证模型已加载，必要时按 LRU 卸载其它模型。"""
    loaded = tts.gpt_models if kind == "gpt" else tts.sovits_models
    lru = _model_lru[kind]
    if key not in loaded:
        if MAX_LOADED_MODELS > 0:
            while len(loaded) >= MAX_LOADED_MODELS:
                victim = next((k for k in lru if k in loaded), None) or next(iter(loaded))
                logger.warning("unloading %s model (LRU): %s", kind, victim)
                if kind == "gpt":
                    tts.unload_gpt_model(victim)
                else:
                    tts.unload_sovits_model(victim)
                lru.pop(victim, None)
        logger.warning("loading %s model: %s", kind, key)
        if kind == "gpt":
            tts.load_gpt_model(key)
        else:
            tts.load_sovits_model(key)
    lru[key] = None
    lru.move_to_end(key)


def _loaded_models() -> dict:
    if _tts is None:
        return {"gpt": [], "sovits": []}
    return {"gpt": list(_tts.gpt_models.keys()), "sovits": list(_tts.sovits_models.keys())}


def _parse_lang_map(raw: str) -> tuple[dict[str, str], list[str]]:
    """解析 GSV_TTS_LANG_MAP，格式 "wuu=zh,yue=zh"（条目用 , ; 或空白分隔，键值用 = 或 :，等号两边的空格忽略）。
    返回 (有效映射, 被忽略的条目)。目标只允许 zh/ja/en/ko，其余忽略而不是让服务起不来。"""
    mapping: dict[str, str] = {}
    ignored: list[str] = []
    for item in re.split(r"[,;\s]+", re.sub(r"\s*([=:])\s*", r"\1", raw or "")):
        if not item:
            continue
        key, sep, value = re.split(r"([=:])", item, maxsplit=1) if re.search(r"[=:]", item) else (item, "", "")
        key, value = key.strip().lower(), value.strip().lower()
        if not sep or not key or value not in ("zh", "ja", "en", "ko"):
            ignored.append(item)
            continue
        mapping[key] = value
    return mapping, ignored


def _model_status() -> dict:
    required = [
        MODELS_DIR / "chinese-hubert-base",
        MODELS_DIR / "g2p",
        MODELS_DIR / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt",
    ]
    optional_defaults = [
        MODELS_DIR / "s1v3.ckpt",
        MODELS_DIR / "s2Gv2ProPlus.pth",
    ]
    missing_required = [str(path) for path in required if not path.exists()]
    missing_defaults = [str(path) for path in optional_defaults if not path.exists()]
    return {
        "models_dir": str(MODELS_DIR),
        "required_ready": not missing_required,
        "missing_required": missing_required,
        "default_weights_ready": not missing_defaults,
        "missing_default_weights": missing_defaults,
    }


def _runtime_status() -> dict:
    try:
        import torch
    except Exception as exc:
        return {
            "torch_available": False,
            "torch_error": str(exc),
        }

    cuda_available = torch.cuda.is_available()
    cuda_devices = []
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            cuda_devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "capability": f"{props.major}.{props.minor}",
                    "memory_gb": round(props.total_memory / (1024**3), 2),
                }
            )

    return {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": len(cuda_devices),
        "cuda_devices": cuda_devices,
    }


def _get_tts():
    global _tts
    with _tts_lock:
        if _tts is not None:
            return _tts

        try:
            from gsv_tts import TTS
        except Exception as exc:
            raise RuntimeError(
                "Unable to import gsv_tts. Install runtime dependencies such as torch "
                "or use an image variant that includes them."
            ) from exc

        # 语言检测把汉字句子判成 wuu/yue 等会导致整句或半句被丢弃；默认不映射，配了才生效，
        # 例如 GSV_TTS_LANG_MAP="wuu=zh,yue=zh"
        lang_map, ignored = _parse_lang_map(os.environ.get("GSV_TTS_LANG_MAP", ""))
        for item in ignored:
            logger.warning("GSV_TTS_LANG_MAP: ignoring invalid entry %r (expected code=zh|ja|en|ko)", item)
        if lang_map:
            from gsv_tts.LangSegment import LangSegment
            LangSegment.set_lang_map(lang_map)
            logger.warning("GSV_TTS_LANG_MAP active: %s", lang_map)

        auto_download = _env_flag_enabled("GSV_TTS_AUTO_DOWNLOAD", False)
        tts = TTS(
            models_dir=str(MODELS_DIR),
            auto_download_models=auto_download,
            use_bert=_env_flag_enabled("GSV_TTS_USE_BERT", False),
            use_flash_attn=_env_flag_enabled("GSV_TTS_USE_FLASH_ATTN", False),
        )
        if _env_flag_enabled("GSV_TTS_LOAD_DEFAULT_WEIGHTS", True):
            # 用规范化后的路径做键，后续请求按同样的键查找，避免同一模型被加载两份
            _ensure_model(tts, "gpt", _default_gpt_key())
            _ensure_model(tts, "sovits", _default_sovits_key())
        try:
            for name, cfg in _load_voices().items():
                if not cfg.get("preload"):
                    continue
                if cfg.get("gpt_model"):
                    _ensure_model(tts, "gpt", _model_key(_config_path(MODELS_DIR, cfg["gpt_model"])))
                if cfg.get("sovits_model"):
                    _ensure_model(tts, "sovits", _model_key(_config_path(MODELS_DIR, cfg["sovits_model"])))
                logger.warning("preloaded voice: %s", name)
        except Exception:
            logger.exception("voice preload failed; continuing with lazy loading")
        _tts = tts
        return _tts


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    voice: str | None = None
    gpt_model: str | None = None
    sovits_model: str | None = None
    speaker_audio: str | None = None
    prompt_audio: str | None = None
    prompt_text: str | None = None
    # 不传则依次取 voice 配置、DEFAULT_TEXT_LANGUAGE / DEFAULT_PROMPT_LANGUAGE
    text_language: str | None = None
    prompt_language: str | None = None
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    noise_scale: float = 0.5
    speed: float = 1.0


class GSVIModelsRequest(BaseModel):
    version: str | None = None


class TTSBatchedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    voice: str | None = None
    gpt_model: str | None = None
    sovits_model: str | None = None
    speaker_audio: str | None = None
    prompt_audio: str | None = None
    prompt_text: str | None = None
    # 不传就落回 DEFAULT_TEXT_LANGUAGE / DEFAULT_PROMPT_LANGUAGE（默认 auto）；
    # 取值同 /api/tts：auto / ja / zh / en，另外接受 _normalize_language 里的别名
    text_language: str | None = None
    prompt_language: str | None = None
    is_cut_text: bool = True
    cut_minlen: int = 10
    cut_mute: float = 0.3
    return_subtitles: bool = False
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    noise_scale: float = 0.5
    speed: float = 1.0


class GSVIInferRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    model_name: str | None = None
    text_lang: str | None = None
    prompt_text_lang: str | None = None
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    speed_facter: float | None = None
    speed_factor: float | None = None


def _normalize_language(value: str | None, default: str = "auto") -> Literal["auto", "ja", "zh", "en"]:
    if not value:
        value = default
    value = value.strip().lower().replace("_", "-")
    aliases = {
        "auto": "auto",
        "ja": "ja",
        "jp": "ja",
        "jpn": "ja",
        "japanese": "ja",
        "日语": "ja",
        "日語": "ja",
        "日本語": "ja",
        "zh": "zh",
        "zh-cn": "zh",
        "zh-hans": "zh",
        "cht": "zh",
        "zh-tw": "zh",
        "zh-hant": "zh",
        "chinese": "zh",
        "中文": "zh",
        "简体中文": "zh",
        "繁体中文": "zh",
        "繁體中文": "zh",
        "en": "en",
        "eng": "en",
        "english": "en",
        "英语": "en",
        "英語": "en",
    }
    if value not in aliases:
        raise HTTPException(status_code=400, detail=f"unsupported language: {value}")
    return aliases[value]


async def _synthesize_to_file(request: TTSRequest) -> Path:
    r = _resolve_synthesis(
        request.voice, request.gpt_model, request.sovits_model,
        request.speaker_audio, request.prompt_audio, request.prompt_text,
        request.text_language, request.prompt_language,
    )

    def _run():
        with _infer_lock:
            tts = _get_tts()
            _ensure_model(tts, "gpt", r["gpt_model"])
            _ensure_model(tts, "sovits", r["sovits_model"])
            clip = tts.infer(
                spk_audio_path=r["speaker_audio"],
                prompt_audio_path=r["prompt_audio"],
                prompt_audio_text=r["prompt_text"],
                text=request.text,
                text_language=r["text_language"],
                prompt_language=r["prompt_language"],
                top_k=request.top_k,
                top_p=request.top_p,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty,
                noise_scale=request.noise_scale,
                speed=request.speed,
                gpt_model=r["gpt_model"],
                sovits_model=r["sovits_model"],
            )
        output = OUTPUTS_DIR / f"tts_{uuid.uuid4().hex}.wav"
        clip.save(str(output))
        return output

    try:
        return await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GSV-TTS-Lite</title>
  <style>
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #18202a; }
    main { max-width: 900px; margin: 0 auto; padding: 32px 18px; }
    h1 { font-size: 28px; margin: 0 0 20px; }
    label { display: block; font-weight: 650; margin: 16px 0 6px; }
    textarea, input { width: 100%; box-sizing: border-box; border: 1px solid #cad1dc; border-radius: 8px; padding: 10px 12px; font: inherit; background: white; }
    textarea { min-height: 130px; resize: vertical; }
    button { margin-top: 18px; border: 0; border-radius: 8px; padding: 11px 16px; font: inherit; font-weight: 700; background: #1f6feb; color: white; cursor: pointer; }
    button:disabled { opacity: .65; cursor: wait; }
    audio { width: 100%; margin-top: 18px; }
    pre { white-space: pre-wrap; background: #fff; border: 1px solid #dde3ea; border-radius: 8px; padding: 12px; }
  </style>
</head>
<body>
<main>
  <h1>GSV-TTS-Lite</h1>
  <form id="form">
    <label>Text</label>
    <textarea id="text" required>こんにちは、これは音声合成のテストです。</textarea>
    <label>Voice (optional, from voices.json)</label>
    <input id="voice" list="voices" placeholder="empty = default" />
    <datalist id="voices"></datalist>
    <label>Speaker audio path (optional override)</label>
    <input id="speaker" placeholder="/refs/speaker.wav" />
    <label>Prompt audio path (optional override)</label>
    <input id="prompt" placeholder="/refs/prompt.wav" />
    <label>Prompt text</label>
    <input id="promptText" placeholder="Text spoken in the prompt audio" />
    <button id="submit">Generate</button>
  </form>
  <audio id="audio" controls hidden></audio>
  <pre id="status"></pre>
</main>
<script>
const form = document.getElementById("form");
const statusEl = document.getElementById("status");
const audio = document.getElementById("audio");
const submit = document.getElementById("submit");
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submit.disabled = true;
  statusEl.textContent = "Generating...";
  audio.hidden = true;
  try {
    const res = await fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.fromEntries(Object.entries({
        text: document.getElementById("text").value,
        voice: document.getElementById("voice").value,
        speaker_audio: document.getElementById("speaker").value,
        prompt_audio: document.getElementById("prompt").value,
        prompt_text: document.getElementById("promptText").value
      }).filter(([, v]) => v)))
    });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    audio.src = URL.createObjectURL(blob);
    audio.hidden = false;
    statusEl.textContent = "Done.";
  } catch (error) {
    statusEl.textContent = String(error);
  } finally {
    submit.disabled = false;
  }
});
fetch("/api/voices").then(r => r.json()).then(j => {
  const list = document.getElementById("voices");
  Object.keys(j.voices || {}).forEach(name => { const o = document.createElement("option"); o.value = name; list.appendChild(o); });
}).catch(() => {});
fetch("/health").then(r => r.json()).then(j => { statusEl.textContent = JSON.stringify(j, null, 2); });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


@app.get("/health")
async def health():
    return {
        "ok": True,
        "auto_download": _env_flag_enabled("GSV_TTS_AUTO_DOWNLOAD", False),
        "tts_loaded": _tts is not None,
        "loaded_models": _loaded_models(),
        "max_loaded_models": MAX_LOADED_MODELS,
        "allow_model_paths": ALLOW_MODEL_PATHS,
        "voices_file": str(VOICES_FILE),
        "runtime": _runtime_status(),
        **_model_status(),
    }


@app.post("/api/tts")
async def synthesize(request: TTSRequest):
    output_path = await _synthesize_to_file(request)
    return FileResponse(output_path, media_type="audio/wav", filename=output_path.name)


@app.get("/api/voices")
async def list_voices():
    loaded = _loaded_models()
    voices = {}
    for name, cfg in _load_voices().items():
        gpt = _model_key(_config_path(MODELS_DIR, cfg["gpt_model"])) if cfg.get("gpt_model") else _default_gpt_key()
        sovits = _model_key(_config_path(MODELS_DIR, cfg["sovits_model"])) if cfg.get("sovits_model") else _default_sovits_key()
        voices[name] = {
            "gpt_model": gpt,
            "sovits_model": sovits,
            "speaker_audio": cfg.get("speaker_audio"),
            "prompt_audio": cfg.get("prompt_audio"),
            "prompt_language": cfg.get("prompt_language"),
            "preload": bool(cfg.get("preload")),
            "loaded": gpt in loaded["gpt"] and sovits in loaded["sovits"],
        }
    return {"voices_file": str(VOICES_FILE), "voices": voices, "loaded_models": loaded}


@app.get("/tts")
async def synthesize_gptsovits_compatible(
    text: str,
    voice: str | None = None,
    gpt_model: str | None = None,
    sovits_model: str | None = None,
    text_language: str | None = None,
    text_lang: str | None = None,
    prompt_language: str | None = None,
    prompt_lang: str | None = None,
    speaker_audio: str | None = None,
    ref_audio_path: str | None = None,
    prompt_audio: str | None = None,
    prompt_text: str | None = None,
    top_k: int = 15,
    top_p: float = 1.0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.35,
    noise_scale: float = 0.5,
    speed: float | None = None,
    speed_factor: float | None = None,
):
    ref_audio = ref_audio_path or prompt_audio
    request = TTSRequest(
        text=text,
        voice=voice,
        gpt_model=gpt_model,
        sovits_model=sovits_model,
        speaker_audio=speaker_audio or ref_audio,
        prompt_audio=ref_audio,
        prompt_text=prompt_text,
        text_language=text_language or text_lang,
        prompt_language=prompt_language or prompt_lang,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        noise_scale=noise_scale,
        speed=speed_factor if speed_factor is not None else (speed if speed is not None else 1.0),
    )
    output_path = await _synthesize_to_file(request)
    return FileResponse(output_path, media_type="audio/wav", filename=output_path.name)


def _gsvi_models_payload(version: str | None = None) -> dict:
    models = {
        GSVI_MODEL_NAME: {
            "speaker_audio": DEFAULT_SPEAKER_AUDIO,
            "prompt_audio": DEFAULT_PROMPT_AUDIO,
            "prompt_language": DEFAULT_PROMPT_LANGUAGE,
        }
    }
    for name, cfg in _load_voices().items():
        models[name] = {
            "speaker_audio": cfg.get("speaker_audio") or cfg.get("prompt_audio"),
            "prompt_audio": cfg.get("prompt_audio") or cfg.get("speaker_audio"),
            "prompt_language": cfg.get("prompt_language") or DEFAULT_PROMPT_LANGUAGE,
        }
    return {"version": version or GSVI_VERSION, "models": models}


@app.get("/version")
async def gsvi_versions():
    return [GSVI_VERSION]


@app.post("/models")
async def gsvi_models(request: GSVIModelsRequest | None = None):
    version = request.version if request else None
    return _gsvi_models_payload(version)


@app.get("/models/{version}")
async def gsvi_models_for_version(version: str):
    return _gsvi_models_payload(version)


@app.post("/infer_single")
async def gsvi_infer_single(payload: GSVIInferRequest, request: Request):
    tts_request = TTSRequest(
        text=payload.text,
        voice=payload.model_name,
        text_language=payload.text_lang,
        top_k=payload.top_k,
        top_p=payload.top_p,
        temperature=payload.temperature,
        repetition_penalty=payload.repetition_penalty,
        speed=payload.speed_facter
        if payload.speed_facter is not None
        else (payload.speed_factor if payload.speed_factor is not None else 1.0),
    )
    output_path = await _synthesize_to_file(tts_request)
    return {
        "msg": "success",
        "audio_url": str(request.url_for("audio", filename=output_path.name)),
    }


@app.post("/tts/batched")
async def tts_batched(request: TTSBatchedRequest):
    r = _resolve_synthesis(
        request.voice, request.gpt_model, request.sovits_model,
        request.speaker_audio, request.prompt_audio, request.prompt_text,
        request.text_language, request.prompt_language,
    )
    speaker_audio, prompt_audio, prompt_text = r["speaker_audio"], r["prompt_audio"], r["prompt_text"]
    text_language, prompt_language = r["text_language"], r["prompt_language"]

    def _run():
        with _infer_lock:
            tts = _get_tts()
            _ensure_model(tts, "gpt", r["gpt_model"])
            _ensure_model(tts, "sovits", r["sovits_model"])
            clips = tts.infer_batched(
                spk_audio_paths=speaker_audio,
                prompt_audio_paths=prompt_audio,
                prompt_audio_texts=prompt_text,
                texts=request.texts,
                text_languages=text_language,
                prompt_languages=prompt_language,
                return_subtitles=request.return_subtitles,
                is_cut_text=request.is_cut_text,
                cut_minlen=request.cut_minlen,
                cut_mute=request.cut_mute,
                top_k=request.top_k,
                top_p=request.top_p,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty,
                noise_scale=request.noise_scale,
                speed=request.speed,
                gpt_model=r["gpt_model"],
                sovits_model=r["sovits_model"],
            )

        filenames = []
        subtitles_list = []
        for clip in clips:
            filename = f"tts_{uuid.uuid4().hex}.wav"
            clip.save(str(OUTPUTS_DIR / filename))
            filenames.append(filename)
            if request.return_subtitles:
                subtitles_list.append(clip.subtitles)
        return filenames, subtitles_list

    try:
        filenames, subtitles_list = await asyncio.to_thread(_run)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result = {
        "success": True,
        "count": len(filenames),
        "filenames": filenames,
        "prompt_text_used": prompt_text,
        "text_language_used": text_language,
        "prompt_language_used": prompt_language,
        "voice_used": r["voice"],
        "gpt_model_used": r["gpt_model"],
        "sovits_model_used": r["sovits_model"],
    }
    if request.return_subtitles:
        result["subtitles"] = subtitles_list
    return result


@app.get("/audio/{filename}")
async def audio(filename: str):
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)
