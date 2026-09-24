from __future__ import annotations

import asyncio
import os
import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
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

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GSV-TTS-Lite Web", version="0.1.0")
_tts = None
_tts_lock = threading.Lock()


def _env_flag_enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


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

        auto_download = _env_flag_enabled("GSV_TTS_AUTO_DOWNLOAD", False)
        tts = TTS(
            models_dir=str(MODELS_DIR),
            auto_download_models=auto_download,
            use_bert=_env_flag_enabled("GSV_TTS_USE_BERT", False),
            use_flash_attn=_env_flag_enabled("GSV_TTS_USE_FLASH_ATTN", False),
        )
        if _env_flag_enabled("GSV_TTS_LOAD_DEFAULT_WEIGHTS", True):
            tts.load_gpt_model()
            tts.load_sovits_model()
        _tts = tts
        return _tts


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    speaker_audio: str | None = None
    prompt_audio: str | None = None
    prompt_text: str | None = None
    text_language: Literal["auto", "ja", "zh", "en"] = "auto"
    prompt_language: Literal["auto", "ja", "zh", "en"] = "auto"
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.35
    noise_scale: float = 0.5
    speed: float = 1.0


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
    speaker_audio = request.speaker_audio or DEFAULT_SPEAKER_AUDIO
    prompt_audio = request.prompt_audio or DEFAULT_PROMPT_AUDIO
    prompt_text = request.prompt_text if request.prompt_text else DEFAULT_PROMPT_TEXT
    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt_text is required unless DEFAULT_PROMPT_TEXT is set.")
    if not Path(speaker_audio).exists():
        raise HTTPException(status_code=400, detail=f"speaker_audio not found: {speaker_audio}")
    if not Path(prompt_audio).exists():
        raise HTTPException(status_code=400, detail=f"prompt_audio not found: {prompt_audio}")

    def _run():
        tts = _get_tts()
        clip = tts.infer(
            spk_audio_path=speaker_audio,
            prompt_audio_path=prompt_audio,
            prompt_audio_text=prompt_text,
            text=request.text,
            text_language=request.text_language,
            prompt_language=request.prompt_language,
            top_k=request.top_k,
            top_p=request.top_p,
            temperature=request.temperature,
            repetition_penalty=request.repetition_penalty,
            noise_scale=request.noise_scale,
            speed=request.speed,
        )
        output = OUTPUTS_DIR / f"tts_{uuid.uuid4().hex}.wav"
        clip.save(str(output))
        return output

    try:
        return await asyncio.to_thread(_run)
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
    <label>Speaker audio path</label>
    <input id="speaker" value="/refs/speaker.wav" />
    <label>Prompt audio path</label>
    <input id="prompt" value="/refs/prompt.wav" />
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
      body: JSON.stringify({
        text: document.getElementById("text").value,
        speaker_audio: document.getElementById("speaker").value,
        prompt_audio: document.getElementById("prompt").value,
        prompt_text: document.getElementById("promptText").value
      })
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
        **_model_status(),
    }


@app.post("/api/tts")
async def synthesize(request: TTSRequest):
    output_path = await _synthesize_to_file(request)
    return FileResponse(output_path, media_type="audio/wav", filename=output_path.name)


@app.get("/tts")
async def synthesize_gptsovits_compatible(
    text: str,
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
        speaker_audio=speaker_audio or ref_audio,
        prompt_audio=ref_audio,
        prompt_text=prompt_text,
        text_language=_normalize_language(text_language or text_lang, DEFAULT_TEXT_LANGUAGE),
        prompt_language=_normalize_language(prompt_language or prompt_lang, DEFAULT_PROMPT_LANGUAGE),
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        noise_scale=noise_scale,
        speed=speed_factor if speed_factor is not None else (speed if speed is not None else 1.0),
    )
    output_path = await _synthesize_to_file(request)
    return FileResponse(output_path, media_type="audio/wav", filename=output_path.name)


@app.get("/audio/{filename}")
async def audio(filename: str):
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(path, media_type="audio/wav", filename=filename)
