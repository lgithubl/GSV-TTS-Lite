from __future__ import annotations

import sys
from pathlib import Path


HF_MODEL_DIRS = (
    "chinese-hubert-base",
    "chinese-roberta-wwm-ext-large",
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_hf_safetensors.py <models-dir>", file=sys.stderr)
        return 2

    root = Path(sys.argv[1])
    if not root.is_dir():
        raise NotADirectoryError(root)

    failures: list[str] = []
    for name in HF_MODEL_DIRS:
        for model_dir in root.rglob(name):
            if not model_dir.is_dir():
                continue
            if list(model_dir.rglob("pytorch_model.bin")):
                failures.append(f"{model_dir} still contains pytorch_model.bin")
            if (model_dir / "config.json").exists() and not (
                (model_dir / "model.safetensors").exists()
                or (model_dir / "cnroberta_int8_dynamic.onnx").exists()
            ):
                failures.append(f"{model_dir} has no model.safetensors or ONNX model")

    if failures:
        print("HuggingFace model package is not torch<2.6 safe:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"HuggingFace model package is safetensors/ONNX safe under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
