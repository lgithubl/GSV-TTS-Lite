from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_with_transformers(bin_path: Path) -> bool:
    model_dir = bin_path.parent
    if not (model_dir / "config.json").exists():
        return False

    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, HubertModel

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if config.model_type == "hubert":
        model = HubertModel.from_pretrained(model_dir, local_files_only=True)
    elif config.model_type in {"bert", "roberta"}:
        model = AutoModelForMaskedLM.from_pretrained(model_dir, local_files_only=True)
    else:
        model = AutoModel.from_pretrained(model_dir, local_files_only=True)

    model.save_pretrained(model_dir, safe_serialization=True)
    bin_path.unlink()
    print(f"converted {bin_path} with transformers save_pretrained")
    return True


def convert_bin(bin_path: Path) -> None:
    target = bin_path.with_name("model.safetensors")
    if target.exists():
        bin_path.unlink()
        print(f"kept existing {target}; removed {bin_path}")
        return

    if convert_with_transformers(bin_path):
        return

    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"{bin_path} did not contain a state dict")
    non_tensor_keys = [key for key, value in state.items() if not torch.is_tensor(value)]
    if non_tensor_keys:
        raise TypeError(
            f"{bin_path} contained non-tensor values: {', '.join(non_tensor_keys[:10])}"
        )

    save_file(state, target, metadata={"format": "pt"})
    bin_path.unlink()
    print(f"converted {bin_path} -> {target}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: convert_hf_bins_to_safetensors.py <models-dir>", file=sys.stderr)
        return 2

    root = Path(sys.argv[1])
    if not root.is_dir():
        raise NotADirectoryError(root)

    bins = sorted(root.rglob("pytorch_model.bin"))
    if not bins:
        print(f"no pytorch_model.bin files found under {root}")
        return 0

    for bin_path in bins:
        convert_bin(bin_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
