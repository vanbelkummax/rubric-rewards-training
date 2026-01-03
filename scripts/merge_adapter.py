#!/usr/bin/env python3
"""
Merge a LoRA adapter into a base model and export as SafeTensors.
"""
import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER_PATH = "models/bioreasoner-dpo"
DEFAULT_OUTPUT_DIR = "models/bioreasoner-2.0-merged"


def resolve_device_map(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "auto"
    if device == "auto":
        return "auto" if torch.cuda.is_available() else "cpu"
    raise ValueError(f"Unsupported device setting: {device}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    device_map = resolve_device_map(args.device)
    dtype = torch.bfloat16

    print("=" * 60)
    print("MERGING LORA ADAPTER INTO BASE MODEL")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {adapter_path}")
    print(f"Output: {args.output}")
    print(f"Device map: {device_map}")
    print(f"DType: {dtype}")

    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print("[2/5] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    print("[3/5] Loading adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("[4/5] Merging weights...")
    model = model.merge_and_unload()

    print("[5/5] Saving merged model...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    total_size = sum(
        os.path.getsize(output_dir / f)
        for f in os.listdir(output_dir)
        if (output_dir / f).is_file()
    )

    print("\n" + "=" * 60)
    print("✅ MERGE COMPLETE")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Size: {total_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
