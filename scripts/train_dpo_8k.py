#!/usr/bin/env python3
"""
BioReasoner DPO Training Script (8K Context, Flash Attention 2)

Distillation mode: Always use Teacher as CHOSEN, Student as REJECTED.

Optimized for RTX 5090 (24GB VRAM) with:
- Flash Attention 2 for efficient 8K context
- Gradient checkpointing to fit in VRAM
- Very low learning rate for stable DPO

Usage:
    python scripts/train_dpo_8k.py \
        --base_model models/bioreasoner-sft \
        --train_data data/dpo_distillation_pairs.jsonl \
        --output_dir models/bioreasoner-dpo \
        --beta 0.1 \
        --epochs 1
"""

import argparse
import json
import torch
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer


SYSTEM_PROMPT = """You are BioReasoner, a scientific hypothesis generator specialized in biomedical research.

Your task is to analyze scientific papers and generate novel hypotheses by:
1. Reasoning step-by-step inside <think> tags
2. Citing specific PMIDs from the provided papers
3. Synthesizing findings across multiple papers
4. Proposing testable hypotheses grounded in the literature

Always structure your response with explicit reasoning followed by a clear hypothesis."""


def load_dpo_data(data_path: str) -> Dataset:
    """Load DPO pairs into HuggingFace Dataset format."""
    samples = []
    with open(data_path) as f:
        for line in f:
            d = json.loads(line)

            # Format prompt with system message
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{d['prompt']}<|im_end|>\n<|im_start|>assistant\n"

            samples.append({
                "prompt": prompt,
                "chosen": d["chosen"],
                "rejected": d["rejected"],
            })

    print(f"Loaded {len(samples)} DPO pairs")
    return Dataset.from_list(samples)


def load_sft_model(model_path: str, base_model_override: str = None, use_4bit: bool = True):
    """Load the SFT model with 4-bit quantization for DPO memory efficiency."""
    print(f"Loading SFT model from {model_path}...")

    # 4-bit quantization config
    bnb_config = None
    if use_4bit:
        print("Using 4-bit quantization for memory efficiency...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    adapter_config = Path(model_path) / "adapter_config.json"

    if adapter_config.exists():
        # LoRA adapter - load base (don't merge, DPO will add new adapter)
        with open(adapter_config) as f:
            config = json.load(f)
        base_model = base_model_override or config.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")

        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            trust_remote_code=True,
        )

        if use_4bit:
            model = prepare_model_for_kbit_training(model)

        print(f"Loading and merging LoRA adapter...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        # Full model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train BioReasoner with DPO")
    parser.add_argument("--base_model", type=str, required=True, help="Path to SFT model")
    parser.add_argument("--base_model_override", type=str, default=None)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/bioreasoner-dpo")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print("=" * 60)
    print("BioReasoner DPO Training (Distillation Mode)")
    print("=" * 60)
    print(f"SFT model: {args.base_model}")
    print(f"Beta (KL penalty): {args.beta}")
    print(f"Max length: {args.max_length}")
    print(f"Effective batch: {args.batch_size * args.gradient_accumulation}")
    print("=" * 60)

    # Load model
    model, tokenizer = load_sft_model(args.base_model, args.base_model_override)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # With PEFT/LoRA, TRL computes reference model internally
    # No need to load separately
    ref_model = None
    print("Using implicit reference model (PEFT mode)")

    # LoRA config for DPO (smaller rank for stability)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load training data
    print(f"\nLoading DPO data from {args.train_data}...")
    train_dataset = load_dpo_data(args.train_data)

    # DPO training config
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )

    # Initialize DPO trainer
    print("\nInitializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,  # TRL 0.24+ API
        peft_config=peft_config,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting DPO training...")
    print("=" * 60)

    start_time = datetime.now()
    trainer.train()
    end_time = datetime.now()

    print("\n" + "=" * 60)
    print(f"Training completed in {end_time - start_time}")
    print("=" * 60)

    # Save
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save config
    config_path = Path(args.output_dir) / "dpo_training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Done! Model saved to: {args.output_dir}")

    # Cleanup
    del model, ref_model, trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
