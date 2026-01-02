import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding


ABSTAIN = "I don't know based on the provided context."


def find_last_assistant_span(input_ids: List[int], tokenizer) -> int:
    raise NotImplementedError


def preprocess(example: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, Any]:
    messages = example["messages"]

    # Full conversation including assistant content
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Prefix up to assistant turn (generation prompt) => where assistant answer starts
    prefix_messages = messages[:-1]  # system + user
    prefix_text = tokenizer.apply_chat_template(prefix_messages + [{"role": "assistant", "content": ""}],
                                                tokenize=False,
                                                add_generation_prompt=True)

    full = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)
    prefix = tokenizer(prefix_text, truncation=True, max_length=max_length, add_special_tokens=False)

    input_ids = full["input_ids"]
    attn = full["attention_mask"]

    # Mask everything BEFORE assistant answer start
    start = len(prefix["input_ids"])
    labels = [-100] * len(input_ids)
    for i in range(start, len(input_ids)):
        labels[i] = input_ids[i]

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune a chat model for RAG-grounded book QA.")
    ap.add_argument("--data_path", required=True, help="Path to formatted JSONL from book_format.py")
    ap.add_argument("--base_model", default='meta-llama/Llama-3.2-1B-Instruct', help="HF base model id/path (e.g. meta-llama/..., mistralai/...)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # LoRA config: target modules varies by architecture; this default is common for Llama/Mistral.
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  
    )
    model = get_peft_model(model, lora)

    ds = load_dataset("json", data_files=args.data_path, split="train")

    tokenized = ds.map(
        lambda ex: preprocess(ex, tokenizer, args.max_length),
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    # Save adapter (recommended) + tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter + tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()
