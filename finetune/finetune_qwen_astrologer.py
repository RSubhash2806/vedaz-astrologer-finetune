"""
Fine-tune Qwen2.5 (or Qwen3) on the Vedaz AI Astrologer chat dataset using LoRA.

Pipeline
--------
1. Robustly parse the provided chat-data file (it is not valid single-document
   JSON / JSONL — records are concatenated with inconsistent separators, some
   real newlines, some literal "\n" characters, no enclosing array). We recover
   every record by locating each `{"messages": ...}` block and decoding it
   independently with json.JSONDecoder.raw_decode.
2. Convert each record into a single training string using the target model's
   chat template (so special tokens / turn markers match what the base model
   expects).
3. LoRA fine-tune with TRL's SFTTrainer (memory-efficient, works on a single
   24GB+ GPU for the 7B model; use the 1.5B/3B variant on smaller GPUs).
4. Save the LoRA adapter, then merge it into the base model to produce a
   single deployable checkpoint (this merged checkpoint is what you point
   vLLM at — see the accompanying hosting write-up).

Requirements
------------
pip install --break-system-packages torch transformers accelerate peft trl \
    bitsandbytes datasets

Tested against: transformers>=4.46, trl>=0.12, peft>=0.13
GPU requirement: this step needs an actual GPU (16GB+ VRAM for 7B in 4-bit,
24GB+ for 7B in bf16 LoRA). It will not run meaningfully on CPU.
"""

import json
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Config — edit these for your run
# ---------------------------------------------------------------------------
RAW_DATA_PATH = "Chat_Data_for_assessment_of_applicants.json"
CLEAN_JSONL_PATH = "vedaz_astrologer_sft.jsonl"

# Swap for "Qwen/Qwen3-8B" (or the size that fits your VPS GPU) once available
# in your environment. Qwen2.5-7B-Instruct is the safe default used below.
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

OUTPUT_DIR = "qwen-vedaz-astrologer-lora"
MERGED_OUTPUT_DIR = "qwen-vedaz-astrologer-merged"

MAX_SEQ_LEN = 2048
USE_4BIT = True  # set False if you have enough VRAM for full bf16 LoRA


# ---------------------------------------------------------------------------
# Step 1: Robustly recover records from the malformed source file
# ---------------------------------------------------------------------------
def parse_messy_chat_file(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    starts = [m.start() for m in re.finditer(r'\{\s*"messages"', text)]
    decoder = json.JSONDecoder(strict=False)  # tolerate stray control chars
    records = []
    for s in starts:
        obj, _ = decoder.raw_decode(text, s)
        records.append(obj)
    return records


def clean_and_dedupe(records: list[dict]) -> list[dict]:
    seen = set()
    cleaned = []
    for r in records:
        msgs = r.get("messages", [])
        if not msgs or msgs[0]["role"] != "system":
            continue
        key = json.dumps(msgs, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"messages": msgs})
    return cleaned


def build_clean_jsonl(raw_path: str, out_path: str) -> int:
    records = parse_messy_chat_file(raw_path)
    records = clean_and_dedupe(records)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


# ---------------------------------------------------------------------------
# Step 2: Load + format with the model's chat template
# ---------------------------------------------------------------------------
def load_dataset_with_template(jsonl_path: str, tokenizer) -> Dataset:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = tokenizer.apply_chat_template(
                rec["messages"], tokenize=False, add_generation_prompt=False
            )
            rows.append({"text": text})
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Step 3: Train
# ---------------------------------------------------------------------------
def main():
    n = build_clean_jsonl(RAW_DATA_PATH, CLEAN_JSONL_PATH)
    print(f"Recovered {n} clean training conversations -> {CLEAN_JSONL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    train_dataset = load_dataset_with_template(CLEAN_JSONL_PATH, tokenizer)
    print(f"Training on {len(train_dataset)} examples")

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,  # keep conversations un-packed; dataset is small
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}")

    # -----------------------------------------------------------------
    # Step 4: Merge LoRA into base weights -> single checkpoint for vLLM
    # -----------------------------------------------------------------
    print("Merging LoRA adapter into base model for deployment...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    merged = PeftModel.from_pretrained(base, OUTPUT_DIR)
    merged = merged.merge_and_unload()
    merged.save_pretrained(MERGED_OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    print(f"Merged model ready for vLLM at: {MERGED_OUTPUT_DIR}")


if __name__ == "__main__":
    main()