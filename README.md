# Vedaz AI Astrologer — Qwen2.5 LoRA Fine-Tune

## Overview

Fine-tunes Qwen/Qwen2.5-7B-Instruct with LoRA on Vedaz's Vedic astrologer

chat dataset, plus two write-ups as specified in the assessment.

## Repo structure

| Path | Description |

|---|---|

| data/original\_dataset\_cleaned\_35records.json | Source dataset, cleaned (raw file had malformed separators — parser in finetune\_qwen\_astrologer.py) |

| data/Chat\_Data\_raw.json | Original raw file as provided |

| finetune/finetune\_qwen\_astrologer.py | Standalone training script (parse → LoRA SFT → merge) |

| finetune/loss\_curve.png | Training loss curve |

| finetune/training\_logs.csv / .txt | Per-step loss logs |

| finetune/sample\_outputs\_before\_after.md | 5 prompts: base model vs fine-tuned, side by side |

| finetune/base\_outputs.json | Raw base model outputs |

| finetune/finetuned\_outputs.json | Raw fine-tuned model outputs |

| writeups/vps\_vllm\_hosting.md | Process for hosting the model on a VPS using vLLM |

| writeups/sample\_astrologer\_conversations.json | 5 manually written training conversations |

## How to reproduce

pip install torch transformers accelerate peft trl==0.19.1 bitsandbytes datasets

python finetune/finetune\_qwen\_astrologer.py
Or open finetune/finetune\_and\_evaluate.ipynb in Google Colab (GPU runtime).

## Note on LoRA adapter

The trained adapter weights are not included in this repo due to file size.

Run the training script to reproduce them locally.
