#!/usr/bin/env python3

"""
Reusable QLoRA training harness for Qwen2.5-VL.

Two modes:

1. Dummy smoke test:
       python3 qlora_train.py --dummy

2. Real JSONL dataset:
       python3 qlora_train.py --data data/train.jsonl

Expected JSONL format:

{"prompt": "What is 2 + 2?", "response": "4"}
{"prompt": "Explain photosynthesis.", "response": "Photosynthesis is..."}

The same harness can be adapted for multimodal examples later.

The base model remains frozen.
Only LoRA adapter parameters are trained.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import torch # type: ignore
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training # type: ignore
from transformers import ( # type: ignore
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

DEFAULT_OUTPUT_DIR = "./checkpoints/qwen2.5-vl-qlora"

DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05

DEFAULT_MAX_LENGTH = 2048

DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUMULATION = 8
DEFAULT_LEARNING_RATE = 2e-4


# ============================================================
# Dummy dataset
# ============================================================

DUMMY_DATA = [
    {
        "prompt": "What is 2 + 2?",
        "response": "2 + 2 equals 4.",
    },
    {
        "prompt": "What is the capital of France?",
        "response": "The capital of France is Paris.",
    },
    {
        "prompt": "Explain machine learning in one sentence.",
        "response": (
            "Machine learning is a method where models learn patterns "
            "from data to make predictions or decisions."
        ),
    },
    {
        "prompt": "What is a GPU?",
        "response": (
            "A GPU is a processor designed to perform many computations "
            "in parallel and is widely used for graphics and machine learning."
        ),
    },
    {
        "prompt": "Why is quantization useful for large language models?",
        "response": (
            "Quantization reduces the numerical precision of model weights, "
            "which can significantly reduce memory usage and sometimes improve "
            "inference efficiency."
        ),
    },
    {
        "prompt": "What does QLoRA mean?",
        "response": (
            "QLoRA combines 4-bit quantization of a pretrained model with "
            "LoRA adapters so that large models can be fine-tuned using much "
            "less GPU memory."
        ),
    },
]


# ============================================================
# Dataset utilities
# ============================================================

def load_dummy_dataset() -> Dataset:
    return Dataset.from_list(DUMMY_DATA)


def load_jsonl(path: str) -> Dataset:
    records: list[dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if "prompt" not in item:
                raise ValueError(
                    f"Line {line_number} is missing 'prompt'."
                )

            if "response" not in item:
                raise ValueError(
                    f"Line {line_number} is missing 'response'."
                )

            records.append(
                {
                    "prompt": str(item["prompt"]),
                    "response": str(item["response"]),
                }
            )

    if not records:
        raise ValueError(f"No records found in {path}")

    return Dataset.from_list(records)


# ============================================================
# QLoRA configuration
# ============================================================

def create_quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def create_lora_config(
    r: int = DEFAULT_LORA_R,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
) -> LoraConfig:

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",

        # Standard Transformer projection layers.
        #
        # If the hackathon model differs, inspect its architecture
        # and change this list.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


# ============================================================
# Model
# ============================================================

class QLoRAModel:
    """
    Handles:

        FP model
          ↓
        4-bit NF4 quantization
          ↓
        prepare for k-bit training
          ↓
        LoRA adapters
          ↓
        trainable model
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        lora_r: int = DEFAULT_LORA_R,
        lora_alpha: int = DEFAULT_LORA_ALPHA,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
    ):

        self.model_id = model_id

        print("=" * 70)
        print("Loading processor")
        print("=" * 70)

        self.processor = AutoProcessor.from_pretrained(
            model_id
        )

        print("=" * 70)
        print("Creating 4-bit quantization configuration")
        print("=" * 70)

        quantization_config = create_quantization_config()

        print("=" * 70)
        print("Loading base model")
        print("=" * 70)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        print("=" * 70)
        print("Preparing model for QLoRA")
        print("=" * 70)

        self.model = prepare_model_for_kbit_training(
            self.model
        )

        print("=" * 70)
        print("Adding LoRA adapters")
        print("=" * 70)

        lora_config = create_lora_config(
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        self.model.add_adapter(lora_config)

        self._print_trainable_parameters()

    def _print_trainable_parameters(self) -> None:

        trainable = 0
        total = 0

        for parameter in self.model.parameters():
            total += parameter.numel()

            if parameter.requires_grad:
                trainable += parameter.numel()

        percentage = (
            100.0 * trainable / total
            if total
            else 0.0
        )

        print()
        print("=" * 70)
        print("Trainable parameters")
        print("=" * 70)
        print(f"Trainable: {trainable:,}")
        print(f"Total:     {total:,}")
        print(f"Percentage: {percentage:.4f}%")
        print()


# ============================================================
# Tokenization
# ============================================================

class QLoRACollator:
    """
    Converts:

        prompt + response

    into:

        input_ids
        attention_mask
        labels

    The prompt tokens are masked with -100 so the loss is
    primarily calculated on the response.
    """

    def __init__(
        self,
        processor,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.processor = processor
        self.max_length = max_length

    def _build_messages(
        self,
        prompt: str,
        response: str | None = None,
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        if response is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": response,
                        }
                    ],
                }
            )

        return messages

    def __call__(self, examples):

        input_ids_list = []
        attention_masks = []
        labels_list = []

        for example in examples:

            prompt = example["prompt"]
            response = example["response"]

            full_messages = self._build_messages(
                prompt,
                response,
            )

            prompt_messages = self._build_messages(
                prompt
            )

            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full = self.processor.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

            prompt_tokens = self.processor.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

            input_ids = full["input_ids"]
            attention_mask = full["attention_mask"]

            labels = input_ids.copy()

            prompt_length = min(
                len(prompt_tokens["input_ids"]),
                len(labels),
            )

            for i in range(prompt_length):
                labels[i] = -100

            input_ids_list.append(input_ids)
            attention_masks.append(attention_mask)
            labels_list.append(labels)

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        pad_token_id = self.processor.tokenizer.pad_token_id

        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id

        max_len = max(
            len(ids)
            for ids in input_ids_list
        )

        padded_input_ids = []
        padded_attention = []
        padded_labels = []

        for input_ids, attention, labels in zip(
            input_ids_list,
            attention_masks,
            labels_list,
        ):

            padding = max_len - len(input_ids)

            padded_input_ids.append(
                input_ids + [pad_token_id] * padding
            )

            padded_attention.append(
                attention + [0] * padding
            )

            padded_labels.append(
                labels + [-100] * padding
            )

        return {
            "input_ids": torch.tensor(
                padded_input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                padded_attention,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                padded_labels,
                dtype=torch.long,
            ),
        }


# ============================================================
# Trainer
# ============================================================

class QLoRATrainer:
    """
    High-level training class.

    Usage:

        trainer = QLoRATrainer(...)
        trainer.train()
    """

    def __init__(
        self,
        model: QLoRAModel,
        dataset: Dataset,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        gradient_accumulation: int = DEFAULT_GRAD_ACCUMULATION,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):

        self.model_wrapper = model
        self.model = model.model
        self.processor = model.processor

        self.output_dir = output_dir

        # ----------------------------------------------------
        # Train/validation split
        # ----------------------------------------------------

        if len(dataset) >= 10:

            split = dataset.train_test_split(
                test_size=0.1,
                seed=42,
            )

            self.train_dataset = split["train"]
            self.eval_dataset = split["test"]

        else:

            # Tiny dummy dataset.
            # Keep a small validation set.
            split = dataset.train_test_split(
                test_size=1,
                seed=42,
            )

            self.train_dataset = split["train"]
            self.eval_dataset = split["test"]

        # ----------------------------------------------------
        # Training arguments
        # ----------------------------------------------------

        self.training_args = TrainingArguments(
            output_dir=output_dir,

            num_train_epochs=epochs,

            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,

            gradient_accumulation_steps=gradient_accumulation,

            learning_rate=learning_rate,

            bf16=True,

            gradient_checkpointing=True,

            optim="paged_adamw_8bit",

            logging_steps=1,

            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,

            eval_strategy="steps",
            eval_steps=50,

            report_to="none",

            remove_unused_columns=False,

            dataloader_pin_memory=True,

            # Useful when experimenting with longer sequences.
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            },
        )

        self.collator = QLoRACollator(
            processor=self.processor,
            max_length=max_length,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.collator,
        )

    def train(self, resume_from_checkpoint: str | None = None):

        print("=" * 70)
        print("Starting QLoRA training")
        print("=" * 70)

        print(f"Train examples: {len(self.train_dataset)}")
        print(f"Eval examples:  {len(self.eval_dataset)}")
        print(f"Output:         {self.output_dir}")

        if torch.cuda.is_available():

            print()
            print(
                "GPU:",
                torch.cuda.get_device_name(0),
            )

            print(
                "VRAM:",
                round(
                    torch.cuda.get_device_properties(0).total_memory
                    / 1024**3,
                    2,
                ),
                "GB",
            )

        print()

        self.trainer.train(
            resume_from_checkpoint=resume_from_checkpoint
        )

        print()
        print("=" * 70)
        print("Saving LoRA adapter")
        print("=" * 70)

        self.trainer.save_model(
            self.output_dir
        )

        self.processor.save_pretrained(
            self.output_dir
        )

        print()
        print("Training complete.")
        print(
            f"Adapter saved to: {self.output_dir}"
        )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Reusable QLoRA training harness"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--data",
        default=None,
        help="JSONL dataset",
    )

    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use built-in dummy dataset",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=DEFAULT_GRAD_ACCUMULATION,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
    )

    parser.add_argument(
        "--lora-r",
        type=int,
        default=DEFAULT_LORA_R,
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=DEFAULT_LORA_ALPHA,
    )

    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=DEFAULT_LORA_DROPOUT,
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint directory to resume from",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    if args.data:

        print(
            f"Loading dataset: {args.data}"
        )

        dataset = load_jsonl(
            args.data
        )

    elif args.dummy:

        print("Using built-in dummy dataset.")

        dataset = load_dummy_dataset()

    else:

        print(
            "No dataset supplied."
        )

        print()
        print(
            "Use:"
        )
        print(
            "  python3 qlora_train.py --dummy"
        )
        print()
        print(
            "or:"
        )
        print(
            "  python3 qlora_train.py --data data/train.jsonl"
        )

        return

    print(
        f"Dataset size: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = QLoRAModel(
        model_id=args.model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = QLoRATrainer(
        model=model,
        dataset=dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    trainer.train(
        resume_from_checkpoint=args.resume
    )


if __name__ == "__main__":
    main()

