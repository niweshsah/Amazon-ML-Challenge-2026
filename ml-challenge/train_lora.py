# train_lora.py

import os
import json
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./qwen-lora-checkpoint"
DATASET_PATH = ""  # Path to JSON/JSONL file; if empty or missing, dummy data is used
MAX_SEQ_LENGTH = 1024
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
LOGGING_STEPS = 1
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 2
SEED = 42
FP16 = False
BF16 = True
GRADIENT_CHECKPOINTING = True
USE_4BIT = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
OPTIMIZER = "paged_adamw_8bit"
LR_SCHEDULER = "cosine"
RESUME_FROM_CHECKPOINT = None
REPORT_TO = "none"
DEVICE_MAP = "auto"
# ==============================================================================


def create_dummy_dataset() -> Dataset:
    """Generates a small instruction-following dataset for sanity checking."""
    dummy_records = [
        {
            "instruction": "Explain quantum computing in simple terms.",
            "input": "",
            "output": "Quantum computing uses quantum bits or qubits, which can exist in multiple states at once, allowing computers to process complex information much faster than standard binary computers.",
        },
        {
            "instruction": "Summarize the given text into one key takeaway.",
            "input": "Fine-tuning large language models using Low-Rank Adaptation (LoRA) significantly reduces GPU memory usage while preserving baseline performance across downstream tasks.",
            "output": "LoRA enables memory-efficient fine-tuning without significant loss in model performance.",
        },
        {
            "instruction": "Write a Python function to check if a number is prime.",
            "input": "",
            "output": "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        },
        {
            "instruction": "Translate the following English sentence into French.",
            "input": "Machine learning enables computers to learn from data.",
            "output": "L'apprentissage automatique permet aux ordinateurs d'apprendre à partir de données.",
        },
    ]
    return Dataset.from_list(dummy_records)


def count_parameters(model):
    """Calculates and reports trainable vs. total parameters."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    perc = 100 * trainable_params / all_params if all_params > 0 else 0
    return trainable_params, all_params, perc


def main():
    
    set_seed(SEED)

    # 1. Dataset Initialization
    if DATASET_PATH and os.path.exists(DATASET_PATH):
        print(f"Loading custom dataset from: {DATASET_PATH}")
        if DATASET_PATH.endswith(".jsonl") or DATASET_PATH.endswith(".json"):
            raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
        else:
            raise ValueError("DATASET_PATH must point to a valid .json or .jsonl file.")
    else:
        print("No valid DATASET_PATH found. Initializing dummy dataset for sanity check.")
        raw_dataset = create_dummy_dataset()

    # 2. Tokenizer Setup
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True, #this is important for Qwen models and other models with custom tokenizers, it allows the tokenizer to load any custom code needed for proper tokenization
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Model & Quantization Configuration
    bnb_config = None
    compute_dtype = torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32)

    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config if USE_4BIT else None,
        device_map=DEVICE_MAP,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    if GRADIENT_CHECKPOINTING:
        base_model.gradient_checkpointing_enable()

    if USE_4BIT:
        base_model = prepare_model_for_kbit_training(base_model)

    # 4. PEFT (LoRA) Setup
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(base_model, peft_config)

    # 5. Preprocessing & Tokenization Logic
    def preprocess_function(examples):
        input_ids_list = []
        
        # Attention masks are used to indicate which tokens should be attended to by the model. # This is important for handling variable-length sequences and padding. The attention mask has the same length as the input IDs, with 1s for tokens that should be attended to and 0s for padding tokens.
        attention_mask_list = []
        labels_list = []

        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]

        for inst, user_in, out in zip(instructions, inputs, outputs):
            
            # The user_content is constructed by combining the instruction and input. 
            # example: 
            #  Instruction: "Summarize the text."
            #  Input: "This is a long text that needs to be summarized."
            #  user_content: "Summarize the text.\n\nInput:\nThis is a long text that needs to be summarized."
            
            
            user_content = f"{inst}\n\nInput:\n{user_in}".strip() if user_in and str(user_in).strip() else str(inst).strip()
            
            
            user_msg = [{"role": "user", "content": user_content}]
            full_msg = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": str(out).strip()},
            ]
            
            # The casual models are trained like next-token prediction, so we need to construct the prompt and full message strings accordingly.
            # we cannot just use the output and compare it with proper output, we need to construct the prompt and full message strings fully.
            
            # If the tokenizer has a chat template, we use it to format the messages. Otherwise, we fall back to a simple "User: ... Assistant: ..." format.
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                prompt_str = tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
                full_str = tokenizer.apply_chat_template(full_msg, tokenize=False, add_generation_prompt=False)
            else:
                prompt_str = f"User: {user_content}\nAssistant: "
                full_str = f"User: {user_content}\nAssistant: {out}"

             # This tokenization step converts the prompt and full message strings into input IDs and attention masks, which are used for model training. The prompt IDs are used to determine which tokens should be masked in the labels, so that the loss is only calculated on the assistant's response.
            prompt_ids = tokenizer(prompt_str, max_length=MAX_SEQ_LENGTH, truncation=True, add_special_tokens=False)["input_ids"]
            full_enc = tokenizer(full_str, max_length=MAX_SEQ_LENGTH, truncation=True, add_special_tokens=False)

            full_ids = full_enc["input_ids"]
            full_mask = full_enc["attention_mask"]

            # Construct labels: mask user prompt tokens with -100 to calculate loss only on response
            labels = list(full_ids)
            prompt_len = min(len(prompt_ids), len(labels))
            
            # Mask the prompt tokens in the labels with -100 so that the loss is only computed on the assistant's response. This is important for training the model to generate appropriate responses without being penalized for the prompt tokens.
            labels[:prompt_len] = [-100] * prompt_len

            input_ids_list.append(full_ids)
            attention_mask_list.append(full_mask)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
        }

    dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing dataset",
    )

    # 6. Print Training Details
    trainable_p, total_p, trainable_perc = count_parameters(model)
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Model Name:                 {MODEL_NAME}")
    print(f"Dataset Path:               {DATASET_PATH if DATASET_PATH else 'Built-in Dummy Dataset'}")
    print(f"Dataset Size:               {len(dataset)} examples")
    print(f"Trainable Parameters:       {trainable_p:,}")
    print(f"Total Parameters:           {total_p:,}")
    print(f"Trainable Percentage:       {trainable_perc:.4f}%")
    print(f"LoRA Rank (r):              {LORA_R}")
    print(f"LoRA Alpha:                 {LORA_ALPHA}")
    print(f"LoRA Dropout:               {LORA_DROPOUT}")
    print(f"Target Modules:             {LORA_TARGET_MODULES}")
    print(f"Precision:                  BF16={BF16}, FP16={FP16}")
    print(f"4-Bit Quantization (QLoRA): {USE_4BIT}")
    print(f"Batch Size (Per Device):    {BATCH_SIZE}")
    print(f"Gradient Accumulation:      {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning Rate:              {LEARNING_RATE}")
    print(f"Epochs:                     {NUM_EPOCHS}")
    print(f"Output Directory:           {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    # 7. Training Arguments & Trainer Setup
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        # warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        fp16=FP16,
        bf16=BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        optim=OPTIMIZER,
        lr_scheduler_type=LR_SCHEDULER,
        report_to=REPORT_TO,
        seed=SEED,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 8. Train & Save Adapter Only
    print("Starting fine-tuning...")
    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

    print(f"\nSaving LoRA adapter and tokenizer to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Success! Trained adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()