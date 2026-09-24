# test_lora.py

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_ADAPTER_PATH = "./qwen-lora-checkpoint"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True
USE_4BIT = True
DEVICE_MAP = "auto"
# ==============================================================================

def generate_response(model, tokenizer, instruction: str, user_input: str = "") -> str:
    """Formats prompt, runs model inference, and returns generated text."""
    prompt_content = f"{instruction}\n\nInput:\n{user_input}".strip() if user_input.strip() else instruction.strip()
    messages = [{"role": "user", "content": prompt_content}]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = f"User: {prompt_content}\nAssistant: "

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE if DO_SAMPLE else 1.0,
            top_p=TOP_P if DO_SAMPLE else 1.0,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the generated response tokens
    generated_tokens = outputs[0][inputs.input_ids.shape[-1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

def main():
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization Config
    bnb_config = None
    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"Loading Base Model: {MODEL_NAME}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config if USE_4BIT else None,
        device_map=DEVICE_MAP,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading LoRA Adapter from: {LORA_ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model.eval()

    # Predefined Test Prompts
    test_prompts = [
        {"instruction": "Explain quantum computing in simple terms.", "input": ""},
        {
            "instruction": "Summarize the given text into one key takeaway.",
            "input": "Fine-tuning large language models using Low-Rank Adaptation (LoRA) significantly reduces GPU memory usage while preserving baseline performance across downstream tasks.",
        },
        {"instruction": "Write a Python function to check if a number is prime.", "input": ""},
    ]

    print("\n" + "=" * 60)
    print("RUNNING PREDEFINED TEST PROMPTS")
    print("=" * 60)

    for idx, test in enumerate(test_prompts, start=1):
        print(f"\n[Test {idx}]")
        print(f"Instruction: {test['instruction']}")
        if test["input"]:
            print(f"Input:       {test['input']}")

        response = generate_response(model, tokenizer, test["instruction"], test["input"])
        print(f"Response:\n{response}")
        print("-" * 60)
        
    print("\nInference complete. Exiting script.")

if __name__ == "__main__":
    main()