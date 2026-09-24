import torch # type: ignore
from transformers import ( # type: ignore
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)


def load_vlm_for_challenge(
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading {model_id} in 4-bit...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_id)

    vram_gb = model.get_memory_footprint() / 1024**3

    print(
        f"Model loaded successfully. "
        f"Base memory footprint: {vram_gb:.2f} GB"
    )

    return model, processor


model, processor = load_vlm_for_challenge()