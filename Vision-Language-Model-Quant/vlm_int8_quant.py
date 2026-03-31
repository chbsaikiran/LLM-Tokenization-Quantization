import gc
import time
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.image_utils import load_image

try:
    from transformers import AutoModelForVision2Seq as VLMModelAutoClass
except ImportError:
    from transformers import AutoModelForImageTextToText as VLMModelAutoClass

# Chosen for 6GB GPUs: very small VLM with strong HF support.
MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"
PROMPT_TEXT = "Describe this image in 3 short bullet points."
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
MAX_NEW_TOKENS = 64

processor = AutoProcessor.from_pretrained(MODEL_NAME)
image = load_image(IMAGE_URL)

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=False,
)


def _format_bytes(n: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def _model_nbytes(model: torch.nn.Module) -> int:
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if getattr(t, "device", None) is not None and str(t.device) == "meta":
            continue
        try:
            total += t.numel() * t.element_size()
        except Exception:
            pass
    return int(total)


def _print_cuda_mem(prefix: str):
    if not torch.cuda.is_available():
        return
    print(f"{prefix} CUDA allocated:", _format_bytes(torch.cuda.memory_allocated()))
    print(f"{prefix} CUDA reserved:", _format_bytes(torch.cuda.memory_reserved()))


def _first_non_meta_param_device(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        if getattr(p, "device", None) is not None and str(p.device) != "meta":
            return p.device
    return torch.device("cpu")


def _build_inputs(model: torch.nn.Module):
    # SmolVLM chat template expects an image placeholder in content.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    batch = processor(text=prompt, images=[image], return_tensors="pt")
    return batch.to(_first_non_meta_param_device(model))


def run_inference(model: torch.nn.Module):
    inputs = _build_inputs(model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return text, elapsed


def compute_loss(model: torch.nn.Module):
    inputs = _build_inputs(model)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item()


print("Loading FP16 VLM...")
model_fp16 = VLMModelAutoClass.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map={"": 0} if torch.cuda.is_available() else "cpu",
)
print("FP16 model (params+buffers):", _format_bytes(_model_nbytes(model_fp16)))
_print_cuda_mem("After FP16 load")
out_fp16, time_fp16 = run_inference(model_fp16)
loss_fp16 = compute_loss(model_fp16)
print("FP16 Time:", time_fp16)
print("Loss FP16:", loss_fp16)
print("\n=== FP16 Output ===")
print(out_fp16)
print("=== End FP16 Output ===\n")

del model_fp16
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Loading INT8 VLM (bitsandbytes)...")
model_int8 = VLMModelAutoClass.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map={"": 0} if torch.cuda.is_available() else "cpu",
)
print("INT8 model (params+buffers):", _format_bytes(_model_nbytes(model_int8)))
_print_cuda_mem("After INT8 load")
out_int8, time_int8 = run_inference(model_int8)
loss_int8 = compute_loss(model_int8)
print("INT8 Time:", time_int8)
print("Loss INT8:", loss_int8)
print("\n=== INT8 Output ===")
print(out_int8)
print("=== End INT8 Output ===\n")
