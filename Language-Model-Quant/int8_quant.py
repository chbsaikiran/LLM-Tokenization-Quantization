import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import time
import gc
# from evaluate import load

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

model_name = DEFAULT_MODEL_NAME
prompt = "Explain DSP optimization in simple terms."
max_new_tokens = 64

tokenizer = AutoTokenizer.from_pretrained(model_name)

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # Keep INT8 weights on GPU (avoid CPU offload paths).
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
            # Some quantized parameter wrappers may not expose element_size cleanly.
            pass
    return int(total)


def _print_cuda_mem(prefix: str):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"{prefix} CUDA allocated:", _format_bytes(allocated))
    print(f"{prefix} CUDA reserved:", _format_bytes(reserved))


def _first_non_meta_param_device(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        if getattr(p, "device", None) is not None and str(p.device) != "meta":
            return p.device
    return torch.device("cpu")


def _inputs_for_model(model: torch.nn.Module, text: str):
    batch = tokenizer(text, return_tensors="pt")
    device = _first_non_meta_param_device(model)
    return batch.to(device)

def run_inference(model, text: str):
    inputs = _inputs_for_model(model, text)
    start = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=False,  # reduce VRAM pressure on small GPUs
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    return tokenizer.decode(outputs[0]), end - start

def compute_loss(model, text):
    batch = _inputs_for_model(model, text)
    with torch.no_grad():
        outputs = model(**batch, labels=batch["input_ids"])
    return outputs.loss.item()

# perplexity = load("perplexity")

texts = [
    "Digital signal processing improves efficiency.",
    "Quantization reduces model size."
]

# ppl_fp16 = perplexity.compute(
#     model_id=model_name,
#     predictions=texts
# )

# print("Perplexity FP16:", ppl_fp16)


def _run_int8():
    print("Loading INT8 model...")
    model_int8 = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        # Force all modules onto a single GPU.
        device_map={"": 0},
    )
    print("INT8 model (params+buffers):", _format_bytes(_model_nbytes(model_int8)))
    _print_cuda_mem("After INT8 load")
    out_int8, time_int8 = run_inference(model_int8, prompt)
    loss_int8 = compute_loss(model_int8, texts[0])
    print("INT8 Time:", time_int8)
    print("Loss INT8:", loss_int8)
    print("\n=== INT8 Output ===")
    print(out_int8)
    print("=== End INT8 Output ===\n")
    return model_int8


print("Loading FP16 model...")
model_fp16 = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map={"": 0} if torch.cuda.is_available() else "cpu",
)
print("FP16 model (params+buffers):", _format_bytes(_model_nbytes(model_fp16)))
_print_cuda_mem("After FP16 load")
out_fp16, time_fp16 = run_inference(model_fp16, prompt)
loss_fp16 = compute_loss(model_fp16, texts[0])
print("FP16 Time:", time_fp16)
print("Loss FP16:", loss_fp16)
print("\n=== FP16 Output ===")
print(out_fp16)
print("=== End FP16 Output ===\n")

# Free VRAM before loading INT8 model.
del model_fp16
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

_run_int8()