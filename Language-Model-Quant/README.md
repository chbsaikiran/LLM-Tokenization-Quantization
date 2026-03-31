# BitsAndBytes INT8 Quantization Notes

This folder contains scripts to compare full-precision and quantized LLM inference.  
This README focuses on `int8_quant.py` and the `bitsandbytes` INT8 flow.

## What BitsAndBytes Quantization Does

`bitsandbytes` provides low-bit linear layers (8-bit and 4-bit) used by Hugging Face Transformers to reduce model memory usage.

For **LLM.int8()** (`load_in_8bit=True`):

- Model linear weights are stored/used in INT8 instead of FP16/FP32 for most matmul work.
- It uses a **mixed-precision outlier strategy**:
  - Most values are processed in INT8.
  - Outlier channels are handled in higher precision to preserve quality.
- This usually cuts model memory significantly while keeping generation quality close to FP16.

In practice, INT8 speed can be faster or slower than FP16 depending on GPU, kernel support, offloading, batch size, and generation settings.

## What `int8_quant.py` Is Doing

The script benchmarks the same prompt with:

1. **FP16 model**
2. **INT8 model via BitsAndBytes**

and prints:

- Approx model size in memory (`params + buffers`)
- CUDA allocated/reserved memory
- Inference time
- A simple loss value on a sample text
- Generated output text

### Step-by-step flow

1. Loads tokenizer and sets:
   - model: `Qwen/Qwen2.5-1.5B-Instruct`
   - prompt and generation length
2. Builds `BitsAndBytesConfig` with:
   - `load_in_8bit=True`
   - `llm_int8_enable_fp32_cpu_offload=False` (keeps quantized path on GPU)
3. Loads and runs **FP16** model first.
4. Frees VRAM (`del`, `gc.collect()`, `torch.cuda.empty_cache()`).
5. Loads and runs **INT8** model with quantization config.
6. Prints metrics and outputs for comparison.

## Why INT8 May Not Always Be Faster

Even when memory is lower, latency can be worse than FP16 if:

- GPU kernels are not fully optimized for the hardware/library combination.
- There is device transfer overhead (CPU offload, if enabled).
- Generation config is overhead-heavy (small batches, short prompts, cache settings).
- Runtime falls back to slower code paths.

So, memory reduction is usually reliable; speedup depends on environment.

## Observed Results From Your Run

From your `python int8_quant.py` run:

| Metric | FP16 | INT8 (bitsandbytes) |
|---|---:|---:|
| Model size (`params+buffers`) | 2.88 GiB | 1.66 GiB |
| CUDA allocated after load | 2.88 GiB | 1.67 GiB |
| CUDA reserved after load | 2.93 GiB | 2.93 GiB |
| Inference time | 1.59 s | 5.75 s |
| Loss (sample text) | 5.03 | 5.59 |

### Why these results look like this

- **Memory dropped a lot (expected):** INT8 reduced model storage from ~2.88 GiB to ~1.66 GiB, which is the main benefit of 8-bit quantization.
- **`allocated` dropped but `reserved` stayed high:** PyTorch caching allocator kept the reserved pool (`2.93 GiB`) even after FP16 cleanup. This is normal and helps avoid repeated allocations.
- **INT8 slower than FP16 in this run:** on this hardware/software combo, bitsandbytes INT8 kernel path had higher runtime overhead than FP16 Tensor Core path for this generation setting.
- **Extra cast warning indicates overhead:** `MatMul8bitLt` warning shows activations were cast from `bfloat16` to `float16` during quantized matmul, which adds work.
- **Loss is slightly higher for INT8:** this is expected because quantization introduces approximation error, even when quality is usually still close.

### Notes about warnings in your output

- `The following generation flags are not valid ... ['temperature', 'top_p', 'top_k']` comes from model generation config defaults and can be ignored for deterministic (`do_sample=False`) decoding.
- Unauthenticated Hugging Face warning only affects download rate limits, not model quality.

## References

- Hugging Face Transformers + bitsandbytes quantization docs:  
  <https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes>
- bitsandbytes LLM.int8() reference:  
  <https://huggingface.co/docs/bitsandbytes/en/reference/nn/linear8bit>
- bitsandbytes quantization primitives overview:  
  <https://huggingface.co/docs/bitsandbytes/main/quantization>
- bitsandbytes repository:  
  <https://github.com/bitsandbytes-foundation/bitsandbytes>
