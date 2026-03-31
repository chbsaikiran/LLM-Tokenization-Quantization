# Vision-Language Model INT8 Quantization

This folder contains a single script, `vlm_int8_quant.py`, that compares a small vision-language model in:

- FP16 (baseline)
- INT8 using BitsAndBytes (`load_in_8bit=True`)

The goal is to measure memory usage, speed, and output quality differences before and after quantization.

## Model and Setup

- Model: `HuggingFaceTB/SmolVLM-256M-Instruct`
- Input image: a public car image from Hugging Face docs dataset
- Prompt: `"Describe this image in 3 short bullet points."`
- Generation: deterministic (`do_sample=False`) with `max_new_tokens=64`
- Quantization config:
  - `load_in_8bit=True`
  - `llm_int8_enable_fp32_cpu_offload=False` (keeps quantized path on GPU)

## What `vlm_int8_quant.py` Does

1. Loads processor and image.
2. Defines helper utilities to:
   - estimate model memory (`params + buffers`)
   - print CUDA allocated/reserved memory
   - build multimodal inputs (chat template + image)
3. Loads FP16 model and runs:
   - inference time benchmark
   - loss computation
   - output text print
4. Frees FP16 model memory.
5. Loads INT8 model (BitsAndBytes) and runs the same metrics.

This gives an apples-to-apples FP16 vs INT8 comparison for the same model, prompt, and image.

## Results From Your Run

| Metric | FP16 | INT8 (BitsAndBytes) |
|---|---:|---:|
| Model size (`params+buffers`) | 489.21 MiB | 300.21 MiB |
| CUDA allocated after load | 489.23 MiB | 310.71 MiB |
| CUDA reserved after load | 544.00 MiB | 636.00 MiB |
| Inference time | 1.74 s | 4.46 s |
| Loss | 20.64 | 20.90 |

## What Each Result Means

- **Model size (`params+buffers`)**
  - Approximate in-memory tensor size of model parameters and buffers.
  - INT8 is smaller (expected), so it helps fit models in low VRAM.

- **CUDA allocated**
  - Memory currently used by active tensors.
  - Lower for INT8, confirming quantized weights reduce active GPU memory usage.

- **CUDA reserved**
  - Memory reserved by PyTorch allocator for reuse.
  - Can stay high (or increase) even if allocated is low; this is normal allocator behavior.

- **Inference time**
  - End-to-end generation time for the same input.
  - INT8 is slower in this run; quantization reduces memory but does not always reduce latency.

- **Loss**
  - Next-token prediction loss on the constructed multimodal input.
  - INT8 has slightly higher loss, which is typical due to quantization approximation.

## Why INT8 Is Slower Here (Even With Lower Memory)

Common causes in this environment:

- Kernel path overhead in 8-bit matmul compared to FP16 path.
- Warning indicates cast overhead:
  - `MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization`
- Small model size (256M) means quantization overhead can dominate compute savings.

So the key takeaway is:

- **INT8 helped memory a lot**
- **FP16 remained faster for this exact hardware/model/runtime setup**

