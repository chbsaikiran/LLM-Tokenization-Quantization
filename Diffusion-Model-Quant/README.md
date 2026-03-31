# Diffusion Model INT8 Quantization

This folder contains a benchmark script, `diffusion_int8_quant.py`, that compares a diffusion pipeline before and after INT8 quantization.

## What This Script Does

The script runs two inference paths on the same prompt:

1. **FP16 baseline**
2. **INT8 path with BitsAndBytes quantization**

Model used:
- `nota-ai/bk-sdm-tiny` (chosen to fit consumer GPUs like 6GB VRAM)

Prompt setup:
- Prompt: `A cinematic photo of a red sports car parked on a street at sunset.`
- Negative prompt: `blurry, low quality, distorted`
- Image size: `512 x 512`
- Denoising steps: `20`

### Pipeline flow

1. Load FP16 diffusion pipeline.
2. Measure and print:
   - model memory estimate (`params + buffers`)
   - CUDA allocated/reserved memory
   - generation time
   - diffusion loss proxy (MSE between predicted and true noise at one timestep)
3. Save generated FP16 image to `fp16_output.png`.
4. Free memory and load INT8 configuration (BitsAndBytes).
5. Load pipeline with quantized text-encoder path and run the same measurements.
6. Save generated INT8 image to `int8_output.png`.

## Results From Your Run

| Metric | FP16 | INT8 |
|---|---:|---:|
| Pipeline size (`params+buffers`) | 1011.08 MiB | 1.74 GiB |
| CUDA allocated after load | 1.58 GiB | 2.90 GiB |
| CUDA reserved after load | 1.59 GiB | 2.96 GiB |
| Inference time (20 steps) | 2.07 s | 4.97 s |
| Loss (diffusion proxy MSE) | 0.2977 | 0.3061 |
| Output image path | `fp16_output.png` | `int8_output.png` |

## How To Read These Results

- **Pipeline size / CUDA memory**
  - Indicates how much model and runtime memory is being used.
  - In this run, INT8 path used more memory because of component dtype/compatibility handling in this specific diffusion setup.

- **Inference time**
  - End-to-end generation latency for the same prompt and settings.
  - INT8 was slower here, showing that quantization does not always improve latency.

- **Loss (proxy MSE)**
  - A noise-prediction proxy metric for one diffusion step.
  - INT8 loss is slightly higher, which is expected from quantization approximation.

## Notes About Warnings In Output

- `UNEXPECTED ... position_ids` in load reports:
  - Benign in this context; commonly appears during cross-checkpoint/component loading and did not block inference.

- `MatMul8bitLt ... cast from torch.float32 to float16`:
  - Shows extra cast overhead in BitsAndBytes path, which can contribute to slower runtime.

- Unauthenticated Hugging Face warning:
  - Only affects download rate limits, not model correctness.
