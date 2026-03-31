import gc
import time
import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from transformers import BitsAndBytesConfig, CLIPTextModel

# Chosen for 6GB GPU: compact SD variant.
MODEL_ID = "nota-ai/bk-sdm-tiny"
PROMPT = "A cinematic photo of a red sports car parked on a street at sunset."
NEGATIVE_PROMPT = "blurry, low quality, distorted"
HEIGHT = 512
WIDTH = 512
STEPS = 20


def _format_bytes(n: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def _module_nbytes(module: torch.nn.Module) -> int:
    total = 0
    for t in list(module.parameters()) + list(module.buffers()):
        if getattr(t, "device", None) is not None and str(t.device) == "meta":
            continue
        try:
            total += t.numel() * t.element_size()
        except Exception:
            pass
    return int(total)


def _pipeline_nbytes(pipe: DiffusionPipeline) -> int:
    total = 0
    for attr in ["unet", "vae", "text_encoder", "text_encoder_2", "transformer"]:
        mod = getattr(pipe, attr, None)
        if isinstance(mod, torch.nn.Module):
            total += _module_nbytes(mod)
    return total


def _print_cuda_mem(prefix: str):
    if not torch.cuda.is_available():
        return
    print(f"{prefix} CUDA allocated:", _format_bytes(torch.cuda.memory_allocated()))
    print(f"{prefix} CUDA reserved:", _format_bytes(torch.cuda.memory_reserved()))


def _diffusion_loss_proxy(pipe: DiffusionPipeline, prompt: str, negative_prompt: str) -> float:
    """MSE(noise_pred, true_noise) for one denoising step."""
    device = pipe._execution_device
    pipe.scheduler.set_timesteps(STEPS, device=device)
    t = pipe.scheduler.timesteps[len(pipe.scheduler.timesteps) // 2]

    with torch.no_grad():
        prompt_embeds, neg_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        latent_h = HEIGHT // 8
        latent_w = WIDTH // 8
        latents = torch.randn(
            (1, pipe.unet.config.in_channels, latent_h, latent_w),
            device=device,
            dtype=pipe.unet.dtype,
        )
        noise = torch.randn_like(latents)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, t)

        # Guidance loss proxy: compare guided prediction with the true added noise.
        latent_in = torch.cat([noisy_latents, noisy_latents], dim=0)
        emb_in = torch.cat([neg_embeds, prompt_embeds], dim=0)
        pred = pipe.unet(latent_in, t, encoder_hidden_states=emb_in).sample
        pred_uncond, pred_text = pred.chunk(2)
        guided = pred_uncond + 7.5 * (pred_text - pred_uncond)
        loss = F.mse_loss(guided.float(), noise.float())
    return float(loss.item())


def _run_generation(pipe: DiffusionPipeline, tag: str):
    generator = torch.Generator(device=pipe._execution_device).manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    out_path = f"{tag}_output.png"
    image.save(out_path)
    return elapsed, out_path


print("Loading FP16 diffusion pipeline...")
pipe_fp16 = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
if torch.cuda.is_available():
    pipe_fp16 = pipe_fp16.to("cuda")
print("FP16 pipeline (params+buffers):", _format_bytes(_pipeline_nbytes(pipe_fp16)))
_print_cuda_mem("After FP16 load")
fp16_time, fp16_img = _run_generation(pipe_fp16, "fp16")
fp16_loss = _diffusion_loss_proxy(pipe_fp16, PROMPT, NEGATIVE_PROMPT)
print("FP16 Time:", fp16_time)
print("Loss FP16 (diffusion proxy MSE):", fp16_loss)
print("FP16 image saved at:", fp16_img)

del pipe_fp16
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nLoading INT8 diffusion pipeline (text encoder quantized with bitsandbytes)...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=False,
)
text_encoder_int8 = CLIPTextModel.from_pretrained(
    MODEL_ID,
    subfolder="text_encoder",
    quantization_config=bnb_config,
    device_map={"": 0} if torch.cuda.is_available() else "cpu",
)
pipe_int8 = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    text_encoder=text_encoder_int8,
    torch_dtype=torch.float32,
)
if torch.cuda.is_available():
    pipe_int8 = pipe_int8.to("cuda")
# Keep UNet/VAE in fp32 for dtype compatibility with this INT8 text-encoder path.
pipe_int8.unet.to(dtype=torch.float32)
pipe_int8.vae.to(dtype=torch.float32)
print("INT8 pipeline (params+buffers):", _format_bytes(_pipeline_nbytes(pipe_int8)))
_print_cuda_mem("After INT8 load")
int8_time, int8_img = _run_generation(pipe_int8, "int8")
int8_loss = _diffusion_loss_proxy(pipe_int8, PROMPT, NEGATIVE_PROMPT)
print("INT8 Time:", int8_time)
print("Loss INT8 (diffusion proxy MSE):", int8_loss)
print("INT8 image saved at:", int8_img)
