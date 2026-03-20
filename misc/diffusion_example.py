import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import trange, tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
)
from diffusers import DiffusionPipeline


def enable_memory_savings(pipe):
    # attempt common memory reductions
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass
    return pipe


def build_fixed_latent(
    pipe, height, width, device, latent_scale=8, seed=0, dtype=torch.float32
):
    """
    Build a fixed random latent tensor to pass to decoder.
    Assumes latent spatial dims = height/latent_scale, width/latent_scale.
    """
    b = 1
    # infer channels from UNet if available
    unet = (
        getattr(pipe, "decoder", None)
        or getattr(pipe, "unet", None)
        or getattr(pipe, "unet", None)
    )
    if unet is None:
        raise RuntimeError(
            "Pipeline does not expose an accessible UNet (decoder/unet)."
        )
    in_ch = getattr(unet, "in_channels", None)
    if in_ch is None:
        # fallback channel for stable diffusion style latents
        in_ch = 4
    latent_h = height // latent_scale
    latent_w = width // latent_scale
    torch_generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (b, in_ch, latent_h, latent_w),
        generator=torch_generator,
        device=device,
        dtype=dtype,
    )
    return latents, torch_generator


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16
steps = 6
seed = 42
guidance_scale = 8.0
num_inference_steps = 40
current_caption = "an image of a young person"
target_caption = "an image of an old person"
interpolation_method = "linear"
model_id = "diffusers/stable-diffusion-2-1-unclip-i2i-l"
# max_theta = 1.3 # 0.67 for slerp, 1.3 for linear (change it to dynamic to suit the image) & 1.0 for rotate
image_path = "samples/test_img.jpg"
height = 512
width = 512
latent_scale = 8

# 1) load pipeline (try UnCLIPPipeline then StableUnCLIPPipeline)
pipe = None
pipe = DiffusionPipeline.from_pretrained(
    model_id, torch_dtype=dtype, trust_remote_code=True
)
print("Loaded UnCLIPPipeline from", model_id)

# reduce memory
pipe = enable_memory_savings(pipe)
pipe = pipe.to(device)

# 2) load CLIP image encoder (match to what pipeline recommends ideally)
# prefer pipeline-provided encoders if present
clip_processor = None
clip_image_encoder = None
# pipeline may have attributes; try a safe default
try:
    clip_model_default = "openai/clip-vit-large-patch14"
    clip_processor = CLIPImageProcessor.from_pretrained(clip_model_default)
    clip_image_encoder = (
        CLIPVisionModelWithProjection.from_pretrained(clip_model_default)
        .to(device)
        .eval()
    )
    print("Loaded default CLIP image encoder:", clip_model_default)
except Exception as e:
    clip_processor = None
    clip_image_encoder = None
    print("Could not load default CLIP encoder:", e)

# 3) get z_i (CLIP image embed)
pil = Image.open(image_path).convert("RGB").resize((width, height))
if clip_processor is None or clip_image_encoder is None:
    raise RuntimeError(
        "No CLIP image encoder available; load one or use a pipeline that exposes one."
    )
clip_inputs = clip_processor(images=pil, return_tensors="pt").to(device)
with torch.no_grad():
    clip_out = clip_image_encoder(**clip_inputs)
    if hasattr(clip_out, "image_embeds"):
        z_i = clip_out.image_embeds
    else:
        # fallback pool
        z_i = clip_out.last_hidden_state.mean(dim=1)

# 4) get text embeddings z_t0 and z_t1 (use CLIP text projection that matches clip_model_default)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = (
    CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    .to(device)
    .eval()
)


def text_embed(text):
    toks = tokenizer(text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = text_encoder(**toks)
        if hasattr(out, "text_embeds"):
            return out.text_embeds
        else:
            return out.last_hidden_state.mean(dim=1)


zt0 = text_embed(current_caption)
zt1 = text_embed(target_caption)

# 5) prepare a fixed random latent (no DDIM inversion)
try:
    latents, generator = build_fixed_latent(
        pipe, height, width, device, latent_scale=latent_scale, seed=seed, dtype=dtype
    )
except Exception as e:
    print(
        "Could not auto-build latents from the pipeline. You may need to set --latent_scale accordingly."
    )
    raise

# 6) sweep and decode
max_theta = 2 * z_i.norm(dim=1) / (zt0.norm(dim=1) + zt1.norm(dim=1))
max_theta = max_theta.item()
out_dir = Path(f"./outputs/woman_{interpolation_method}_{max_theta:.2f}/")
out_dir.mkdir(parents=True, exist_ok=True)
thetas = np.linspace(0.0, max_theta, steps)
saved = []
for i, theta in enumerate(tqdm(thetas, desc="text-diff sweep")):
    zt = z_i + (theta * (zt1 - zt0))
    # zt = zt/(zt.norm(dim=1, keepdim=True) + 1e-8)
    # try to call pipeline with latents argument (most UnCLIP/StabUnCLIP accept image_embeds + latents)
    try:
        zt = zt.type(latents.dtype)
        out = pipe(
            image_embeds=zt,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        images = out.images if hasattr(out, "images") else out
    except TypeError as e:
        # pipeline doesn't accept latents; fallback to fixed generator approach (seed -> reproducible)
        print(
            "Pipeline did not accept latents=... ; falling back to fixed RNG generator (seed). Error:",
            e,
        )
        # use generator to seed the internal noise; this is less exact than passing latents but reproducible.
        gen = torch.Generator(device=device).manual_seed(seed)
        out = pipe(
            image_embeds=zt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=gen,
        )
        images = out.images if hasattr(out, "images") else out
    img = images[0] if isinstance(images, (list, tuple)) else images
    out_path = Path(out_dir) / f"textdiff_theta{theta:.3f}_{i:03d}.png"
    img.save(out_path)
    saved.append(str(out_path))

# optional gif
frames = [Image.open(p) for p in saved]
gif_path = Path(out_dir) / "textdiff_noddim.gif"
frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0)
print("Saved gif:", gif_path)

print("Saved frames:", saved)
