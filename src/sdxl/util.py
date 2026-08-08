"""Model loading and image decoding utilities for SDXL."""

from __future__ import annotations

import os

import torch
from diffusers import DDIMScheduler, StableDiffusionXLPipeline


def require_local_path(path: str, label: str) -> str:
    """Require a local diffusers model directory."""
    if not path:
        raise ValueError(f"{label} path must be provided")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    return path


def load_pipeline(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionXLPipeline:
    """Load the local SDXL pipeline and replace its sampler with deterministic DDIM."""
    model_path = require_local_path(model_path, "SDXL model")
    if device.type != "cuda":
        raise RuntimeError("SDXL inference currently requires CUDA")

    print(f"Loading SDXL pipeline from {model_path}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        local_files_only=True,
        add_watermarker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()
    return pipe


@torch.inference_mode()
def decode_latents(pipe: StableDiffusionXLPipeline, latents: torch.Tensor):
    """Decode SDXL latents with the same VAE upcasting behavior as diffusers."""
    needs_upcasting = (
        pipe.vae.dtype == torch.float16
        and bool(getattr(pipe.vae.config, "force_upcast", False))
    )
    if needs_upcasting:
        pipe.upcast_vae()
        latents = latents.to(next(iter(pipe.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != pipe.vae.dtype:
        latents = latents.to(pipe.vae.dtype)

    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]

    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)

    return pipe.image_processor.postprocess(image, output_type="pil")[0]

