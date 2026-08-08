"""Model loading and image decoding utilities for Stable Diffusion 1.5."""

from __future__ import annotations

import os

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline


def require_local_path(path: str, label: str) -> str:
    if not path:
        raise ValueError(f"{label} path must be provided")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    return path


def load_pipeline(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionPipeline:
    """Load the local SD1.5 pipeline and replace PNDM with deterministic DDIM."""
    model_path = require_local_path(model_path, "SD1.5 model")
    if device.type != "cuda":
        raise RuntimeError("Stable Diffusion 1.5 inference currently requires CUDA")

    print(f"Loading Stable Diffusion 1.5 pipeline from {model_path}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    return pipe


@torch.inference_mode()
def decode_latents(pipe: StableDiffusionPipeline, latents: torch.Tensor):
    latents = latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]

