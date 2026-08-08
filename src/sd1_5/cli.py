"""CLI-style entry point for SD1.5 feature extraction and SAE-guided generation."""

from __future__ import annotations

from glob import iglob
import os
import re
import time

import numpy as np
import torch
from tqdm import tqdm

from sae.model import SparseAutoencoder
from src.sd1_5.sampling import FEATURE_DIM, do_sampling
from src.sd1_5.util import decode_latents, load_pipeline


VALID_MODES = {"normal", "save_feature", "mean_ablation", "sae_style_guide"}


def load_prompts_from_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _next_output_index(output_dir: str) -> int:
    files = [
        path
        for path in iglob(os.path.join(output_dir, "img_*.jpg"))
        if re.search(r"img_[0-9]+\.jpg$", path)
    ]
    return max((int(path.rsplit("_", 1)[-1].split(".")[0]) for path in files), default=-1) + 1


def _lookup_style_code(style_data: dict, timestep: int, target_class: int):
    per_step = style_data.get(timestep, style_data.get(str(timestep)))
    if per_step is None:
        raise KeyError(f"No style codes for timestep {timestep}")
    code = per_step.get(target_class, per_step.get(str(target_class)))
    if code is None:
        raise KeyError(f"No style code for timestep {timestep}, class {target_class}")
    return code


def _load_sae_config(
    checkpoint_dir: str,
    prototype_path: str,
    ablation_steps: list[int],
    target_class: int,
    alpha: float,
    device: torch.device,
) -> dict:
    if not ablation_steps:
        raise ValueError("sae_style_guide requires non-empty ablation_steps")
    style_data = torch.load(prototype_path, map_location="cpu", weights_only=True)
    config = {"models": {}, "style_codes": {}, "alpha": alpha}
    for step in sorted(set(ablation_steps)):
        timestep = step - 1
        path = os.path.join(checkpoint_dir, f"step_{timestep:02d}")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"SD1.5 SAE checkpoint not found: {path}")
        sae = SparseAutoencoder.load_from_disk(path, map_location=device).to(device).eval()
        if sae.d_model != FEATURE_DIM:
            raise ValueError(
                f"SD1.5 requires SAE d_model={FEATURE_DIM}, got {sae.d_model} at {path}"
            )
        norm = np.load(os.path.join(path, "norm_params.npz"))
        mu = torch.as_tensor(norm["mu"], dtype=torch.float32, device=device)
        sigma = torch.as_tensor(norm["sigma"], dtype=torch.float32, device=device)
        if mu.numel() != FEATURE_DIM or sigma.numel() != FEATURE_DIM:
            raise ValueError(f"Invalid SD1.5 normalization shape at {path}")
        config["models"][timestep] = {"sae": sae, "mu": mu, "sigma": sigma}
        config["style_codes"][timestep] = _lookup_style_code(
            style_data, timestep, target_class
        ).to(device)
    return config


@torch.inference_mode()
def main(
    # Place the local Diffusers model here or pass model_path explicitly.
    model_path: str = "models/sd1.5",
    width: int = 256,
    height: int = 256,
    steps: int = 50,
    cfg_scale: float = 7.5,
    seed: int = 2026,
    prompt: str = "a photo of a cat",
    prompt_file: str | None = None,
    output_dir: str = "output/sd1_5",
    feature_dir: str = "feature",
    mode: str = "normal",
    feature_name: str = "sd1_5_step_updates_style",
    ablation_steps: list[int] | None = None,
    mean_feature_path: str = "feature/sd1_5_mean_step_updates.pt",
    sae_checkpoint_dir: str = "checkpoints/sd1_5",
    prototype_path: str = "feature/sd1_5_style_prototypes.pt",
    target_class: int = 9,
    sae_alpha: float = 0.8,
):
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported SD1.5 mode: {mode}")
    if steps <= 0:
        raise ValueError(f"SD1.5 steps must be positive, got {steps}")
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError("SD1.5 width and height must be positive multiples of 16")
    if not 0 <= target_class < 10:
        raise ValueError(f"SD1.5 target_class must be in 0..9, got {target_class}")
    if not 0.0 <= sae_alpha <= 1.0:
        raise ValueError(f"SD1.5 sae_alpha must be between 0 and 1, got {sae_alpha}")
    ablation_steps = ablation_steps or []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if prompt_file:
        if not os.path.isfile(prompt_file):
            raise FileNotFoundError(f"Prompt file does not exist: {prompt_file}")
        prompts = load_prompts_from_file(prompt_file)
        if not prompts:
            raise ValueError(f"Prompt file is empty: {prompt_file}")
    else:
        prompts = [prompt]

    sae_config = None
    if mode == "sae_style_guide":
        sae_config = _load_sae_config(
            sae_checkpoint_dir,
            prototype_path,
            ablation_steps,
            target_class,
            sae_alpha,
            device,
        )

    mean_features = None
    if mode == "mean_ablation":
        if not ablation_steps:
            raise ValueError("mean_ablation requires non-empty ablation_steps")
        if not os.path.isfile(mean_feature_path):
            raise FileNotFoundError(f"SD1.5 mean feature file not found: {mean_feature_path}")
        mean_features = torch.load(mean_feature_path, map_location="cpu", weights_only=True)
        if len(mean_features) < steps:
            raise ValueError(
                f"Mean feature file has {len(mean_features)} steps, expected at least {steps}"
            )

    pipe = load_pipeline(model_path, device=device, dtype=dtype)
    os.makedirs(output_dir, exist_ok=True)
    output_index = _next_output_index(output_dir)
    all_features = {}

    for current_prompt in tqdm(prompts, desc="Generating SD1.5"):
        start = time.perf_counter()
        latents, features = do_sampling(
            pipe=pipe,
            prompt=current_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            mode=mode,
            ablation_steps=ablation_steps,
            mean_features=mean_features,
            sae_config=sae_config,
        )
        image = decode_latents(pipe, latents)
        output_path = os.path.join(output_dir, f"img_{output_index}.jpg")
        image.save(output_path)
        print(f"SD1.5 saved {output_path} in {time.perf_counter() - start:.1f}s")

        if mode == "save_feature":
            all_features[output_index] = {
                "idx": output_index,
                "prompt": current_prompt,
                "features": features,
            }
        output_index += 1

    if mode == "save_feature":
        os.makedirs(feature_dir, exist_ok=True)
        feature_path = os.path.join(feature_dir, f"{feature_name}_{seed}.pt")
        torch.save(all_features, feature_path)
        print(f"Saved {len(all_features)} SD1.5 feature trajectories to {feature_path}")


if __name__ == "__main__":
    main()
