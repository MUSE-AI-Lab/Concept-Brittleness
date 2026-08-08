"""
CLI entry point for PixArt-α inference.
Mirrors the pattern of src/flux1/cli.py with four modes:
  - normal:          standard generation
  - save_feature:    save intermediate features during generation
  - mean_ablation:   apply mean ablation at specified steps
  - sae_style_guide: encode noise via SAE, blend with style reference, decode back
"""

import os
import re
import time
from glob import iglob

import numpy as np
import torch
from PIL import Image

from src.pixart.util import load_pixart_model, load_vae, load_t5
from src.pixart.sampling import prepare_cond, do_sampling
from sae.model import SparseAutoencoder


def load_prompts_from_file(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


# ─────────────────────── Default configuration ──────────────────────────

PIXART_256_DEFAULTS = {
    "image_size": 256,
    "steps": 20,
    "cfg_scale": 4.5,
    "vae_scale_factor": 0.18215,
}


# ─────────────────────── Main entry point ───────────────────────────────

@torch.inference_mode()
def main(
    # Model paths
    # Place PixArt components here or pass explicit relative paths.
    model_path: str = "models/pixart/PixArt-XL-2-256x256.pth",
    vae_path: str = "models/pixart/sd-vae-ft-ema",
    t5_path: str = "models/pixart/t5-v1_1-xxl",
    # Generation parameters
    image_size: int = 256,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int = 2026,
    # Prompt parameters
    prompt: str = "a photo of a cat",
    prompt_file: str = "config/3.txt",
    # Output parameters
    output_dir: str = "output",
    feature_dir: str = "feature",
    # Mode parameters
    mode: str = "normal",
    feature_name: str = "pixart_step_updates_style",
    ablation_steps: list | None = None,
    # SAE parameters
    sae_checkpoint_dir: str = "checkpoints",
    prototype_path: str = "feature/pixart_style_prototypes.pt",
    mean_feature_path: str = "feature/pixart_mean_step_updates.pt",
    target_class: int = 9,
    sae_alpha: float = 0.85,
):
    """Sample from PixArt-α-256.

    Args:
        model_path:  Path to PixArt-XL-2 ``.pth`` checkpoint
        vae_path:    Path to ``sd-vae-ft-ema`` directory (diffusers format)
        t5_path:     Path to ``t5-v1_1-xxl`` directory
        image_size:  Image resolution (default 256)
        steps:       Sampling steps (default 20)
        cfg_scale:   CFG scale (default 4.5)
        seed:        Random seed
        prompt_file: Text file with one prompt per line
        output_dir:  Directory for output images
        mode:        'normal', 'save_feature', 'mean_ablation', 'sae_style_guide'
        ablation_steps: 1-based step indices for intervention
    """
    if ablation_steps is None:
        ablation_steps = []

    _steps = steps or PIXART_256_DEFAULTS["steps"]
    _cfg = cfg_scale or PIXART_256_DEFAULTS["cfg_scale"]
    vae_sf = PIXART_256_DEFAULTS["vae_scale_factor"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load SAE models for sae_style_guide mode ────────────────────
    sae_config = None
    if mode == "sae_style_guide":
        print(f"Loading SAE models for style guide (class={target_class}, alpha={sae_alpha})...")
        sae_config = {'models': {}, 'style_codes': {}, 'alpha': sae_alpha}

        style_data = torch.load(prototype_path, map_location="cpu", weights_only=True)

        for step in ablation_steps:
            timestep = step - 1
            ckpt_dir = os.path.join(sae_checkpoint_dir, f"step_{timestep:02d}")
            if not os.path.exists(ckpt_dir):
                print(f"  [WARN] SAE checkpoint not found: {ckpt_dir}, skipping step {step}")
                continue

            sae = SparseAutoencoder.load_from_disk(ckpt_dir, map_location=device)
            sae.to(device)
            sae.eval()

            norm = np.load(os.path.join(ckpt_dir, "norm_params.npz"))
            mu = torch.tensor(norm["mu"], dtype=torch.float32, device=device)
            sigma = torch.tensor(norm["sigma"], dtype=torch.float32, device=device)

            sae_config['models'][timestep] = {'sae': sae, 'mu': mu, 'sigma': sigma}

            if timestep in style_data and target_class in style_data[timestep]:
                sae_config['style_codes'][timestep] = style_data[timestep][target_class].to(device)
            else:
                print(f"  [WARN] No style codes for timestep {timestep}, class {target_class}")

        print(f"  Loaded {len(sae_config['models'])} SAE models for {len(ablation_steps)} steps")

    # ── Load prompts ────────────────────────────────────────────────
    if prompt_file and os.path.exists(prompt_file):
        all_prompts = load_prompts_from_file(prompt_file)
        print(f"Loaded {len(all_prompts)} prompts from {prompt_file}")
    else:
        all_prompts = [prompt]

    all_features = {}
    print(f"Running PixArt-α-{image_size} in '{mode}' mode")
    if mode == "mean_ablation" and ablation_steps:
        print(f"Ablation steps: {ablation_steps}")

    mean_features = None
    if mode == "mean_ablation":
        if not ablation_steps:
            raise ValueError("mean_ablation requires non-empty ablation_steps")
        if not os.path.isfile(mean_feature_path):
            raise FileNotFoundError(
                f"PixArt mean step-update file not found: {mean_feature_path}"
            )
        mean_features = torch.load(
            mean_feature_path, map_location="cpu", weights_only=True
        )

    # ── Load models ─────────────────────────────────────────────────
    t5 = load_t5(t5_path, device=device, torch_dtype=torch.float)
    model = load_pixart_model(model_path, image_size=image_size,
                              device=device, dtype=torch.float16)
    vae = load_vae(vae_path, device=device)

    # ── Setup output directory ──────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "img_*.jpg")
    fns = [fn for fn in iglob(output_pattern)
           if re.search(r"img_[0-9]+\.jpg$", fn)]
    idx = max((int(fn.split("_")[-1].split(".")[0]) for fn in fns), default=-1) + 1

    # ── Set seed ────────────────────────────────────────────────────
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    latent_size = image_size // 8

    # ── Generate images ─────────────────────────────────────────────
    for i, cur_prompt in enumerate(all_prompts):
        print(f"[{i+1}/{len(all_prompts)}] seed={seed}: {cur_prompt}")
        t0 = time.perf_counter()

        # Encode prompt
        caption_embs, null_y, emb_masks = prepare_cond(
            model, t5, [cur_prompt], device)

        # Create initial noise
        z = torch.randn(1, 4, latent_size, latent_size, device=device)

        # Sampling
        sampled, features = do_sampling(
            model=model,
            z=z,
            caption_embs=caption_embs,
            null_y=null_y,
            emb_masks=emb_masks,
            cfg_scale=_cfg,
            steps=_steps,
            device=device,
            mode=mode,
            ablation_steps=ablation_steps,
            sae_config=sae_config,
            mean_features=mean_features,
        )

        # Decode latent → image
        with torch.no_grad():
            image = vae.decode(sampled / vae_sf).sample

        image = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)[0]
        image_np = (255.0 * image.cpu().permute(1, 2, 0).numpy()).astype(np.uint8)
        out_image = Image.fromarray(image_np)

        t1 = time.perf_counter()
        fn = os.path.join(output_dir, f"img_{idx}.jpg")
        print(f"  Done in {t1 - t0:.1f}s → {fn}")
        out_image.save(fn)

        # Save features
        if mode == "save_feature" and features:
            all_features[idx] = {
                "idx": idx,
                "prompt": cur_prompt,
                "features": features,
            }
            print(f"  Saved features for image {idx}")

        idx += 1

    # ── Persist features ────────────────────────────────────────────
    if mode == "save_feature" and all_features:
        os.makedirs(feature_dir, exist_ok=True)
        feature_path = os.path.join(feature_dir, f"{feature_name}_{seed}.pt")
        torch.save(all_features, feature_path)
        print(f"Saved all {len(all_features)} features to {feature_path}")

    print("Done!")


if __name__ == "__main__":
    main()
