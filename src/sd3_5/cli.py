"""
CLI entry point for SD3.5 Large inference.
Mirrors the pattern of src/flux1/cli.py with four modes:
  - normal: standard generation
  - save_feature: save intermediate features during generation
  - mean_ablation: apply mean ablation at specified steps
  - sae_style_guide: encode noise via SAE, blend with style reference, decode back
"""

import os
import re
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.sd3_5.util import (
    load_tokenizer,
    load_clip_g,
    load_clip_l,
    load_t5xxl,
    load_sd3_model,
    load_vae,
    SD35_LARGE_CONFIG,
)
from src.sd3_5.sampling import (
    get_empty_latent,
    get_cond,
    do_sampling,
)
from src.sd3_5.sd3_impls import SD3LatentFormat
from sae.model import SparseAutoencoder


def load_prompts_from_file(file_path: str) -> list[str]:
    """Load prompts from a text file, one prompt per line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


@torch.inference_mode()
def main(
    # Place SD3.5 weights here or pass explicit relative paths.
    model_path: str = "models/sd3.5/sd3.5_large.safetensors",
    model_folder: str = "models/sd3.5",
    vae_path: str = None,
    clip_l_tokenizer_path: str = None,
    t5_tokenizer_path: str = None,
    # Generation parameters
    width: int = 256,
    height: int = 256,
    steps: int = None,
    cfg_scale: float = None,
    sampler: str = None,
    seed: int = 2026,
    # Prompt parameters
    prompt: str = "a photo of a cat",
    prompt_file: str = "config/3.txt",
    # Output parameters
    output_dir: str = "output",
    feature_dir: str = "feature",
    # Mode parameters
    mode: str = "normal",
    feature_name: str = "sd3_5_step_updates_style",
    ablation_steps: list = None,
    # SAE parameters
    sae_checkpoint_dir: str = "checkpoints",
    prototype_path: str = "feature/sd3_5_style_prototypes.pt",
    mean_feature_path: str = "feature/sd3_5_mean_step_updates.pt",
    target_class: int = 9,
    sae_alpha: float = 0.85,
    # Other
    text_encoder_device: str = "cpu",
    verbose: bool = False,
):
    """
    Sample the SD3.5 Large model.

    Args:
        model_path: Path to the SD3.5 Large safetensors model file
        model_folder: Folder containing text encoder safetensors (clip_g, clip_l, t5xxl)
        vae_path: Path to a separate VAE safetensors file (None = use main model file)
        clip_l_tokenizer_path: Local directory for the CLIP tokenizer
        t5_tokenizer_path: Local directory for the T5 tokenizer
        width: Width of generated image in pixels
        height: Height of generated image in pixels
        steps: Number of sampling steps (default 40 for SD3.5 Large)
        cfg_scale: CFG scale (default 4.5 for SD3.5 Large)
        sampler: Sampler to use ('euler' or 'dpmpp_2m')
        seed: Random seed
        prompt: Text prompt for generation
        prompt_file: Path to text file with prompts (one per line)
        output_dir: Directory to save output images
        feature_dir: Directory to save features
        mode: Generation mode ('normal', 'save_feature', 'mean_ablation', 'sae_style_guide')
        feature_name: Name prefix for saved features
        ablation_steps: List of step indices for ablation/intervention
        sae_checkpoint_dir: Directory containing SAE checkpoints
        prototype_path: Path to precomputed class-level SAE prototypes
        mean_feature_path: Path to mean denoising step updates
        target_class: Target style class index (0-9)
        sae_alpha: SAE latent blending strength (0=original, 1=full style)
        text_encoder_device: Device for text encoders
        verbose: Print verbose output
    """
    if ablation_steps is None:
        ablation_steps = []

    # Apply SD3.5 Large defaults
    config = SD35_LARGE_CONFIG
    _steps = steps or config["steps"]
    _cfg = cfg_scale or config["cfg"]
    _sampler = sampler or config["sampler"]
    _shift = config["shift"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load SAE models and style reference codes for sae_style_guide mode ──
    sae_config = None
    if mode == "sae_style_guide":
        print(f"Loading SAE models for style guide (class={target_class}, alpha={sae_alpha})...")
        sae_config = {'models': {}, 'style_codes': {}, 'alpha': sae_alpha}

        # Load precomputed style reference codes
        style_data = torch.load(prototype_path, map_location="cpu", weights_only=True)

        for step in ablation_steps:
            timestep = step - 1  # step_count is 1-based, SAE timestep is 0-based
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

    # Load prompts from file
    if prompt_file and os.path.exists(prompt_file):
        all_prompts = load_prompts_from_file(prompt_file)
        print(f"Loaded {len(all_prompts)} prompts from {prompt_file}")
    else:
        all_prompts = [prompt]

    # Initialize features dictionary
    all_features = {}

    print(f"Running SD3.5 Large in '{mode}' mode")
    if mode == "mean_ablation" and ablation_steps:
        print(f"Ablation steps: {ablation_steps}")

    mean_features = None
    if mode == "mean_ablation":
        if not ablation_steps:
            raise ValueError("mean_ablation requires non-empty ablation_steps")
        if not os.path.isfile(mean_feature_path):
            raise FileNotFoundError(
                f"SD3.5 mean step-update file not found: {mean_feature_path}"
            )
        mean_features = torch.load(
            mean_feature_path, map_location="cpu", weights_only=True
        )

    # ── Load models ──
    tokenizer = load_tokenizer(
        model_folder=model_folder,
        clip_l_tokenizer_path=clip_l_tokenizer_path,
        t5_tokenizer_path=t5_tokenizer_path,
    )
    t5xxl = load_t5xxl(model_folder, text_encoder_device, torch.float32)
    clip_l = load_clip_l(model_folder)
    clip_g = load_clip_g(model_folder, text_encoder_device)
    sd3_model = load_sd3_model(model_path, shift=_shift, verbose=verbose, device=device)
    vae = load_vae(vae_path or model_path)

    # ── Output directory ──
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        output_pattern = os.path.join(output_dir, "img_*.jpg")
        from glob import iglob
        fns = [fn for fn in iglob(output_pattern) if re.search(r"img_[0-9]+\.jpg$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    # ── Generate images ──
    neg_cond = get_cond(tokenizer, clip_l.model if hasattr(clip_l, 'model') else clip_l,
                        clip_g.model if hasattr(clip_g, 'model') else clip_g,
                        t5xxl.model if hasattr(t5xxl, 'model') else t5xxl, "")

    for i, cur_prompt in enumerate(tqdm(all_prompts, desc="Generating")):
        print(f"Generating with seed {seed}: {cur_prompt}")
        t0 = time.perf_counter()

        # Encode prompt
        conditioning = get_cond(
            tokenizer,
            clip_l.model if hasattr(clip_l, 'model') else clip_l,
            clip_g.model if hasattr(clip_g, 'model') else clip_g,
            t5xxl.model if hasattr(t5xxl, 'model') else t5xxl,
            cur_prompt,
        )

        # Prepare latent
        latent = get_empty_latent(1, width, height, seed, "cpu").cuda()

        # Do sampling
        sampled_latent, features = do_sampling(
            model=sd3_model,
            latent=latent,
            seed=seed,
            conditioning=conditioning,
            neg_cond=neg_cond,
            steps=_steps,
            cfg_scale=_cfg,
            sampler=_sampler,
            denoise=1.0,
            mode=mode,
            ablation_steps=ablation_steps,
            sae_config=sae_config,
            mean_features=mean_features,
        )

        # Decode to image
        sampled_latent = sampled_latent.cuda()
        vae.cuda()
        image = vae.decode(sampled_latent)
        image = image.float()
        vae.cpu()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
        decoded_np = decoded_np.astype(np.uint8)
        out_image = Image.fromarray(decoded_np)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        fn = os.path.join(output_dir, f"img_{idx}.jpg")
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
        out_image.save(fn)

        # Save features if in save_feature mode
        if mode == "save_feature" and features:
            all_features[idx] = {
                "idx": idx,
                "prompt": cur_prompt,
                "features": features,
            }
            print(f"Saved features for image {idx}")

        idx += 1

    # Save all features to a single file when in save_feature mode
    if mode == "save_feature" and all_features:
        os.makedirs(feature_dir, exist_ok=True)
        feature_path = os.path.join(feature_dir, f"{feature_name}_{seed}.pt")
        torch.save(all_features, feature_path)
        print(f"Saved all {len(all_features)} features to {feature_path}")

    print("Done!")


if __name__ == "__main__":
    main()
