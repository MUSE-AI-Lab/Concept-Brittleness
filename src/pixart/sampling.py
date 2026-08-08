"""
Sampling functions for PixArt-α with four modes:
  - normal:          standard DPM-Solver++ (order 2, 20 steps)
  - save_feature:    save per-step noise updates
  - mean_ablation:   blend step noise with precomputed mean
  - sae_style_guide: encode step noise via SAE, blend with style reference
"""

import torch
from tqdm import tqdm

from src.pixart.dpm_solver import (
    get_named_beta_schedule,
    NoiseScheduleVP,
    model_wrapper,
    DPM_Solver,
    DPMS,
)


# ─────────────────────── Conditioning helper ────────────────────────────

PATCH_SIZE = 2


def patchify_latent(x: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Convert [B, C, H, W] latent maps into [B, N, D] patch tokens."""
    b, c, h, w = x.shape
    p = patch_size
    assert h % p == 0 and w % p == 0, f"Latent spatial size must be divisible by {p}"

    x = x.reshape(b, c, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // p) * (w // p), c * p * p)


def unpatchify_latent(
    tokens: torch.Tensor,
    channels: int,
    height: int,
    width: int,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """Convert [B, N, D] patch tokens back into [B, C, H, W] latent maps."""
    b, n, d = tokens.shape
    p = patch_size
    h = height // p
    w = width // p
    assert n == h * w, f"Expected {h * w} tokens, got {n}"
    assert d == channels * p * p, f"Expected token dim {channels * p * p}, got {d}"

    x = tokens.reshape(b, h, w, channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, channels, height, width)


def as_latent_like(feature: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Accept either saved token features [B, N, D] or latent maps [B, C, H, W]."""
    if feature.ndim == 4:
        return feature.to(device=reference.device, dtype=reference.dtype)
    if feature.ndim == 3:
        _, channels, height, width = reference.shape
        tokens = feature.to(device=reference.device, dtype=reference.dtype)
        return unpatchify_latent(tokens, channels=channels, height=height, width=width)
    raise ValueError(f"Unsupported feature shape: {tuple(feature.shape)}")


def prepare_cond(model, t5_embedder, prompts, device):
    """Encode prompts into caption embeddings and build the null embedding.

    Returns:
        caption_embs:  (B, 1, L, 4096)   float
        null_y:        (B, 1, L, 4096)    float   (unconditional embedding)
        emb_masks:     (B, L)             int     attention mask
    """
    caption_embs, emb_masks = t5_embedder.get_text_embeddings(prompts)
    caption_embs = caption_embs.float()[:, None]  # (B, 1, L, C)

    # null_y: unconditional embedding from the model's learned y_embedding
    null_y = (model.y_embedder.y_embedding[None]
              .repeat(len(prompts), 1, 1)[:, None])   # (B, 1, L, C)

    return caption_embs, null_y, emb_masks


# ─────────────────────── Main sampling function ─────────────────────────

def do_sampling(
    model,
    z: torch.Tensor,
    caption_embs: torch.Tensor,
    null_y: torch.Tensor,
    emb_masks: torch.Tensor,
    cfg_scale: float,
    steps: int,
    device: torch.device,
    # mode parameters
    mode: str = "normal",
    ablation_steps: list | None = None,
    sae_config: dict | None = None,
    mean_features=None,
) -> tuple[torch.Tensor, list]:
    """Run the denoising loop.

    Args:
        model:        PixArt model (already on *device* and in eval mode)
        z:            initial noise  (B, 4, H/8, W/8)
        caption_embs: (B, 1, L, C)  conditional text embedding
        null_y:       (B, 1, L, C)  unconditional text embedding
        emb_masks:    (B, L)        attention mask
        cfg_scale:    classifier-free guidance scale
        steps:        number of sampling steps
        device:       torch device
        mode:         one of "normal", "save_feature", "mean_ablation", "sae_style_guide"
        ablation_steps: list of 1-based step indices for intervention
        sae_config:   dict with 'models', 'style_codes', 'alpha' (for sae_style_guide)

    Returns:
        (denoised_latent, all_features)
    """
    if ablation_steps is None:
        ablation_steps = []

    all_features: list[torch.Tensor] = []

    model_kwargs = dict(mask=emb_masks)

    # ── Normal mode: use full DPM-Solver++ order-2 directly ─────────
    if mode == "normal":
        dpm_solver = DPMS(
            model.forward_with_dpmsolver,
            condition=caption_embs,
            uncondition=null_y,
            cfg_scale=cfg_scale,
            model_kwargs=model_kwargs,
        )
        x = dpm_solver.sample(
            z, steps=steps, order=2,
            skip_type="time_uniform", method="multistep",
        )
        return x, all_features

    # ── Modes requiring step-by-step control ────────────────────────
    # Build noise schedule + model wrapper manually
    betas = get_named_beta_schedule("linear", 1000)
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(betas))

    model_fn = model_wrapper(
        model.forward_with_dpmsolver,
        noise_schedule,
        model_type="noise",
        model_kwargs=model_kwargs,
        guidance_type="classifier-free",
        condition=caption_embs,
        unconditional_condition=null_y,
        guidance_scale=cfg_scale,
    )

    dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
    timesteps = dpm_solver.get_time_steps(
        "time_uniform", noise_schedule.T,
        1. / noise_schedule.total_N, steps, device,
    )

    # First-order DDIM loop for per-step access
    x = z
    step_count = 0

    for i in tqdm(range(len(timesteps) - 1), desc="PixArt sampling"):
        step_count += 1
        s_t, t_t = timesteps[i], timesteps[i + 1]

        x_new = dpm_solver.dpm_solver_first_update(x, s_t, t_t)
        step_noise = x_new - x

        # ── mean ablation ───────────────────────────────────────────
        if mode == "mean_ablation" and step_count in ablation_steps:
            if mean_features is None:
                raise ValueError("mean_ablation requires loaded mean step updates")
            ref_noise = as_latent_like(mean_features[step_count - 1], step_noise)
            step_noise = 0.5 * step_noise + 0.5 * ref_noise
            x_new = x + step_noise
            print(f"Applied mean ablation at step {step_count}")

        # ── SAE style guide ─────────────────────────────────────────
        if (mode == "sae_style_guide"
                and sae_config is not None
                and step_count in ablation_steps):
            timestep = step_count - 1   # SAE checkpoints are 0-based
            if (timestep in sae_config['models']
                    and timestep in sae_config['style_codes']):
                info = sae_config['models'][timestep]
                sae = info['sae']
                mu = info['mu']
                sigma_norm = info['sigma']
                alpha = sae_config['alpha']
                ref_code = sae_config['style_codes'][timestep].to(
                    device=step_noise.device, dtype=torch.float32)

                noise_tokens = patchify_latent(step_noise).squeeze(0).float()
                noise_norm = (noise_tokens - mu) / (sigma_norm + 1e-8)
                z_enc = sae.encode(noise_norm)
                z_blended = (1.0 - alpha) * z_enc + alpha * ref_code
                recons = sae.decoder(z_blended) + sae.pre_bias
                noise_new = recons * (sigma_norm + 1e-8) + mu

                step_noise = unpatchify_latent(
                    noise_new.unsqueeze(0),
                    channels=step_noise.shape[1],
                    height=step_noise.shape[2],
                    width=step_noise.shape[3],
                ).to(dtype=x.dtype)
                x_new = x + step_noise
                print(f"Applied SAE style guide at step {step_count} (alpha={alpha:.2f})")

        # ── save features ───────────────────────────────────────────
        if mode == "save_feature":
            feature = patchify_latent(step_noise).detach().cpu().to(dtype=torch.float32)
            all_features.append(feature)

        x = x_new

    return x, all_features
