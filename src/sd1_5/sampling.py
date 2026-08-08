"""DDIM sampling for SD1.5 with step-update feature extraction and intervention."""

from __future__ import annotations

import torch
from tqdm import tqdm


PATCH_SIZE = 2
FEATURE_DIM = 16
VALID_MODES = {"normal", "save_feature", "mean_ablation", "sae_style_guide"}


def patchify_latent(x: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Convert latent maps [B,C,H,W] to row-major patch tokens [B,N,C*p*p]."""
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D latent tensor, got shape {tuple(x.shape)}")
    batch, channels, height, width = x.shape
    p = patch_size
    if height % p or width % p:
        raise ValueError(f"Latent size {(height, width)} must be divisible by patch size {p}")
    x = x.reshape(batch, channels, height // p, p, width // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(batch, (height // p) * (width // p), channels * p * p)


def unpatchify_latent(
    tokens: torch.Tensor,
    channels: int,
    height: int,
    width: int,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    """Invert :func:`patchify_latent`."""
    if tokens.ndim != 3:
        raise ValueError(f"Expected 3D tokens, got shape {tuple(tokens.shape)}")
    batch, token_count, feature_dim = tokens.shape
    p = patch_size
    token_h, token_w = height // p, width // p
    if token_count != token_h * token_w:
        raise ValueError(f"Expected {token_h * token_w} tokens, got {token_count}")
    if feature_dim != channels * p * p:
        raise ValueError(f"Expected token dim {channels * p * p}, got {feature_dim}")
    x = tokens.reshape(batch, token_h, token_w, channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(batch, channels, height, width)


def as_latent_like(feature: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(feature, torch.Tensor):
        feature = torch.as_tensor(feature)
    if feature.ndim == 4:
        if tuple(feature.shape) != tuple(reference.shape):
            raise ValueError(
                f"Mean feature shape {tuple(feature.shape)} does not match update {tuple(reference.shape)}"
            )
        return feature.to(device=reference.device, dtype=reference.dtype)
    if feature.ndim == 3:
        _, channels, height, width = reference.shape
        return unpatchify_latent(
            feature.to(device=reference.device, dtype=reference.dtype),
            channels=channels,
            height=height,
            width=width,
        )
    raise ValueError(f"Unsupported feature shape: {tuple(feature.shape)}")


def _encode_prompt(pipe, prompt: str, device: torch.device) -> torch.Tensor:
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
    )
    return torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)


def _apply_sae_update(
    step_update: torch.Tensor,
    timestep: int,
    sae_config: dict,
) -> torch.Tensor:
    info = sae_config["models"][timestep]
    sae = info["sae"]
    mu = info["mu"].to(device=step_update.device, dtype=torch.float32)
    sigma = info["sigma"].to(device=step_update.device, dtype=torch.float32)
    alpha = float(sae_config["alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"SAE alpha must be between 0 and 1, got {alpha}")
    ref_code = sae_config["style_codes"][timestep].to(
        device=step_update.device, dtype=torch.float32
    )

    tokens = patchify_latent(step_update).squeeze(0).float()
    if tokens.shape[-1] != mu.numel():
        raise ValueError(
            f"SD1.5 update dim {tokens.shape[-1]} does not match SAE normalization dim {mu.numel()}"
        )
    if tuple(ref_code.shape) != (tokens.shape[0], sae.n_dirs):
        raise ValueError(
            f"Style code shape {tuple(ref_code.shape)} does not match "
            f"expected {(tokens.shape[0], sae.n_dirs)}"
        )

    normalized = (tokens - mu) / (sigma + 1e-8)
    code = sae.encode(normalized)
    blended = (1.0 - alpha) * code + alpha * ref_code
    reconstructed = sae.decoder(blended) + sae.pre_bias
    guided_tokens = reconstructed * (sigma + 1e-8) + mu
    return unpatchify_latent(
        guided_tokens.unsqueeze(0),
        channels=step_update.shape[1],
        height=step_update.shape[2],
        width=step_update.shape[3],
    ).to(dtype=step_update.dtype)


@torch.inference_mode()
def do_sampling(
    pipe,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    mode: str = "normal",
    ablation_steps: list[int] | None = None,
    mean_features=None,
    sae_config: dict | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run deterministic DDIM and expose the CFG-conditioned update at every step."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported SD1.5 sampling mode: {mode}")
    if steps <= 0:
        raise ValueError(f"SD1.5 steps must be positive, got {steps}")
    if width <= 0 or height <= 0:
        raise ValueError(f"SD1.5 width and height must be positive, got {(width, height)}")
    if width % 16 or height % 16:
        raise ValueError("SD1.5 width and height must be divisible by 16")
    ablation_steps = set(ablation_steps or [])
    invalid_steps = sorted(step for step in ablation_steps if step < 1 or step > steps)
    if invalid_steps:
        raise ValueError(f"Ablation steps outside 1..{steps}: {invalid_steps}")

    device = pipe._execution_device
    prompt_embeds = _encode_prompt(pipe, prompt, device)
    pipe.scheduler.set_timesteps(steps, device=device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.unet.config.in_channels,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=None,
    )

    all_features: list[torch.Tensor] = []
    for step_index, timestep in enumerate(
        tqdm(pipe.scheduler.timesteps, desc="SD1.5 DDIM sampling"), start=1
    ):
        latent_input = torch.cat([latents, latents], dim=0)
        latent_input = pipe.scheduler.scale_model_input(latent_input, timestep)
        noise_pred = pipe.unet(
            latent_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + cfg_scale * (noise_text - noise_uncond)

        next_latents = pipe.scheduler.step(
            noise_pred,
            timestep,
            latents,
            eta=0.0,
            generator=generator,
            return_dict=True,
        ).prev_sample
        step_update = next_latents - latents

        if mode == "mean_ablation" and step_index in ablation_steps:
            if mean_features is None:
                raise ValueError("mean_ablation requires loaded mean features")
            reference = as_latent_like(mean_features[step_index - 1], step_update)
            step_update = 0.5 * step_update + 0.5 * reference

        if mode == "sae_style_guide" and step_index in ablation_steps:
            sae_timestep = step_index - 1
            if sae_config is None:
                raise ValueError("sae_style_guide requires an SAE configuration")
            if (
                sae_timestep not in sae_config["models"]
                or sae_timestep not in sae_config["style_codes"]
            ):
                raise KeyError(f"Missing SD1.5 SAE or style code for timestep {sae_timestep}")
            step_update = _apply_sae_update(step_update, sae_timestep, sae_config)

        if mode == "save_feature":
            all_features.append(
                patchify_latent(step_update).detach().cpu().to(torch.float32)
            )

        latents = latents + step_update

    return latents, all_features
