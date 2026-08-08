"""
Sampling functions for SD3.5 with mode support.
Mirrors the pattern of src/flux1/sampling.py with four modes:
  - normal: standard generation
  - save_feature: save intermediate features during generation
  - mean_ablation: apply mean ablation at specified steps
  - sae_style_guide: encode noise via SAE, blend with style reference, decode back
"""

import math
import torch
import numpy as np
from tqdm import tqdm

from src.sd3_5.sd3_impls import (
    BaseModel,
    CFGDenoiser,
    SD3LatentFormat,
    ModelSamplingDiscreteFlow,
    sample_euler,
    sample_dpmpp_2m,
    append_dims,
    to_d,
)


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


def get_empty_latent(batch_size: int, width: int, height: int, seed: int, device: str = "cuda") -> torch.Tensor:
    """Create empty latent noise tensor."""
    shape = (batch_size, 16, height // 8, width // 8)
    latents = torch.zeros(shape, device=device)
    for i in range(shape[0]):
        prng = torch.Generator(device=device).manual_seed(int(seed + i))
        latents[i] = torch.randn(shape[1:], generator=prng, device=device)
    return latents


def get_noise(seed: int, latent: torch.Tensor) -> torch.Tensor:
    """Generate noise tensor with given seed matching latent shape."""
    generator = torch.manual_seed(seed)
    return torch.randn(
        latent.size(),
        dtype=torch.float32,
        layout=latent.layout,
        generator=generator,
        device="cpu",
    ).to(latent.dtype)


def get_sigmas(sampling: ModelSamplingDiscreteFlow, steps: int) -> torch.FloatTensor:
    """Compute sigma schedule for the given number of steps."""
    start = sampling.timestep(sampling.sigma_max)
    end = sampling.timestep(sampling.sigma_min)
    timesteps = torch.linspace(start, end, steps)
    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(sampling.sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def max_denoise(model_sampling: ModelSamplingDiscreteFlow, sigmas: torch.Tensor) -> bool:
    """Check if we should apply max denoising."""
    max_sigma = float(model_sampling.sigma_max)
    sigma = float(sigmas[0])
    return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma


def get_cond(tokenizer, clip_l, clip_g, t5xxl, prompt: str):
    """Encode a text prompt into conditioning tensors."""
    tokens = tokenizer.tokenize_with_weights(prompt)
    l_out, l_pooled = clip_l.encode_token_weights(tokens["l"])
    g_out, g_pooled = clip_g.encode_token_weights(tokens["g"])
    t5_out, t5_pooled = t5xxl.encode_token_weights(tokens["t5xxl"])
    lg_out = torch.cat([l_out, g_out], dim=-1)
    lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
    return torch.cat([lg_out, t5_out], dim=-2), torch.cat((l_pooled, g_pooled), dim=-1)


def fix_cond(cond):
    """Move conditioning to cuda and convert to half precision."""
    cond_tensor, pooled = cond[0].half().cuda(), cond[1].half().cuda()
    return {"c_crossattn": cond_tensor, "y": pooled}


def do_sampling(
    model: BaseModel,
    latent: torch.Tensor,
    seed: int,
    conditioning,
    neg_cond,
    steps: int,
    cfg_scale: float,
    sampler: str = "dpmpp_2m",
    denoise: float = 1.0,
    # Mode parameters
    mode: str = "normal",
    ablation_steps: list = None,
    sae_config: dict = None,
    mean_features=None,
) -> tuple:
    """
    Perform the sampling loop with mode support.
    
    Args:
        model: The SD3.5 BaseModel
        latent: Input latent tensor
        seed: Random seed
        conditioning: Positive conditioning dict
        neg_cond: Negative conditioning dict
        steps: Number of sampling steps
        cfg_scale: CFG scale
        sampler: Sampler name ('euler' or 'dpmpp_2m')
        denoise: Denoising strength
        mode: One of 'normal', 'save_feature', 'mean_ablation', 'sae_style_guide'
        ablation_steps: List of step indices for ablation/intervention
        sae_config: Dict for sae_style_guide mode
    
    Returns:
        (latent, all_features): The denoised latent and list of saved features
    """
    if ablation_steps is None:
        ablation_steps = []
    
    all_features = []
    
    latent = latent.half().cuda()
    model = model.cuda()
    noise = get_noise(seed, latent).cuda()
    sigmas = get_sigmas(model.model_sampling, steps).cuda()
    sigmas = sigmas[int(steps * (1 - denoise)):]
    
    conditioning = fix_cond(conditioning)
    neg_cond = fix_cond(neg_cond)
    
    noise_scaled = model.model_sampling.noise_scaling(
        sigmas[0], noise, latent, max_denoise(model.model_sampling, sigmas)
    )
    
    # For normal mode without any special processing, use the original sampler directly
    if mode == "normal" and not ablation_steps:
        denoiser = CFGDenoiser(model)
        extra_args = {
            "cond": conditioning,
            "uncond": neg_cond,
            "cond_scale": cfg_scale,
        }
        sample_fn = sample_dpmpp_2m if sampler == "dpmpp_2m" else sample_euler
        latent = sample_fn(denoiser, noise_scaled, sigmas, extra_args=extra_args)
        latent = SD3LatentFormat().process_out(latent)
        model = model.cpu()
        return latent, all_features
    
    # For modes that need step-by-step control, use a manual sampling loop.
    x = noise_scaled
    s_in = x.new_ones([x.shape[0]])
    denoiser = CFGDenoiser(model)
    extra_args = {
        "cond": conditioning,
        "uncond": neg_cond,
        "cond_scale": cfg_scale,
    }
    
    old_denoised = None  # For DPM++ 2M
    step_count = 0
    
    for i in tqdm(range(len(sigmas) - 1)):
        step_count += 1
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        with torch.autocast("cuda", dtype=torch.float16):
            denoised = denoiser(x, sigma * s_in, **extra_args)
        
        if sampler == "dpmpp_2m":
            sigma_fn = lambda t: t.neg().exp()
            t_fn = lambda sigma: sigma.log().neg()
            t, t_next = t_fn(sigma), t_fn(sigma_next)
            h = t_next - t
            if old_denoised is None or sigma_next == 0:
                noise_pred = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                noise_pred = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
            
            # The "noise" for this step is the difference
            step_noise = noise_pred - x
            old_denoised = denoised
        else:  # euler
            d = to_d(x, sigma, denoised)
            dt = sigma_next - sigma
            step_noise = d * dt
            noise_pred = x + step_noise
        
        # Apply mean ablation if in mean_ablation mode and current step is in ablation_steps
        if mode == "mean_ablation" and step_count in ablation_steps:
            if mean_features is None:
                raise ValueError("mean_ablation requires loaded mean step updates")
            ref_noise = as_latent_like(mean_features[step_count - 1], step_noise)
            step_noise = 0.5 * step_noise + 0.5 * ref_noise
            noise_pred = x + step_noise
            print(f"Applied mean ablation at step {step_count}")
        
        # Apply SAE style guide
        if mode == "sae_style_guide" and sae_config is not None and step_count in ablation_steps:
            timestep = step_count - 1  # steps are 1-based, SAE timesteps are 0-based
            if timestep in sae_config['models'] and timestep in sae_config['style_codes']:
                info = sae_config['models'][timestep]
                sae = info['sae']
                mu = info['mu']
                sigma_norm = info['sigma']
                alpha = sae_config['alpha']
                ref_code = sae_config['style_codes'][timestep].to(
                    device=step_noise.device, dtype=torch.float32
                )

                noise_tokens = patchify_latent(step_noise).squeeze(0).float()
                noise_norm = (noise_tokens - mu) / (sigma_norm + 1e-8)
                z = sae.encode(noise_norm)
                z_blended = (1.0 - alpha) * z + alpha * ref_code
                recons = sae.decoder(z_blended) + sae.pre_bias
                noise_new = recons * (sigma_norm + 1e-8) + mu

                step_noise = unpatchify_latent(
                    noise_new.unsqueeze(0),
                    channels=step_noise.shape[1],
                    height=step_noise.shape[2],
                    width=step_noise.shape[3],
                ).to(dtype=x.dtype)
                noise_pred = x + step_noise
                print(f"Applied SAE style guide at step {step_count} (alpha={alpha:.2f})")
        
        # Save features if in save_feature mode
        if mode == "save_feature":
            feature = patchify_latent(step_noise).detach().cpu().to(dtype=torch.float32)
            all_features.append(feature)
        
        x = noise_pred
    
    latent = SD3LatentFormat().process_out(x)
    model = model.cpu()
    return latent, all_features
