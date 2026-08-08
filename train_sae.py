"""
Phase 1: Train one SAE per timestep (50 total).

Usage:
    python train_sae.py                        # train all 50 steps
    python train_sae.py --steps 0 5 10 25 49   # train specific steps
    python train_sae.py --k 10 --epochs 200    # paper defaults
"""

import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from types import SimpleNamespace
from typing import Optional

from sae.model import SparseAutoencoder, unit_norm_decoder_, unit_norm_decoder_grad_adjustment_
from sae.dataset import (
    load_feature_data, extract_timestep_tokens,
    compute_norm_params, normalize, TokenDataset,
    N_CLASSES, SAMPLES_PER_CLASS, STYLE_NAMES
)


# ── Geometric median (from stable-concepts) ─────────────────────────────────

def weighted_average(points: torch.Tensor, weights: torch.Tensor):
    weights = weights / weights.sum()
    return (points * weights.view(-1, 1)).sum(dim=0)


def compute_geometric_median(
    points: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    maxiter: int = 100,
    ftol: float = 1e-20,
):
    with torch.no_grad():
        if weights is None:
            weights = torch.ones(points.shape[0], device=points.device)
        median = weighted_average(points, weights)

        for _ in range(maxiter):
            prev = median.clone()
            norms = torch.linalg.norm(points - median.view(1, -1), dim=1)
            new_weights = weights / torch.clamp(norms, min=eps)
            median = weighted_average(points, new_weights)
            if torch.linalg.norm(median - prev) < ftol:
                break

    return median


# ── Loss functions ───────────────────────────────────────────────────────────

def mse(recons: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return ((recons - x) ** 2).mean()


def normalized_mse(recon: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    xs_mu = xs.mean(dim=0)
    loss = mse(recon, xs) / mse(xs_mu[None, :].broadcast_to(xs.shape), xs)
    return loss


def explained_variance(recons: torch.Tensor, x: torch.Tensor) -> float:
    diff_var = torch.var(x - recons, dim=0, unbiased=False)
    x_var = torch.var(x, dim=0, unbiased=False)
    ev = (1 - diff_var / (x_var + 1e-8)).mean()
    return ev.item()


# ── Training one timestep ────────────────────────────────────────────────────

def train_one_timestep(
    data: dict,
    timestep: int,
    cfg: argparse.Namespace,
    device: str = "cuda",
):
    """Train SAE for a single timestep."""

    print(f"\n{'='*60}")
    print(f"  Training SAE for timestep {timestep}")
    print(f"{'='*60}")

    # ── 1. Extract and normalize tokens ──────────────────────────────────
    tokens, labels = extract_timestep_tokens(data, timestep)
    mu, sigma = compute_norm_params(tokens)
    tokens_norm = normalize(tokens, mu, sigma)

    print(f"  Tokens: {tokens_norm.shape} (= {tokens_norm.shape[0]//256} samples x 256 tokens)")
    print(f"  Mean range: [{mu.min():.4f}, {mu.max():.4f}]")
    print(f"  Std range:  [{sigma.min():.4f}, {sigma.max():.4f}]")

    # ── 2. Initialize SAE ────────────────────────────────────────────────
    ae = SparseAutoencoder(
        n_dirs_local=cfg.n_dirs,
        d_model=cfg.d_model,
        k=cfg.k,
        auxk=cfg.auxk,
        dead_steps_threshold=cfg.dead_steps_threshold,
        feature_dropout=cfg.feature_dropout,
    ).to(device)

    # Initialize pre_bias with geometric median
    sample_for_init = tokens_norm[:min(32768, len(tokens_norm))].float().to(device)
    ae.pre_bias.data = compute_geometric_median(sample_for_init)

    # Compute MSE scale for normalized loss
    mse_scale = 1.0 / ((sample_for_init.mean(dim=0) - sample_for_init) ** 2).mean().item()
    del sample_for_init

    # ── 3. Optimizer ─────────────────────────────────────────────────────
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr, eps=6.25e-10)

    # ── 4. Training loop ─────────────────────────────────────────────────
    dataset = TokenDataset(tokens_norm, cfg.batch_size, device)
    n_batches_per_epoch = len(dataset)
    total_steps = cfg.epochs * n_batches_per_epoch

    history = {
        "loss": [], "recon_mse": [], "auxk_loss": [],
        "explained_var": [], "dead_ratio_5k": [],
    }

    global_step = 0
    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        n_batch = 0

        for batch in dataset.iterate():
            recons, info = ae(batch)

            # ── Reconstruction loss
            recon_loss = mse_scale * mse(recons, batch)

            # ── AuxK loss
            if cfg.auxk is not None and info["auxk_inds"] is not None:
                auxk_recon = ae.decode_sparse(info["auxk_inds"], info["auxk_vals"])
                residual = batch - recons.detach() + ae.pre_bias.detach()
                auxk_loss = cfg.auxk_coef * normalized_mse(auxk_recon, residual).nan_to_num(0)
            else:
                auxk_loss = torch.tensor(0.0, device=device)

            loss = recon_loss + auxk_loss

            loss.backward()
            unit_norm_decoder_(ae)
            unit_norm_decoder_grad_adjustment_(ae)
            opt.step()
            opt.zero_grad()

            epoch_loss += loss.item()
            n_batch += 1
            global_step += 1

            # ── Logging
            if global_step % cfg.log_interval == 0 or global_step == 1:
                ev = explained_variance(recons, batch)
                bs = batch.shape[0]
                dead_5k = (ae.stats_last_nonzero > 5000 // bs).float().mean().item()

                history["loss"].append(loss.item())
                history["recon_mse"].append(recon_loss.item())
                history["auxk_loss"].append(auxk_loss.item())
                history["explained_var"].append(ev)
                history["dead_ratio_5k"].append(dead_5k)

                if global_step % (cfg.log_interval * 5) == 0:
                    elapsed = time.time() - t0
                    print(f"  [step {global_step:5d}/{total_steps}] "
                          f"loss={loss.item():.4f}  recon={recon_loss.item():.4f}  "
                          f"auxk={auxk_loss.item():.4f}  EV={ev:.4f}  "
                          f"dead={dead_5k:.3f}  "
                          f"({elapsed:.0f}s)")

            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()

        avg_loss = epoch_loss / max(n_batch, 1)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{cfg.epochs}: avg_loss={avg_loss:.5f}")

    elapsed_total = time.time() - t0
    print(f"  Training complete: {elapsed_total:.1f}s, best_loss={best_loss:.5f}")

    # ── 5. Save ──────────────────────────────────────────────────────────
    save_dir = os.path.join(cfg.checkpoint_dir, f"step_{timestep:02d}")
    ae.save_to_disk(save_dir)

    # Save normalization params
    np.savez(
        os.path.join(save_dir, "norm_params.npz"),
        mu=mu.numpy(),
        sigma=sigma.numpy(),
    )

    # Save training history
    np.savez(
        os.path.join(save_dir, "history.npz"),
        **{k: np.array(v) for k, v in history.items()}
    )

    print(f"  Saved to: {save_dir}")

    return ae, history


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train per-timestep Style SAEs")
    p.add_argument("--data", type=str, default="feature/step_updates_style_subset.pt",
                   help="Path to saved denoising step updates (.pt)")
    p.add_argument("--allow_unsafe_pickle", action="store_true",
                   help="Allow loading a trusted legacy object-array .npy file")
    p.add_argument("--steps", type=int, nargs="*", default=None,
                   help="Which timesteps to train (default: all 50)")
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_dirs", type=int, default=256,
                   help="Number of SAE latent features (4x expansion)")
    p.add_argument("--k", type=int, default=10, help="TopK sparsity")
    p.add_argument("--auxk", type=int, default=128, help="AuxK for dead features")
    p.add_argument("--auxk_coef", type=float, default=1/32, help="AuxK loss weight")
    p.add_argument("--batch_size", type=int, default=40960)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--dead_steps_threshold", type=int, default=50)
    p.add_argument("--feature_dropout", type=float, default=0.0,
                   help="Dropout probability for top-k features during training (0-1). "
                        "Randomly drops activated features to encourage diverse usage. "
                        "E.g. 0.1 drops 10%% of top-k features per forward pass.")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="Which style class IDs (0-9) to use for training. "
                        "Default: all classes. E.g. --classes 0 2 7")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    cfg = parse_args()

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Load data once
    print(f"Loading data from {cfg.data} ...")
    data = load_feature_data(cfg.data, allow_unsafe_pickle=cfg.allow_unsafe_pickle)
    print(f"  Loaded {len(data)} samples")

    # Filter by classes if specified
    if cfg.classes is not None:
        selected = sorted(cfg.classes)
        cls_set = set(selected)
        class_names = [STYLE_NAMES[c] for c in selected]
        data = {sid: v for sid, v in data.items()
                if sid // SAMPLES_PER_CLASS in cls_set}
        print(f"  Filtered to {len(data)} samples from {len(selected)} classes: "
              f"{selected} ({', '.join(class_names)})")
    else:
        print(f"  Using all {N_CLASSES} classes")

    # Which timesteps
    timesteps = cfg.steps if cfg.steps is not None else list(range(50))
    print(f"  Will train {len(timesteps)} SAEs: steps {timesteps}")

    # Check device
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  CUDA not available, falling back to CPU")
        device = "cpu"

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Train each timestep
    all_histories = {}
    t0_global = time.time()

    for t in timesteps:
        ae, hist = train_one_timestep(data, t, cfg, device=device)
        all_histories[t] = hist
        del ae
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    total_time = time.time() - t0_global
    print(f"\nAll done! Total time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # Save summary
    summary_path = os.path.join(cfg.checkpoint_dir, "training_summary.npz")
    np.savez(summary_path, timesteps=np.array(timesteps))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
