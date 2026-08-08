"""
Dataset utilities for loading noise features and creating token-level batches.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple


# ── Style class definitions ──────────────────────────────────────────────────
STYLE_NAMES = [
    "abstract art", "realistic photo", "cartoon style",
    "vintage photo", "modern design", "black and white photo",
    "3D render", "oil painting", "pixel art", "ink sketch"
]
N_CLASSES = 10
SAMPLES_PER_CLASS = 208


def get_style_label(sample_idx: int) -> int:
    """Return style class id (0-3) for a given sample index (1248-1951)."""
    # return (sample_idx-1248) // SAMPLES_PER_CLASS
    return sample_idx // SAMPLES_PER_CLASS


def load_feature_data(path: str, allow_unsafe_pickle: bool = False) -> dict:
    """Load feature trajectories while avoiding pickle by default.

    New trajectories use ``.pt`` and are loaded with ``weights_only=True``.
    Legacy object-array ``.npy`` files require explicit opt-in because NumPy
    must invoke pickle to deserialize them; only enable that for trusted files.
    """
    feature_path = Path(path)
    if feature_path.suffix == ".pt":
        data = torch.load(feature_path, map_location="cpu", weights_only=True)
    elif feature_path.suffix == ".npy":
        if not allow_unsafe_pickle:
            raise ValueError(
                "Legacy .npy feature files require pickle. Convert the file to .pt "
                "or pass --allow_unsafe_pickle only for a trusted local file."
            )
        data = np.load(feature_path, allow_pickle=True).item()
    else:
        raise ValueError(f"Unsupported feature file type: {feature_path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary of feature trajectories in {path}")
    # Normalize keys to int
    result = {}
    for k, v in data.items():
        result[int(k)] = v
    return result


def load_npy_data(path: str, allow_unsafe_pickle: bool = False) -> dict:
    """Compatibility alias for :func:`load_feature_data`."""
    return load_feature_data(path, allow_unsafe_pickle=allow_unsafe_pickle)


def extract_timestep_tokens(
    data: dict,
    timestep: int,
    sample_ids: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract token-level features for a specific timestep.

    Returns:
        tokens: [N_samples * tokens_per_sample, feature_dim] float32 tensor
        labels: [N_samples] int tensor (style class for each sample)
    """
    if sample_ids is None:
        sample_ids = sorted(data.keys())

    all_tokens = []
    all_labels = []

    for sid in sample_ids:
        sample = data[sid]
        feat = sample['features'][timestep]  # [1, tokens, feature_dim]
        if isinstance(feat, torch.Tensor):
            feat = feat.squeeze(0).float()  # [256, 64]
        else:
            feat = torch.tensor(feat).squeeze(0).float()
        all_tokens.append(feat)
        all_labels.append(get_style_label(sid))

    tokens = torch.cat(all_tokens, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)  # [N]

    return tokens, labels


def compute_norm_params(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-dim mean and std for z-score normalization."""
    mu = tokens.mean(dim=0)
    sigma = tokens.std(dim=0)
    return mu, sigma


def normalize(tokens: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Apply z-score normalization."""
    return (tokens - mu) / (sigma + eps)


def denormalize(tokens: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Reverse z-score normalization."""
    return tokens * (sigma + eps) + mu


class TokenDataset:
    """
    Iterable dataset that yields shuffled token-level batches.
    """

    def __init__(self, tokens: torch.Tensor, batch_size: int, device: str = "cuda"):
        """
        Args:
            tokens: [N, d] normalized token features
            batch_size: batch size for training
            device: target device
        """
        self.tokens = tokens
        self.batch_size = batch_size
        self.device = device
        self.n = tokens.shape[0]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def iterate(self):
        """Yield shuffled batches for one epoch."""
        perm = torch.randperm(self.n)
        for start in range(0, self.n, self.batch_size):
            end = min(start + self.batch_size, self.n)
            idx = perm[start:end]
            batch = self.tokens[idx].to(self.device)
            yield batch
