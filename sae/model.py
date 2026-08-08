"""
TopK Sparse Autoencoder with AuxK dead feature prevention.
Adapted from stable-concepts (https://github.com/... /sparse_autoencoder).
Modified for token-level denoising step-update features.
"""

import torch
import torch.nn as nn
import os
import json


class SparseAutoencoder(nn.Module):
    """
    Top-K Sparse Autoencoder:
        latents = relu(topk(encoder(x - pre_bias) + latent_bias))
        recons  = decoder(latents) + pre_bias
    """

    def __init__(
        self,
        n_dirs_local: int,
        d_model: int,
        k: int,
        auxk: int | None,
        dead_steps_threshold: int,
        feature_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_dirs_local = n_dirs_local
        self.d_model = d_model
        self.k = k
        self.auxk = auxk
        self.dead_steps_threshold = dead_steps_threshold
        self.feature_dropout = feature_dropout

        self.encoder = nn.Linear(d_model, n_dirs_local, bias=False)
        self.decoder = nn.Linear(n_dirs_local, d_model, bias=False)

        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.latent_bias = nn.Parameter(torch.zeros(n_dirs_local))

        # Track how long each feature has not been activated
        self.stats_last_nonzero: torch.Tensor
        self.register_buffer(
            "stats_last_nonzero",
            torch.zeros(n_dirs_local, dtype=torch.long)
        )

        def auxk_mask_fn(x):
            dead_mask = self.stats_last_nonzero > dead_steps_threshold
            x.data *= dead_mask  # inplace to save memory
            return x

        self.auxk_mask_fn = auxk_mask_fn

        # Tied initialization
        self.decoder.weight.data = self.encoder.weight.data.T.clone()
        self.decoder.weight.data = self.decoder.weight.data.T.contiguous().T

        unit_norm_decoder_(self)

    @property
    def n_dirs(self):
        return self.n_dirs_local

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, x):
        """Encode input to sparse latent (inference only, no stats update)."""
        x = x - self.pre_bias
        latents_pre_act = self.encoder(x) + self.latent_bias

        vals, inds = torch.topk(latents_pre_act, k=self.k, dim=-1)

        latents = torch.zeros_like(latents_pre_act)
        latents.scatter_(-1, inds, torch.relu(vals))

        return latents

    def forward(self, x):
        """Forward pass with stats tracking and AuxK."""
        x = x - self.pre_bias
        latents_pre_act = self.encoder(x) + self.latent_bias
        vals, inds = torch.topk(latents_pre_act, k=self.k, dim=-1)

        # Update activation stats
        tmp = torch.zeros_like(self.stats_last_nonzero)
        tmp.scatter_add_(
            0,
            inds.reshape(-1),
            (vals > 1e-3).to(tmp.dtype).reshape(-1),
        )
        self.stats_last_nonzero *= 1 - tmp.clamp(max=1)
        self.stats_last_nonzero += 1

        # AuxK: select top-k among dead features
        if self.auxk is not None:
            auxk_vals, auxk_inds = torch.topk(
                self.auxk_mask_fn(latents_pre_act),
                k=self.auxk,
                dim=-1
            )
        else:
            auxk_inds = None
            auxk_vals = None

        vals = torch.relu(vals)
        if auxk_vals is not None:
            auxk_vals = torch.relu(auxk_vals)

        # Feature dropout: randomly zero out some top-k features during training
        # to encourage more diverse feature usage and reduce dead features.
        # Surviving values are scaled by 1/(1-p) to preserve expected magnitude.
        if self.training and self.feature_dropout > 0:
            drop_mask = torch.bernoulli(
                torch.full_like(vals, 1.0 - self.feature_dropout)
            )
            vals = vals * drop_mask / (1.0 - self.feature_dropout)

        # Sparse reconstruction using COO sparse mm
        rows, cols = latents_pre_act.size()
        row_indices = torch.arange(rows, device=inds.device).unsqueeze(1).expand(-1, self.k).reshape(-1)
        vals_flat = vals.reshape(-1)
        inds_flat = inds.reshape(-1)

        indices = torch.stack([row_indices, inds_flat])
        sparse_tensor = torch.sparse_coo_tensor(indices, vals_flat, torch.Size([rows, cols]))
        recons = torch.sparse.mm(sparse_tensor, self.decoder.weight.T) + self.pre_bias

        return recons, {
            "inds": inds,
            "vals": vals,
            "auxk_inds": auxk_inds,
            "auxk_vals": auxk_vals,
        }

    def decode_sparse(self, inds, vals):
        """Decode from sparse indices and values."""
        rows, cols = inds.shape[0], self.n_dirs
        row_indices = torch.arange(rows, device=inds.device).unsqueeze(1).expand(-1, inds.shape[1]).reshape(-1)
        vals_flat = vals.reshape(-1)
        inds_flat = inds.reshape(-1)

        indices = torch.stack([row_indices, inds_flat])
        sparse_tensor = torch.sparse_coo_tensor(indices, vals_flat, torch.Size([rows, cols]))
        recons = torch.sparse.mm(sparse_tensor, self.decoder.weight.T) + self.pre_bias
        return recons

    def save_to_disk(self, path: str):
        """Save model config and weights."""
        cfg = {
            "n_dirs_local": self.n_dirs_local,
            "d_model": self.d_model,
            "k": self.k,
            "auxk": self.auxk,
            "dead_steps_threshold": self.dead_steps_threshold,
            "feature_dropout": self.feature_dropout,
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(cfg, f, indent=2)
        torch.save(
            {"state_dict": self.state_dict()},
            os.path.join(path, "state_dict.pth")
        )

    @classmethod
    def load_from_disk(cls, path: str, map_location="cpu"):
        """Load model from saved config and weights."""
        with open(os.path.join(path, "config.json"), 'r') as f:
            cfg = json.load(f)
        ae = cls(
            n_dirs_local=cfg["n_dirs_local"],
            d_model=cfg["d_model"],
            k=cfg["k"],
            auxk=cfg["auxk"],
            dead_steps_threshold=cfg["dead_steps_threshold"],
            feature_dropout=cfg.get("feature_dropout", 0.0),
        )
        state_dict = torch.load(
            os.path.join(path, "state_dict.pth"),
            map_location=map_location,
            weights_only=True,
        )["state_dict"]
        ae.load_state_dict(state_dict)
        return ae


def unit_norm_decoder_(autoencoder: SparseAutoencoder) -> None:
    """Unit normalize decoder columns."""
    norms = autoencoder.decoder.weight.data.norm(dim=0, keepdim=True)
    autoencoder.decoder.weight.data /= norms.clamp(min=1e-8)


def unit_norm_decoder_grad_adjustment_(autoencoder: SparseAutoencoder) -> None:
    """Project out gradient component parallel to decoder columns."""
    if autoencoder.decoder.weight.grad is None:
        return
    autoencoder.decoder.weight.grad += (
        torch.einsum(
            "bn,bn->n",
            autoencoder.decoder.weight.data,
            autoencoder.decoder.weight.grad
        ) * autoencoder.decoder.weight.data * -1
    )
