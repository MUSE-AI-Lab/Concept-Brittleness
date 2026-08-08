"""
Configuration for Step-wise Style SAE training.
"""
from dataclasses import dataclass, field
from typing import Optional
import json
import os


@dataclass
class SAEConfig:
    """Configuration for a single SAE."""
    d_model: int = 64           # token dimension
    n_dirs: int = 256           # number of latent features (4x expansion)
    k: int = 10                 # TopK sparsity
    auxk: int = 128             # AuxK for dead feature prevention
    auxk_coef: float = 1 / 32  # AuxK loss coefficient
    lr: float = 1e-4            # learning rate
    eps: float = 6.25e-10       # Adam epsilon
    dead_steps_threshold: int = 5000  # steps to consider feature dead


@dataclass
class TrainConfig:
    """Configuration for the full training pipeline."""
    # Data
    data_path: str = "feature/step_updates_style_2026.pt"
    use_nonstyle: bool = False  # whether to also include nonstyle data
    nonstyle_path: str = "feature/step_updates_nonstyle_2026.pt"

    # Training
    batch_size: int = 4096
    n_epochs: int = 30
    log_interval: int = 50       # log every N steps
    save_interval: int = 500     # save checkpoint every N steps

    # SAE
    sae: SAEConfig = field(default_factory=SAEConfig)

    # Paths
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results_sae"

    # Which timesteps to train (None = all 50)
    timesteps: Optional[list] = None

    # Style info
    n_classes: int = 10
    samples_per_class: int = 50
    n_tokens: int = 256
    style_names: list = field(default_factory=lambda: [
        "abstract art", "realistic photo", "cartoon style",
        "vintage photo", "modern design", "black and white photo",
        "3D render", "oil painting", "pixel art", "ink sketch"
    ])

    def get_timesteps(self):
        if self.timesteps is not None:
            return self.timesteps
        return list(range(50))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        cfg_dict = {
            "data_path": self.data_path,
            "use_nonstyle": self.use_nonstyle,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "sae": {
                "d_model": self.sae.d_model,
                "n_dirs": self.sae.n_dirs,
                "k": self.sae.k,
                "auxk": self.sae.auxk,
                "auxk_coef": self.sae.auxk_coef,
                "lr": self.sae.lr,
                "dead_steps_threshold": self.sae.dead_steps_threshold,
            },
            "n_classes": self.n_classes,
            "samples_per_class": self.samples_per_class,
            "style_names": self.style_names,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        sae_d = d.pop("sae", {})
        cfg = cls(**{k: v for k, v in d.items() if k != "sae"})
        cfg.sae = SAEConfig(**sae_d)
        return cfg
