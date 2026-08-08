"""
Model-loading and text-encoding utilities for PixArt-α.

Dependencies:
  - torch
  - transformers  (T5EncoderModel, AutoTokenizer)
  - diffusers     (AutoencoderKL)
"""

import os
import torch
from transformers import T5EncoderModel, AutoTokenizer
from diffusers.models import AutoencoderKL

from src.pixart.model import PixArt_XL_2


def _require_local_path(path: str, label: str) -> str:
    """Require a local file/directory path to exist before loading."""
    if not path:
        raise ValueError(f"{label} path must be provided.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    return path


# ─────────────────────── T5 text embedder ───────────────────────────────

class T5Embedder:
    """Loads T5-v1.1-XXL and produces text embeddings for PixArt-α."""

    def __init__(self, device, t5_path, torch_dtype=None, model_max_length=120):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.model_max_length = model_max_length
        t5_path = _require_local_path(t5_path, "PixArt T5")

        print(f"Loading T5 tokenizer from {t5_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(t5_path, local_files_only=True)

        print(f"Loading T5 model from {t5_path}")
        self.model = T5EncoderModel.from_pretrained(
            t5_path,
            low_cpu_mem_usage=True,
            local_files_only=True,
            torch_dtype=self.torch_dtype,
            device_map={'shared': self.device, 'encoder': self.device},
        ).eval()

    def get_text_embeddings(self, texts):
        """Encode a list of strings into T5 hidden states and attention masks.

        Returns:
            embs:  (B, model_max_length, 4096)
            masks: (B, model_max_length)   1=real token, 0=padding
        """
        texts = [t.strip() for t in texts]
        tok = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt',
        )
        with torch.no_grad():
            embs = self.model(
                input_ids=tok['input_ids'].to(self.device),
                attention_mask=tok['attention_mask'].to(self.device),
            )['last_hidden_state'].detach()
        return embs, tok['attention_mask'].to(self.device)


# ─────────────────────── Model loaders ──────────────────────────────────

def load_pixart_model(model_path: str, image_size: int = 256,
                      device: str = "cuda", dtype=torch.float16):
    """Load a PixArt-α XL/2 checkpoint.

    The checkpoint is expected to be a ``.pth`` file with key ``'state_dict'``
    (the standard PixArt-alpha release format).
    """
    latent_size = image_size // 8
    lewei_scale = {256: 1, 512: 1, 1024: 2}.get(image_size, 1)

    model = PixArt_XL_2(input_size=latent_size, lewei_scale=lewei_scale).to(device)

    model_path = _require_local_path(model_path, "PixArt checkpoint")
    print(f"Loading PixArt model from {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    # pos_embed is recomputed from sin-cos; delete to avoid shape mismatch
    state_dict.pop('pos_embed', None)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    model.eval()
    model.to(dtype)
    return model


def load_vae(vae_path: str, device: str = "cuda"):
    """Load the SD-VAE-ft-ema (``AutoencoderKL`` from diffusers)."""
    vae_path = _require_local_path(vae_path, "PixArt VAE")
    print(f"Loading VAE from {vae_path}")
    return AutoencoderKL.from_pretrained(vae_path, local_files_only=True).to(device)


def load_t5(t5_path: str, device: str = "cuda",
            torch_dtype=None, model_max_length: int = 120):
    """Load T5 embedder."""
    return T5Embedder(device=device, t5_path=t5_path,
                      torch_dtype=torch_dtype,
                      model_max_length=model_max_length)
