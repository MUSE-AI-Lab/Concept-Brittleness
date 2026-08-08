"""
Utility functions for loading SD3.5 Large model components.
Mirrors the pattern of src/flux1/util.py.
"""

import os
import math

import torch
from safetensors import safe_open

from src.sd3_5.other_impls import (
    SD3Tokenizer,
    SDClipModel,
    SDXLClipG,
    T5XXLModel,
)
from src.sd3_5.sd3_impls import (
    SDVAE,
    BaseModel,
    SD3LatentFormat,
)


#################################################################################################
### Weight loading helper
#################################################################################################


def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    """Apply the weights in a safetensors file to the pytorch module."""
    for key in ckpt.keys():
        model_key = key
        if remap is not None and key in remap:
            model_key = remap[key]
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix):].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(f"Skipping key '{model_key}' in safetensors file as '{p}' does not exist in python model")
                        break
            if obj is None:
                continue
            try:
                tensor = ckpt.get_tensor(key).to(device=device)
                if dtype is not None and tensor.dtype != torch.int32:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                if obj.shape != tensor.shape:
                    print(f"W: shape mismatch for key {model_key}, {obj.shape} != {tensor.shape}")
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e


#################################################################################################
### Model component configs (for SD3.5 Large only)
#################################################################################################

CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}

CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}

T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}

# SD3.5 Large configuration
SD35_LARGE_CONFIG = {
    "shift": 3.0,
    "steps": 40,
    "cfg": 4.5,
    "sampler": "dpmpp_2m",
}


#################################################################################################
### Loading functions
#################################################################################################


def require_local_path(path: str, label: str) -> str:
    """Require a local file/directory path to exist before loading."""
    if not path:
        raise ValueError(f"{label} path must be provided.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    return path


def resolve_tokenizer_path(model_folder: str, explicit_path: str | None, candidates: list[str], label: str) -> str:
    """Resolve a tokenizer directory from an explicit path or common local folder names."""
    if explicit_path:
        return require_local_path(explicit_path, label)

    searched_paths = []
    for candidate in candidates:
        path = os.path.join(model_folder, candidate)
        searched_paths.append(path)
        if os.path.exists(path):
            return path

    candidate_list = ", ".join(searched_paths)
    raise FileNotFoundError(
        f"Could not find {label} locally. Provide the path explicitly or place it in one of: {candidate_list}"
    )


def load_tokenizer(model_folder: str, clip_l_tokenizer_path: str | None = None,
                   t5_tokenizer_path: str | None = None) -> SD3Tokenizer:
    """Load the SD3 tokenizer (CLIP-L, CLIP-G, T5-XXL tokenizers)."""
    print("Loading SD3.5 tokenizers...")
    model_folder = require_local_path(model_folder, "SD3.5 model folder")
    clip_l_tokenizer_path = resolve_tokenizer_path(
        model_folder,
        clip_l_tokenizer_path,
        ["tokenizer", "text_encoder", "clip-vit-large-patch14", "clip_l_tokenizer"],
        "SD3.5 CLIP tokenizer",
    )
    t5_tokenizer_path = resolve_tokenizer_path(
        model_folder,
        t5_tokenizer_path,
        ["tokenizer_3", "text_encoder_3", "t5-v1_1-xxl", "t5xxl_tokenizer"],
        "SD3.5 T5 tokenizer",
    )
    return SD3Tokenizer(clip_l_tokenizer_path, t5_tokenizer_path)


def resolve_file_path(base_dir: str, explicit_path: str | None, candidates: list[str], label: str) -> str:
    """Resolve a local file from an explicit path or common candidate locations."""
    if explicit_path:
        return require_local_path(explicit_path, label)

    searched_paths = []
    for candidate in candidates:
        path = os.path.join(base_dir, candidate)
        searched_paths.append(path)
        if os.path.exists(path):
            return path

    candidate_list = ", ".join(searched_paths)
    raise FileNotFoundError(
        f"Could not find {label} locally. Provide the path explicitly or place it in one of: {candidate_list}"
    )


def load_clip_g(model_folder: str, device: str = "cpu") -> SDXLClipG:
    """Load OpenCLIP bigG text encoder."""
    model_folder = require_local_path(model_folder, "SD3.5 model folder")
    print("Loading OpenCLIP bigG...")
    clip_g_path = resolve_file_path(
        model_folder,
        None,
        ["clip_g.safetensors", "text_encoders/clip_g.safetensors"],
        "SD3.5 clip_g",
    )
    with safe_open(clip_g_path, framework="pt", device="cpu") as f:
        model = SDXLClipG(CLIPG_CONFIG, device=device, dtype=torch.float32)
        load_into(f, model.transformer, "", device, torch.float32)
    return model


def load_clip_l(model_folder: str) -> SDClipModel:
    """Load OpenAI CLIP-L text encoder."""
    model_folder = require_local_path(model_folder, "SD3.5 model folder")
    print("Loading OpenAI CLIP-L...")
    clip_l_path = resolve_file_path(
        model_folder,
        None,
        ["clip_l.safetensors", "text_encoders/clip_l.safetensors"],
        "SD3.5 clip_l",
    )
    with safe_open(clip_l_path, framework="pt", device="cpu") as f:
        model = SDClipModel(
            layer="hidden",
            layer_idx=-2,
            device="cpu",
            dtype=torch.float32,
            layer_norm_hidden_state=False,
            return_projected_pooled=False,
            textmodel_json_config=CLIPL_CONFIG,
        )
        load_into(f, model.transformer, "", "cpu", torch.float32)
    return model


def load_t5xxl(model_folder: str, device: str = "cpu", dtype=torch.float32) -> T5XXLModel:
    """Load Google T5-v1.1-XXL text encoder."""
    model_folder = require_local_path(model_folder, "SD3.5 model folder")
    print("Loading Google T5-v1-XXL...")
    t5_path = resolve_file_path(
        model_folder,
        None,
        [
            "t5xxl.safetensors",
            "t5xxl_fp16.safetensors",
            "text_encoders/t5xxl.safetensors",
            "text_encoders/t5xxl_fp16.safetensors",
        ],
        "SD3.5 t5xxl",
    )
    with safe_open(t5_path, framework="pt", device="cpu") as f:
        model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
        load_into(f, model.transformer, "", device, dtype)
    return model


def load_sd3_model(model_path: str, shift: float = 3.0, verbose: bool = False, device: str = "cuda") -> BaseModel:
    """Load the SD3.5 Large diffusion model."""
    model_path = require_local_path(model_path, "SD3.5 checkpoint")
    print(f"Loading SD3.5 model {os.path.basename(model_path)}...")
    with safe_open(model_path, framework="pt", device="cpu") as f:
        model = BaseModel(
            shift=shift,
            file=f,
            prefix="model.diffusion_model.",
            device=device,
            dtype=torch.float16,
            verbose=verbose,
        ).eval()
        load_into(f, model, "model.", device, torch.float16)
    return model


def load_vae(model_path: str, dtype: torch.dtype = torch.float16) -> SDVAE:
    """Load the SD3 VAE model."""
    model_path = require_local_path(model_path, "SD3.5 VAE")
    print("Loading VAE model...")
    with safe_open(model_path, framework="pt", device="cpu") as f:
        vae = SDVAE(device="cpu", dtype=dtype).eval().cpu()
        prefix = ""
        if any(k.startswith("first_stage_model.") for k in f.keys()):
            prefix = "first_stage_model."
        load_into(f, vae, prefix, "cpu", dtype)
    return vae
