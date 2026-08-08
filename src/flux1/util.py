import math
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange
from safetensors.torch import load_file as load_sft

from src.flux1.model import Flux, FluxParams
from src.flux1.modules.autoencoder import AutoEncoder, AutoEncoderParams
from src.flux1.modules.conditioner import HFEmbedder

@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    repo_id: str
    repo_flow: str
    repo_ae: str
    lora_repo_id: str | None = None
    lora_filename: str | None = None

configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-dev-kontext": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-Kontext-dev",
        repo_flow="flux1-kontext-dev.safetensors",
        repo_ae="ae.safetensors",
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}

PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

def aspect_ratio_to_height_width(aspect_ratio: str, area: int = 1024**2) -> tuple[int, int]:
    width = float(aspect_ratio.split(":")[0])
    height = float(aspect_ratio.split(":")[1])
    ratio = width / height
    width = round(math.sqrt(area * ratio))
    height = round(math.sqrt(area / ratio))
    return 16 * (width // 16), 16 * (height // 16)

def _require_model_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found: {path}. Configure the relative model path in "
            "generate.py or place the files as documented in models/README.md."
        )
    return path


def load_flow_model(
    name: str,
    device: str | torch.device = "cuda",
    verbose: bool = True,
    model_dir: str | Path = "models/flux1-dev",
) -> Flux:
    # Loading Flux
    print("Init model")
    config = configs[name]

    ckpt_path = _require_model_path(
        Path(model_dir) / config.repo_flow,
        f"{name} checkpoint",
    )

    with torch.device("meta"):
        model = Flux(config.params).to(torch.bfloat16)
        
    print(f"Loading checkpoint: {ckpt_path}")
    # load_sft doesn't support torch.device
    sd = load_sft(str(ckpt_path), device=str(device))
    sd = optionally_expand_state_dict(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)

    return model

def load_t5(
    device: str | torch.device = "cuda",
    max_length: int = 512,
    model_dir: str | Path = "models/flux1-dev",
) -> HFEmbedder:
    print(f"Trying to load T5...")
    model_dir = _require_model_path(Path(model_dir), "FLUX model directory")
    return HFEmbedder(
        model_path=model_dir / "text_encoder_2",
        tokenizer_path=model_dir / "tokenizer_2",
        is_clip=False,
        max_length=max_length,
        torch_dtype=torch.bfloat16,
    ).to(device)

def load_clip(
    device: str | torch.device = "cuda",
    model_dir: str | Path = "models/flux1-dev",
) -> HFEmbedder:
    print(f"Trying to load CLIP...")
    model_dir = _require_model_path(Path(model_dir), "FLUX model directory")
    return HFEmbedder(
        model_path=model_dir / "text_encoder",
        tokenizer_path=model_dir / "tokenizer",
        is_clip=True,
        max_length=77,
        torch_dtype=torch.bfloat16,
    ).to(device)

def load_ae(
    name: str,
    device: str | torch.device = "cuda",
    model_dir: str | Path = "models/flux1-dev",
) -> AutoEncoder:
    config = configs[name]
    ckpt_path = _require_model_path(
        Path(model_dir) / config.repo_ae,
        f"{name} autoencoder checkpoint",
    )

    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta"):
        ae = AutoEncoder(config.ae_params)

    print(f"Loading AE checkpoint: {ckpt_path}")
    sd = load_sft(str(ckpt_path), device=str(device))
    missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
    return ae


def optionally_expand_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
    """
    Optionally expand the state dict to match the model's parameters shapes.
    """
    for name, param in model.named_parameters():
        if name in state_dict:
            if state_dict[name].shape != param.shape:
                print(
                    f"Expanding '{name}' with shape {state_dict[name].shape} to model parameter with shape {param.shape}."
                )
                # expand with zeros:
                expanded_state_dict_weight = torch.zeros_like(param, device=state_dict[name].device)
                slices = tuple(slice(0, dim) for dim in state_dict[name].shape)
                expanded_state_dict_weight[slices] = state_dict[name]
                state_dict[name] = expanded_state_dict_weight

    return state_dict
