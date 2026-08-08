import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import torch
from torchvision.utils import save_image as torch_save_image
import numpy as np

from src.flux1.sampling import denoise, get_noise, get_schedule, prepare, unpack
from src.flux1.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)
from sae.model import SparseAutoencoder

@dataclass
class SamplingOptions:
    prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

def parse_prompt(options: SamplingOptions) -> SamplingOptions | None:
    user_question = "Next prompt (write /h for help, /q to quit and leave empty to repeat):\n"
    usage = (
        "Usage: Either write your prompt directly, leave this field empty "
        "to repeat the prompt or write a command starting with a slash:\n"
        "- '/w <width>' will set the width of the generated image\n"
        "- '/h <height>' will set the height of the generated image\n"
        "- '/s <seed>' sets the next seed\n"
        "- '/g <guidance>' sets the guidance (flux-dev only)\n"
        "- '/n <steps>' sets the number of steps\n"
        "- '/q' to quit"
    )

    while (prompt := input(user_question)).startswith("/"):
        if prompt.startswith("/w"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, width = prompt.split()
            options.width = 16 * (int(width) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height * options.width / 1e6:.2f}MP)"
            )
        elif prompt.startswith("/h"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, height = prompt.split()
            options.height = 16 * (int(height) // 16)
            print(
                f"Setting resolution to {options.width} x {options.height} "
                f"({options.height * options.width / 1e6:.2f}MP)"
            )
        elif prompt.startswith("/g"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, guidance = prompt.split()
            options.guidance = float(guidance)
            print(f"Setting guidance to {options.guidance}")
        elif prompt.startswith("/s"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, seed = prompt.split()
            options.seed = int(seed)
            print(f"Setting seed to {options.seed}")
        elif prompt.startswith("/n"):
            if prompt.count(" ") != 1:
                print(f"Got invalid command '{prompt}'\n{usage}")
                continue
            _, steps = prompt.split()
            options.num_steps = int(steps)
            print(f"Setting number of steps to {options.num_steps}")
        elif prompt.startswith("/q"):
            print("Quitting")
            return None
        else:
            if not prompt.startswith("/h"):
                print(f"Got invalid command '{prompt}'\n{usage}")
            print(usage)
    if prompt != "":
        options.prompt = prompt
    return options

def load_prompts_from_file(file_path: str) -> list[str]:
    """Load prompts from a text file, one prompt per line."""
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

@torch.inference_mode()
def main(
    name: str = "flux-dev",
    # Place FLUX weights here or pass a different relative directory.
    model_dir: str = "models/flux1-dev",
    width: int = 256,
    height: int = 256,
    seed: int | None = 2026,
    prompt: str = ("A photo of a white sailboat, sails up."),
    prompt_file: str = "config/3.txt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    guidance: float = 2.5,
    offload: bool = False,
    output_dir: str = "output",
    feature_dir: str = "feature",
    trt: bool = False,
    mode: str = "normal",
    feature_name: str = "flux1_step_updates_style",
    ablation_steps: list[int] = None,
    sae_checkpoint_dir: str = "checkpoints",
    prototype_path: str = "feature/flux1_style_prototypes.pt",
    mean_feature_path: str = "feature/flux1_mean_step_updates.pt",
    target_class: int = 9,
    sae_alpha: float = 0.5,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        trt: use TensorRT backend for optimized inference
        mode: generation mode - 'normal', 'save_feature', 'mean_ablation', or 'sae_style_guide'
        ablation_steps: list of step indices for mean ablation or SAE style guide mode
        sae_checkpoint_dir: directory containing SAE checkpoints (step_00, step_01, ...)
        prototype_path: path to precomputed class-level SAE prototypes
        mean_feature_path: path to mean denoising step updates
        target_class: target style class index (0-9)
        sae_alpha: blending strength for SAE style guide (0=original, 1=full style)
    """
    
    if ablation_steps is None:
        ablation_steps = []
    if trt:
        raise NotImplementedError(
            "TensorRT execution is not configured in this open-source entry point"
        )

    # ── Load SAE models and style reference codes for sae_style_guide mode ──
    sae_config = None
    if mode == "sae_style_guide":
        print(f"Loading SAE models for style guide (class={target_class}, alpha={sae_alpha})...")
        sae_config = {'models': {}, 'style_codes': {}, 'alpha': sae_alpha}

        # Load precomputed style reference codes
        style_data = torch.load(prototype_path, map_location="cpu", weights_only=True)

        for step in ablation_steps:
            timestep = step - 1  # FLUX step_count is 1-based, SAE timestep is 0-based
            ckpt_dir = os.path.join(sae_checkpoint_dir, f"step_{timestep:02d}")
            if not os.path.exists(ckpt_dir):
                print(f"  [WARN] SAE checkpoint not found: {ckpt_dir}, skipping step {step}")
                continue

            sae = SparseAutoencoder.load_from_disk(ckpt_dir, map_location=device)
            sae.to(device)
            sae.eval()

            norm = np.load(os.path.join(ckpt_dir, "norm_params.npz"))
            mu = torch.tensor(norm["mu"], dtype=torch.float32, device=device)
            sigma = torch.tensor(norm["sigma"], dtype=torch.float32, device=device)

            sae_config['models'][timestep] = {'sae': sae, 'mu': mu, 'sigma': sigma}

            if timestep in style_data and target_class in style_data[timestep]:
                sae_config['style_codes'][timestep] = style_data[timestep][target_class].to(device)
            else:
                print(f"  [WARN] No style codes for timestep {timestep}, class {target_class}")

        print(f"  Loaded {len(sae_config['models'])} SAE models for {len(ablation_steps)} steps")

    # Load prompts from file
    if prompt_file and os.path.exists(prompt_file):
        all_prompts = load_prompts_from_file(prompt_file)
        print(f"Loaded {len(all_prompts)} prompts from {prompt_file}")
    else:
        prompt = prompt.split("|")
        if len(prompt) == 1:
            all_prompts = [prompt[0]]
        else:
            all_prompts = prompt

    assert not (
        loop and len(all_prompts) > 1
    ), "Do not provide multiple prompts and set loop to True"
    
    # Initialize features dictionary to store all features with ids
    all_features = {}
    
    print(f"Running in '{mode}' mode")
    if mode == "mean_ablation" and ablation_steps:
        print(f"Ablation steps: {ablation_steps}")

    mean_features = None
    if mode == "mean_ablation":
        if not ablation_steps:
            raise ValueError("mean_ablation requires non-empty ablation_steps")
        if not os.path.isfile(mean_feature_path):
            raise FileNotFoundError(f"FLUX mean step-update file not found: {mean_feature_path}")
        mean_features = torch.load(
            mean_feature_path, map_location="cpu", weights_only=True
        )

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 50

    # allow for packing and conversion to latent space
    height = 16 * (height // 16)
    width = 16 * (width // 16)

    output_name = os.path.join(output_dir, "img_{idx}.png")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    else:
        fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.png$", fn)]
        if len(fns) > 0:
            idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
        else:
            idx = 0

    t5 = load_t5(
        torch_device,
        max_length=256 if name == "flux-schnell" else 512,
        model_dir=model_dir,
    )
    clip = load_clip(torch_device, model_dir=model_dir)
    model = load_flow_model(
        name, device="cpu" if offload else torch_device, model_dir=model_dir
    )
    ae = load_ae(
        name, device="cpu" if offload else torch_device, model_dir=model_dir
    )

    rng = torch.Generator(device="cpu")
    current_prompt_idx = 0
    opts = SamplingOptions(
        prompt=all_prompts[current_prompt_idx],
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )
    if loop:
        opts = parse_prompt(opts)

    while opts is not None:
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.prompt}")
        t0 = time.perf_counter()

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        opts.seed = None
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)
        inp = prepare(t5, clip, x, prompt=opts.prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))
        # print(timesteps)

        # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # denoise initial noise
        x, features = denoise(
            model, **inp, timesteps=timesteps, guidance=opts.guidance,
            mode=mode, ablation_steps=ablation_steps,
            sae_config=sae_config,
            mean_features=mean_features,
        )

        # offload model, load autoencoder to gpu
        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        fn = output_name.format(idx=idx)
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")

        torch_save_image(x, fn, normalize=True)

        # Save features if in save_feature mode
        if mode == "save_feature" and features:
            all_features[idx] = {
                "idx": idx,
                "prompt": opts.prompt,
                "features": features,
            }
            print(f"Saved features for image {idx}")
        
        idx += 1
        current_prompt_idx += 1

        if loop:
            print("-" * 80)
            opts = parse_prompt(opts)
        elif current_prompt_idx < len(all_prompts):
            opts.prompt = all_prompts[current_prompt_idx]
            opts.seed = seed  # Reset seed for each prompt
        else:
            opts = None

    # Save all features to a single file when in save_feature mode
    if mode == "save_feature" and all_features:
        os.makedirs(feature_dir) if not os.path.exists(feature_dir) else None
        feature_path = os.path.join(feature_dir, f"{feature_name}_{seed}.pt")
        torch.save(all_features, feature_path)
        print(f"Saved all {len(all_features)} features to {feature_path}")

    print("Done!")


if __name__ == "__main__":
    main()
