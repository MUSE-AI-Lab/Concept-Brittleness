import argparse

from src.flux1.cli import main as flux_main
from src.sd3_5.cli import main as sd35_main
from src.pixart.cli import main as pixart_main
from src.sdxl.cli import main as sdxl_main
from src.sd1_5.cli import main as sd15_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="flux",
                        choices=["flux", "sd3.5", "pixart", "sdxl", "sd1.5"],
                        help="Diffusion model to use")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--prompt", type=str, default="a photo of a cat")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Optional text file with one prompt per line")
    parser.add_argument("--img_cond_path", type=str, default="output/img_0.jpg")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mode", type=str, default="normal",
                        choices=["normal", "save_feature", "mean_ablation", "sae_style_guide"])
    parser.add_argument("--feature_name", type=str, default="step_updates_style",
                        help="Name prefix for saved denoising step updates")
    parser.add_argument("--ablation_steps", type=str, default="",
                        help="Comma-separated step indices for ablation/intervention (e.g., '1,2,3')")
    parser.add_argument("--sae_checkpoint_dir", type=str, default=None,
                        help="SAE checkpoints; defaults to a model-specific directory")
    parser.add_argument("--prototype_path", "--style_ref_path", dest="prototype_path",
                        type=str, default=None,
                        help="Class-level SAE prototype file; defaults by model")
    parser.add_argument("--mean_feature_path", type=str, default=None,
                        help="Mean update features used by mean_ablation")
    parser.add_argument("--target_class", type=int, default=9,
                        help="Target style class index (0-9). Default: 9 (ink sketch)")
    parser.add_argument("--sae_alpha", type=float, default=0.8,
                        help="SAE latent blending strength (0=original, 1=full style)")
    # Put local model weights under models/ or override these relative paths.
    parser.add_argument("--flux_model_dir", type=str, default="models/flux1-dev",
                        help="FLUX.1-dev model directory")
    # SD3.5-specific paths. See models/README.md for the expected layout.
    parser.add_argument("--sd35_model_path", type=str,
                        default="models/sd3.5/sd3.5_large.safetensors",
                        help="Path to SD3.5 Large safetensors model file")
    parser.add_argument("--sd35_model_folder", type=str, default="models/sd3.5",
                        help="Directory containing SD3.5 text encoders and tokenizers")
    parser.add_argument("--sd35_vae_path", type=str, default=None,
                        help="Path to separate VAE file (None = use main model)")
    parser.add_argument("--sd35_clip_l_tokenizer_path", type=str, default=None,
                        help="Optional SD3.5 CLIP tokenizer override")
    parser.add_argument("--sd35_t5_tokenizer_path", type=str, default=None,
                        help="Optional SD3.5 T5 tokenizer override")
    parser.add_argument("--width", type=int, default=256,
                        help="Image width (SD3.5 default: 1024)")
    parser.add_argument("--height", type=int, default=256,
                        help="Image height (SD3.5 default: 1024)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of sampling steps (None = model default)")
    parser.add_argument("--cfg_scale", type=float, default=None,
                        help="CFG scale (None = model default)")
    parser.add_argument("--sampler", type=str, default=None,
                        choices=["euler", "dpmpp_2m", "ddim"],
                        help="Sampler to use (None = model default)")
    # PixArt specific arguments
    parser.add_argument("--pixart_model_path", type=str,
                        default="models/pixart/PixArt-XL-2-256x256.pth",
                        help="Path to PixArt .pth checkpoint")
    parser.add_argument("--pixart_vae_path", type=str,
                        default="models/pixart/sd-vae-ft-ema",
                        help="Path to sd-vae-ft-ema directory (diffusers format)")
    parser.add_argument("--pixart_t5_path", type=str,
                        default="models/pixart/t5-v1_1-xxl",
                        help="Path to t5-v1_1-xxl directory")
    parser.add_argument("--image_size", type=int, default=256,
                        help="PixArt image size (default: 256)")
    # Diffusers-format Stable Diffusion model directories
    parser.add_argument("--sdxl_model_path", type=str,
                        default="models/sdxl",
                        help="Path to the local SDXL diffusers directory")
    parser.add_argument("--sd15_model_path", type=str,
                        default="models/sd1.5",
                        help="Path to the local SD1.5 diffusers directory")
    args = parser.parse_args()

    # Parse ablation steps if provided
    ablation_steps = []
    if args.ablation_steps:
        ablation_steps = [int(s.strip()) for s in args.ablation_steps.split(",") if s.strip()]
    if not 0 <= args.target_class <= 9:
        parser.error("--target_class must be between 0 and 9")
    if not 0.0 <= args.sae_alpha <= 1.0:
        parser.error("--sae_alpha must be between 0 and 1")

    sae_checkpoint_dir = args.sae_checkpoint_dir or {
        "flux": "checkpoints/flux1",
        "pixart": "checkpoints/pixart",
        "sd3.5": "checkpoints/sd3_5",
        "sdxl": "checkpoints/sdxl",
        "sd1.5": "checkpoints/sd1_5",
    }.get(args.model, "checkpoints")
    prototype_path = args.prototype_path or {
        "flux": "feature/flux1_style_prototypes.pt",
        "pixart": "feature/pixart_style_prototypes.pt",
        "sd3.5": "feature/sd3_5_style_prototypes.pt",
        "sdxl": "feature/sdxl_style_prototypes.pt",
        "sd1.5": "feature/sd1_5_style_prototypes.pt",
    }[args.model]
    mean_feature_path = args.mean_feature_path or {
        "flux": "feature/flux1_mean_step_updates.pt",
        "pixart": "feature/pixart_mean_step_updates.pt",
        "sd3.5": "feature/sd3_5_mean_step_updates.pt",
        "sdxl": "feature/sdxl_mean_step_updates.pt",
        "sd1.5": "feature/sd1_5_mean_step_updates.pt",
    }[args.model]

    if args.model == "flux":
        flux_main(
            model_dir=args.flux_model_dir,
            width=args.width,
            height=args.height,
            output_dir=args.output_dir,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            seed=args.seed,
            mode=args.mode,
            feature_name=args.feature_name,
            ablation_steps=ablation_steps,
            sae_checkpoint_dir=sae_checkpoint_dir,
            prototype_path=prototype_path,
            mean_feature_path=mean_feature_path,
            target_class=args.target_class,
            sae_alpha=args.sae_alpha,
        )
    elif args.model == "pixart":
        pixart_main(
            model_path=args.pixart_model_path,
            vae_path=args.pixart_vae_path,
            t5_path=args.pixart_t5_path,
            image_size=args.image_size,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            output_dir=args.output_dir,
            feature_dir="feature",
            mode=args.mode,
            feature_name=args.feature_name,
            ablation_steps=ablation_steps,
            sae_checkpoint_dir=sae_checkpoint_dir,
            prototype_path=prototype_path,
            mean_feature_path=mean_feature_path,
            target_class=args.target_class,
            sae_alpha=args.sae_alpha,
        )
    elif args.model == "sd3.5":
        if args.sampler == "ddim":
            parser.error("SD3.5 supports --sampler euler or dpmpp_2m")
        sd35_main(
            model_path=args.sd35_model_path,
            model_folder=args.sd35_model_folder,
            vae_path=args.sd35_vae_path,
            clip_l_tokenizer_path=args.sd35_clip_l_tokenizer_path,
            t5_tokenizer_path=args.sd35_t5_tokenizer_path,
            width=args.width,
            height=args.height,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            sampler=args.sampler,
            seed=args.seed,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            output_dir=args.output_dir,
            feature_dir="feature",
            mode=args.mode,
            feature_name=args.feature_name,
            ablation_steps=ablation_steps,
            sae_checkpoint_dir=sae_checkpoint_dir,
            prototype_path=prototype_path,
            mean_feature_path=mean_feature_path,
            target_class=args.target_class,
            sae_alpha=args.sae_alpha,
        )
    elif args.model == "sdxl":
        if args.sampler not in (None, "ddim"):
            parser.error("SDXL follows the paper's DDIM trajectory; use --sampler ddim")
        sdxl_main(
            model_path=args.sdxl_model_path,
            width=args.width,
            height=args.height,
            steps=50 if args.steps is None else args.steps,
            cfg_scale=args.cfg_scale if args.cfg_scale is not None else 5.0,
            seed=args.seed,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            output_dir=args.output_dir,
            feature_dir="feature",
            mode=args.mode,
            feature_name=args.feature_name,
            ablation_steps=ablation_steps,
            mean_feature_path=mean_feature_path,
            sae_checkpoint_dir=sae_checkpoint_dir,
            prototype_path=prototype_path,
            target_class=args.target_class,
            sae_alpha=args.sae_alpha,
        )
    elif args.model == "sd1.5":
        if args.sampler not in (None, "ddim"):
            parser.error("SD1.5 follows the paper's DDIM trajectory; use --sampler ddim")
        sd15_main(
            model_path=args.sd15_model_path,
            width=args.width,
            height=args.height,
            steps=50 if args.steps is None else args.steps,
            cfg_scale=args.cfg_scale if args.cfg_scale is not None else 7.5,
            seed=args.seed,
            prompt=args.prompt,
            prompt_file=args.prompt_file,
            output_dir=args.output_dir,
            feature_dir="feature",
            mode=args.mode,
            feature_name=args.feature_name,
            ablation_steps=ablation_steps,
            mean_feature_path=mean_feature_path,
            sae_checkpoint_dir=sae_checkpoint_dir,
            prototype_path=prototype_path,
            target_class=args.target_class,
            sae_alpha=args.sae_alpha,
        )
