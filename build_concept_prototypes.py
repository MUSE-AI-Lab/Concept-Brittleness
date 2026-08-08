"""
Build per-class SAE concept prototypes for prototype-guided correction.

For each timestep and style class, encodes all samples through the
corresponding SAE and computes the mean latent code (per token position).

Output: style_prototypes.pt
    {
        timestep(int): {
            class_id(int): mean_code Tensor [256, n_dirs]
        },
        ...
    }

Usage:
    python build_concept_prototypes.py \
        --data_path feature/step_updates_style_filtered.pt \
        --checkpoint_dir checkpoints \
        --output_path feature/style_prototypes.pt
"""

import os
import argparse

import torch
import numpy as np
from sae.model import SparseAutoencoder
from sae.dataset import load_feature_data, get_style_label, N_CLASSES, STYLE_NAMES


def main():
    parser = argparse.ArgumentParser(description="Build per-class SAE concept prototypes")
    parser.add_argument("--data_path", type=str,
                        default="feature/step_updates_style_filtered.pt")
    parser.add_argument("--allow_unsafe_pickle", action="store_true",
                        help="Allow loading a trusted legacy object-array .npy file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_path", type=str, default="feature/style_prototypes.pt")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Loading style data...")
    data = load_feature_data(
        args.data_path, allow_unsafe_pickle=args.allow_unsafe_pickle
    )
    sample_ids = sorted(data.keys())
    print(f"  {len(sample_ids)} samples loaded")

    # Group sample IDs by class
    class_samples = {c: [] for c in range(N_CLASSES)}
    for sid in sample_ids:
        class_samples[get_style_label(sid)].append(sid)
    for c in range(N_CLASSES):
        print(f"  Class {c} ({STYLE_NAMES[c]}): {len(class_samples[c])} samples")

    style_prototypes = {}
    prototype_width = None

    for t in range(args.num_steps):
        ckpt_dir = os.path.join(args.checkpoint_dir, f"step_{t:02d}")
        if not os.path.exists(ckpt_dir):
            print(f"[WARN] Checkpoint not found: {ckpt_dir}, skipping step {t}")
            continue

        # Load SAE
        sae = SparseAutoencoder.load_from_disk(ckpt_dir, map_location=args.device)
        sae.to(args.device)
        sae.eval()
        prototype_width = sae.n_dirs

        # Load normalization params
        norm = np.load(os.path.join(ckpt_dir, "norm_params.npz"))
        mu = torch.tensor(norm["mu"], dtype=torch.float32, device=args.device)
        sigma = torch.tensor(norm["sigma"], dtype=torch.float32, device=args.device)

        # Accumulate codes per class
        class_code_accum = {c: [] for c in range(N_CLASSES)}

        for sid in sample_ids:
            feat = data[sid]['features'][t]  # [1, tokens, feature_dim]
            if isinstance(feat, torch.Tensor):
                tokens = feat.squeeze(0).float().to(args.device)
            else:
                tokens = torch.tensor(feat, dtype=torch.float32, device=args.device).squeeze(0)

            # Normalize
            tokens_norm = (tokens - mu) / (sigma + 1e-8)

            # Encode
            with torch.no_grad():
                codes = sae.encode(tokens_norm)  # [256, n_dirs]

            class_id = get_style_label(sid)
            class_code_accum[class_id].append(codes)

        # Compute per-class mean (per token position)
        step_codes = {}
        for c in range(N_CLASSES):
            if class_code_accum[c]:
                stacked = torch.stack(class_code_accum[c])
                step_codes[c] = stacked.mean(dim=0).cpu()  # [tokens, n_dirs]

        style_prototypes[t] = step_codes
        print(f"  Step {t:02d}: done ({len(sample_ids)} samples → {N_CLASSES} class means)")

        # Free GPU memory
        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(style_prototypes, args.output_path)
    print(f"\nSaved to {args.output_path}")
    print(
        "Structure: prototypes[timestep][class_id] → "
        f"Tensor [tokens, {prototype_width or 'n_dirs'}]"
    )
    print(f"\nClass mapping:")
    for c in range(N_CLASSES):
        print(f"  {c}: {STYLE_NAMES[c]}")


if __name__ == "__main__":
    main()
