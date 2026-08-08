"""
Select reliable reference samples using style-class SAE code statistics.

Workflow:
1. Load one or more trained timestep SAEs.
2. Encode every sample.
3. Aggregate token-level codes into one sample-level code vector.
4. Compute per-class code mean/std.
5. Score each sample by whitened distance to its class center.
6. Record indices that are far from or close to the class center.

Notes:
- Token aggregation defaults to mean, which matches the current analysis code.
- If multiple timesteps are provided, sample codes are averaged across steps.
- "near" samples are not inherently bad; they are just highly prototypical.
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch

from sae.dataset import STYLE_NAMES, get_style_label, load_feature_data, normalize
from sae.model import SparseAutoencoder


def parse_args():
    p = argparse.ArgumentParser(description="Select reference samples from SAE statistics")
    p.add_argument("--data", type=str, default="feature/step_updates_style.pt",
                   help="Path to saved denoising step updates (.pt)")
    p.add_argument("--allow_unsafe_pickle", action="store_true",
                   help="Allow loading a trusted legacy object-array .npy file")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/flux1",
                   help="Directory containing SAE checkpoints step_xx")
    p.add_argument("--steps", type=int, nargs="+", required=True,
                   help="Timesteps whose SAEs will be used")
    p.add_argument("--output_dir", type=str, default="reference_selection",
                   help="Directory to save scores and index lists")
    p.add_argument("--token_reduce", choices=["mean", "max"], default="mean",
                   help="How to aggregate token-level codes into one sample vector")
    p.add_argument("--step_reduce", choices=["mean"], default="mean",
                   help="How to combine multiple timestep sample vectors")
    p.add_argument("--far_threshold", type=float, default=1,
                   help="Mark samples with score > threshold as far/outliers")
    p.add_argument("--near_threshold", type=float, default=0.75,
                   help="Mark samples with score < threshold as near/prototypes")
    p.add_argument("--min_class_size", type=int, default=8,
                   help="Skip classes with fewer than this many samples")
    p.add_argument("--filtered_output", type=str, default=None,
                   help="Optional output .pt path after removing selected indices")
    p.add_argument("--remove_far", action="store_true",
                   help="Remove far samples when writing filtered_output")
    p.add_argument("--remove_near", action="store_true",
                   help="Remove near samples when writing filtered_output")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def aggregate_tokens(codes: torch.Tensor, method: str) -> torch.Tensor:
    if method == "mean":
        return codes.mean(dim=0)
    if method == "max":
        return codes.max(dim=0).values
    raise ValueError(f"Unknown token_reduce: {method}")


def load_sae_with_norm(checkpoint_dir: str, step: int, device: str):
    step_dir = os.path.join(checkpoint_dir, f"step_{step:02d}")
    sae = SparseAutoencoder.load_from_disk(step_dir, map_location=device).to(device)
    sae.eval()
    norm = np.load(os.path.join(step_dir, "norm_params.npz"))
    mu = torch.tensor(norm["mu"], dtype=torch.float32, device=device)
    sigma = torch.tensor(norm["sigma"], dtype=torch.float32, device=device)
    return sae, mu, sigma


def compute_class_stats(sample_codes: dict[int, torch.Tensor], labels: dict[int, int], min_class_size: int):
    per_class_ids = defaultdict(list)
    for sid, label in labels.items():
        per_class_ids[label].append(sid)

    class_stats = {}
    for label, sample_ids in per_class_ids.items():
        if len(sample_ids) < min_class_size:
            continue
        stacked = torch.stack([sample_codes[sid] for sid in sample_ids], dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False).clamp(min=1e-6)
        class_stats[label] = {
            "mean": mean,
            "std": std,
            "count": len(sample_ids),
        }
    return class_stats


def sample_score(code: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> float:
    z = (code - mean) / std
    return torch.sqrt((z ** 2).mean()).item()


def write_scores_csv(rows: list[dict], path: str):
    fieldnames = [
        "sample_id", "class_id", "class_name", "score",
        "is_far", "is_near",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.filtered_output is not None and not (args.remove_far or args.remove_near):
        raise ValueError("filtered_output requires --remove_far and/or --remove_near")

    print(f"Loading data from {args.data}")
    data = load_feature_data(
        args.data, allow_unsafe_pickle=args.allow_unsafe_pickle
    )
    sample_ids = sorted(data.keys())
    print(f"  Loaded {len(sample_ids)} samples")
    print(f"Using steps: {args.steps}")
    print(f"Device: {args.device}")

    sample_codes = {}
    labels = {}
    step_accum = {sid: [] for sid in sample_ids}

    for sid in sample_ids:
        labels[sid] = get_style_label(sid)

    for step in args.steps:
        print(f"Encoding step {step:02d}")
        sae, mu, sigma = load_sae_with_norm(args.checkpoint_dir, step, args.device)

        for i, sid in enumerate(sample_ids, start=1):
            feat = data[sid]["features"][step]
            if isinstance(feat, torch.Tensor):
                tokens = feat.squeeze(0).float().to(args.device)
            else:
                tokens = torch.tensor(feat, dtype=torch.float32, device=args.device).squeeze(0)

            tokens_norm = normalize(tokens, mu, sigma)
            with torch.no_grad():
                codes = sae.encode(tokens_norm)
            step_accum[sid].append(aggregate_tokens(codes, args.token_reduce).cpu())

            if i % 200 == 0 or i == len(sample_ids):
                print(f"  Step {step:02d}: encoded {i}/{len(sample_ids)} samples")

        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()

    for sid in sample_ids:
        sample_codes[sid] = torch.stack(step_accum[sid], dim=0).mean(dim=0)

    class_stats = compute_class_stats(sample_codes, labels, args.min_class_size)

    rows = []
    far_by_class = defaultdict(list)
    near_by_class = defaultdict(list)

    for sid in sample_ids:
        label = labels[sid]
        if label not in class_stats:
            continue

        stats = class_stats[label]
        score = sample_score(sample_codes[sid], stats["mean"], stats["std"])
        is_far = score > args.far_threshold
        is_near = score < args.near_threshold

        rows.append({
            "sample_id": sid,
            "class_id": label,
            "class_name": STYLE_NAMES[label],
            "score": f"{score:.6f}",
            "is_far": int(is_far),
            "is_near": int(is_near),
        })

        if is_far:
            far_by_class[label].append((sid, score))
        if is_near:
            near_by_class[label].append((sid, score))

    rows.sort(key=lambda x: (int(x["class_id"]), float(x["score"])))
    for label in far_by_class:
        far_by_class[label].sort(key=lambda x: x[1], reverse=True)
    for label in near_by_class:
        near_by_class[label].sort(key=lambda x: x[1])

    scores_csv = os.path.join(args.output_dir, "sample_scores.csv")
    write_scores_csv(rows, scores_csv)

    summary = {
        "data": args.data,
        "checkpoint_dir": args.checkpoint_dir,
        "steps": args.steps,
        "token_reduce": args.token_reduce,
        "step_reduce": args.step_reduce,
        "far_threshold": args.far_threshold,
        "near_threshold": args.near_threshold,
        "classes": {},
    }

    far_indices = []
    near_indices = []

    for label, stats in class_stats.items():
        far_ids = [sid for sid, _ in far_by_class[label]]
        near_ids = [sid for sid, _ in near_by_class[label]]
        far_indices.extend(far_ids)
        near_indices.extend(near_ids)
        summary["classes"][str(label)] = {
            "class_name": STYLE_NAMES[label],
            "count": stats["count"],
            "far_count": len(far_ids),
            "near_count": len(near_ids),
            "far_indices": far_ids,
            "near_indices": near_ids,
        }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output_dir, "far_indices.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, sorted(far_indices))))
        f.write("\n")

    with open(os.path.join(args.output_dir, "near_indices.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, sorted(near_indices))))
        f.write("\n")

    print(f"Saved score table to {scores_csv}")
    print(f"Saved summary to {os.path.join(args.output_dir, 'summary.json')}")
    print(f"Far samples: {len(far_indices)}")
    print(f"Near samples: {len(near_indices)}")

    if args.filtered_output is not None:
        if not args.filtered_output.endswith(".pt"):
            raise ValueError("filtered_output must use the safe .pt format")
        remove_ids = set()
        if args.remove_far:
            remove_ids.update(far_indices)
        if args.remove_near:
            remove_ids.update(near_indices)

        filtered = {sid: sample for sid, sample in data.items() if sid not in remove_ids}
        os.makedirs(os.path.dirname(args.filtered_output) or ".", exist_ok=True)
        torch.save(filtered, args.filtered_output)
        print(f"Saved filtered data to {args.filtered_output}")
        print(f"Removed {len(remove_ids)} samples, kept {len(filtered)}")


if __name__ == "__main__":
    main()
