"""Run the full zero-shot benchmark from `python -m visionlang`."""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from visionlang.data import load_flickr8k
from visionlang.metrics import (caption_length_analysis, grouped_bootstrap_ci,
                                metrics_from_ranks, retrieval_ranks)
from visionlang.models import CLIPEncoder, MODELS
from visionlang.reporting import plot_recall, write_csv, write_json


def create_parser():
    parser = argparse.ArgumentParser(description="Zero-shot CLIP/Flickr8k research benchmark")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=["clip-b32", "clip-b16"])
    parser.add_argument("--max-images", type=int, default=0, help="0=full test split (1000 images); 100=quick smoke")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--corruptions", nargs="*", choices=["blur", "grayscale"], default=["blur", "grayscale"])
    parser.add_argument("--outdir", default="results")
    return parser


def main():
    args = create_parser().parse_args()
    if args.batch_size < 1 or args.text_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("Loading official Flickr8k test split; first download may be ~1.1 GB.", flush=True)
    images, captions, paired, names = load_flickr8k(split="test", max_images=args.max_images, seed=args.seed)
    paired = np.asarray(paired, dtype=np.int64)
    rows, ci_rows, lengths, failures = [], [], [], []
    for model in args.models:
        print(f"Encoding model {model} ...", flush=True)
        encoder = CLIPEncoder(model, device=args.device)
        text_vec = encoder.texts(captions, batch_size=args.text_batch_size)
        for corruption in ["none", *args.corruptions]:
            print(f"  {model} with image condition {corruption} ...", flush=True)
            image_vec = encoder.images(images, batch_size=args.batch_size, corruption=corruption)
            similarity = (text_vec @ image_vec.T).astype(np.float32)
            t2i_ranks, i2t_ranks = retrieval_ranks(similarity, paired)
            for direction, ranks in [("text_to_image", t2i_ranks), ("image_to_text", i2t_ranks)]:
                rows.append({"model": model, "corruption": corruption,
                             "direction": direction, **metrics_from_ranks(ranks)})
                ci = grouped_bootstrap_ci(ranks, paired if direction == "text_to_image" else None,
                                          n_bootstrap=args.bootstrap, seed=args.seed)
                ci_rows.append({"model": model, "corruption": corruption,
                                "direction": direction, "ci95": ci})
            if corruption == "none":
                for r in caption_length_analysis(t2i_ranks, captions):
                    lengths.append({"model": model, **r})
                bad_indices = np.flatnonzero(t2i_ranks > 1)
                bad_indices = sorted(bad_indices, key=lambda i: (-int(t2i_ranks[i]), int(i)))[:30]
                for j in bad_indices:
                    top = int(np.argmax(similarity[j]))
                    failures.append({"model": model, "caption": captions[j],
                                     "true_image": names[paired[j]], "top_predicted_image": names[top],
                                     "rank_of_true_image": int(t2i_ranks[j]),
                                     "top_similarity": round(float(similarity[j, top]), 5),
                                     "true_similarity": round(float(similarity[j, paired[j]]), 5)})
                # Can serve a local Gradio demo; no Flickr photos redistributed in this zip.
                np.savez_compressed(out / f"index_{model}.npz", image_embeddings=image_vec,
                                    caption_embeddings=text_vec, captions=np.asarray(captions),
                                    paired=paired, filenames=np.asarray(names))
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(out / "metrics.csv", rows)
    write_csv(out / "caption_length.csv", lengths)
    write_csv(out / "failures.csv", failures)
    write_json(out / "confidence_intervals.json", ci_rows)
    plot_recall(out / "recall_curves.png", rows)
    import transformers
    import datasets
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "intro/flickr8k", "split": "test", "n_images": len(images),
        "n_captions": len(captions), "subset_not_comparable_to_full_benchmark": len(images) != 1000,
        "seed": args.seed, "models": {key: MODELS[key] for key in args.models},
        "conditions": ["none", *args.corruptions], "bootstrap": args.bootstrap,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__, "numpy_version": np.__version__,
        "python_version": platform.python_version(), "device_requested": args.device,
        "gpu_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timing_seconds": round(time.time()-started, 2),
        "note": "95% CIs bootstrap image groups conditional on the fixed candidate gallery. No fine-tuning; pretrained frozen CLIP."
    }
    write_json(out / "run_metadata.json", metadata)
    print("\n=== ACTUAL MEASURED RESULTS (no fabricated scores) ===")
    for r in rows:
        print(f'{r["model"]:10s} {r["corruption"]:10s} {r["direction"]:14s} '
              f'R@1={r["R@1"]:.2f}%  R@5={r["R@5"]:.2f}%  R@10={r["R@10"]:.2f}%')
    print(f"\nSaved reports into {out.resolve()}")
    if len(images) != 1000:
        print("WARNING: This is a SMALL-SUBSET DEMO, NOT the comparable Flickr8k test benchmark.")


if __name__ == "__main__":
    main()
