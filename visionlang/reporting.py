"""Plain CSV/JSON/Markdown reporting with provenance, no invented scores."""
from __future__ import annotations
import csv
import json
from pathlib import Path


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_recall(path, rows):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in rows:
        if row["corruption"] != "none":
            continue
        ax.plot([1, 5, 10], [row[f"R@{k}"] for k in (1, 5, 10)], marker="o", label=f'{row["model"]}: {row["direction"]}')
    ax.set(xlabel="K", ylabel="Recall@K (%)", title="Zero-shot Flickr8k image-text retrieval")
    ax.set_xticks([1, 5, 10])
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
