"""Tokenise Flickr30k captions with CLIP / Llama / DeepSeek-V3 tokenisers,
plot histograms, and print a markdown summary table.

Output:
- paper/figs-tmlr/rebuttal/flickr_token_hist.{pdf,png}
- paper/figs-tmlr/rebuttal/flickr_token_counts.csv (raw counts for reproducibility)
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "paper" / "figs-tmlr" / "rebuttal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_ARCH_CAP = 77
CLIP_OUR_CAP = 30


def load_flickr_captions() -> tuple[list[str], list[list[str]]]:
    """Returns (per_caption_list, per_image_5cap_list).

    nlphuji/flickr30k stores 5 captions per image as a stringified list in column 'raw'.
    """
    csv_path = hf_hub_download(
        repo_id="nlphuji/flickr30k",
        filename="flickr_annotations_30k.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(csv_path)
    per_image = [ast.literal_eval(s) for s in df["raw"].tolist()]
    per_caption = [c for caps in per_image for c in caps]
    return per_caption, per_image


def load_filtered_captions() -> tuple[list[str], list[list[str]]]:
    """The actual filtered subset used in the paper.

    `data/flickr_csv/flickr_cpt_{1..5}.txt` are 14,109 lines each, where line i
    is caption-i for image i in the filtered set. Filter (per
    20250319_storeFlickr.py): all 5 captions per image have 10 ≤ L ≤ 30 Llama-3.1 BPE.
    """
    base = REPO_ROOT / "data" / "flickr_csv"
    cap_lists = []
    for i in range(1, 6):
        with open(base / f"flickr_cpt_{i}.txt") as f:
            cap_lists.append([ln.rstrip("\n") for ln in f])
    n = len(cap_lists[0])
    assert all(len(c) == n for c in cap_lists), "caption files have different lengths"
    per_image = [[cap_lists[k][i] for k in range(5)] for i in range(n)]
    per_caption = [c for caps in per_image for c in caps]
    return per_caption, per_image


def tokenise_lengths(tokeniser, texts: list[str]) -> np.ndarray:
    encoded = tokeniser(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    return np.array([len(ids) for ids in encoded["input_ids"]], dtype=np.int32)


def summarise(name: str, lengths: np.ndarray) -> dict:
    return {
        "tokeniser": name,
        "n": int(lengths.size),
        "median": int(np.median(lengths)),
        "p95": int(np.percentile(lengths, 95)),
        "max": int(lengths.max()),
        "frac_gt_30_pct": 100.0 * (lengths > 30).mean(),
        "frac_gt_77_pct": 100.0 * (lengths > 77).mean(),
    }


def main():
    print("Loading Flickr30k (full nlphuji) …")
    full_per_caption, full_per_image = load_flickr_captions()
    full_concat = [" ".join(caps) for caps in full_per_image]
    print(f"  {len(full_per_caption):,} captions across {len(full_concat):,} images")

    print("Loading filtered subset (paper's actual extraction set) …")
    filt_per_caption, filt_per_image = load_filtered_captions()
    filt_concat = [" ".join(caps) for caps in filt_per_image]
    print(f"  {len(filt_per_caption):,} captions across {len(filt_concat):,} images")

    tokenisers = {
        "CLIP-BPE": AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
        "Llama-3.1": AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B"),
        "DeepSeek-V3": AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True),
    }

    rows = []
    raw = {}
    splits = {
        "full": (full_per_caption, full_concat),
        "filtered": (filt_per_caption, filt_concat),
    }
    for split_name, (per_cap_texts, concat_texts) in splits.items():
        for name, tok in tokenisers.items():
            print(f"Tokenising {split_name} with {name} …")
            per_cap = tokenise_lengths(tok, per_cap_texts)
            concat = tokenise_lengths(tok, concat_texts)
            raw[(split_name, name, "per_caption")] = per_cap
            raw[(split_name, name, "concat_5")] = concat
            rows.append({"split": split_name, "input": "per_caption", **summarise(name, per_cap)})
            rows.append({"split": split_name, "input": "concat_5", **summarise(name, concat)})

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "flickr_token_counts.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Histograms — 2 rows × 3 cols, stacking full (light) and filtered (dark) per panel
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey="row")
    names = list(tokenisers.keys())
    colors = {"CLIP-BPE": "#1f77b4", "Llama-3.1": "#d62728", "DeepSeek-V3": "#2ca02c"}

    # Row 0: per-caption
    for j, name in enumerate(names):
        ax = axes[0, j]
        bins = np.arange(0, 90, 1)
        ax.hist(raw[("full", name, "per_caption")], bins=bins, color=colors[name],
                alpha=0.35, edgecolor="white", linewidth=0.3, label="full Flickr30k")
        ax.hist(raw[("filtered", name, "per_caption")], bins=bins, color=colors[name],
                alpha=0.95, edgecolor="white", linewidth=0.3, label="filtered (paper)")
        ax.axvline(CLIP_OUR_CAP, color="black", ls="--", lw=1.0, label=f"extraction cap = {CLIP_OUR_CAP}")
        ax.axvline(CLIP_ARCH_CAP, color="grey", ls=":", lw=1.0, label=f"CLIP arch = {CLIP_ARCH_CAP}")
        ax.set_title(f"{name}  (per caption)")
        ax.set_xlabel("tokens")
        ax.set_xlim(0, 90)
        if j == 0:
            ax.set_ylabel("# captions")
        ax.legend(fontsize=7, loc="upper right")

    # Row 1: 5-caption concat
    for j, name in enumerate(names):
        ax = axes[1, j]
        bins = np.arange(0, 230, 4)
        ax.hist(raw[("full", name, "concat_5")], bins=bins, color=colors[name],
                alpha=0.35, edgecolor="white", linewidth=0.3, label="full Flickr30k")
        ax.hist(raw[("filtered", name, "concat_5")], bins=bins, color=colors[name],
                alpha=0.95, edgecolor="white", linewidth=0.3, label="filtered (paper)")
        ax.axvline(CLIP_OUR_CAP, color="black", ls="--", lw=1.0)
        ax.axvline(CLIP_ARCH_CAP, color="grey", ls=":", lw=1.0)
        ax.set_title(f"{name}  (5 captions concatenated)")
        ax.set_xlabel("tokens")
        ax.set_xlim(0, 230)
        if j == 0:
            ax.set_ylabel("# images")
        if j == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Flickr30k token-length distribution: full vs filtered subset (10–30 Llama-BPE per caption)", fontsize=11)
    fig.tight_layout()

    pdf_path = OUT_DIR / "flickr_token_hist.pdf"
    png_path = OUT_DIR / "flickr_token_hist.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=160)
    print(f"\nWrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
