"""Build the keep mask for the max_length=77 CLIP-text rebuttal experiment.

For each of the first `subset_size` images in `/home/datasets/flickr30k/`,
tokenise the 5-caption concatenation (the `txt` field that
`extract_hidden_clip.py` actually feeds to CLIPProcessor) with
`openai/clip-vit-large-patch14`'s tokeniser. Mark images whose total token
count (including [BOS]+[EOT]) is ≤ `max_length` — i.e. those for which no
truncation occurred at extraction time.

The output is a length-`subset_size` boolean .npy that lines up row-by-row
with the mean-pooled safetensors produced by `preprocess_mean.py`. Apply at
II time via `compute_ii_mean.py --keep_indices …`.

Usage:
    python build_clip_max77_keep.py
    # → data/clip_max77_keep.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from transformers import CLIPTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/home/datasets/flickr30k")
    parser.add_argument("--subset_size", type=int, default=5000,
                        help="Must match the subset_size used at extraction time.")
    parser.add_argument("--max_length", type=int, default=77,
                        help="Token-count cap that defines 'no truncation'.")
    parser.add_argument("--tokenizer", type=str,
                        default="openai/clip-vit-large-patch14")
    parser.add_argument("--output", type=str, default="data/clip_max77_keep.npy")
    args = parser.parse_args()

    print(f"Loading {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)
    n = min(args.subset_size, len(ds))
    print(f"  using first {n:,} of {len(ds):,} images")

    print(f"Tokenising with {args.tokenizer}")
    tok = CLIPTokenizer.from_pretrained(args.tokenizer)
    texts = ds["txt"][:n]
    ids = tok(texts, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
    lengths = np.array([len(t) for t in ids], dtype=np.int32)

    keep = lengths <= args.max_length
    n_keep = int(keep.sum())
    print(f"\nToken-length stats (5-cap concat, CLIP-BPE incl. [BOS]+[EOT]):")
    print(f"  min={lengths.min()}  median={int(np.median(lengths))}  "
          f"95p={int(np.percentile(lengths, 95))}  max={lengths.max()}")
    print(f"  ≤ {args.max_length}: {n_keep:,}/{n:,} ({100*n_keep/n:.2f}%)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, keep)
    print(f"\nSaved mask → {out_path} (dtype={keep.dtype}, shape={keep.shape})")


if __name__ == "__main__":
    main()
