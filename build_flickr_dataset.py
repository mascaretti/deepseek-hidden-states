"""Build the filtered Flickr30k dataset used throughout the paper.

Faithful port of `clip-multimodal-medical/src/scripts/20250319_storeFlickr.py`
inlined here so the dataset provenance lives next to the scripts that consume it.

Loads `clip-benchmark/wds_flickr30k`, splits each `txt` (newline-joined 5-caption
string) into `cpt_1..cpt_5` columns, filters to images where *every* caption has
`min_tokens ≤ Llama-3.1 BPE ≤ max_tokens` (defaults 10 and 30), and writes:

    <output_dir>/flickr30k/                            # HF arrow dataset (jpg, txt, cpt_1..5)
    <output_dir>/flickr_csv/flickr_cpt_{1..5}.txt      # plain-text caption files

Result with the defaults: 14,109 of 31,014 images retained.

Differences from the original script:
- Default tokenizer is NousResearch/Meta-Llama-3.1-8B (identical vocab to
  meta-llama/Llama-3.1-8B, no HF token required); pass --tokenizer to override.
- Caption files are written as plain `.txt` (one caption per line, no header,
  no index column), matching the format consumed by extract_hidden_states.py.
  The original wrote pandas-style `.csv`.

Usage:
    python build_flickr_dataset.py                                  # → data/
    python build_flickr_dataset.py --output_dir /home/datasets      # cluster paths
    python build_flickr_dataset.py --limit 200 --output_dir /tmp/x  # smoke test

Reproducibility check (after running on cluster):
    diff <(sort /tmp/x/flickr_csv/flickr_cpt_1.txt) \
         <(sort /home/datasets/flickr_csv/flickr_cpt_1.txt)
should be empty when --limit is not set.
"""
from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Parent dir; outputs go to <output_dir>/flickr30k/ and "
                             "<output_dir>/flickr_csv/.")
    parser.add_argument("--tokenizer", type=str,
                        default="NousResearch/Meta-Llama-3.1-8B",
                        help="HF tokenizer for the per-caption length filter.")
    parser.add_argument("--min_tokens", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None,
                        help="Subsample the raw dataset to this size before "
                             "filtering (smoke test).")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    arrow_out = out_root / "flickr30k"
    csv_out = out_root / "flickr_csv"
    csv_out.mkdir(parents=True, exist_ok=True)

    num_proc = max(multiprocessing.cpu_count() - 1, 1)
    print(f"num_proc={num_proc}")

    print("Loading clip-benchmark/wds_flickr30k …")
    ds = load_dataset("clip-benchmark/wds_flickr30k", split="train", num_proc=num_proc)
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"  loaded: {len(ds):,} images")

    print("Splitting txt → cpt_1..cpt_5 …")

    def split_captions(ex):
        caps = ex["txt"].split("\n")
        return {f"cpt_{i + 1}": v for i, v in enumerate(caps)}

    ds = ds.map(split_captions, num_proc=num_proc)

    print(f"Filtering by per-caption length: "
          f"{args.min_tokens} ≤ {args.tokenizer} BPE ≤ {args.max_tokens}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for i in range(1, 6):
        col = f"cpt_{i}"

        def in_range(examples, col=col):
            ids = tok(examples[col], add_special_tokens=True, truncation=False)["input_ids"]
            return [args.min_tokens <= len(t) <= args.max_tokens for t in ids]

        ds = ds.filter(in_range, batched=True, batch_size=1000, num_proc=num_proc)
        print(f"  after {col}: {len(ds):,}")

    print(f"\nSaving HF arrow dataset → {arrow_out}")
    ds.save_to_disk(str(arrow_out))

    print(f"Writing per-caption text files → {csv_out}/")
    for i in range(1, 6):
        path = csv_out / f"flickr_cpt_{i}.txt"
        with open(path, "w") as f:
            for cap in ds[f"cpt_{i}"]:
                f.write(cap + "\n")
        print(f"  {path}: {len(ds):,} lines")

    print("\nDone.")


if __name__ == "__main__":
    main()
