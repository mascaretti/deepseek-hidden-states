"""Verify the CLIP-text max_length=77 smoke extraction.

Run after `sbatch run_extract_clip_max77.sbatch` (smoke defaults: subset=20).
Asserts the output chunks exist with the expected dict keys and tensor shapes.

Usage:
    python verify_clip_max77_smoke.py
    # or, with a custom path:
    python verify_clip_max77_smoke.py /path/to/.../text
"""
import pickle
import sys
from pathlib import Path

DEFAULT_DIR = Path(
    "/home/laios/image-information-imbalance/openai-clip-vit-large-patch14_max77/text"
)
EXPECTED_BATCH = 10
EXPECTED_SEQ = 77
EXPECTED_HIDDEN = 768
EXPECTED_LAYERS = 13  # CLIP-L text: 12 hidden + 1 embedding

text_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
chunks = sorted(text_dir.glob("chunk_*.pkl"))
print(f"Dir: {text_dir}")
print(f"Found {len(chunks)} chunk(s)")
assert len(chunks) >= 1, f"No chunks at {text_dir}"

with open(chunks[0], "rb") as f:
    d = pickle.load(f)

assert isinstance(d, dict), f"chunk_0 is {type(d)}, expected dict"
keys = sorted(d.keys())
expected_keys = [f"layer_{j}" for j in range(EXPECTED_LAYERS)]
assert keys == sorted(expected_keys), f"keys mismatch: {keys}"

for k in expected_keys:
    t = d[k]
    shape = tuple(t.shape)
    assert shape == (EXPECTED_BATCH, EXPECTED_SEQ, EXPECTED_HIDDEN), (
        f"{k}: shape {shape} != ({EXPECTED_BATCH},{EXPECTED_SEQ},{EXPECTED_HIDDEN})"
    )

print(f"OK: {len(chunks)} chunk(s); chunk_0 has {len(keys)} layers, "
      f"each shape ({EXPECTED_BATCH}, {EXPECTED_SEQ}, {EXPECTED_HIDDEN}); dtype {d[keys[0]].dtype}")
