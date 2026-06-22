# Semantic Information in Deep Representations — Information Imbalance & CKA

This repository measures **Information Imbalance (II)** and **CKA** between the
hidden states of large unimodal models — DeepSeek‑V3, Llama‑3.1, CLIP (text &
image), DinoV2, ImageGPT — on Flickr30k captions and images, plus ImageNet‑1k
same‑class image pairs. II is an asymmetric, rank‑based statistic of *directed
predictability* between two representation spaces: `Δ(X→Y) ≈ 0` means neighbourhoods
in `X` predict those in `Y`; `Δ(X→Y) ≈ 1` means they carry no information about each
other. The asymmetry `Δ(X→Y) ≠ Δ(Y→X)` is the quantity of interest.

The accompanying paper lives in `paper/` (the active branch `image-captions` is
the TMLR rebuttal track). For agent‑oriented working notes and conventions, see
[`CLAUDE.md`](CLAUDE.md); for an audit of which scripts are active vs superseded,
see [`REPORT.org`](REPORT.org).

---

## Setup

```bash
# Local analysis/plotting environment (matplotlib, scienceplots, safetensors, jax)
.venv/bin/python <script>.py        # the system python lacks matplotlib

# On the cluster (slurm), activate the JAX env instead:
source /home/$USER/myjax/bin/activate
```

Source files sync to the cluster **via git**; data artefacts (pickles, npz,
safetensors, figures) sync via **rclone** — never the reverse.

---

## The core pipeline: extract → preprocess → compute II

Run a stage locally if it's cheap, on the cluster otherwise. Each compute script
has a `run_<name>.sbatch` driver (many are slurm arrays — see the
`case ${SLURM_ARRAY_TASK_ID}` blocks).

```
  Flickr30k captions+images / ImageNet-1k same-class pairs
                          │
        ┌─────────────────┼──────────────────────────┐
        ▼                 ▼                           ▼
 extract_hidden_    extract_hidden_clip.py    extract_hidden_images_
   states.py         (CLIP text + image)        states.py
 (Llama, sglang)     --max_length 30|77        (DinoV2 / ImageGPT)
        │                 │                           │
        └─────────────────┴──────────────────────────┘
                          │   per-token hidden states  (chunk_N pickle)
                          ▼
        ┌────────────────────────────────────────────┐
        │ preprocess_mean.py                          │
        │  winsorise 5/95  →  mean-pool token window  │
        │  --last_k --first_k --skip-first --concat   │
        └────────────────────────────────────────────┘
                          │   mean_chunk_N.safetensors  {layer_N: (n_samples, embed)}
                          ▼
        ┌────────────────────────────────────────────┐
        │ compute_ii_mean.py   (two folders A, B)     │
        │  build_information_matrix → ranks/cards     │
        │  → II + jackknife SE per layer              │
        │  --metrics hamming l2   --keep_indices …    │
        └────────────────────────────────────────────┘
                          │   layer_NN.safetensors
                          │   {inf_imb, inf_imb_se, ranks, cards, r_ranks, r_cards}
                          ▼
        notebooks/*_plot.py  →  paper/figs-tmlr/{,rebuttal/}*.pdf

  side-branch (CLIP token-cap rebuttal):
   build_clip_max77_keep.py ─► data/clip_max77_keep.npy ─► --keep_indices
                                  (filters images whose 5-caption concat ≤ 77 BPE)
```


| Stage | Entry point | Cluster driver | What it does |
|---|---|---|---|
| 1. Extract | `extract_hidden_states.py` | — | Per‑token hidden states for Llama (via sglang) |
| | `extract_hidden_clip.py` | `run_extract_clip_max77.sbatch` | CLIP text + image; text is the 5‑caption concatenation per image |
| | `extract_hidden_images_states.py` | `run_extract_images.sbatch` | DinoV2 / ImageGPT / generic HF vision |
| 2. Preprocess | `preprocess_mean.py` | `run_preprocess_mean.sbatch`, `run_preprocess_clip_max77.sbatch` | Winsorise (5/95) then mean‑pool over a token window → `mean_chunk_N.safetensors` |
| 3. Compute II | `compute_ii_mean.py` | `run_ii_mean.sbatch` (32‑task array) | II + jackknife SE per layer → `layer_NN.safetensors` (`inf_imb`, `inf_imb_se`, per‑row `ranks`, `cards`) |

Key `preprocess_mean.py` flags: `--last_k`, `--first_k`, `--skip-first` (drop BOS),
`--concat`. Key `compute_ii_mean.py` flags: `--metrics {hamming,l2}` (default:
**both**), `--mean_center` / `--l2_normalize` (these name the l2 output folder —
e.g. `--mean_center --l2_normalize` → `l2_centered_normalized/`), `--layer_A`
(one or more), and `--keep_indices <bool.npy>` to subset samples.

Dataset construction: `build_flickr_dataset.py`.

### Two II implementations — pick the right one
- **`information_imbalance_module.py`** (local JAX) — used by the whole mean
  pipeline. `build_information_imbalance` returns scalar fwd/bwd; `build_information_matrix`
  returns per‑row ranks/cards for post‑hoc jackknife (this is what gives you
  per‑sample **residuals**: `ρ_i = 2·ranks[i]/cards[i]/(n_col−1)`).
- **`information_imbalance`** (pip‑installed NumPy) — used only by
  `notebooks/20251201_mat_*.py`. Different API — check imports before assuming.

Supporting metric libs: `cka.py`, `rsa.py` (Spearman/RSA), `alignment_metrics_lib.py`.

---

## Figures

| Figure | Compute | Plot | Notes |
|---|---|---|---|
| **Fig. 1** (headline II‑vs‑CKA) | `notebooks/figure1_compute.py` | `notebooks/figure1_plot.py` | Synthetic Gaussians; II + CKA + RSA + Neighborhood Overlap, jackknife. Driver: `slurm/launch_figure1.sh`. Cached: `notebooks/figure1_data.npz`. |
| CLIP rebuttal (token‑cap control) | — (reads `results-mean/`) | `notebooks/clip_max77_plot.py`, `notebooks/clip_four_cell_plot.py`, `notebooks/clip_vs_deepseek_filtered_plot.py` | Render to `paper/figs-tmlr/rebuttal/` |
| CKA at p=10⁵ | `legacy/run_cka.py` (pickles in `infimbgauss/`) | `notebooks/cka_p1e5_plot.py` | |
| Flickr token lengths | — | `notebooks/flickr_token_lengths.py` | Self‑contained (downloads Flickr30k) |

Typical render:
```bash
.venv/bin/python notebooks/figure1_plot.py --input notebooks/figure1_data.npz \
    --output paper/figs-tmlr/figure1_ii_cka
```

---

## CLIP rebuttal pipeline (current TMLR work)

Tests whether CLIP's 77‑token cap explains its higher II. The four cells are
`max_length ∈ {30, 77} × dataset ∈ {all, ≤77‑BPE subset}`:

- `build_clip_max77_keep.py` (+ `run_build_clip_max77_keep.sbatch`) — builds
  `data/clip_max77_keep.npy`, the 5,000‑bool mask of images whose 5‑caption
  concatenation fits CLIP's 77‑token cap (3,330 of 5,000).
- `run_extract_clip_max77.sbatch` → `run_preprocess_clip_max77.sbatch` →
  `run_ii_clip_max{30,77}.sbatch` (cells selected via `SUFFIX` / `KEEP_INDICES`).
- `run_ii_deepseek_dino_filtered.sbatch` — DeepSeek↔DinoV2 on the filtered subset.

Example (array job with env overrides):
```bash
sbatch --export=ALL,KEEP_INDICES=data/clip_max77_keep.npy,SUFFIX=_filtered \
    run_ii_clip_max30.sbatch
```
Smoke check: `verify_clip_max77_smoke.py`.

---

## Tests & diagnostics

```bash
.venv/bin/python -m pytest tests/test_preprocessing.py -xvs
```
Distance/geometry diagnostics: `analyze_distances.py`, `plot_distance_diagnostics.py`,
`validate_sweep.py`, `test_twoNN.py`.

---

## Repository layout

| Path | Contents |
|---|---|
| `*.py` (root) | Pipeline entry points + metric libraries |
| `run_*.sbatch`, `slurm/` | Cluster drivers (slurm arrays) |
| `notebooks/` | Figure compute/plot scripts (dated experiment scripts too) |
| `data/` | Local copies of cluster results (`results-mean/`), keep‑masks — **gitignored** |
| `paper/` | TMLR paper source + figures (`figs-tmlr/`, `figs-tmlr/rebuttal/`) |
| `legacy/` | Superseded compute scripts — see `REPORT.org` before reusing |
| `infimbgauss/` | Gaussian/Bernoulli CKA pickles |

Outputs land in `${IMAGE_BASE}/results-mean/<pair>{,_filtered}/<metric>/layer_NN.safetensors`,
synced locally under `data/results-mean/`.

> ⚠️ Don't reach into `legacy/` unless you've confirmed the active path doesn't
> already cover what you need (`REPORT.org` documents every superseded file).
