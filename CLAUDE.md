# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules
1. Smoke test everything.
2. Avoid creating new functions if possible: code is expensive to review.
3. We need to understand what we expect from what we run before we run it.
4. We take small steps.
5. Run locally what's fast, on the cluster everything else.
6. Before creating new code, we check if we can re-use what we have.
7. Source files (`.sbatch`, `.py`, `.sh`) sync to the cluster via git, **not** rclone. rclone is for data artefacts only (pickles, npz, safetensors, figures).

## What this codebase does

Measures **Information Imbalance (II)** and **CKA** between hidden states of large unimodal models (DeepSeek-V3, Llama-3.1-8B, CLIP text/image, DinoV2, ImageGPT) on Flickr30k captions+images. The published paper is **not** in this repo — it lives in the Overleaf-synced project `~/Documenti/Semantics-TMLR/` (this repo renders figures into its `figs-tmlr/`). The active branch (`image-captions`) is the TMLR rebuttal track.

## Workflow: extract → preprocess → compute II

The current ("mean") pipeline replaces an older family of `check_asymmetry*.py` / `compute_information_imbalance*.py` scripts that still live in `legacy/`. **Don't reach for `legacy/` unless you've checked the active path doesn't already cover it.**

1. **Extract** per-token hidden states (one chunk pickle per batch):
   - `extract_hidden_states.py` — Llama via sglang
   - `extract_hidden_clip.py` — CLIP text + image (text uses `max_length=30` historically; left-padded — see `extract_hidden_clip.py:51-58`)
   - `extract_hidden_images_states.py` — DinoV2 / ImageGPT / generic HF vision

2. **Preprocess** with `preprocess_mean.py`: winsorise (5/95 quantile) then mean-pool over a token window. Auto-detects on-disk format (DeepSeek pickle vs image-model pickle). Key flags: `--last_k`, `--first_k`, `--skip-first` (drop BOS), `--concat`. Output: `mean_chunk_N.safetensors` with `{layer_N: (n_samples, embed)}`.

3. **Compute II** with `compute_ii_mean.py`: loads two mean-pooled folders, runs `build_information_matrix` (raw ranks/cards) and computes II + jackknife SE per layer. Supports `--keep_indices <bool.npy>` to subset images. Metrics: `l2_centered_normalized` (default), `hamming` (binarised). Output: `layer_NN.safetensors` with `inf_imb`, `inf_imb_se`, `ranks`, `cards`.

   Driver: `run_ii_mean.sbatch` (32-task array, one model pair per task).

### Two II implementations — pick the right one
- **`information_imbalance_module.py`** (local, JAX) — used by everything in the active mean pipeline (`compute_ii_mean.py`, `notebooks/figure1_compute.py`). Has both `build_information_imbalance` (returns scalar fwd/bwd tuple) and `build_information_matrix` (returns per-row ranks/cards for post-hoc jackknife).
- **`information_imbalance`** (pip-installed, NumPy) — used by `notebooks/20251201_mat_*.py`. Different API (`build_information_imbalance(distance_fn, k, maxk, subset_size, rng)`). Easy to confuse — check imports before assuming.

## Figure 1 (the paper's headline II-vs-CKA figure) is fully scripted

- `notebooks/figure1_compute.py` — both panels (A: rank-asymmetry, B: shared-features at multiple `p`); II + CKA + Neighborhood Overlap; jackknife over `--n_reps`. Saves npz only.
- `notebooks/figure1_plot.py` — renders the figure from the npz.
- `slurm/launch_figure1.sh` — sbatch driver. Edit `--p_values` etc. inline.

The cached `notebooks/figure1_data.npz` is at `p_values=[100, 5000]`, `n_reps=10`, `n_samples=2500`.

## CLIP rebuttal pipeline (current TMLR work)

- `build_clip_max77_keep.py` (+ sbatch) — builds `data/clip_max77_keep.npy`, a 5,000-bool mask of images whose 5-caption concatenation fits in CLIP's 77-token cap (3,330 of 5,000).
- `run_extract_clip_max77.sbatch` — re-extracts CLIP-text at `max_length=77`.
- `run_preprocess_clip_max77.sbatch` — mean-pool the max=77 extraction.
- `run_ii_clip_max{30,77}.sbatch` — II for the four cells (max=30/77 × all/filtered) via `SUFFIX` and `KEEP_INDICES` env vars.
- `run_ii_deepseek_dino_filtered.sbatch` — DeepSeek↔DinoV2 on the filtered subset, `LAYER_A=36` (≈60% relative depth = layer 36 of 61).
- Plot scripts under `notebooks/clip_*_plot.py`.

Outputs land in `${IMAGE_BASE}/results-mean/<pair>{,_filtered}/<metric>/layer_NN.safetensors`. Locally synced under `data/results-mean/`.

## Commands

```bash
# Local environment (matplotlib, scienceplots, safetensors, jax)
.venv/bin/python <script>.py     # always use the venv, system python lacks matplotlib

# Run a single test
.venv/bin/python -m pytest tests/test_preprocessing.py::<name> -xvs

# Render a paper figure
.venv/bin/python notebooks/<plot>.py --results_dir data/results-mean --output ~/Documenti/Semantics-TMLR/figs-tmlr/rebuttal/<name>

# Inspect a results safetensors
.venv/bin/python -c "from safetensors.numpy import load_file; print(load_file('<path>').keys())"
```

### Cluster (slurm)
- User account: `mascaretti`. Repo on cluster: `/home/mascaretti/deepseek-hidden-states`.
- Shared data root: `/home/laios/image-information-imbalance` (`IMAGE_BASE` in sbatches).
- Activate: `source /home/$USER/myjax/bin/activate` (JAX env).
- Submit: `sbatch run_<name>.sbatch`. Many sbatches are SLURM arrays — see `case ${SLURM_ARRAY_TASK_ID}` blocks.
- Common env-var overrides: `SUFFIX`, `KEEP_INDICES`, `METRICS`. E.g.:
  ```
  sbatch --export=ALL,KEEP_INDICES=data/clip_max77_keep.npy,SUFFIX=_filtered run_ii_clip_max30.sbatch
  ```
- Logs: `logs/<jobname>_<jobid>{,_<arrayid>}.{out,err}`.

### Syncing data from cluster
rclone remotes are configured (`udine`, `claio`):
```
rclone copy udine:/home/laios/image-information-imbalance/results-mean/<dir>/ data/results-mean/<dir>/ -P
```
Don't rclone source files — use git.

## Where things live

- `~/Documenti/Semantics-TMLR/` — the Overleaf-synced paper project (source + figures). Figures render into its `figs-tmlr/` (and `figs-tmlr/rebuttal/`). The repo's own `paper/` copy was removed as a stale, un-synced duplicate.
- `infimbgauss/` — pickles from the Gaussian/Bernoulli CKA experiments (`legacy/run_cka.py`).
- `data/` — local copies of cluster results (`results-mean/`), keep-masks (`clip_max77_keep.npy`), tokeniser artefacts.
- `legacy/` — superseded compute scripts (REPORT.org documents what each file replaced and why).
- `REPORT.org` — exhaustive audit of which scripts are active vs superseded. Consult before assuming a file is the right entry point.

## Conventions

- sbatch headers: 10 cpus, 128 GB RAM, 1 GPU, 4–24 h walltime, mail to `amascare@sissa.it`.
- Layer indices are 0-based in code but 1-based in paper text.
- DeepSeek "≈ 60% relative depth" = layer 36 of 61.
- CLIP text encoder is fed the 5-caption concatenation per image, **not** per-caption (`extract_hidden_clip.py:54-61`).
- `scienceplots` is imported with try/except so plots render even when LaTeX/scienceplots is missing.
