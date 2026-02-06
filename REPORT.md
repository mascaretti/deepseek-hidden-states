# deepseek-hidden-states – code review and pipeline report

## Overview
- Purpose: quantify information imbalance (II) between hidden states of text and vision models (DeepSeek-V3, LLaMA 3.1, CLIP text/image, DinoV2, ImageGPT) using JAX kernels, then bootstrap and plot results.
- Pipeline (current): extract activations → compute II matrices per comparison → bootstrap ranks/cards → plot; orchestrated via SLURM jobs pointing to data under `/home/laios/*` and outputs under `/home/laios/image-information-imbalance`.

## Active entry points (SBATCH-driven, minimal set)
- `run_asymmetry_array.sbatch` → `compute_asymmetry_unified.py` + `bootstrap_analysis.py` (single-layer DeepSeek/CLIP/Dino comparisons).
- `overnight_multilayer.sbatch` → `run_overnight_analysis.sh` → `compute_asymmetry_multi_layer.py` (uses collectors from `compute_asymmetry_unified.py`) + `bootstrap_multilayer.py` + `plot_results_simple.py` (plus inline matplotlib).
- `submit_deepseek_llama.sbatch` → `compute_deepseek_llama_comparison.py` (reuses unified collectors, `bootstrap_analysis.py`, plotting).
- `send_ii.sbatch` → `20250505_gaussianII.py` (synthetic II simulation).
- Everything else is currently not invoked by SLURM and can be treated as legacy until re-hooked.

## Code review findings (ordered by severity)
- `compute_information_imbalance_imageimage.py:137-141` and `compute_information_imbalance_images_jackknife.py:137-141`: save filenames use `n_tokens` which is out of scope, so the last value from the loop is undefined at save time → crash or wrong filenames. Should use the loop variable explicitly.
- `compute_platonic_images.py:109`: `build_mutual_k_NN_alignment` returns `(alignment, inf_imb_XY, inf_imb_YX)` but the code assigns them to `(inf_imb, reciprocal_inf_imb, mutual_imb)`, so “inf_imb” actually holds alignment and the two II directions are swapped/misnamed.
- `compute_asymmetry_unified.py:255-261`: comparison-type routing labels `deepseek_clip_text` as `text_image`, so text/text runs are preprocessed like text→image (token flattening still works but semantics and naming are wrong; impacts downstream interpretation).
- `run_asymmetry_array.sbatch:109`: stray literal `4` after the “Starting bootstrap analysis...” echo will break the script before bootstrap, so batches won’t run bootstrap on the cluster.
- Loader/format mismatch risk: `collect_clip_text_data` expects `text-chunk_*` in the root folder, while `extract_hidden_clip.py` writes `text/chunk_*` (and optionally `image-chunk_*`). If data are stored with the subfolder convention, current loaders miss them; conversely, if only `chunk_*` exists, CLIP text/image separation fails.
- Multi-layer assumptions: `compute_asymmetry_multi_layer.py:58-61` requires `layer_1` to exist for sample sizing and skips `layer_0`; missing layers will crash. Token windows (9 for text, 200 for images) are hard-coded in multiple places, increasing divergence risk.

## Current pipeline (as implemented)
1) Extract activations: `extract_hidden_states.py` (DeepSeek/LLaMA text), `extract_hidden_images_states.py` (ImageGPT/Dino/CLIP vision), `extract_hidden_clip.py` (CLIP text or vision). Outputs are pickle chunks (`chunk_*`, `image-chunk_*`, `text-chunk_*` depending on script).
2) Compute II:
   - Single-layer: `compute_asymmetry_unified.py` routes comparisons (`deepseek_clip_text`, `clip_text_image`, `dino_clip_image`, `deepseek_dino`, `deepseek_clip_image`), flattens last tokens, binarizes, and writes per-target-layer `.safetensors`.
   - Multi-layer sweeps: `compute_asymmetry_multi_layer.py` iterates selected source layers, saves subfolders plus metadata.
   - DeepSeek vs LLaMA: `compute_deepseek_llama_comparison.py` wraps unified helpers, bootstraps, and plots.
3) Bootstrap: `bootstrap_analysis.py` (single-layer) and `bootstrap_multilayer.py` (multi-layer) resample ranks/cards to `.npy` plus summary JSON.
4) Plotting: `plot_results_simple.py`, `plot_all_results.py`, `plot_results.py` read bootstrap outputs and emit PDFs/PNGs; `run_overnight_analysis.sh` also generates inlined plots per comparison.

## Refactor directions (behavior-preserving)
- Consolidate loaders into one module with explicit schema (chunk prefix, token window, dtype handling) and reuse across all compute scripts; remove on-the-fly script generation in `run_overnight_analysis.sh`.
- Centralize config (paths, subset sizes, token counts, layer IDs, seeds) and pass into compute/bootstrap/plot steps instead of hard-coded `/home/laios/...` scattered across scripts.
- Normalize comparison typing (text_text vs text_image vs image_image) in one place to avoid misrouting (fix `deepseek_clip_text` case).
- Add a lightweight preflight validator (presence of expected chunk files, layer keys, token lengths) before heavy JAX work; include metadata JSON alongside each safetensors run (models, layers, tokens, subset size, seed).
- Standardize filenames and prefixes for CLIP text/image outputs to match loaders; document the expected layout.

## How to sanity-check without expensive runs
- File presence: `verify_data.py` to confirm activation chunk counts; ensure CLIP text/image prefixes match loader expectations.
- Tiny smoke: run `compute_asymmetry_unified.py` with a small `subset_size`/`n_files` override (manual edit or wrapper) to ensure safetensors are emitted and shapes match; check `layer_1` exists.
- Bootstrap shape check: run `bootstrap_analysis.py` with `n_replicas=10` on the smoke output to confirm forward/backward arrays shape `(n_layers_B-1, n_replicas)` and reasonable values.
- Self-check: run DeepSeek self-comparison (layer 52 → all) to verify II≈0 on the diagonal (`compute_deepseek_llama_comparison.py --include_validation` has this path).

## Data and outputs
- Inputs: pickled torch tensors (text: `meta_info.hidden_states[0]`; images/CLIP: dict of layer tensors). Naming is critical (`chunk_*`, `image-chunk_*`, `text-chunk_*`).
- Outputs: per-layer safetensors with `ranks/cards` (and reciprocal), bootstrap `.npy` pairs named `{model_a}_{model_b}_{layer}.npy`, summary JSON, and plots under `plots/`.

