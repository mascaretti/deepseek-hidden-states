#!/usr/bin/env python
"""
Per-layer PCA of mean-pooled activations — the same matrices II is computed on.

Input:  mean_chunk_N.safetensors from preprocess_mean.py  (via
        compute_ii_mean.load_mean_activations, so --subset_size / --keep_indices
        select exactly the rows II would have used).

For each layer the (N, d) centred matrix is decomposed on its smaller side:
    d <= N :  eigh of the (d, d) covariance      -> eigenvalues, loadings V
    d >  N :  eigh of the (N, N) Gram matrix     -> eigenvalues, scores U*sqrt(l*(N-1))
Both give the covariance-convention spectrum l_k = S_k^2 / (N - 1), i.e.
Var(score_k) == l_k.  float64 throughout; round-off negatives are clipped to 0.

Output:
  --output_npz     spectrum + dimensionality diagnostics (small, always written)
  --project_k K    additionally writes the top-K PC scores as
                   <project_out>/mean_chunk_0.safetensors -> {layer_N: (N, K)}
                   which compute_ii_mean.py reads unchanged.

NOTE: when --keep_indices is used the projections are already filtered — do not
      pass --keep_indices again to compute_ii_mean.py on that folder.

Usage:
  python compute_pca_mean.py \
      --input_path /home/laios/image-information-imbalance/windsorised_mean/facebook-dinov2-large \
      --output_npz results/pca/dinov2-large.npz \
      --project_k 100 \
      --project_out /home/laios/image-information-imbalance/pca_mean/facebook-dinov2-large_k100
"""

import argparse
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from tqdm import tqdm

from compute_ii_mean import load_mean_activations


def pca_layer(x, n_components):
    """PCA of one (N, d) matrix.

    Returns (eigenvalues, scores): eigenvalues are the full descending
    covariance spectrum, length min(N, d); scores is (N, n_components) or None
    when n_components is 0.
    """
    xc = x.astype(np.float64)
    xc -= xc.mean(axis=0)
    n, d = xc.shape

    if d <= n:
        w, v = np.linalg.eigh(xc.T @ xc / (n - 1))          # (d, d), ascending
        eigenvalues = np.maximum(w[::-1], 0.0)
        scores = xc @ v[:, ::-1][:, :n_components] if n_components else None
    else:
        w, u = np.linalg.eigh(xc @ xc.T / (n - 1))          # (N, N), ascending
        eigenvalues = np.maximum(w[::-1], 0.0)
        scores = (u[:, ::-1][:, :n_components]
                  * np.sqrt(eigenvalues[:n_components] * (n - 1))) if n_components else None

    return eigenvalues, scores


def main():
    parser = argparse.ArgumentParser(
        description="Per-layer PCA of mean-pooled activations"
    )
    parser.add_argument("--input_path", type=str, required=True,
                        help="Directory with mean_chunk_N.safetensors")
    parser.add_argument("--output_npz", type=str, required=True,
                        help="Where to write the spectrum/diagnostics npz")
    parser.add_argument("--subset_size", type=int, default=5000)
    parser.add_argument("--keep_indices", type=str, default=None,
                        help="Optional length-subset_size boolean .npy mask")
    parser.add_argument("--project_k", type=int, default=0,
                        help="Number of PCs to write out (0 = spectrum only)")
    parser.add_argument("--project_out", type=str, default=None,
                        help="Directory for the projected safetensors "
                             "(required when --project_k > 0)")
    args = parser.parse_args()

    if args.project_k and not args.project_out:
        parser.error("--project_k requires --project_out")

    keep_indices = None
    if args.keep_indices:
        keep_indices = np.load(args.keep_indices).astype(bool)
        print(f"Keep mask: {int(keep_indices.sum()):,}/{len(keep_indices):,} kept "
              f"({100*keep_indices.mean():.2f}%) — {args.keep_indices}")

    print("Loading activations...")
    act = load_mean_activations(args.input_path, subset_size=args.subset_size,
                                keep_indices=keep_indices)

    layers = sorted((int(k.split("_")[1]) for k in act), key=int)
    n_samples, embed_dim = act["layer_0"].shape
    n_eig = min(n_samples, embed_dim)
    k = min(args.project_k, n_eig)
    if k != args.project_k:
        print(f"--project_k {args.project_k} > min(N, d) = {n_eig}; using {k}")

    print(f"\n{len(layers)} layers, N={n_samples}, d={embed_dim}, "
          f"{n_eig} eigenvalues/layer, side={'covariance' if embed_dim <= n_samples else 'Gram'}")

    eigenvalues = np.zeros((len(layers), n_eig), dtype=np.float64)
    projections = {}
    for row, layer in enumerate(tqdm(layers, desc="PCA")):
        eig, scores = pca_layer(act[f"layer_{layer}"], k)
        eigenvalues[row] = eig
        if k:
            projections[f"layer_{layer}"] = np.ascontiguousarray(scores, dtype=np.float32)

    # Dimensionality diagnostics from the spectrum
    total_variance = eigenvalues.sum(axis=1)
    p = eigenvalues / total_variance[:, None]
    evr_cumsum = np.cumsum(p, axis=1)
    spectral_entropy = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0), axis=1)
    participation_ratio = total_variance ** 2 / np.sum(eigenvalues ** 2, axis=1)
    # +1: searchsorted gives the count of PCs strictly below the threshold
    n_pcs = {f"n_pcs_{int(t*100)}": np.array(
        [np.searchsorted(evr_cumsum[i], t) + 1 for i in range(len(layers))], dtype=np.int32)
        for t in (0.90, 0.95, 0.99)}

    Path(args.output_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        layers=np.array(layers, dtype=np.int32),
        n_samples=np.array(n_samples),
        embed_dim=np.array(embed_dim),
        eigenvalues=eigenvalues.astype(np.float32),
        evr_cumsum=evr_cumsum.astype(np.float32),
        total_variance=total_variance,
        spectral_entropy=spectral_entropy,
        participation_ratio=participation_ratio,
        effective_rank=np.exp(spectral_entropy),
        **n_pcs,
    )
    print(f"\nSaved {args.output_npz}")

    step = max(1, len(layers) // 8)
    print(f"\n{'layer':>6}  {'PR':>8}  {'eff.rank':>9}  {'H_spec':>7}  "
          f"{'n90':>5}  {'n95':>5}  {'n99':>5}")
    for i in range(0, len(layers), step):
        print(f"{layers[i]:>6}  {participation_ratio[i]:>8.2f}  {np.exp(spectral_entropy[i]):>9.2f}  "
              f"{spectral_entropy[i]:>7.3f}  {n_pcs['n_pcs_90'][i]:>5}  "
              f"{n_pcs['n_pcs_95'][i]:>5}  {n_pcs['n_pcs_99'][i]:>5}")

    if k:
        Path(args.project_out).mkdir(parents=True, exist_ok=True)
        out_file = Path(args.project_out, "mean_chunk_0.safetensors")
        save_file(projections, str(out_file))
        print(f"\nSaved {out_file}  ({len(projections)} layers, {n_samples}x{k})")


if __name__ == "__main__":
    main()
