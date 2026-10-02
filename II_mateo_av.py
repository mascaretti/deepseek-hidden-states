import jax
import jax.numpy as jnp
from dadapy._utils.metric_comparisons import _return_imbalance
import numpy as np
import pickle
from pathlib import Path
from typing import List
from tqdm import tqdm
import warnings
import torch
import dadapy

import typer

app = typer.Typer()
def information_imbalance(indices1: np.ndarray, indices2: np.ndarray, rng, k: int) -> list:
    """Compute information imbalance between given indices and indices from a path."""    
    assert len(indices1) == len(indices2), \
        "Mismatch in number of samples, impossible to perform information imbalance"
    
    return [
        _return_imbalance(indices1, indices2, rng, k=k),
        _return_imbalance(indices2, indices1, rng, k=k)
    ]


def get_quantiles(a,alphamin,alphamax):
  qmin = jnp.quantile(a,q=alphamin,axis=1)
  qmax = jnp.quantile(a,q=alphamax,axis=1)
  return qmin,qmax

def clip(act):
    alphamin = 0.05
    alphamax = 0.95
    
    if len(act.shape)==3:
      reshape = True
    else:
      reshape = False

    if reshape:
      B,T,E = act.shape
      act = jnp.reshape(act,shape=(B,T*E))
    
    qmin,qmax = get_quantiles(act,alphamin,alphamax)
    act = jnp.clip(act.T,min=qmin,max=qmax).T

    # if reshape:
    #   act = jnp.reshape(act,shape=(B,T,E))
    return act

def normalized_L2_distance(x, y):
    x /= jnp.linalg.norm(x)
    y /= jnp.linalg.norm(y)
    u = (x-y)
    return jnp.sqrt(jnp.sum(u*u))

def pairwise_distances(distance, xs, ys):
  return jax.vmap(lambda x: jax.vmap(lambda y: distance(x, y))(xs))(ys).T

@app.command()
def main(input_path, 
         output_folder:str=None,
         n_files:int=1,
         diagonal_only:bool=False,  # New option to compute only the diagonal
         rank:int=0,
         ):
    # collect all pickles
    files = sorted(Path(input_path).glob("*.pkl"))
    #shuffle
    # rng = np.random.default_rng(42)
    # rng.shuffle(files)
    assert len(files) > n_files* (rank+1), f"Not enough files to process. {len(files)} files found, but {n_files*(rank+1)} files are required."
    files = files[n_files*rank:n_files*(rank+1)]

    acts = []
    for file in tqdm(list(files), desc="Collect File"):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            _a = {k: torch.tensor(v) for k, v in data[0].items()}
            _b = {k: torch.tensor(v) for k, v in data[1].items()}
            acts.append((_a, _b))
    inf_imb_matrix = []
    reciprocal_inf_imb_matrix = []
    all_activations_A = {f"layer_{i}": torch.cat([act[0][f"layer_{i}"] for act in acts], dim=0) for i in range(len(acts[0][0].keys()))}
    all_activations_B = {f"layer_{i}": torch.cat([act[1][f"layer_{i}"] for act in acts], dim=0) for i in range(len(acts[0][0].keys()))}
    all_activations_A = {k: v.mean(dim=1, keepdim=False).detach().cpu().numpy() for k, v in all_activations_A.items()}
    all_activations_B = {k: v.mean(dim=1, keepdim=False).detach().cpu().numpy() for k, v in all_activations_B.items()}

    sample_size = all_activations_B["layer_0"].shape[0]
    rows = all_activations_A.keys()
    cols = all_activations_B.keys()

    # key = jax.random.key(42)
    # key, subkey = jax.random.split(key)
    # key = np.random.default_rng(42)

    for row in tqdm(rows, desc="Compute Distances"):
        for col in tqdm(cols, desc="Computing layer_B", leave=False):
            if diagonal_only and row != col:
                continue
            act_A = all_activations_A[row]
            act_B = all_activations_B[col]
            # print(f"act_A.shape: {act_A.shape}, act_B.shape: {act_B.shape}")

            # Clip activations
            act_A = np.array(clip(act_A))
            act_B = np.array(clip(act_B))
            # print(f"act_A.shape: {act_A.shape}, act_B.shape: {act_B.shape}")
            # input("continue?")

            dt_A = dadapy.Data(act_A)
            dt_B = dadapy.Data(act_B)
        
            dt_A.compute_distances(maxk=sample_size-1, metric="euclidean")
            dt_B.compute_distances(maxk=sample_size-1, metric="euclidean")
            # print(dt_A.dist_indices[0], dt_B.dist_indices[0])

            # print(distA.shape, distB.shape)
            # input("continue?")

            # ii_ab, ii_ba = information_imbalance(dt_A.dist_indices, dt_B.dist_indices, key, k=sample_size-1)
            ii=dt_A.return_information_imbalace(act_B, subset_size=sample_size-1)
            ii_ab = ii[0][0]
            ii_ba = ii[1][0]
            tqdm.write(f"(II(A-->B) {ii_ab:.4f} ; II(B-->A) {ii_ba:.4f})")
            inf_imb_matrix.append((row, col, ii_ab))
            reciprocal_inf_imb_matrix.append((row, col, ii_ba))

    # Save the results
    if output_folder is None:
        output_folder = Path(input_path).parent / "information_imbalance"
    else:
        output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / f"clipped_II_AB_{rank}.pkl", 'wb') as f:
        pickle.dump(inf_imb_matrix, f)
    with open(output_folder / f"clipped_II_BA_{rank}.pkl", 'wb') as f:
        pickle.dump(reciprocal_inf_imb_matrix, f)
            
if __name__ == "__main__":
    app()