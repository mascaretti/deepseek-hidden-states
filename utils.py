import re
from pathlib import Path
import torch
import jax.numpy as jnp

from einops import rearrange

def extract_index(file):
    match = re.search(r'chunk_(\d+)\.pkl', file.name)
    return int(match.group(1)) if match else float('inf')  # push non-matching files to the end

def list_folder(path, desc="chunk_"):
    folder = Path(path)

    #* collect all the files starting with *chunk_* in the input_path
    files = sorted(
        [f for f in folder.iterdir() if f.name.startswith(desc)],
        key=extract_index)

    return files

def bf16_torch_to_jax(tensor):
    import jax.numpy as jnp
    intermediate_tensor = tensor.view(torch.uint16)
    tensor = jnp.array(intermediate_tensor).view('bfloat16')
    return tensor

def flatten_tokens_features(tensor):
    tensor = rearrange(tensor, 'batch tokens embed -> batch (tokens embed)')
    return tensor

def binarize(a, eps=1e-6, dtype=jnp.int32):
    a = jnp.sign(a + eps)
    return a.astype(dtype)
