import sys,os
sys.path.append('../../')
sys.path.append('../')


os.environ["JAX_PLATFORMS"] = "cpu"
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from information_imbalance_module import (normalized_L2_distance, 
                                          hamming_distance, 
                                          build_mutual_k_NN_alignment, 
                                          build_information_imbalance,
                                          separate_samples,
                                          pairwise_distances,
                                          mapped_compute_ranks,
                                          mapped_relative_ranks,
                                          )
import jax,jax.numpy as jnp
from utils import (bf16_torch_to_jax, 
                   flatten_tokens_features, 
                   list_folder, 
                   binarize, 
                   clip,
                   makefolder,
                   )
import pickle
from tqdm import tqdm
import torch
from datapaths import *
import argparse
from II_santiago import collect_data
from time import time


start = time()
N_list = jnp.arange(10,100,5,dtype=int)
N_list = jnp.concatenate(jnp.array([N_list,
                                    jnp.arange(100,1000,50,dtype=int)])
                        )
Ns_list = [1000]
seed_list = np.arange(0, 10,dtype=int)
Nbits_list = [1]
k_II = 1

for Nbits_id,Nbits in enumerate(Nbits_list):
    for Ns_id,Ns in enumerate(Ns_list):
        distance_fn = normalized_L2_distance if Nbits == 0 else hamming_distance
        key = jax.random.key(42)
        key, subkey = jax.random.split(key)
        information_imbalance_fn = build_information_imbalance(key=subkey, 
                                                                sample_size=Ns, 
                                                                distance_fn=distance_fn,
                                                                k=k_II)  
        for N_id,N in enumerate(N_list):
            IIs = np.zeros(shape=(len(seed_list),2))
            print(f'{N=}',flush=True)                                                      
            for seed in seed_list:
                X = np.random.normal(loc=0, scale=1, size=(Ns, N))
                k = N   # Number of additional random features
                random_features = np.random.normal(loc=0, scale=1, size=(Ns,k))
                Y = np.concatenate((random_features,X), axis=1)
                if Nbits == 1:
                    Y = binarize(Y)
                    X = binarize(X)
                IIs[seed,:] = information_imbalance_fn(X, Y)
                outputfolder = makefolder(base='results/',
                                        create_folder=True,
                                        N=N,
                                        Ns=Ns,
                                        Nbits=Nbits,
                                        k_II=k_II,
                                        )
            np.savetxt(outputfolder + 'IIs.txt', IIs)


print(f'Total time: {time() - start:.2f} seconds', flush=True)