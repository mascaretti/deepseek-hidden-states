import sys,os
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.append('../')
import numpy as np

import jax
from utils import (
                   makefolder,
                   emb_dims,
                   depths,
                   )
from datapaths import *
from Spins.spins import *
import matplotlib.pyplot as plt


depths = {"deepseek":61,
          "llama":32}
emb_dims = {"deepseek":7168,
          "llama":4096}

model = sys.argv[1] # deepseek or llama
n_tokens_list = [int(sys.argv[2])]
print(f'{n_tokens_list=}')
seed_list = [0,1,2]
print(f'{seed_list=}')

Nbits_list = [1]
avg_flags = [0]
n_files = 500
languageA = 'english'
languageB = 'spanish'
languages = ['source']
min_token_length = 40

layers = np.arange(1,depths[model]+1,1,dtype=int)


for layer_idx in layers:
    print(f'{layer_idx=}')
    for n_tokens in n_tokens_list:
        print(f'{n_tokens=}')
        n_features = n_tokens * emb_dims[model]
        for seed_id, jack_seed in enumerate(seed_list):
            print(f'{jack_seed=}')
            for language_id,language in enumerate(languages):
                output_folder0 = makefolder(base=f'./results/santiago/',
                                            create_folder=True,
                                            modelA=model,
                                            modelB=model,
                                            n_files=n_files,
                                            min_token_length=min_token_length,
                                            languageA=languageA,
                                            languageB=languageB,
                                            language=language,
                                            )
                H = Hamming(crossed_distances=1)
                histfolder = makefolder(base=output_folder0 + '/hist/',
                                        n_tokens=n_tokens,
                                        layer_idx=layer_idx,
                                        seed=jack_seed,)
                H.D_histogram(compute_flag=0,
                            resultsfolder=histfolder,
                            )
                alphamin = .15    
                alphamax_list = [.4] #np.arange(.55,.55+eps,.05)        
                Nsteps = int(3e3)   
                export_results = 1  
                export_logKLs = 1   
                optfolder0 = makefolder(base=output_folder0 + '/opt/',
                                    n_tokens=n_tokens,
                                    layer_idx=layer_idx,
                                    seed=jack_seed,
                                    )
                steps_initial = jnp.array([1E-1],dtype=jnp.double)
                for alphamax_id,alphamax in enumerate(alphamax_list):
                    B = BID(
                        H=H,
                        alphamin=alphamin,
                        alphamax=alphamax,
                        Nsteps=Nsteps,
                        export_results=export_results,
                        export_logKLs=export_logKLs,
                        optfolder0=optfolder0,
                        L=n_features,
                        steps_initial=steps_initial,
                    )
                    B.computeBID()  