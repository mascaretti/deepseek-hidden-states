import os,sys
import torch
import sglang as sgl
import pickle
import time
import socket

def get_master_address():
    hostnames = os.popen("scontrol show hostname $SLURM_JOB_NODELIST").read().split()
    return socket.gethostbyname(hostnames[0])

dist_init_addr = f"{get_master_address()}:8000"


def batch_generator(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def main(model, x, languageA, languageB, batch_size=10, n_prompts=None): # x = "source" or "target"
    
    if model == 'deepseek':
        model_path = "/home/rende/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/86518964eaef84e3fdd98e9861759a1384f9c29d"
        tp_size = 16
        nnodes = 2
    elif model == 'llama':
        model_path = "/home/rende/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
        tp_size = 1
        nnodes = 1

    if languageA == "en" and languageB == "it":
        file_path = f"/home/laios/activations/opus_english_italian/deepseek_{x}.txt"
    elif languageA == "de" and languageB == "en":
        file_path = f"/home/laios/activations/opus_german_english/deepseek_{x}.txt"
    else:
        file_path = f"/home/laios/Data/Text/sentences/corpus_Helsinki-NLP/opus_books/DeepSeek-V3/languages_{languageA}-{languageB}/min_token_length_40/max_token_length_80/{x}.txt"

    output_folder_path = f"./{model}_{languageA}_{languageB}_{x}" #os.path.splitext(file_path)[0]  # removes .txt extension

    with open(file_path, "r") as f:
        prompts = [line.strip() for line in f]
    if n_prompts is not None:
        prompts = prompts[:n_prompts]

    # Create an LLM.
    NODE_RANK = int(os.environ.get("SLURM_NODEID", 0))

    llm = sgl.Engine(
        model_path=model_path,
        tp_size=tp_size,
        return_hidden_states=True,
        nnodes=nnodes,
        dist_init_addr=dist_init_addr,
        node_rank=NODE_RANK,
        grammar_backend="xgrammar",
        disable_radix_cache=True,
    ) 

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 2,
    }

    os.makedirs(output_folder_path, exist_ok=True)

    for i, p in enumerate(batch_generator(prompts, batch_size)):
        start = time.time()
        outputs = llm.generate(p, sampling_params=sampling_params)
        with open(f"{output_folder_path}/chunk_{i}.pkl", "wb") as f:
            pickle.dump(outputs, f)

        t_step = time.time() - start
        print(f"iter {i} | t_step = {t_step:.2f}", flush=True)

    llm.shutdown()

# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine

if __name__ == "__main__":
    model = sys.argv[1] # deepseek or llama
    x = sys.argv[2] # source or target
    languageA = sys.argv[3] # in acronym form, e.g., en
    languageB = sys.argv[4] # in acronym form, e.g., es
    n_prompts = int(sys.argv[5]) if len(sys.argv) > 5 else None
    main(model, x, languageA, languageB, n_prompts=n_prompts)
