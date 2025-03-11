import torch
import sglang as sgl
import os
import pickle
from tqdm import tqdm
from datasets import load_dataset

def batch_generator(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def main(what="filckr", batch_size=10):
    # Read the "flickr.csv" file line by line
    prompts = []
    with open("/home/rende/deepseek/filckr8k_filtered_4080.csv", "r") as file:
        for line in file:
            prompts.append(line.strip())

    # Create an LLM.
    NODE_RANK = int(os.environ["SLURM_NODEID"])
    llm = sgl.Engine(
        model_path="/home/rende/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/86518964eaef84e3fdd98e9861759a1384f9c29d",
        tp_size=16,
        return_hidden_states=True,
        nnodes=2,
        dist_init_addr="10.149.146.53:8000",
        node_rank=NODE_RANK,
        grammar_backend="xgrammar",
    ) 

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 2,
    }

    all_outputs = []
    for p in tqdm(batch_generator(prompts, batch_size)):
        outputs = llm.generate(p, sampling_params=sampling_params)
        all_outputs.append(outputs)
    
    with open(f"{what}_outputs.pkl", "wb") as f:
        pickle.dump(all_outputs, f)

    llm.shutdown()

# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()