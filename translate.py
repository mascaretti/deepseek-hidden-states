import torch
import sglang as sgl
import os
import pickle
from tqdm import tqdm
import time

def batch_generator(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def main(batch_size=10):
    file_path = "./source.txt"

    question = "Can you translate the following sentence into Chinese? Provide only the answer.\n"

    with open(file_path, "r") as f:
        prompts = [question + line.strip() for line in f]

    # Create an LLM.
    NODE_RANK = int(os.environ["SLURM_NODEID"])
    llm = sgl.Engine(
        model_path="/home/rende/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/86518964eaef84e3fdd98e9861759a1384f9c29d",
        tp_size=16,
        return_hidden_states=False,
        nnodes=2,
        dist_init_addr="10.149.146.53:8000",
        node_rank=NODE_RANK,
        grammar_backend="xgrammar",
    ) 

    folder_path = "./chinese_translations" #os.path.splitext(file_path)[0]  # removes .txt extension
    os.makedirs(folder_path, exist_ok=True)

    for i, p in enumerate(batch_generator(prompts, batch_size)):
        start = time.time()
        outputs = llm.generate(p)
        with open(f"{folder_path}/chunk_{i}.pkl", "wb") as f:
            pickle.dump(outputs, f)

        t_step = time.time() - start
        print(f"iter {i} | t_step = {t_step:.2f}", flush=True)

    llm.shutdown()

# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()