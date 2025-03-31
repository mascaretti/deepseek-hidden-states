import torch
import sglang as sgl
import os
import pickle
from tqdm import tqdm
from datasets import load_dataset
import time

def batch_generator(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

def main(batch_size=30):
    #file_path = "/home/laios/Data/Text/tokens/corpus_Helsinki-NLP/opus_books/LLM_A_Llama3/model_A_id_0/languageA_English/min_token_length_40/max_token_length_50/source.txt"
    file_path = "/home/laios/Data/Text/tokens/corpus_Helsinki-NLP/opus_books/LLM_A_Llama3/model_A_id_0/LLM_B_Llama3/model_B_id_0/languageA_English/languageB_Spanish/min_token_length_40/max_token_length_50/target.txt"
    output_folder_path = "./deepseek_spanish" #os.path.splitext(file_path)[0]  # removes .txt extension

    with open(file_path, "r") as f:
        prompts = [line.strip() for line in f]

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
    main()