import pickle

from pathlib import Path
import re

path = "../chinese_translations"
folder = Path(path)

def extract_index(file):
    match = re.search(r'chunk_(\d+)\.pkl', file.name)
    return int(match.group(1)) if match else float('inf')  # push non-matching files to the end

files = sorted(
    [f for f in folder.iterdir() if f.is_file() and f.name.startswith("chunk_")],
    key=extract_index)

for file in files[:2]:
    print(file.name)
    data = pickle.load(open(path + "/" + file.name, 'rb'))
    prompts, outputs = data

    for prompt, sentence in zip(prompts, outputs):
        # hidden_states = sentence['meta_info']['hidden_states'][0]
        # print(hidden_states.shape)
        print("--------")
        print("====== PROMPT ========")
        print(prompt)
        print("****** ANSWER ********")
        print(sentence['text'])
