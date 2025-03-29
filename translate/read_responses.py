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

for file in files:
    data = pickle.load(open(path + "/" + file.name, 'rb'))
    prompts, outputs = data

    for prompt, sentence in zip(prompts, outputs):
        # print("====== PROMPT ========")
        # print(prompt)
        # print("****** ANSWER ********")
        print(sentence['text'].replace("\n", ""))
