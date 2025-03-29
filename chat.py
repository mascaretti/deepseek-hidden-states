from sglang.utils import print_highlight
from tqdm import tqdm
import requests
url = f"http://127.0.0.1:30000/v1/chat/completions"

import sys
f = open('chinese.txt', 'w')
sys.stdout = f #* redirect the outputs in this file

file_path = "./source.txt"

with open(file_path, 'r') as file:
    for line in tqdm(file):
        line = line.strip()

        question = "Can you translate the following sentence into Chinese? Provide only the answer.\n"
        prompt = question + line

        print("======== Prompt ========", flush=True)
        print(prompt)

        data = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": [{"role": "user", "content": question + line}],
        }

        print("******** Answer ********")

        response = requests.post(url, json=data)
        print_highlight(response.json()['choices'][0]['message']['content'])

        print("========================")

