import requests
from sglang.test.test_utils import is_in_ci
if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

from sglang.utils import wait_for_server, print_highlight, terminate_process
import os

def main():
    NODE_RANK = int(os.environ["SLURM_NODEID"])
    server_process, port = launch_server_cmd(
        f"python3 -m sglang.launch_server --model /home/rende/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/86518964eaef84e3fdd98e9861759a1384f9c29d --trust-remote-code --tp 16 --dist-init-addr 10.149.146.53:8000 --nnodes 2 --grammar-backend xgrammar --node-rank {NODE_RANK}"
    )

    wait_for_server(f"http://localhost:{port}")

    from sglang.utils import print_highlight
    from tqdm import tqdm
    import requests
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

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

if __name__ == "__main__":
    main()