import sys,os
sys.path.append('../')
from utils import makefolder
from datapaths import *



models = ['deepseek','llama']
languageA = 'english'

min_token_length = 40
mask = 2
n_loaded_tokens = min_token_length - mask
dbg = 0

if dbg == 0:
    n_files = 500
elif dbg == 1:
    n_files = 2


for model_id,model in enumerate(models):
    input_path = input_paths[model][languageA]
    output_folder = makefolder(base=f'./results/santiago/',
                                create_folder=True,
                                model=model,
                                n_files=n_files,
                                languageA=languageA,
                                mask=mask,
                                )

    cmd = (
        f"python3 token_II_santiago.py "
        f"--input_path={input_path} "
        f"--output_folder={output_folder} "
        f"--n_files={n_files} "
        f"--mask={mask} "
        f"--n_loaded_tokens={n_loaded_tokens} "
    )

    os.system(cmd)