import pickle
import numpy as np
from pathlib import Path
import re,os,sys
from time import time

def R_path(lower_case_language,model):
  if model == 'llama':
    path = f"/home/laios/activations/llama-8b/llama_{lower_case_language}"
  elif model =='deepseek':
    path = f"/home/laios/activations/deepseek/deepseek_{lower_case_language}"
  folder = Path(path)
  return folder,path

def extract_index(file):
    match = re.search(r'chunk_(\d+)\.pkl', file.name)
    return int(match.group(1)) if match else float('inf')  # push non-matching files to the end

def load_files(lower_case_language,
               model,
               Nsamples,
               batch_size,
               Ntokens=20,
               emb_dim=7168,
               Nlayers=62,
               initial_file_id=0
               ):

  start = time()
  folder,path = R_path(lower_case_language)
  files = sorted(
      [f for f in folder.iterdir() if f.is_file() and f.name.startswith("chunk_")],
      key=extract_index)

  for file_index,file in enumerate(files[initial_file_id:Nsamples//batch_size]):
    print(f'{file_index=}')
    a_batch = np.zeros(shape=(batch_size,Nlayers,Ntokens,emb_dim))
    data = pickle.load(open(path + "/" + file.name, 'rb'))
    for sentence_index,sentence in enumerate(data):
      a_batch[sentence_index] = sentence['meta_info']['hidden_states'][0][:,-Ntokens:,:].float().numpy()
    for layer_id in range(Nlayers):
      act_folder = f'activations/{model}/{lower_case_language}/b_{file_index+initial_file_id}/'
      os.makedirs(act_folder,exist_ok=True)
      np.save(f'{act_folder}l_{layer_id}',a_batch[:,layer_id,:,:])
  print(f'importing took {(time()-start)/60.:.1f} m')
  return 

lower_case_language = 'spanish'
lower_case_language = 'english'

load_files(lower_case_language,
           model="deepseek",
           Nsamples=7000,
           batch_size=10,
           initial_file_id=0)
# load_files(lower_case_language,
#            "llama",
#            Nsamples=7000,
#            batch_size=10,
#            Ntokens=20,
#            emb_dim=4096,
#            Nlayers=33,
#            initial_file_id=0)