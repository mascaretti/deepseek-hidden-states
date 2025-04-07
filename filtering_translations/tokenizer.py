import sys
from _paths import *
from _model import *
from _utils import *
import numpy as np
from time import time
start = time()

print(f'{sys.argv=}')
dbg = int(sys.argv[1])
LLM_A = sys.argv[2]
model_A_id = 1 #int(sys.argv[2])
LLM_B = sys.argv[3]
model_B_id = model_A_id #int(sys.argv[4])

languageA = sys.argv[4]
languageB = sys.argv[5]
# languageA = 'German'
# languageB = 'Hungarian'

model_A_name = get_models()[LLM_A][model_A_id]
model_B_name = get_models()[LLM_B][model_B_id]

corpus = "Helsinki-NLP/opus_books"

min_token_length = int(sys.argv[6])
max_token_length = int(sys.argv[7])

indeces_print = [0,1,2]
### DATA
tokens_folder = makefolder(base=path0 + f'tokens/',
                                  create_folder=True,
                                  corpus=corpus,
                                  LLMs=f'{model_A_name}-{model_B_name}',
                                  languages=f'{languageA}-{languageB}',
                                  min_token_length=min_token_length,
                                  max_token_length=max_token_length,
                                  )

languageA_acronym = get_acronym(languageA)
languageB_acronym = get_acronym(languageB)
dataset = load_dataset(corpus,
                      f'{languageA_acronym}-{languageB_acronym}',
                      cache_dir=cache_dir,
                      )
if dbg:
  k = 200
else:
  k = len(dataset["train"])
  # k = 50000
dataset = dataset["train"].select(range(k))
print(f'{len(dataset)=}')

### MODEL A
_, tokenizer = get_model(
                        modelname = model_A_name,
                        cache_dir=get_model_dir(LLM_A),
                        load_model=False,
                        )
# config = get_config(model_A_name,cache_dir=get_model_dir(LLM_A))
if tokenizer.pad_token == None:
  tokenizer.pad_token = tokenizer.eos_token
  
print(f'{tokenizer.pad_token_id=}')
print(f'{tokenizer.bos_token_id=}')

### SOURCE TOKENIZATION
print(f'{languageA=}')

X = dataset.map(lambda x : get_source(x,languageA_acronym))
dataset = X 

tokenization = tokenizer(dataset['source'],
                        return_tensors="np",
                        padding='longest',
                        return_attention_mask=True,
                        padding_side='left',
                        )
dataset = dataset.add_column('source_input_ids',tokenization['input_ids'].tolist())
dataset = dataset.add_column('source_attention_mask',tokenization['attention_mask'].tolist()) 


### SOURCE MIN TOKEN-LENGTH FILTERING
print(f'min_token_length filtering:')
X = dataset.filter(lambda x : filter_by_min_token_length(x,min_token_length,'source'))
dataset = X
print(f'max_token_length filtering:')
### SOURCE MAX TOKEN-LENGTH FILTERING
X = dataset.filter(lambda x : filter_by_max_token_length(x,max_token_length,'source'))
dataset = X

### TARGET TOKENIZATION
print(f'{languageB=}')
X = dataset.map(lambda x : get_target(x,languageB_acronym))
dataset = X # stupid intermediate step, this can be improved...

### MODEL B
_, tokenizer = get_model(
                        modelname = model_B_name,
                        cache_dir=get_model_dir(LLM_B),
                        load_model=False,
                        )
if tokenizer.pad_token_id == None:
  # config = get_config(model_B_name,cache_dir=get_model_dir(LLM_B))
  tokenizer.pad_token = tokenizer.eos_token
print(f'{tokenizer.pad_token_id=}')
print(f'{tokenizer.bos_token_id=}')

tokenization = tokenizer(dataset['target'],
                        return_tensors="np",
                        padding='longest',
                        return_attention_mask=True,
                        padding_side='left',
                        )
dataset = dataset.add_column('target_input_ids',tokenization['input_ids'].tolist())
dataset = dataset.add_column('target_attention_mask',tokenization['attention_mask'].tolist()) 

### TARGET MIN TOKEN-LENGTH FILTERING
print(f'min_token_length filtering:')
X = dataset.filter(lambda x : filter_by_min_token_length(x,min_token_length,'target'))
dataset = X
print(f'max_token_length filtering:')
### TARGET MAX TOKEN-LENGTH FILTERING
X = dataset.filter(lambda x : filter_by_max_token_length(x,2*min_token_length,'target'))
dataset = X

print(f'translations:')
for idx in indeces_print:
  print(dataset['source'][idx])
  print(dataset['target'][idx])
  print()
print('\n')

### SOURCE EXPORTING 

source_input_ids = np.array(dataset['source_input_ids'])[:,-max_token_length:] # saving some space here removing padding.
source_attention_mask = np.array(dataset['source_attention_mask'])[:,-max_token_length:]

print(f'{source_input_ids.shape=}')
print(f'{source_attention_mask.shape=}')
print(f'{source_input_ids[0]=}')
print(f'{source_input_ids[1]=}')

np.savetxt(tokens_folder + f'source_token_ids.txt',source_input_ids,fmt='%d')
np.savetxt(tokens_folder + f'source_attention_mask.txt',source_attention_mask,fmt='%d')

### SAVING FILTERED SOURCE TXT 
with open(f"{tokens_folder}source.txt", "w") as f:
    f.writelines("\n".join(dataset['source']))
print(f'{len(dataset["source"])=}')

### TARGET EXPORTING
empirical_max_token_length_B = np.array(dataset['target_attention_mask']).sum(axis=1).max()
target_input_ids = np.array(dataset['target_input_ids'])[:,-empirical_max_token_length_B:]
target_attention_mask = np.array(dataset['target_attention_mask'])[:,-empirical_max_token_length_B:]

print(f'{target_input_ids.shape=}')
print(f'{target_attention_mask.shape=}')
print(f'{target_input_ids[0]=}')
print(f'{target_input_ids[1]=}')
np.savetxt(tokens_folder + f'target_token_ids.txt',target_input_ids,fmt='%d')
np.savetxt(tokens_folder + f'target_attention_mask.txt',target_attention_mask,fmt='%d')

### SAVING FILTERED TARGET TXT 
with open(f"{tokens_folder}target.txt", "w") as f:
    f.writelines("\n".join(dataset['target']))  # Each string on a new line

print(f'this took {(time()-start)/60.} m')