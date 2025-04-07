import sys,os
import numpy as np
from _paths import *
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          AutoConfig)

from torch import device,cuda
device = device('cuda' if cuda.is_available() else 'cpu')
print(f'{device=}')
if cuda.is_available():
  print(f"Available GPUs: {cuda.device_count()}")


def get_models():
  models = {
    'Llama3':["meta-llama/Llama-3.2-1B",
              "meta-llama/Llama-3.1-8B",
              "meta-llama/Llama-3.1-70B"],
    'DS_Llama3':['placeholder',
                 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
                 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'],
    'DS_Qwen':['deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
               'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
               'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
               'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'],
    'deepseek':['deepseek',
                'deepseek',
                ]
  }
  return models

def get_model_dir(LLM):
  if LLM == 'deepseek':
    return deep_seek_tokenizer_path
  else:  
    return cache_dir

def get_model(
              modelname = None,
              cache_dir = None,
              load_model = True,
              load_tokenizer = True):
  model = None
  tokenizer = None

  if load_model:
    model = AutoModelForCausalLM.from_pretrained(modelname,
                                                device_map="auto",
                                                cache_dir=cache_dir,
                                                )#.to(device)
  
  if load_tokenizer:
    if modelname != 'deepseek':
      tokenizer = AutoTokenizer.from_pretrained(modelname,
                                              padding_side='left',
                                              device_map="auto",
                                              cache_dir=cache_dir,
                                              )      
    else:
      tokenizer = AutoTokenizer.from_pretrained(deep_seek_tokenizer_path,
                                          padding_side='left',
                                          device_map="auto",)

  return (model,tokenizer)

def get_config(modelname,cache_dir):
  config = AutoConfig.from_pretrained(modelname,cache_dir=cache_dir)
  return config

def get_embedding_dim(LLM,config,layer_id):
  emb_dim = config.hidden_size
  if LLM == 'OPT' and layer_id == config.num_hidden_layers: # remember python starts from 0...
    emb_dim =  config.word_embed_proj_dim
  return emb_dim