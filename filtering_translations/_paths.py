import sys,os

# cache_dir = os.getenv("SCRATCH") + f'/hf_cache'
cache_dir = f'/home/laios/hf_cache/'

deep_seek_tokenizer_path = "/home/laios/deepseek-hidden-states/filtering_translations/deepseek_tokenizer"

# path0 = os.getenv("SCRATCH") + "/Data/Text/"
path0 = "/home/laios/Data/Text/"

def makefolder(base='./',
               create_folder=False,
               precision=5,
               **kwargs,
               ):
  folder = base
  for key, value in kwargs.items():
    if value != None:
      if isinstance(value,float) == True:
        folder += key + f'_{value:.{precision}f}/'
      else:
        folder += key + f'_{value}/'
  if create_folder:
    os.makedirs(folder,exist_ok=True)
  return folder



