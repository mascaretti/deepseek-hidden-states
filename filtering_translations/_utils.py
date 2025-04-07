import os
import sys
from _paths import *

from datasets import load_dataset

def get_acronym(language):
  if language == 'Spanish':
    acronym = 'es'
  elif language == 'German':
     acronym = 'de'
  elif language == 'Italian':
    acronym = 'it'
  elif language == 'English':
    acronym = 'en'
  elif language == 'Russian':
    acronym = 'ru'
  elif language == 'Hungarian':
    acronym = 'hu'
  else:
    print(f'ERROR IN LANGUAGE NAME')
  return acronym

def get_instructions(dataset,
                    target_language,
                    source_language_acronym,
                    ):
  X = {'instructions' : 
      'The sentence "' + dataset['translation'][source_language_acronym] + \
      f'" is translated to {target_language} as: "'
      }
  return X


def get_instructed_translations(dataset,
                    target_language,
                    source_language_acronym,
                    target_language_acronym,):
  X = {'instructed_translations' : 
      'The sentence "' + dataset['translation'][source_language_acronym] + \
      f'" is translated to {target_language} as: "' + \
      dataset["translation"][target_language_acronym] + '"'
      }
  return X

def get_target(dataset,
              target_language_acronym,):
  X = {'target':
       dataset["translation"][target_language_acronym]
       }
  return X

def get_source(dataset,
              source_language_acronym,):
  X = {'source':
       dataset["translation"][source_language_acronym]
       }
  return X

def filter_by_min_token_length(dataset,min_token_length,x):
    "x is either source or target"
    return sum(dataset[f'{x}_attention_mask']) >= min_token_length

def filter_by_max_token_length(dataset,max_token_length,x):
    "x is either source or target"
    return sum(dataset[f'{x}_attention_mask']) <= max_token_length