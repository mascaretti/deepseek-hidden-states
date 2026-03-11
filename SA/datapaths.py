#### format is [model][language][min_token_length]

input_paths = {}

###  models
input_paths["deepseek"] = {}
input_paths["llama"] = {}
input_paths["bert"] = {}

### source languages
input_paths["deepseek"]["english"] = {"spanish":   {"source":{},"target":{}}, 
                                      "italian":   {"source":{},"target":{}},
                                      "hungarian": {"source":{},"target":{}},
                                      "dutch":     {"source":{},"target":{}},
                                      "french":    {"source":{},"target":{}},
                                      }
input_paths["deepseek"]["spanish"] = {"italian": {"source":{},"target":{}},}
input_paths["deepseek"]["german"] = {"hungarian": {"source":{},"target":{}},
                                     "english":   {"source":{},"target":{}},}

input_paths["llama"]["english"] = {"spanish": {"source":{},"target":{}}, 
                                   "italian": {"source":{},"target":{}},
                                   "french": {"source":{},"target":{}},
                                   "dutch": {"source":{},"target":{}},
                                   "hungarian": {"source":{},"target":{}},
                                   }
input_paths["llama"]["spanish"] = {"italian": {"source":{},"target":{}},}
input_paths["llama"]["german"] = {"english": {"source":{},"target":{}}}

input_paths["bert"]["english"] = {"spanish": {"source":{},"target":{}},}

### min_token_length
input_paths["deepseek"]["english"]["spanish"]['source']['40'] = "/home/laios/activations/opus_english_spanish/deepseek/deepseek_english"
input_paths["deepseek"]["english"]["spanish"]['target']['40'] = "/home/laios/activations/opus_english_spanish/deepseek/deepseek_spanish"
input_paths["deepseek"]["english"]["italian"]['source']['40'] = "/home/laios/activations/opus_english_italian/deepseek_source"
input_paths["deepseek"]["english"]["italian"]['target']['40'] = "/home/laios/activations/opus_english_italian/deepseek_target"
input_paths["deepseek"]["german"]["english"]['source']['40'] = "/home/laios/activations/opus_german_english/deepseek_german_english_source"
input_paths["deepseek"]["german"]["english"]['target']['40'] = "/home/laios/activations/opus_german_english/deepseek_german_english_target"
input_paths["deepseek"]["english"]["hungarian"]['source']['40'] = "/home/laios/activations/opus_english_hungarian/deepseek_en_hu_source"
input_paths["deepseek"]["english"]["hungarian"]['target']['40'] = "/home/laios/activations/opus_english_hungarian/deepseek_en_hu_target"
input_paths["deepseek"]["english"]["dutch"]['source']['40'] = "/home/laios/activations/opus_english_dutch/deepseek_en_nl_source"
input_paths["deepseek"]["english"]["dutch"]['target']['40'] = "/home/laios/activations/opus_english_dutch/deepseek_en_nl_target"
input_paths["deepseek"]["english"]["french"]['source']['40'] = "/home/laios/activations/ric/results_en_fr_source"
input_paths["deepseek"]["english"]["french"]['target']['40'] = "/home/laios/activations/ric/results_en_fr_target"


input_paths["deepseek"]["english"]["spanish"]["source"]['100'] = "/home/laios/activations/opus_english_spanish/deepseek/100_200_tokens/deepseek_source"
input_paths["deepseek"]["english"]["spanish"]["target"]['100'] = "/home/laios/activations/opus_english_spanish/deepseek/100_200_tokens/deepseek_target"

input_paths["llama"]["german"]["english"]["source"]['40'] = "/home/laios/activations/opus_german_english/llama_german_english_source"
input_paths["llama"]["german"]["english"]["target"]['40'] = "/home/laios/activations/opus_german_english/llama_german_english_target"
input_paths["llama"]["english"]["spanish"]["source"]['40'] = "/home/laios/activations/opus_english_spanish/llama_en_es_source"
input_paths["llama"]["english"]["spanish"]["target"]['40'] = "/home/laios/activations/opus_english_spanish/llama_en_es_target"
input_paths["llama"]["english"]["spanish"]["source"]['100'] = "/home/laios/activations/opus_english_spanish/llama-8b/tokens_100_200/llama_english_100"
input_paths["llama"]["english"]["spanish"]["target"]['100'] = "/home/laios/activations/opus_english_spanish/llama-8b/tokens_100_200/llama_spanish_100"
input_paths["llama"]["english"]["italian"]["source"]['40'] = "/home/laios/activations/opus_english_italian/llama_en_it_source"
input_paths["llama"]["english"]["italian"]["target"]['40'] = "/home/laios/activations/opus_english_italian/llama_en_it_target"
input_paths["llama"]["english"]["french"]["source"]['40'] = "/home/laios/activations/opus_english_french/llama_en_fr_source"
input_paths["llama"]["english"]["french"]["target"]['40'] = "/home/laios/activations/opus_english_french/llama_en_fr_target"
input_paths["llama"]["english"]["dutch"]["source"]['40'] = "/home/laios/activations/opus_english_dutch/llama_en_nl_source"
input_paths["llama"]["english"]["dutch"]["target"]['40'] = "/home/laios/activations/opus_english_dutch/llama_en_nl_target"
input_paths["llama"]["english"]["hungarian"]["source"]['40'] = "/home/laios/activations/opus_english_hungarian/llama_en_hu_source"
input_paths["llama"]["english"]["hungarian"]["target"]['40'] = "/home/laios/activations/opus_english_hungarian/llama_en_hu_target"


input_paths["deepseek"]["spanish"]["italian"]["source"]['40'] = "/home/laios/activations/opus_spanish_italian/deepseek_source"
input_paths["deepseek"]["spanish"]["italian"]["target"]['40'] = "/home/laios/activations/opus_spanish_italian/deepseek_target"
input_paths["deepseek"]["german"]["hungarian"]["source"]['40'] = f"/home/laios/activations/opus_german_hungarian/deepseek_german"
input_paths["deepseek"]["german"]["hungarian"]["target"]['40'] = f"/home/laios/activations/opus_german_hungarian/deepseek_hungarian"

input_paths["bert"]["english"]["spanish"]["source"]['40'] = "/home/laios/BERT_english_activations"