#### format is [model][language][min_token_length]

input_paths = {}

###  models
input_paths["deepseek"] = {}
input_paths["llama"] = {}

### languages
input_paths["deepseek"]["english"] = {}
input_paths["deepseek"]["spanish"] = {}
input_paths["deepseek"]["german"] = {}
input_paths["deepseek"]["hungarian"] = {}

input_paths["llama"]["english"] = {}
input_paths["llama"]["spanish"] = {}
input_paths["llama"]["italian"] = {}

### min_token_length
input_paths["deepseek"]["english"]['40'] = "/home/laios/activations/opus_english_spanish/deepseek/deepseek_english"
input_paths["deepseek"]["spanish"]['40'] = "/home/laios/activations/opus_english_spanish/deepseek/deepseek_spanish"
input_paths["deepseek"]["english"]['100'] = "/home/laios/activations/opus_english_spanish/deepseek/100_200_tokens/deepseek_source"
input_paths["deepseek"]["spanish"]['100'] = "/home/laios/activations/opus_english_spanish/deepseek/100_200_tokens/deepseek_target"

input_paths["llama"]["english"]['40'] = "/home/laios/activations/opus_english_spanish/llama-8b/llama_english"
input_paths["llama"]["spanish"]['40'] = "/home/laios/activations/opus_english_spanish/llama-8b/llama_spanish"
input_paths["llama"]["english"]['100'] = "/home/laios/activations/opus_english_spanish/llama-8b/tokens_100_200/llama_english_100"
input_paths["llama"]["spanish"]['100'] = "/home/laios/activations/opus_english_spanish/llama-8b/tokens_100_200/llama_spanish_100"

input_paths["deepseek"]["german"]['40'] = f"/home/laios/activations/opus_german_hungarian/deepseek_german"
input_paths["deepseek"]["hungarian"]['40'] = f"/home/laios/activations/opus_german_hungarian/deepseek_hungarian"
