## Generated using GPT, and then reviewed and modified by the user

import torch
from torch import nn
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import TrainerCallback, TrainerState, TrainerControl
from huggingface_hub import hf_hub_download
import sys
import os
from huggingface_hub import InferenceClient,login
import copy
import random

from pathlib import Path
import argparse

token = ""
login(token=token)
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
wandb.init(mode="disabled")

from utils.model_utils import load_model_and_tokenizer, set_seed
seed = 93
set_seed(seed)

current_path = Path(__file__)
sys.path.append(str(current_path.parent.parent))

cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir) #sys.path.append(cur_dir)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process model argument.')
    parser.add_argument('--model_a', type=str, required=True, help='Specify Model A')
    parser.add_argument('--model_b', type=str, required=True, help='Specify Model B')
    args = parser.parse_args()
    model_a = args.model_a
    model_b = args.model_b



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name_map = {
    "pythia70m": "EleutherAI/pythia-70m-deduped",
    "pythia160m": "EleutherAI/pythia-160m-deduped",
    "pythia410m": "EleutherAI/pythia-410m-deduped",
    "pythia1.4b": "EleutherAI/pythia-1.4b-deduped",
    "pythia2.8b": "EleutherAI/pythia-2.8b-deduped",
    "pythia6.9b": "EleutherAI/pythia-6.9b-deduped",
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
    "opt350m": "facebook/opt-350m",
    "opt1.3b": "facebook/opt-1.3b",
    "opt2.7b": "facebook/opt-2.7b",
    "opt6.7b": "facebook/opt-6.7b",
}

for key in model_name_map.keys():
    if key in model_a:
        model_name1 = model_name_map[key]
    if key in model_b:
        model_name2 = model_name_map[key]


cache_dir = "/om2/user/dbaek/MODELS"


tokenizer1 = AutoTokenizer.from_pretrained(model_name1, cache_dir=cache_dir, trust_remote_code=True)
tokenizer2 = AutoTokenizer.from_pretrained(model_name2, cache_dir=cache_dir, trust_remote_code=True)

if tokenizer1.pad_token_id is None:
    tokenizer1.pad_token_id = tokenizer1.eos_token_id

if tokenizer2.pad_token_id is None:
    tokenizer2.pad_token_id = tokenizer2.eos_token_id

model1 = AutoModelForCausalLM.from_pretrained(model_name1,cache_dir=cache_dir)
model2 = AutoModelForCausalLM.from_pretrained(model_name2,cache_dir=cache_dir)


for param in model1.parameters():
    param.requires_grad = False
for param in model2.parameters():
    param.requires_grad = False

print(model1)
print(model2)
sys.stdout.flush()

class CustomLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(CustomLinear, self).__init__(in_features, out_features, bias, dtype=torch.float32)

    def forward(self, x, **kwargs):
        x = super(CustomLinear, self).forward(x)
        ret = (x,)
        if 'output_attentions' in kwargs:
            ret += (None,)
        if 'use_cache' in kwargs:
            ret += (None,)
        return ret

stitching_layer = CustomLinear(model2.config.hidden_size, model1.config.hidden_size)


stitching_layer.load_state_dict(torch.load(f"../results/{model_b}_embed_to_{model_a}_seed{seed}_randemb0/{model_b}_embed_to_{model_a}_seed{seed}_randemb0_stitched_model.pth"))
stitching_layer.to(device)


if "pythia" in model_a:
    model1.gpt_neox.layers = nn.ModuleList(nn.ModuleList([stitching_layer]) + model1.gpt_neox.layers)
    model1.gpt_neox.embed_in = model2.gpt_neox.embed_in

model1.config.num_hidden_layers = len(model1.gpt_neox.layers) if "pythia" in model_a else len(model1.transformer.h) if "gpt" in model_a else len(model1.model.decoder.layers) if "opt" in model_a else 0
model1 = model1.to(device)
print(model1)


n_shots = 10
seed = 42

dataset_name = "english-french"
test_split = 0.3
root_data_dir = "../dataset_files"
prefixes = {"input":"Q:", "output":"A:", "instructions":""}
separators = {"input":"\n", "output":"\n\n", "instructions":""}


import os, re, json
import torch, numpy as np

import sys
sys.path.append('..')
torch.set_grad_enabled(False)
from src.utils.eval_utils import compute_dataset_baseline, make_valid_path_name
from src.utils.prompt_utils import load_dataset

model_config={
    "n_heads":model1.config.num_attention_heads,
    "n_layers":model1.config.num_hidden_layers,
    "resid_dim":model1.config.hidden_size,
    "name_or_path":model1.config._name_or_path,
    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model1.config.num_hidden_layers)],
    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model1.config.num_hidden_layers)],
    "prepend_bos":False
}

set_seed(seed)


print(f"Computing model baseline results for {n_shots}-shots")
dataset_name_list = [
    "english-french",
    "english-german",
    "english-spanish",
    "country-capital",
    "present-past",
    "singular-plural",
    "word_length",
    "person-sport",
    "product-company",
    "person-instrument",
    "person-occupation"
]

os.makedirs(f"../results/{model_b}_embed_to_{model_a}_seed{seed}_randemb0", exist_ok=True)

for dataset_name in dataset_name_list:
    dataset = load_dataset(dataset_name, root_data_dir=root_data_dir, test_size=test_split, seed=seed)
    baseline_results = compute_dataset_baseline(dataset, model1, model_config, tokenizer1, n_shots=n_shots, seed=seed, prefixes=prefixes, separators=separators)        
    print(baseline_results)

    save_path_root = f"../results/"
    baseline_file_name = make_valid_path_name(f'{save_path_root}/{model_b}_embed_to_{model_a}_seed{seed}_randemb0/{model_b}_embed_to_{model_a}_seed{seed}_randemb0_icl_{dataset_name}.json')
    with open(baseline_file_name, 'w') as results_file:
        json.dump(baseline_results, results_file, indent=2)