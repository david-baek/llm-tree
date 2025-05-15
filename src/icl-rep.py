# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F

import json

from utils import *
from transformer_lens import HookedTransformer

import pybabynames as bn
# Load 2020 data:
df2020 = bn.babynames[bn.babynames.year == 2000]
df2020 = pd.concat([
    df2020[df2020.sex == 'F'].sort_values('n', ascending=False).head(200),
    df2020[df2020.sex == 'M'].sort_values('n', ascending=False).head(200)
])
# Shuffle the dataframe deterministically
df2020 = df2020.sample(frac=1, random_state=42).reset_index(drop=True)
# Top 10 baby names in 2020:
print(df2020.head())
print(df2020.sort_values('n', ascending=False).head(100)['name'])
# %%

all_model_list = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/gpt-j-6B",
    "microsoft/Phi-3-mini-4k-instruct",
    "01-ai/Yi-6B-Chat"
]

import argparse
parser = argparse.ArgumentParser(description="Knowledge Graph Fine-tuning")
parser.add_argument("--model_idx", type=int, required=True, help="Index of the model to use from the list.")
args = parser.parse_args()

assert 0 <= args.model_idx < len(all_model_list), f"Model index must be between 0 and {len(all_model_list) - 1}."

model_name= all_model_list[args.model_idx]
print(f"Using model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# hf_model = AutoModelForCausalLM.from_pretrained(
#     model_name,#f"./fine-tuned/{model_name.split('/')[-1]}",
# #    load_in_8bit=True,
#     device_map="auto",
#     local_files_only=True,
# )
model = HookedTransformer.from_pretrained_no_processing(model_name, dtype=torch.float16, n_devices=1, device="cuda:0")
print(model.cfg)

single_token_names = []
for name in df2020['name']:
    tokens = tokenizer.tokenize(name)
    if len(tokens) == 1:
        single_token_names.append(name)
        
# %%
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}"""

for data_seed in [int(x) for x in np.linspace(100, 999, 5, dtype=int)]:
    parents = generate_random_tree(15, seed = int(data_seed), tree_type='balanced')
    pairs, labels = build_dataset(parents)

    current_names = single_token_names.copy()
    random.seed(data_seed)
    np.random.seed(data_seed)
    random.shuffle(current_names)
    
    
    instructions = []
    inputs = []
    outputs = []
    prompt_list = []
    
    batch_size = 4  # Define the batch size
    
    all_reps = {}
    decoded_outputs = []  # Initialize the list to store all decoded outputs
    for batch_start in range(0, len(parents), batch_size):
        batch_end = min(batch_start + batch_size, len(parents))
        batch_prompt_list = []

        for i in range(batch_start, batch_end):
            cur_inst = "Answer a question about the family tree relationship based on the given data. If it's a yes/no question, answer with only one word: 'Yes' or 'No.' If it's a 'who' question, answer with the person's name(s)."
            cur_input = generate_graph_description(parents, current_names) + "\n" + f"Question: Is {current_names[i]}"
            batch_prompt_list.append(alpaca_prompt.format(cur_inst, cur_input))
            
        input_ids = tokenizer(batch_prompt_list, return_tensors="pt")['input_ids'].to("cuda")

        with torch.no_grad():
            _, cache = model.run_with_cache(batch_prompt_list, names_filter=lambda x: x.endswith(".hook_resid_pre") or x.endswith(".hook_resid_post"))

            for layer_num in range(model.cfg.n_layers):
                if layer_num == 0:
                    rep = cache[f'blocks.{layer_num}.hook_resid_pre']
                else:
                    rep = cache[f'blocks.{layer_num - 1}.hook_resid_post']
                    
                rep = rep[:, -1, :]
                if layer_num not in all_reps:
                    all_reps[layer_num] = []
                all_reps[layer_num].extend(rep.cpu().numpy().tolist())
        # with torch.no_grad():
        #     outputs = hf_model(input_ids, output_hidden_states=True)
        #     for layer_num in range(len(outputs.hidden_states)):
        #         if layer_num not in all_reps:
        #             all_reps[layer_num] = []
        #         if layer_num == hf_model.config.num_hidden_layers//2:
        #             print(outputs.hidden_states[layer_num].shape)
        #         hidden_states = outputs.hidden_states[layer_num][:, -1, :]
        #         all_reps[layer_num].extend(hidden_states.cpu().numpy().tolist())

        # print(np.array(all_reps[0]).shape)
        # print(rep_hf.shape)

        # print(torch.norm(rep_hf.cpu() - torch.tensor(all_reps[hf_model.config.num_hidden_layers//2][-batch_size:]).cpu()))
        torch.cuda.empty_cache()
    
    with open(f'./data/llm_real_reps_{data_seed}_{model_name.split("/")[-1]}.json', 'w') as f:
        json.dump(all_reps, f, indent=4)
# %%
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# for data_seed in [42, 100, 195, 283]:
#     all_linear = []
#     all_aat = []
#     for layer_num in range(model.config.num_hidden_layers):
#         parents = generate_random_tree(30, seed = data_seed)
#         pairs, labels = build_dataset(parents)
        
#         with open(f'./data/llm_reps_{data_seed}_{model_name.split("/")[-1]}.json', 'r') as f:
#             all_reps = json.load(f)
            
#         emb_weight = all_reps[str(layer_num)]
#         emb_weight = np.array(emb_weight)
        
#         pca = PCA(n_components=20)
#         emb_weight = pca.fit_transform(emb_weight)
        
#         linear_probe_results = train_linear_probe_for_epsilons(
#             E=emb_weight,
#             pairs=pairs,
#             labels=labels,
#             epsilons=[0],
#             emb_dim=2,
#             hidden_dim=emb_weight.shape[1],
#             batch_size=16,
#             train_seed=42,
#             lr=1e-3,
#             max_epochs=300,
#             device='cuda' if torch.cuda.is_available() else 'cpu',
#         )
#         linear_probe_f1 = linear_probe_results[0]

#         # Compute AAT probe F1 score
#         aat_probe_results = train_AAT_for_epsilons(
#             E=emb_weight,
#             pairs=pairs,
#             labels=labels,
#             epsilons=[0],
#             emb_dim=2,
#             hidden_dim=emb_weight.shape[1],
#             batch_size=16,
#             train_seed=42,
#             lr=1e-3,
#             max_epochs=500,
#             device='cuda' if torch.cuda.is_available() else 'cpu',
#         )
#         print(f"seed {data_seed} layer {layer_num} linear probe f1: {linear_probe_f1} aat probe f1: {aat_probe_results[0]}")
#         all_linear.append(linear_probe_f1)
#         all_aat.append(aat_probe_results[0])
#     plt.figure()
#     plt.plot(all_linear, label='Linear Probe F1')
#     plt.plot(all_aat, label='AAT Probe F1')
#     plt.show()
# %%
# import json
# import numpy as np
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# for data_seed in [int(x) for x in np.linspace(100, 999, 5, dtype=int)]:
#     with open(f'./data/llm_reps_{data_seed}_{MODEL_NAME.split("/")[-1]}.json', 'r') as f:
#         data = json.load(f)

#     true_correct = sum(1 for pred, label in zip(data['predictions'], data['true_labels']) if pred == 1 and label == 1)
#     false_correct = sum(1 for pred, label in zip(data['predictions'], data['true_labels']) if pred == 0 and label == 0)

#     all_true = sum(1 for label in data['true_labels'] if label == 1)
#     all_false = sum(1 for label in data['true_labels'] if label == 0)
    
#     print(f"seed {data_seed} true correct: {true_correct}/{all_true} false correct: {false_correct}/{all_false}")

#     # Find two correct pairs with different answers but the same second person
#     correct_pairs = [(pred, label, idx) for idx, (pred, label) in enumerate(zip(data['predictions'], data['true_labels'])) if pred == label]
#     found_pair = False
#     for i in range(len(correct_pairs)):
#         for j in range(i + 1, len(correct_pairs)):
#             if correct_pairs[i][0] != correct_pairs[j][0] and data['true_labels'][correct_pairs[i][2]] == data['true_labels'][correct_pairs[j][2]]:
#                 print(f"Found correct pairs with different answers but same second person: {correct_pairs[i]}, {correct_pairs[j]}")
#                 found_pair = True
#                 break
#         if found_pair:
#             break
        
#     print(data)
# %%
