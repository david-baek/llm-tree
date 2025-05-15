# %%

from transformer_lens import HookedTransformer
from utils import *

all_model_list = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "facebook/opt-6.7b",
    "Qwen/Qwen2.5-7B-Instruct",
]


import argparse
parser = argparse.ArgumentParser(description="Knowledge Graph Fine-tuning")
parser.add_argument("--model_idx", type=int, required=True, help="Index of the model to use from the list.")
parser.add_argument("--shuffle", type=int, required=True, help="Whether to shuffle the names.")
args = parser.parse_args()

assert 0 <= args.model_idx < len(all_model_list), f"Model index must be between 0 and {len(all_model_list) - 1}."

MODEL_NAME = all_model_list[args.model_idx]
print(f"Using model: {MODEL_NAME}")

if args.shuffle == 1:
    shuffle = True
else:
    shuffle = False

model = HookedTransformer.from_pretrained(MODEL_NAME, dtype=torch.float16, device='cuda:0')
tokenizer = model.tokenizer

# %%
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

# %%
import json
from datasets import Dataset
import pandas as pd
import numpy as np
import random

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

single_token_names = []
for name in df2020['name']:
    tokens = tokenizer.tokenize(name)
    if len(tokens) == 1:
        single_token_names.append(name)
       

instructions = []
inputs = []
outputs = []

all_results = {}
 
for data_seed in [int(x) for x in np.linspace(100, 999, 5, dtype=int)]:
    parents = generate_random_tree(15, seed = data_seed, tree_type='balanced')
    pairs, labels = build_dataset(parents)

    current_names = single_token_names.copy()
    random.seed(data_seed)
    np.random.seed(data_seed)
    random.shuffle(current_names)

    all_prompts = []

    for i in range(len(pairs)):
        instructions.append("Answer a question about the family tree relationship based on the given data. If it's a yes/no question, answer with only one word: 'Yes' or 'No.' If it's a 'who' question, answer with the person's name(s).")
        inputs.append(generate_graph_description(parents, current_names, shuffle=shuffle) + "\n" + f"Question: Is {current_names[pairs[i][0]]} a direct descendant of {current_names[pairs[i][1]]}?")
        outputs.append("Yes" if labels[i] == 1 else "No")
        all_prompts.append(alpaca_prompt.format(instructions[-1], inputs[-1], ""))

    yes_token_id = tokenizer.encode("Yes")[-1]
    no_token_id = tokenizer.encode("No")[-1]

    correct = 0
    incorrect = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in tqdm(range(len(all_prompts))):
        logits, cache = model.run_with_cache(all_prompts[i])
        next_token_id = logits[0, -1, :].argmax(dim=-1).item()
        if next_token_id == yes_token_id and labels[i] == 1:
            tp += 1
        elif next_token_id == yes_token_id and labels[i] == 0:
            fp += 1
        elif next_token_id == no_token_id and labels[i] == 0:
            tn += 1
        elif next_token_id == no_token_id and labels[i] == 1:
            fn += 1
        if i % 100 == 0 and i != 0:
            print(f"{i}: Model: {MODEL_NAME}, Data Seed: {data_seed}, current accuracy: {(tp + tn) / (tp + tn + fp + fn)}, tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}, f1: {2 * tp / (2 * tp + fp + fn)}")
    print(f"Finished Model: {MODEL_NAME}, Data Seed: {data_seed}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn)}")
    print(f"Precision: {tp / (tp + fp)}")
    print(f"Recall: {tp / (tp + fn)}")
    print(f"F1 Score: {2 * tp / (2 * tp + fp + fn)}")
    all_results[data_seed] = {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': tp / (tp + fp),
        'recall': tp / (tp + fn),
        'f1': 2 * tp / (2 * tp + fp + fn),
    }

with open(f'./data/model_eval_results_{MODEL_NAME.split("/")[-1]}_{1 if shuffle else 0}.json', 'w') as f:
    json.dump(all_results, f, indent=4)


