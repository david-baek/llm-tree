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

from transformer_lens import HookedTransformer

import sys
import json

from utils import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# %%

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

alpaca_prompt_input_only = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}"""

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
parser.add_argument("--exp_type", type=str, required=True, choices=["layer", "model"])
args = parser.parse_args()

assert 0 <= args.model_idx < len(all_model_list), f"Model index must be between 0 and {len(all_model_list) - 1}."
model_name = all_model_list[args.model_idx]

print(f"Using model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# hf_model = AutoModelForCausalLM.from_pretrained(
#     model_name,#f"./fine-tuned/{model_name.split('/')[-1]}",
# #    load_in_8bit=True,
#     device_map="auto",
#     local_files_only=True,
# )
model = HookedTransformer.from_pretrained_no_processing(model_name, dtype=torch.float16, n_devices=1, device="cuda:0")

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
        
all_layer_list = list(range(0, model.cfg.n_layers * 2 // 3, 2))
all_ablation_list = ["pca2", "multi0", "full"]

if args.exp_type == "model":
    all_layer_list = [model.cfg.n_layers // 3]
# %%

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
    for i in range(len(pairs)):
        instructions.append("Answer a question about the family tree relationship based on the given data. If it's a yes/no question, answer with only one word: 'Yes' or 'No.' If it's a 'who' question, answer with the person's name(s).")
        inputs.append(generate_graph_description(parents, current_names) + "\n" + f"Question: Is {current_names[pairs[i][0]]} a direct descendant of {current_names[pairs[i][1]]}?")
        outputs.append("Yes" if labels[i] == 1 else "No")
    
    with open(f'./data/llm_real_reps_{data_seed}_{model_name.split("/")[-1]}.json', 'r') as f:
        all_reps = json.load(f)

    '''
    Generate 20 random pairs of clean and corrupted prompts
    '''
    n_samples = 20
    clean_prompts = []
    corrupted_prompts = []
    clean_labels = []
    
    sample_cnt = 0
    check_pairs = {}
    samples_info = []
    yes_token_id = tokenizer.encode("Yes")[-1]
    no_token_id = tokenizer.encode("No")[-1]
    print(tokenizer.encode("Yes"), yes_token_id)

    one_label_indices = [i for i, lbl in enumerate(labels) if lbl == 1]

    random.shuffle(one_label_indices)
    print(one_label_indices)

    for i in range(len(one_label_indices)):
        pair_idx = one_label_indices[i]
        x, y = pairs[pair_idx]
        label = labels[pair_idx]

        available_names = list(range(len(parents)))
        available_names.remove(x)
        if y in available_names:
            available_names.remove(y)
        if label == 1:  # If x is a descendant of y, remove y and all its descendants
            descendants = compute_descendants(parents)[y]
            for d in descendants:
                if d in available_names:
                    available_names.remove(d)
        print(i, x, y, available_names)
        if len(available_names) == 0:
            continue

        random.shuffle(available_names)
        for corrupted_name in available_names[:1]:
            corrupted_x = corrupted_name
            if (corrupted_x, x, y) in check_pairs or (x, corrupted_x, y) in check_pairs:
                continue

            clean_input = generate_graph_description(parents, current_names) + "\n" + f"Question: Is {current_names[x]} a direct descendant of {current_names[y]}?"
            
            # Create clean prompt
            clean_prompt = alpaca_prompt.format(
                instructions[pair_idx],
                clean_input,
                ""
            )
            
            corrupted_input = generate_graph_description(parents, current_names) + "\n" + \
                f"Question: Is {current_names[corrupted_x]} a direct descendant of {current_names[y]}?"
            
            corrupted_prompt = alpaca_prompt.format(
                instructions[pair_idx],
                corrupted_input,
                ""
            )

            # with torch.no_grad():
            #     clean_output = model(clean_prompt)[:, -1, :]
            #     clean_yes_no = clean_output[0, [yes_token_id, no_token_id]]
            #     clean_logit_diff = (clean_yes_no[0].item() - clean_yes_no[1].item()) if label == 1 else (clean_yes_no[1].item() - clean_yes_no[0].item())

            #     corrupted_output = model(corrupted_prompt)[:, -1, :]
            #     corrupted_yes_no = corrupted_output[0, [yes_token_id, no_token_id]]
            #     corrupted_logit_diff = (corrupted_yes_no[1].item() - corrupted_yes_no[0].item()) if label == 1 else (corrupted_yes_no[0].item() - corrupted_yes_no[1].item())

            # print(clean_logit_diff, corrupted_logit_diff, corrupted_x, x, y, label)
            # sys.stdout.flush()
                
            # if clean_logit_diff > 0 and corrupted_logit_diff > 0:
            check_pairs[(corrupted_x, x, y)] = 1
            samples_info.append((corrupted_x, x, y))
            print(corrupted_x, x, y, label)
            if sample_cnt % 2 == 0:
                clean_labels.append(label)
                clean_prompts.append(clean_prompt)
                corrupted_prompts.append(corrupted_prompt)
            else:
                clean_labels.append(1 - label)
                clean_prompts.append(corrupted_prompt)
                corrupted_prompts.append(clean_prompt)
            sample_cnt += 1

    n_samples = sample_cnt

    print(f"Processed {sample_cnt} samples")


    '''
    Load representations and train a probe
    '''
        
    for layer_num in all_layer_list:
        hook_point = f"blocks.{layer_num}.hook_resid_pre" if layer_num == 0 else f"blocks.{layer_num - 1}.hook_resid_post"
        for ablation_type in all_ablation_list:
            emb_weight = all_reps[str(layer_num)]
            emb_weight = np.array(emb_weight)
            
            d_pca = 10
            pca = PCA(n_components=d_pca)
            emb_weight_pca = pca.fit_transform(emb_weight)

            
            if ablation_type == "order":
                linear_probe_results = train_ordering_probe(
                    E=emb_weight_pca,
                    emb_dim=2,
                    hidden_dim=emb_weight_pca.shape[1],
                    batch_size=16,
                    train_seed=42,
                    lr=1e-3,
                    max_epochs=300,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                )
            elif ablation_type.startswith("multi"):
                linear_probe_results = train_AAT_for_epsilons(
                    emb_weight_pca,               # numpy array, shape (n_nodes, emb_dim)x
                    pairs,           # numpy array, shape (N_pairs, 2)
                    labels,          # numpy array, shape (N_pairs,)
                    [0 if ablation_type == "multi0" else 0.3],        # list of float
                    emb_dim=2,
                    hidden_dim=emb_weight_pca.shape[1],
                    batch_size=32,
                    train_seed=10,
                    lr=1e-3,
                    max_epochs=3000
                )
            else:      
                linear_probe_results = train_linear_probe_for_epsilons(
                    E=emb_weight_pca,
                    pairs=pairs,
                    labels=labels,
                    epsilons=[0.01],
                    emb_dim=2,
                    hidden_dim=emb_weight_pca.shape[1],
                    batch_size=16,
                    train_seed=42,
                    lr=1e-3,
                    max_epochs=300,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                )


            prompt_until_first_person = alpaca_prompt_input_only.format(
                instructions[0],
                generate_graph_description(parents, current_names) + "\n" + f"Question: Is {current_names[0]}"
            )
            def make_patch_fn(corrupted_cache):
                last_token_idx = len(model.to_tokens(prompt_until_first_person)[0]) - 1
                print("last_token_idx", last_token_idx)
                # Precompute the PCA basis and probe vector once
                if not ablation_type.startswith("multi"):
                    w = linear_probe_results['linear_probe'].coef_.reshape(-1)  # (d,)
                    w_norm_sq = float(w.dot(w))

                    
                    pca_projection_matrix = torch.tensor(pca.components_.T, dtype=torch.float32)

                    def patch_fn(resid, hook):
                        """
                        Patches the residual stream at a specific token position by replacing its PCA subspace
                        projection with that from the corrupted activation.

                        Args:
                            resid: Tensor of shape (B, S, D) - the clean residual stream
                            hook: Hook object (not used directly here beyond context)

                        Returns:
                            resid: Modified residual stream tensor
                        """
                        clean_np   = resid.detach().cpu().numpy()  
                        corrupt_np = corrupted_cache[hook_point].detach().cpu().numpy()

                        B, S, D = clean_np.shape
                        # 2) Flatten to (B*S, D) for PCA
                        clean_flat   = clean_np.reshape(B*S, D)
                        corrupt_flat = corrupt_np.reshape(B*S, D)

                        # 3) PCA transform
                        clean_pca   = pca.transform(clean_flat)    # (B*S, d_pca)
                        corrupt_pca = pca.transform(corrupt_flat)  # (B*S, d_pca)

                        avg_pca = np.tile(np.mean(emb_weight_pca, axis=0), (B*S, 1)) 
                        # 4) Compute residual in original space = original − recon
                        recon_clean = pca.inverse_transform(clean_pca)  # (B*S, D)
                        residual    = clean_flat - recon_clean          # (B*S, D)

                        # 5) Compute projection coefficients
                        #    alpha_i = w·clean_pca_i / ‖w‖²,  beta_i = w·corrupt_pca_i / ‖w‖²
                        alphas = (clean_pca @ w)   / w_norm_sq          # (B*S,)
                        betas  = (corrupt_pca @ w) / w_norm_sq          # (B*S,)

                        avg_alpha = (avg_pca @ w) / w_norm_sq

                        # 6) Build modified PCA vectors
                        proj = alphas[:, None] * w                       # (B*S, d_pca)
                        add  = betas[:, None]  * w                       # (B*S, d_pca)
                        if ablation_type == "linear" or ablation_type == "order":
                            new_pca = clean_pca - proj + add                 # (B*S, d_pca)
                        elif ablation_type.startswith("pca"):
                            if ablation_type == "pca_ablate":
                                new_pca = clean_pca.copy()
                                new_pca[:, :2] = 0
                            else:
                                replace_pca = int(ablation_type.split("pca")[-1])
                                new_pca = clean_pca.copy()
                                new_pca[:, :replace_pca] = corrupt_pca[:, :replace_pca]
                        elif ablation_type == "none":
                            new_pca = clean_pca
                        elif ablation_type == "full":
                            new_pca = corrupt_pca
                        elif ablation_type == "average":
                            new_pca = np.tile(np.mean(emb_weight_pca, axis=0), (B*S, 1))
                        
                        else:
                            raise ValueError(f"Invalid ablation type: {ablation_type}")


                        # 7) Inverse PCA → (B*S, D) and add residual
                        recon_new  = pca.inverse_transform(new_pca)      # (B*S, D)
                        patched_flat = recon_new  + residual             # (B*S, D)

                        # 8) Reshape back to (B, S, D)
                        patched_np = patched_flat.reshape(B, S, D)
                        if ablation_type == "full":
                            patched_np = corrupt_np

                        # 9) Write only token position t into resid
                        t = last_token_idx
                        with torch.no_grad():
                            patch_tensor = torch.from_numpy(patched_np[:, t, :]).to(resid.device)
                            resid[:, t, :] = patch_tensor

                        return resid
                    return patch_fn
                elif ablation_type.startswith("multi"):
                    aat_projection_matrix = linear_probe_results['probe']['linear.weight']  # (k, d_pca), in PCA space

                    if ablation_type == "multi-random":
                        aat_projection_matrix = torch.randn(aat_projection_matrix.shape[0], aat_projection_matrix.shape[1])

                    # Convert aat_projection_matrix to NumPy for PCA operations
                    V = aat_projection_matrix.detach().cpu().numpy()  # (k, d_pca), rows span the subspace
                    Q, _ = np.linalg.qr(V.T)  # Q: (d_pca, r), orthonormal columns, r = min(k, d_pca)

                    def patch_aat_fn(resid, hook):
                        """
                        Patches the residual stream at a specific token position by replacing its projection
                        onto the subspace defined by aat_projection_matrix with that from the corrupted activation.

                        Args:
                            resid: Tensor of shape (B, S, D) - the clean residual stream
                            hook: Hook object (not used directly here beyond context)

                        Returns:
                            resid: Modified residual stream tensor
                        """
                        clean_np   = resid.detach().cpu().numpy()  
                        corrupt_np = corrupted_cache[hook_point].detach().cpu().numpy()

                        B, S, D = clean_np.shape
                        # Flatten to (B*S, D) for PCA
                        clean_flat   = clean_np.reshape(B*S, D)
                        corrupt_flat = corrupt_np.reshape(B*S, D)

                        avg_pca = np.tile(np.mean(emb_weight_pca, axis=0), (B*S, 1))

                        # PCA transform
                        clean_pca   = pca.transform(clean_flat)    # (B*S, d_pca)
                        corrupt_pca = pca.transform(corrupt_flat)  # (B*S, d_pca)

                        # Compute residual in original space = original − recon
                        recon_clean = pca.inverse_transform(clean_pca)  # (B*S, D)
                        residual    = clean_flat - recon_clean          # (B*S, D)

                        # Compute projections onto the subspace
                        proj_clean   = clean_pca @ Q @ Q.T    # (B*S, d_pca)
                        proj_corrupt = corrupt_pca @ Q @ Q.T  # (B*S, d_pca)
                        proj_avg = avg_pca @ Q @ Q.T

                        # Build modified PCA vectors by swapping the subspace projection
                        new_pca = clean_pca - proj_clean + proj_corrupt  # (B*S, d_pca)
                        if ablation_type == "multi_ablate":
                            new_pca = clean_pca - proj_clean

                        # Inverse PCA → (B*S, D) and add residual
                        recon_new    = pca.inverse_transform(new_pca)    # (B*S, D)
                        patched_flat = recon_new + residual              # (B*S, D)

                        # Reshape back to (B, S, D)
                        patched_np = patched_flat.reshape(B, S, D)

                        # Write only token position t into resid
                        t = last_token_idx
                        with torch.no_grad():
                            patch_tensor = torch.from_numpy(patched_np[:, t, :]).to(resid.device)
                            resid[:, t, :] = patch_tensor

                        return resid
                    return patch_aat_fn
                else:
                    raise ValueError(f"Invalid ablation type: {ablation_type}")

                
            
            
            # Process prompts in batches
            batch_size = 1
            clean_logit_diffs = []
            corrupted_logit_diffs = []
            patched_logit_diffs = []

        
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                
                # Prepare batch inputs
                clean_batch = clean_prompts[batch_start:batch_end]
                corrupted_batch = corrupted_prompts[batch_start:batch_end]
                
                batch_labels = clean_labels[batch_start:batch_end]
                
                with torch.no_grad():
                    # Get clean logits
                    clean_output = model(clean_batch)[:, -1, :]
                    clean_yes_no = clean_output[:, [yes_token_id, no_token_id]]
                    
                    clean_logit_diff = [
                        (yes_no[0].item() - yes_no[1].item()) if label == 1 else (yes_no[1].item() - yes_no[0].item())
                        for yes_no, label in zip(clean_yes_no, batch_labels)
                    ]
                    clean_logit_diffs.extend(clean_logit_diff)

                    _, clean_cache = model.run_with_cache(clean_batch, names_filter=lambda x: x.endswith(".hook_resid_pre") or x.endswith(".hook_resid_post"))

                    # Get corrupted logits
                    corrupted_output = model(corrupted_batch)[:, -1, :]
                    corrupted_yes_no = corrupted_output[:, [yes_token_id, no_token_id]]
                    corrupted_logit_diff = [
                        (yes_no[0].item() - yes_no[1].item()) if label == 1 else (yes_no[1].item() - yes_no[0].item())
                        for yes_no, label in zip(corrupted_yes_no, batch_labels)
                    ]
                    corrupted_logit_diffs.extend(corrupted_logit_diff)

                    # Get patched logits
                    fwd_hooks = [(hook_point, make_patch_fn(clean_cache))]
                    patched_output = model.run_with_hooks(corrupted_batch, fwd_hooks=fwd_hooks)[:, -1, :]
                    patched_yes_no = patched_output[:, [yes_token_id, no_token_id]]
                    
                    print(patched_yes_no, corrupted_yes_no, clean_yes_no)
                    patched_logit_diff = [
                        (yes_no[0].item() - yes_no[1].item()) if label == 1 else (yes_no[1].item() - yes_no[0].item())
                        for yes_no, label in zip(patched_yes_no, batch_labels)
                    ]
                    
                    patched_logit_diffs.extend(patched_logit_diff)

                del clean_batch, corrupted_batch, clean_output, corrupted_output, patched_output, clean_cache
                torch.cuda.empty_cache()
                if batch_start % 10 == 0:
                    print(f"Processed {batch_start} samples")

            logit_diffs = np.array(patched_logit_diffs) - np.array(corrupted_logit_diffs)
            max_logit_diff = np.array(clean_logit_diffs) - np.array(corrupted_logit_diffs)
            
            norm_logit_diffs = logit_diffs / max_logit_diff
            
            with open(f'./data/ablation_{data_seed}_{model_name.split("/")[-1]}_{layer_num}_{ablation_type}.json', 'w') as f:
                json.dump({
                    'clean_logit_diffs': clean_logit_diffs,
                    'corrupted_logit_diffs': corrupted_logit_diffs,
                    'patched_logit_diffs': patched_logit_diffs,
                }, f, indent=4)

            print(np.mean(patched_logit_diffs), np.std(patched_logit_diffs))
            print(np.mean(logit_diffs), np.std(logit_diffs))
            # Create a hook function to intervene with ablated representations

            torch.cuda.empty_cache()
            print(f"DONE: {data_seed} {layer_num} {ablation_type}")
