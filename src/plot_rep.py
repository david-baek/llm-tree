import pybabynames as bn
import pandas as pd
from brokenaxes import brokenaxes

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

model_name = "meta-llama/Llama-3.1-8B-Instruct"
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from utils import *
model = HookedTransformer.from_pretrained_no_processing(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


single_token_names = []
for name in df2020['name']:
    tokens = tokenizer.tokenize(name)
    if len(tokens) == 1:
        single_token_names.append(name)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}"""
def compute_depth(parents):
    depth = np.zeros(len(parents))
    for i in range(len(parents)):
        if parents[i] != -1:
            depth[i] = depth[parents[i]] + 1
    return depth

## Create 3x5 subplot grid
with plt.rc_context({
    'font.family': 'sans-serif',
    'font.size': 5,
    'axes.labelsize': 7.25,
    'axes.titlesize': 7.25,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5.5,
}):
    fig, axes = plt.subplots(2, 5, figsize=(5.5, 2.3))
    layer_num = model.cfg.n_layers // 3
    seed_list = [int(x) for x in np.linspace(100, 999, 5, dtype=int)]
    for data_seed_idx in range(len(seed_list)):
        data_seed = seed_list[data_seed_idx]
        parents = generate_random_tree(15, seed=data_seed, tree_type='balanced')
        pairs, labels = build_dataset(parents)

        current_names = single_token_names.copy()
        random.seed(data_seed)
        np.random.seed(data_seed)
        random.shuffle(current_names)

        prompts = []
        batch_size = 4
        all_reps = []
        for batch_start in range(0, len(parents), batch_size):
            batch_end = min(batch_start + batch_size, len(parents))
            batch_prompt_list = []

            for i in range(batch_start, batch_end):
                cur_inst = "You are a helpful assistant. You will be asked to answer a question about an ancestor-descendant relationship in the given family tree."
                cur_input = generate_graph_description(parents, current_names) + "\n" + f"Is {current_names[i]}"
                if i == 0:
                    print(cur_input)
                batch_prompt_list.append(alpaca_prompt.format(cur_inst, cur_input))
                
            input_ids = tokenizer(batch_prompt_list, return_tensors="pt")['input_ids'].to("cuda")

            with torch.no_grad():
                _, cache = model.run_with_cache(batch_prompt_list, names_filter=lambda x: x == f"blocks.{layer_num - 1}.hook_resid_post")
                rep = cache[f'blocks.{layer_num -1}.hook_resid_post']
                rep = rep[:, -1, :].cpu().numpy()
                all_reps.append(rep)

        all_reps = np.concatenate(all_reps, axis=0)
        print(all_reps.shape)
        
        # Original PCA
        pca = PCA(n_components=10)
        rep_pca = pca.fit_transform(all_reps)

#        axes[0, data_seed_idx].scatter(rep_pca[:, 0], rep_pca[:, 1], c=compute_depth(parents), cmap='viridis')
        axes[0, data_seed_idx].scatter(rep_pca[:, 0], rep_pca[:, 1], c=compute_depth(parents), cmap='viridis', s=7)
        for i in range(len(parents)):
            if parents[i] != -1:
                depth_color = compute_depth(parents)[i]
                axes[0, data_seed_idx].annotate(
                    '', 
                    xy=(rep_pca[i, 0], rep_pca[i, 1]), 
                    xytext=(rep_pca[parents[i], 0], rep_pca[parents[i], 1]),
                    arrowprops=dict(arrowstyle='->', color=plt.cm.viridis(depth_color / max(compute_depth(parents))), lw=1.5)
                )
        axes[0, data_seed_idx].set_frame_on(False)
        
        # Train AAT with eps=0.0
        aat_0_results = train_AAT_for_epsilons(
            rep_pca,               # numpy array, shape (n_nodes, emb_dim)
            pairs,           # numpy array, shape (N_pairs, 2)
            labels,          # numpy array, shape (N_pairs,)
            epsilons=[0.0],        # list of float
            emb_dim=2,
            hidden_dim=rep_pca.shape[1],
            max_epochs=3000,
        )
        aat_projection_matrix = aat_0_results['probe']['linear.weight'].cpu().numpy()
        print(aat_projection_matrix.shape)
        rep_aat_0 = rep_pca @ aat_projection_matrix.T
        axes[1, data_seed_idx].scatter(rep_aat_0[:, 0], rep_aat_0[:, 1], c=compute_depth(parents), cmap='viridis', s=7)
#        axes[1, data_seed_idx].set_title(f'f1={aat_0_results["results"][0.0]:.2f}')
        for i in range(len(parents)):
            if parents[i] != -1:
                depth_color = compute_depth(parents)[i]
                axes[1, data_seed_idx].annotate(
                    '', 
                    xy=(rep_aat_0[i, 0], rep_aat_0[i, 1]), 
                    xytext=(rep_aat_0[parents[i], 0], rep_aat_0[parents[i], 1]),
                    arrowprops=dict(arrowstyle='->', color=plt.cm.viridis(depth_color / max(compute_depth(parents))), lw=1.5)
                )
        axes[1, data_seed_idx].set_frame_on(False)
        
#         # Train AAT with eps=0.3
#         aat_3_results = train_AAT_for_epsilons(
#             rep_pca,               # numpy array, shape (n_nodes, emb_dim)
#             pairs,           # numpy array, shape (N_pairs, 2)
#             labels,          # numpy array, shape (N_pairs,)
#             epsilons=[0.02],        # list of float
#             emb_dim=2,
#             hidden_dim=rep_pca.shape[1],
#             max_epochs=3000,
#         )
#         aat_projection_matrix = aat_3_results['probe']['linear.weight'].cpu().numpy()
#         rep_aat_3 = rep_pca @ aat_projection_matrix.T

#         axes[2, data_seed_idx].scatter(rep_aat_3[:, 0], rep_aat_3[:, 1], c=compute_depth(parents), cmap='viridis')
# #        axes[2, data_seed_idx].set_title(f'f1={aat_3_results["results"][0.02]:.2f}')
#         for i in range(len(parents)):
#             if parents[i] != -1:
#                 depth_color = compute_depth(parents)[i]
#                 axes[2, data_seed_idx].annotate(
#                     '', 
#                     xy=(rep_aat_3[i, 0], rep_aat_3[i, 1]), 
#                     xytext=(rep_aat_3[parents[i], 0], rep_aat_3[parents[i], 1]),
#                     arrowprops=dict(arrowstyle='->', color=plt.cm.viridis(depth_color / max(compute_depth(parents))), lw=1.5)
#                 )
#         axes[2, data_seed_idx].set_frame_on(False)
        
        torch.cuda.empty_cache()

        axes[0, data_seed_idx].set_xticks([])
        axes[0, data_seed_idx].set_yticks([])
        axes[1, data_seed_idx].set_xticks([])
        axes[1, data_seed_idx].set_yticks([])
        # axes[2, data_seed_idx].set_xticks([])
        # axes[2, data_seed_idx].set_yticks([])
        if data_seed_idx == 0:
            axes[0, data_seed_idx].set_ylabel('PCA')
            axes[1, data_seed_idx].set_ylabel('Cone')
            # axes[2, data_seed_idx].set_ylabel('AAT with eps=0.3')
        

    plt.savefig('pca_vis.png', bbox_inches='tight')
    plt.savefig('pca_vis.pdf', bbox_inches='tight')

    