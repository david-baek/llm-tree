import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils.parametrizations as parametrize

from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from collections import defaultdict
from itertools import product
from tqdm import tqdm
import argparse
import networkx as nx
from matplotlib.patches import FancyArrowPatch


def generate_graph_description(parents, names, seed=42, shuffle=False):
    description = "Family Tree:\n"
    children_list = [[] for _ in range(len(parents))]
    for child, parent in enumerate(parents):
        if parent >= 0:
            children_list[parent].append(child)

    parent_list = list(range(len(parents)))
    if shuffle:
        random.seed(seed)
        random.shuffle(parent_list)
    
    for parent in parent_list:
        children = children_list[parent]
        if children:
            description += f"{names[parent]}'s children: [{', '.join(names[child] for child in children)}]\n"
    return description


def compute_descendants(parents):
    """Return for each node the set of all its descendants."""
    n = len(parents)
    children = defaultdict(list)
    for child, p in enumerate(parents):
        if p >= 0:
            children[p].append(child)
    descendants = [set() for i in range(n)]
    def dfs(u):
        for v in children[u]:
            descendants[u].add(v)
            dfs(v)
            descendants[u].update(descendants[v])
    for u in range(n):
        dfs(u)
    return descendants

class PairDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]


def train_ordering_probe(
    E,                  # numpy array, shape (n_nodes, emb_dim)
    epsilons=[0.01],    # list of float
    emb_dim=2,         # int
    hidden_dim=None,    # int or None
    batch_size=32,     # int
    train_seed=42,     # int
    lr=1e-3,          # float
    max_epochs=300,    # int
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train a logistic regression probe to predict ordering of nodes based on their indices.
    
    Args:
        E: Node embeddings matrix of shape (n_nodes, emb_dim)
        epsilons: List of epsilon values to try
        emb_dim: Dimension of probe output
        hidden_dim: Hidden dimension (defaults to E.shape[1])
        batch_size: Training batch size
        train_seed: Random seed for training
        lr: Learning rate
        max_epochs: Maximum training epochs
        device: Device to train on
    
    Returns:
        Dictionary containing:
        - 'logistic_probe': Best trained logistic probe
        - 'results': Dictionary mapping epsilon to final loss
    """
    if hidden_dim is None:
        hidden_dim = E.shape[1]
        
    n_nodes = E.shape[0]
    
    # Generate all ordered pairs and their target labels
    pairs = []
    targets = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                pairs.append(E[i] - E[j])
                targets.append(1 if i < j else 0)
                
    pairs = np.array(pairs)
    targets = np.array(targets)

    X_train, y_train = pairs, targets
    X_test, y_test = pairs, targets

    log_reg = LogisticRegression(random_state=train_seed, max_iter=10000, penalty='l1', fit_intercept=False, solver='liblinear')
    log_reg.fit(X_train, y_train)

    # Evaluate on test set
    y_test_pred = log_reg.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # # Print results
    # ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
    # plt.title("Confusion Matrix")
    # plt.show()

    results = {}
    # Store results (using a single epsilon value for compatibility)
    for eps in epsilons:
        results[eps] = test_f1
        
    results['linear_probe'] = log_reg

    return results



def build_dataset(parents):
    n = len(parents)
    descendants = compute_descendants(parents)
    pairs = []
    labels = []
    # all ordered pairs (i, j)
    for i, j in product(range(n), range(n)):
        if i != j:
            pairs.append((i, j))
            labels.append(1 if i in descendants[j] else 0)
    return np.array(pairs, dtype=np.int64), np.array(labels, dtype=np.float32)


class DiffAATProbe(nn.Module):
    def __init__(self, emb_dim, hidden_dim, epsilon):
        super().__init__()
        # Linear term
        self.linear = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.linear.weight.data.copy_(torch.eye(emb_dim, hidden_dim))
        
        # D matrix as a parameter
        self.D = nn.Parameter(torch.eye(hidden_dim, hidden_dim).unsqueeze(-1).repeat(1, 1, emb_dim))
        
        # Apply spectral normalization to D to ensure spectral norm <= 1
        torch.nn.utils.parametrizations.spectral_norm(self, name='D', n_power_iterations=1)
        
        # Quadratic correction absorbed into embeddings upstream
        self.epsilon = epsilon
        
        # Trainable width parameter for sigmoid
        self.sigmoid_width = nn.Parameter(torch.tensor(1.0), requires_grad=True)
#        torch.nn.utils.parametrize.register_parametrization(self, 'sigmoid_width', torch.nn.Softplus())
        

    def forward(self, Ex, Ey):
        # Ex, Ey: (batch, emb_dim)
        # Apply linear transformation and quadratic form with D
        Ex = self.linear(Ex) + self.epsilon *torch.einsum('bi,bj,ijk->bk', Ex, Ex, self.D)  # (batch, hidden_dim)
        Ey = self.linear(Ey) + self.epsilon *torch.einsum('bi,bj,ijk->bk', Ey, Ey, self.D)  # (batch, hidden_dim)
        # Ex = self.linear(Ex)
        # Ey = self.linear(Ey)
        
        # Compute coordinate-wise sigmoid differences
        # Calculate differences for each coordinate
        differences = Ex - Ey  # (batch, emb_dim)
        
        # Apply sigmoid to each coordinate difference
        sigmoid_values = torch.sigmoid(differences / torch.max(self.sigmoid_width, torch.tensor(1e-3)))
        
        # Compute the mean of sigmoid values across all coordinates
        p = torch.mean(sigmoid_values, dim=1)  # (batch,)
        p = sigmoid_values[:, 0] * sigmoid_values[:, 1]
        return p

class PairDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = torch.tensor(pairs, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.pairs[idx], self.labels[idx]

def train_linear_probe_for_epsilons(
    E,               # numpy array, shape (n_nodes, emb_dim)
    pairs,           # numpy array, shape (N_pairs, 2)
    labels,          # numpy array, shape (N_pairs,)
    epsilons,        # list of float (not used for linear probe but kept for signature consistency)
    emb_dim,
    hidden_dim=32,   # not used for linear probe
    batch_size=64,   # not used for linear probe
    train_seed=10,
    lr=1e-3,         # not used for linear probe
    max_epochs=100,  # not used for linear probe
    device='cpu'     # not used for linear probe
):
    results = {}
    # Flatten pairs into features using embeddings E
    X = E[pairs[:,0]] - E[pairs[:,1]]#np.hstack([E[pairs[:, 0]], E[pairs[:, 1]]])  # Concatenate embeddings of both nodes
    y = labels

    # Split into train/test sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=train_seed, stratify=y
    # )
    X_train, y_train = X, y  # Use all pairs for training
    X_test, y_test = X, y  # Use all pairs for testing

    # Train logistic regression as a linear probe
    log_reg = LogisticRegression(random_state=train_seed, max_iter=10000, penalty='l1', fit_intercept=False, solver='liblinear')
    log_reg.fit(X_train, y_train)

    # Evaluate on test set
    y_test_pred = log_reg.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # # Print results
    # ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
    # plt.title("Confusion Matrix")
    # plt.show()

    # Store results (using a single epsilon value for compatibility)
    for eps in epsilons:
        results[eps] = test_f1
        
    results['linear_probe'] = log_reg

    return results

def train_AAT_for_epsilons(
    E,               # numpy array, shape (n_nodes, emb_dim)
    pairs,           # numpy array, shape (N_pairs, 2)
    labels,          # numpy array, shape (N_pairs,)
    epsilons,        # list of float
    emb_dim,
    hidden_dim=32,
    batch_size=64,
    train_seed=10,
    lr=1e-3,
    max_epochs=100,
    device='cpu'
):
    results = {}
    # split into train+val/test
    # p_train, p_test, y_train, y_test = train_test_split(
    #     pairs, labels, test_size=0.2, random_state=train_seed, stratify=labels
    # )
    
    p_train, y_train = pairs, labels  # use all pairs for training
    p_test, y_test = pairs, labels  # use all pairs for testing

    train_ds = PairDataset(p_train, y_train)
    test_ds  = PairDataset(p_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # move embedding matrix to torch
    E_tensor = torch.tensor(E, dtype=torch.float32, device=device)
    E_tensor.requires_grad = False

    results['results'] = {}

    for eps in epsilons:
        best_val_f1 = -10
        best_model  = None

        probe   = DiffAATProbe(emb_dim, hidden_dim, epsilon=eps).to(device)
        optimizer = AdamW(probe.parameters(), lr=lr)
        loss_fn   = nn.BCELoss()

        pbar = tqdm(range(1, max_epochs+1), desc=f"eps={eps:.2e}", unit='epoch')
        for epoch in pbar:
            # --- train epoch ---
            probe.train()
            # Assuming positive samples are rare
            
            for batch_pairs, batch_labels in train_loader:
                xb, yb = batch_pairs[:,0].to(device), batch_pairs[:,1].to(device)
                ex = E_tensor[xb]
                ey = E_tensor[yb]
                preds = probe(ex, ey)


                loss  = loss_fn(preds, batch_labels.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # --- validate ---
            probe.eval()
            all_preds, all_trues = [], []
            with torch.no_grad():
                for batch_pairs, batch_labels in train_loader:
                    xb, yb = batch_pairs[:,0].to(device), batch_pairs[:,1].to(device)
                    ex = E_tensor[xb]; ey = E_tensor[yb]
                    preds = probe(ex, ey)
                    preds = (preds > 0.5).cpu().numpy().astype(int)
                    all_preds.append(preds)
                    all_trues.append(batch_labels.numpy().astype(int))
            val_f1 = f1_score(
                np.concatenate(all_trues),
                np.concatenate(all_preds),
            )

            # track best
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model  = copy.deepcopy(probe.state_dict())

            pbar.set_postfix(val_f1=best_val_f1)
        pbar.close()
        print(f"Best val F1: {best_val_f1}")

        # load best on test
        probe.load_state_dict(best_model)
        probe.eval()
        all_preds, all_trues = [], []
        raw_preds = []
        with torch.no_grad():
            for batch_pairs, batch_labels in test_loader:
                xb, yb = batch_pairs[:,0].to(device), batch_pairs[:,1].to(device)
                ex = E_tensor[xb]; ey = E_tensor[yb]
                preds = probe(ex, ey)
                raw_preds.append(preds.cpu().numpy().astype(float))
                preds = (preds > 0.5).cpu().numpy().astype(int)
                all_preds.append(preds)
                all_trues.append(batch_labels.numpy().astype(int))
        test_f1 = f1_score(
            np.concatenate(all_trues),
            np.concatenate(all_preds),
        )
        print(best_val_f1, test_f1)

        results['results'][eps] = test_f1
        
    results['probe'] = copy.deepcopy(probe.state_dict())
    results['raw_preds'] = np.concatenate(raw_preds)
    results['all_trues'] = np.concatenate(all_trues)
    return results



def generate_random_tree(num_nodes, seed=42, tree_type='random'):
    """Return parent list of length num_nodes, root has parent = -1."""
    parents = [-1]  # node 0 is root
    
    random.seed(seed)  # for reproducibility
    np.random.seed(seed)
    
    for i in range(1, num_nodes):
        if tree_type == 'balanced':
            # Balanced binary tree: attach to the parent node in a complete binary tree
            parents.append((i + 1) // 2 - 1)
        else:
            # Random tree: attach to any existing node
            p = random.randrange(0, i)
            parents.append(p)
    with open(f'./data/parents_{tree_type}_{num_nodes}_{seed}.json', 'w') as f:
        json.dump(parents, f, indent=4)
    return parents


# === 2. Model definition ===

class DescendantProbe(nn.Module):
    def __init__(self, num_nodes, emb_dim=16, hidden_dim=50):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.zeros_(self.emb.weight)
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x, y):
        # x, y: [batch]
        ex = self.emb(x)
        ey = self.emb(y)
        h = torch.cat([ex, ey], dim=1)
        return self.mlp(h).squeeze(1)

# === 3. Training / evaluation ===
def train_one_seed(
    seed,
    num_nodes=31,
    emb_dim=2,
    batch_size=16,
    max_epochs=10000,
    patience=30,
    device='cpu',
    tree_type='random',
):
    # ─── 1) Reproducibility ─────────────────────────────────────────────────────
    seed = int(seed)  # ensure seed is an integer
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ─── 2) Build tree & dataset ───────────────────────────────────────────────
    parents = generate_random_tree(num_nodes, tree_type=tree_type)
    pairs, labels = build_dataset(parents)
    
    p_train, y_train = pairs, labels  # use all pairs for training
    p_val, y_val = pairs, labels  # use all pairs for validation
    p_test, y_test = pairs, labels  # use all pairs for testing

    # stratified split
    # p_train, p_test, y_train, y_test = train_test_split(
    #     pairs, labels, test_size=0.1, random_state=seed, stratify=labels
    # )
    # p_val, y_val = p_train, y_train

    train_ds = PairDataset(p_train, y_train)
    val_ds   = PairDataset(p_val,   y_val)
    test_ds  = PairDataset(p_test,  y_test)

    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # ─── 3) Model, loss, optimizer, scheduler ────────────────────────────────
    model = DescendantProbe(num_nodes, emb_dim).to(device)

    # use plain BCE (expects probabilities in [0,1])
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
#    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    # ─── 4) Training loop with early stopping ─────────────────────────────────
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    linear_probe_f1 = 0
    aat_probe_f1 = 0
    
    linear_probe_f1_list = []
    aat_probe_f1_list = []
    train_acc_list = []

    pbar = tqdm(range(1, max_epochs + 2), desc=f"Seed {seed}", unit='epoch')
    for epoch in pbar:
        # — train —
        model.train()
        train_loss = 0.0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        for (x_pair, y_lbl) in train_loader:
            xa = x_pair[:,0].to(device)
            ya = x_pair[:,1].to(device)
            yb = y_lbl.to(device)

            optimizer.zero_grad()
            logits = model(xa, ya)
            probs  = torch.sigmoid(logits)        # convert logits → probabilities
            loss   = criterion(probs, yb.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * yb.size(0)
        train_loss /= len(train_ds)

        # — validate —
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_pair, y_lbl) in val_loader:
                xa = x_pair[:,0].to(device)
                ya = x_pair[:,1].to(device)
                yb = y_lbl.to(device).float()
                logits = model(xa, ya)
                probs  = torch.sigmoid(logits)
                val_loss += criterion(probs, yb).item() * yb.size(0)
        val_loss /= len(val_loader.dataset)

#        scheduler.step(val_loss)

        if epoch % 50 == 1:
#            print(f"[Seed {seed}] Epoch {epoch}/{max_epochs} Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")
            
            # Compute train F1 score
            model.eval()
            all_train_preds = []
            all_train_trues = []
            with torch.no_grad():
                for (x_pair, y_lbl) in train_loader:
                    xa = x_pair[:, 0].to(device)
                    ya = x_pair[:, 1].to(device)
                    yb = y_lbl.to(device).float()
                    logits = model(xa, ya)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).cpu().numpy().astype(int)
                    all_train_preds.append(preds)
                    all_train_trues.append(yb.cpu().numpy().astype(int))
            train_f1 = f1_score(
                np.concatenate(all_train_trues),
                np.concatenate(all_train_preds)
            )

            # Compute validation F1 score
            all_val_preds = []
            all_val_trues = []
            with torch.no_grad():
                for (x_pair, y_lbl) in val_loader:
                    xa = x_pair[:, 0].to(device)
                    ya = x_pair[:, 1].to(device)
                    yb = y_lbl.to(device).float()
                    logits = model(xa, ya)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).cpu().numpy().astype(int)
                    all_val_preds.append(preds)
                    all_val_trues.append(yb.cpu().numpy().astype(int))
            val_f1 = f1_score(
                np.concatenate(all_val_trues),
                np.concatenate(all_val_preds)
            )

            
        # Append accuracies to progress bar
        # pbar.set_postfix(
        #     train=f"{train_loss:.4f}",
        #     val=f"{val_loss:.4f}",
        #     train_f1=f"{train_f1:.4f}",
        #     val_f1=f"{val_f1:.4f}",
        #     linear_f1=f"{linear_probe_f1:.4f}",
        #     aat_f1=f"{aat_probe_f1:.4f}"
        # )
            
        # if epoch % 50 == 1:
        #     # Compute linear probe F1 score
        #     linear_probe_results = train_linear_probe_for_epsilons(
        #         E=model.emb.weight.data.cpu().numpy(),
        #         pairs=p_train,
        #         labels=y_train,
        #         epsilons=[0],  # epsilon not used for AAT probe here
        #         emb_dim=emb_dim,
        #         hidden_dim=emb_dim,
        #         batch_size=16,
        #         train_seed=seed,
        #         lr=1e-3,
        #         max_epochs=300,
        #         device=device
        #     )
        #     linear_probe_f1 = linear_probe_results[0]

        #     # Compute AAT probe F1 score
        #     aat_probe_results = train_AAT_for_epsilons(
        #         E=model.emb.weight.data.cpu().numpy(),
        #         pairs=p_train,
        #         labels=y_train,
        #         epsilons=[0],  # epsilon not used for AAT probe here
        #         emb_dim=emb_dim,
        #         hidden_dim=emb_dim,
        #         batch_size=16,
        #         train_seed=seed,
        #         lr=1e-3,
        #         max_epochs=500,
        #         device=device
        #     )
        #     aat_probe_f1 = aat_probe_results[0]
        #     linear_probe_f1_list.append(linear_probe_f1)
        #     aat_probe_f1_list.append(aat_probe_f1)
        #     train_acc_list.append(train_f1)

        # #Append scores to progress bar
        # pbar.set_postfix(
        #     train_f1=f"{train_f1:.4f}",
        #     linear_f1=f"{linear_probe_f1:.4f}",
        #     aat_f1=f"{aat_probe_f1:.4f}"
        # )

        # early stopping
        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'./data/best_model_mlp_{tree_type}_seed{seed}.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Seed {seed}] Early stopping at epoch {epoch}")
                break
        # pbar.set_postfix(
        #     train=f"{train_loss:.4f}",
        #     val=f"{val_loss:.4f}"
        # )
    pbar.close()

    # ─── 5) Load best model & evaluate on test set ────────────────────────────
    model.load_state_dict(torch.load(f'./data/best_model_mlp_{tree_type}_seed{seed}.pt'))
    model.eval()

    all_preds = []
    all_trues = []
    with torch.no_grad():
        for (x_pair, y_lbl) in test_loader:
            xa = x_pair[:,0].to(device)
            ya = x_pair[:,1].to(device)
            logits = model(xa, ya)
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).cpu().numpy().astype(int)
            all_preds.append(preds)
            all_trues.append(y_lbl.numpy().astype(int))

    # Compute AAT probe F1 score
    aat_probe_results = train_AAT_for_epsilons(
        E=model.emb.weight.data.cpu().numpy(),
        pairs=pairs,
        labels=labels,
        epsilons=[0],  # epsilon not used for AAT probe here
        emb_dim=emb_dim,
        hidden_dim=emb_dim,
        batch_size=16,
        train_seed=seed,
        lr=1e-3,
        max_epochs=500,
        device=device
    )
    aat_probe_f1 = aat_probe_results['results'][0]
        

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)
    test_acc = accuracy_score(y_true, y_pred)
    test_f1  = f1_score(y_true, y_pred)

    # # ─── 6) Hypothesis cone‐check & PCA plot ─────────────────────────────────
    # emb_weights = model.emb.weight.data.cpu().numpy()
    # positives = [(i,j) for (i,j),lab in zip(p_test, y_test) if lab==1]
    # in_cone = sum(int(np.dot(emb_weights[i], emb_weights[j])>0) for i,j in positives)
    # cone_prop = in_cone / len(positives)
    

    # os.makedirs('./figures', exist_ok=True)
    # Z = PCA(n_components=2).fit_transform(emb_weights)
    # # Rotate so that node 0 has the largest y-coordinate
    # max_y_angle = np.arctan2(Z[0, 1], Z[0, 0]) - np.pi / 2
    # rotation_matrix = np.array([
    #     [np.cos(-max_y_angle), -np.sin(-max_y_angle)],
    #     [np.sin(-max_y_angle),  np.cos(-max_y_angle)]
    # ])
    # Z = Z @ rotation_matrix.T
    # plt.figure(figsize=(5,5))
    # for i, p in enumerate(parents):
    #     if p >= 0:
    #         plt.plot([Z[i,0],Z[p,0]], [Z[i,1],Z[p,1]],
    #                  color='gray', linewidth=0.5, alpha=0.7)
    #     plt.text(Z[i,0], Z[i,1], str(i), fontsize=8, ha='center', va='center')
    # plt.scatter(Z[:,0], Z[:,1],
    #             c=[parents[i] for i in range(num_nodes)],
    #             cmap='tab20', s=30, edgecolor='k', linewidth=0.2)
    # plt.title(f'Seed {seed} PCA of Embeddings')
    # plt.xlabel('PC1'); plt.ylabel('PC2')
    # plt.tight_layout()
    # plt.savefig(f'./figures/pca_seed_{seed}.png', dpi=150)
    # plt.close()
    
    results = {}
    # results['linear_probe_f1_list'] = linear_probe_f1_list
    # results['aat_probe_f1_list'] = aat_probe_f1_list
    # results['train_acc_list'] = train_acc_list
    results['test_acc'] = test_acc
    results['test_f1'] = test_f1
    results['aat_probe_f1'] = aat_probe_f1
    return results


def visualize_tree(parents, names=None, save_path=None):
    """
    Visualize a tree given the parents array without using networkx.

    Args:
        parents (list): List of parent indices, where -1 indicates the root.
        names (list): Optional list of node names. If None, nodes are labeled by their indices.
        save_path (str): Optional path to save the visualization as an image. If None, the plot is shown.
    """
    import matplotlib.pyplot as plt

    n = len(parents)
    if names is None:
        names = [str(i) for i in range(n)]

    # Create a layout for the tree
    levels = {}
    positions = {}
    for i, parent in enumerate(parents):
        level = 0
        current = i
        while parents[current] != -1:
            current = parents[current]
            level += 1
        levels.setdefault(level, []).append(i)

    y_spacing = 2
    x_spacing = 2
    for level, nodes in levels.items():
        x_start = -(len(nodes) - 1) * x_spacing / 2
        for i, node in enumerate(nodes):
            positions[node] = (x_start + i * x_spacing, -level * y_spacing)

    # Plot the tree
    fig, ax = plt.subplots(figsize=(8, 6))
    for child, parent in enumerate(parents):
        if parent >= 0:
            x1, y1 = positions[parent]
            x2, y2 = positions[child]
            arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', color='black', mutation_scale=10)
            ax.add_patch(arrow)

    for node, (x, y) in positions.items():
        ax.text(x, y, names[node], ha='center', va='center', fontsize=10, bbox=dict(boxstyle="circle", facecolor="lightblue", edgecolor="black"))

    ax.set_xlim(-len(parents), len(parents))
    ax.set_ylim(-len(levels) * y_spacing - 1, 1)
    ax.axis('off')

    if save_path:
        plt.savefig(save_path, format="png", dpi=300)
        print(f"Tree visualization saved to {save_path}")
    else:
        plt.show()

# Function to perform ablation and measure logit changes
def perform_ablation(ablation_type, emb_weight, probe_weights, pairs, labels):
    # Create a copy of embeddings for ablation
    ablated_emb = emb_weight.copy()
    
    if ablation_type == 'zero':
        # Project out the direction of the linear probe (zero ablation)
        for i in range(len(ablated_emb)):
            projection = np.dot(ablated_emb[i], probe_weights.T) * probe_weights / np.sum(probe_weights**2)
            ablated_emb[i] = ablated_emb[i] - projection.flatten()
    
    elif ablation_type == 'random':
        # Replace the probe direction with random values (random ablation)
        for i in range(len(ablated_emb)):
            projection = np.dot(ablated_emb[i], probe_weights.T) * probe_weights / np.sum(probe_weights**2)
            random_values = np.random.randn(*projection.shape)
            # Normalize to have same magnitude as original projection
            random_values = random_values * np.linalg.norm(projection) / np.linalg.norm(random_values) if np.linalg.norm(random_values) > 0 else random_values
            ablated_emb[i] = ablated_emb[i] - projection.flatten() + random_values.flatten()
    
    elif ablation_type == 'mean':
        # Replace the probe direction with mean value (mean ablation)
        projections = []
        for i in range(len(ablated_emb)):
            projection = np.dot(ablated_emb[i], probe_weights.T) * probe_weights / np.sum(probe_weights**2)
            projections.append(projection.flatten())
        
        mean_projection = np.mean(projections, axis=0)
        
        for i in range(len(ablated_emb)):
            ablated_emb[i] = ablated_emb[i] - projections[i] + mean_projection
    else:
        raise ValueError(f"Invalid ablation type: {ablation_type}")
    
    return ablated_emb

# %%

# %%
