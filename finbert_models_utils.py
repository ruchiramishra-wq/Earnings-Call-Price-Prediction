import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import os
from torch.utils.data import Dataset
import glob
from config import CACHE_DIR
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
import numpy as np

class CachedDataset(Dataset):
    """
    Dataset for loading cached FinBERT embeddings and labels.
    """
    def __init__(self, cache_dir):
        self.paths = glob.glob(os.path.join(cache_dir, "*.pt"))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        obj = torch.load(self.paths[idx], map_location="cpu")
        return obj["Z"].float(), torch.tensor(obj["y"], dtype=torch.float32)


class CachedZFinDataset(Dataset):
    """
    Dataset for loading cached FinBERT embeddings, financial features, and labels.
    """
    def __init__(self, cache_dir):
        self.paths = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        obj = torch.load(self.paths[idx], map_location="cpu")
        Z = obj["Z"].float()  # (C,768)
        y = torch.tensor(obj["y"], dtype=torch.float32)
        fin = obj["fin_features"].float().view(-1)
        return Z, fin, y
    

def collate_pad(batch):
    """Pads variable-length sequences in the batch with zeros to help with batching.""" 
    
    # batch = [(Z1, y1), (Z2, y2), ...]
    Z_list, y_list = zip(*batch)

    B = len(Z_list)
    dim = Z_list[0].shape[1]
    C_max = max(z.shape[0] for z in Z_list)

    Z_pad = torch.zeros(B, C_max, dim)
    mask  = torch.zeros(B, C_max)

    for i, Z in enumerate(Z_list):
        C = Z.shape[0]
        Z_pad[i, :C] = Z
        mask[i, :C] = 1.0

    y = torch.tensor(y_list, dtype=torch.float32)
    return Z_pad, mask, y


def collate_pad_chunks_with_fin(batch):
    """Pads variable-length sequences in the batch with zeros to help with batching. Also stacks financial features."""
    # batch: [(Z, fin, y), ...]
    Z_list, fin_list, y_list = zip(*batch)

    B = len(Z_list)
    dim = Z_list[0].shape[1]
    C_max = max(z.shape[0] for z in Z_list)

    Z_pad = torch.zeros(B, C_max, dim, dtype=torch.float32)
    mask  = torch.zeros(B, C_max, dtype=torch.float32)

    for i, Z in enumerate(Z_list):
        C = Z.shape[0]
        Z_pad[i, :C] = Z
        mask[i, :C] = 1.0

    fin = torch.stack([f.view(-1) for f in fin_list]).float()  # (B,K)
    y = torch.tensor(y_list, dtype=torch.float32)              # (B,)

    return Z_pad, mask, fin, y



class MeanPoolClassifier(nn.Module):
    """A simple mean-pooling classifier. Takes the mean of all chunk vectors to return one embedding vector per transcript
    
    Args:
        dim (int): Dimension of input features.
        hidden (int): Dimension of hidden layer.
        dropout (float): Dropout rate

    Returns:
        torch.Tensor: Output logits of shape (B,). 
    """

    def __init__(self, dim=768, hidden=256, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, Z, mask):
        # Z: (B,C,768), mask: (B,C)
        mask3 = mask.unsqueeze(-1)  # (B,C,1)
        doc = (Z * mask3).sum(dim=1) / mask3.sum(dim=1).clamp(min=1e-9)
        x = F.relu(self.fc1(doc))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)  # (B,)


class AttnPoolClassifier(nn.Module):
    """
    A simple attention-pooling classifier. Learns attention weights over chunk vectors to return one embedding vector per transcript.
    Args:
        dim (int): Dimension of input features.
        hidden (int): Dimension of hidden layer.
        dropout (float): Dropout rate

    Returns:
        torch.Tensor: Output logits of shape (B,).        

    """
    def __init__(self, dim=768, hidden=256, dropout=0.2):
        super().__init__()
        self.attn = nn.Parameter(torch.randn(dim) * 0.02)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, Z, mask):
        # Z: (B,C,768)
        scores = torch.einsum("bcd,d->bc", Z, self.attn)  # (B,C)
        scores = scores.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(scores, dim=1)
        doc = torch.einsum("bc,bcd->bd", alpha, Z)        # (B,768)
        x = F.relu(self.fc1(doc))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)

class AttnMLPPoolClassifier(nn.Module):
    """
    An attention-pooling classifier with non-linearity applied to attention. Learns attention weights over chunk vectors to return one embedding vector per transcript.

    Args:
        dim (int): Dimension of input features. 
        attn_hidden (int): Dimension of attention hidden layer.
        hidden (int): Dimension of hidden layer.
        dropout (float): Dropout rate

    Returns:
        torch.Tensor: Output logits of shape (B,).        
    """
    def __init__(self, dim=768, attn_hidden=256, hidden=256, dropout=0.2):
        super().__init__()
        self.W = nn.Linear(dim, attn_hidden)
        self.v = nn.Linear(attn_hidden, 1, bias=False)

        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, Z, mask):
        # Z: (B,C,768), mask: (B,C)
        h = torch.tanh(self.W(Z))              # (B,C,H)
        scores = self.v(h).squeeze(-1)         # (B,C)
        scores = scores.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(scores, dim=1)   # (B,C)

        doc = torch.einsum("bc,bcd->bd", alpha, Z)  # (B,768)

        x = F.relu(self.fc1(doc))
        x = self.drop(x)
        return self.fc2(x).squeeze(-1)
      

class AttnPoolTwoTower(nn.Module):
    """
    An attention-pooling two-tower classifier. Learns attention weights over chunk vectors to return one embedding vector per transcript, and combines it with financial features.
    Uses two separate projection heads for document and financial features before combining them for final classification.

    Args:
        dim (int): Dimension of input features. 
        fin_dim (int): Dimension of financial features.
        hidden (int): Dimension of hidden layer.
        dropout (float): Dropout rate

    Returns:
        torch.Tensor: Output logits of shape (B,).        

    """
    def __init__(self, dim=768, fin_dim=4, hidden=256, dropout=0.2):
        super().__init__()
        self.attn = nn.Parameter(torch.randn(dim) * 0.02)

        self.doc_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fin_proj = nn.Sequential(
            nn.LayerNorm(fin_dim),          # optional
            nn.Linear(fin_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, Z, mask, fin):
        scores = torch.einsum("bcd,d->bc", Z, self.attn)
        scores = scores.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(scores, dim=1)
        doc = torch.einsum("bc,bcd->bd", alpha, Z)  # (B,768)

        a = self.doc_proj(doc)   # (B,hidden)
        b = self.fin_proj(fin)   # (B,hidden)

        x = torch.cat([a, b], dim=1)
        return self.out(x).squeeze(-1)


loss_fn = nn.BCEWithLogitsLoss()

@torch.no_grad()
def eval_loop_auc(model, loader, device):
    """
    Evaluation loop that computes average loss and AUC over the dataset.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to run the evaluation on.

    Returns:
        tuple: Logits,Average loss and AUC score.        
    """
    model.eval()
    total_loss, n = 0.0, 0

    all_logits = []
    all_labels = []

    for Z, mask, y in loader:
        Z = Z.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logit = model(Z, mask)              # (B,)
        loss = loss_fn(logit, y)

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        all_logits.append(logit.cpu())
        all_labels.append(y.cpu())

    avg_loss = total_loss / max(1, n)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    auc = roc_auc_score(labels, logits)

    return logits,avg_loss, auc


@torch.no_grad()
def eval_loop_auc_fin(model, loader, device):
    """
Evaluation loop that computes y_scores, average loss and AUC over the dataset, for models that take financial features and transcript embeddings.

    Args:
        model (nn.Module): The model to evaluate.   
        loader (DataLoader): DataLoader for the evaluation dataset.
        device: Device to run the evaluation on.

    Returns:
        tuple: y_scores, Average loss and AUC score.        
    """
    model.eval()
    total_loss, n = 0.0, 0

    all_logits = []
    all_labels = []

    for Z, mask,fin, y in loader:
        Z = Z.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        fin=fin.to(device,non_blocking=True)
        y = y.to(device, non_blocking=True)

        logit = model(Z, mask,fin)              # (B,)
        loss = loss_fn(logit, y)

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        all_logits.append(logit.cpu())
        all_labels.append(y.cpu())

    avg_loss = total_loss / max(1, n)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    auc = roc_auc_score(labels, logits)

    return logits,avg_loss, auc


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    device,
    max_epochs=50,
    patience=7,
    lr=1e-3,
    weight_decay=1e-2,
    save_path="best.pt",
):
    """
Training loop with early stopping based on validation AUC. Uses AdamW optimizer. 
Stops training if validation AUC does not improve for a specified number of epochs (patience).
    
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the training on.
        max_epochs (int): Maximum number of epochs to train.
        patience (int): Number of epochs to wait for improvement before stopping.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        save_path (str): Path to save the best model weights.

    Returns:
        nn.Module: The trained model with the best weights loaded.        
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    bad_epochs = 0

    best_auc = -float("inf")
    patience = 7
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n = 0.0, 0

        for Z, mask, y in train_loader:
            Z = Z.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logit = model(Z, mask)
            loss = loss_fn(logit, y)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        train_loss = total_loss / n
        val_logits,val_loss, val_auc = eval_loop_auc(model, val_loader, device)

        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.3f}"
        )
    
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            bad_epochs = 0
            torch.save(model.state_dict(), save_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping on AUC.")
                break

    # load best weights before returning
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


def train_with_early_stopping_fin(
    model,
    train_loader,
    val_loader,
    device,
    max_epochs=50,
    patience=7,
    lr=1e-3,
    weight_decay=1e-2,
    save_path="best.pt",
):
    """
Training loop with early stopping based on validation AUC, for models that take financial features and transcript embeddings. Uses AdamW optimizer. 
Stops training if validation AUC does not improve for a specified number of epochs (patience).

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset. 
        device (torch.device): Device to run the training on.
        max_epochs (int): Maximum number of epochs to train.
        patience (int): Number of epochs to wait for improvement before stopping.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
        save_path (str): Path to save the best model weights.

    Returns:
        nn.Module: The trained model with the best weights loaded.
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    bad_epochs = 0

    best_auc = -float("inf")
    patience = 7
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss, n = 0.0, 0

        for Z, mask,fin, y in train_loader:
            Z = Z.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            fin=fin.to(device,non_blocking=True)
            y = y.to(device, non_blocking=True)

            logit = model(Z, mask,fin)
            loss = loss_fn(logit, y)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        train_loss = total_loss / n
        val_logits,val_loss, val_auc = eval_loop_auc_fin(model, val_loader, device)

        print(
            f"epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.3f}"
        )
    
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            bad_epochs = 0
            torch.save(model.state_dict(), save_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping on AUC.")
                break

    # load best weights before returning
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model


# def load_cached_finbert_dataset(split,return_days=1,cache_dir=CACHE_DIR,batch_size=16,shuffle=True):
#     """
#     Loads a cached FinBERT dataset from disk.

#     Args:
#         cache_dir: Directory where cached embeddings are stored.
#         split: One of "train", "val", or "test".
#         batch_size: Batch size for DataLoader.
#         shuffle: Whether to shuffle the data.
#         collate_fn: Function to collate batches 
#     Returns:
#         DataLoader for the specified split.
#     """
#     from torch.utils.data import DataLoader

#     split_cache_dir = os.path.join(cache_dir, f"cache_{split}_{return_days}")
#     dataset = CachedDataset(split_cache_dir)
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         collate_fn=collate_pad)
#     return dataloader

def load_cached_finbert_dataset(split, return_days=1, cache_dir=None, batch_size=64, shuffle=False):
    """
    Load cached FinBERT embeddings dataset.
    
    Tries two directory structures:
    1. cache_dir/split/return_days/*.pt  (new structure)
    2. cache_dir/cache_{split}_{return_days}/*.pt  (old structure from notebook)
    """
    if cache_dir is None:
        raise ValueError("cache_dir must be provided")

    # Try new structure first
    split_cache_dir = os.path.join(cache_dir)
    
    # Fallback to notebook structure if needed
    if not os.path.exists(split_cache_dir) or len(glob.glob(os.path.join(split_cache_dir, "*.pt"))) == 0:
        split_cache_dir = os.path.join(cache_dir, f"cache_{split}_{return_days}")
    try:
        dataset = CachedDataset(split_cache_dir)
    except Exception as e:
        dataset = CachedDataset(os.path.join(cache_dir, split, str(return_days), f"cache_{split}_{return_days}"))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_pad)

def load_cached_finbert_fin_dataset(split,return_days=1,cache_dir=CACHE_DIR,batch_size=16,shuffle=True):
    """
    Loads a cached FinBERT dataset with financial features from disk.

    Args:
        cache_dir: Directory where cached embeddings are stored.
        split: One of "train", "val", or "test".
        batch_size: Batch size for DataLoader.
        shuffle: Whether to shuffle the data.
        collate_fn: Function to collate batches 
    Returns:
        DataLoader for the specified split.
    """
    from torch.utils.data import DataLoader

    split_cache_dir = os.path.join(cache_dir, f"cache_{split}_{return_days}")
    dataset = CachedZFinDataset(split_cache_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_pad_chunks_with_fin)
    return dataloader

def bootstrap_auc_se(y_true, y_scores, n_bootstraps=1000, random_seed=42):
    """
    Computes bootstrap confidence intervals and standard error for AUC.

    Args:
        y_true (array-like): True binary labels.
        y_scores (array-like): Predicted scores.
        n_bootstraps (int): Number of bootstrap samples.
        random_seed (int): Random seed for reproducibility.
    Returns:
        tuple: (lower_bound, upper_bound) of 95% confidence interval for AUC
        int: standard error of AUC
    """
    import numpy as np

    rng = np.random.RandomState(random_seed)
    bootstrapped_scores = []

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower_bound = sorted_scores[int(0.025 * len(sorted_scores))]
    upper_bound = sorted_scores[int(0.975 * len(sorted_scores))]
    se = np.std(bootstrapped_scores)

    return (lower_bound, upper_bound), se

def call_model(Model="AttnMLPPoolClassifier",dim=768, attn_hidden=256, hidden=256, dropout=0.2,return_period=1):
    """
    trains the specifed model at the given return period and returns the trained model, test loss and test AUC confidence intervals.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = load_cached_finbert_dataset(
        cache_dir=CACHE_DIR, split="train",return_days=1 ,batch_size=16, shuffle=True
    )
    val_loader = load_cached_finbert_dataset(
        cache_dir=CACHE_DIR, split="val",return_days=1,batch_size=16, shuffle=False
    )
    test_loader = load_cached_finbert_dataset(
        cache_dir=CACHE_DIR, split="test",return_days=1,batch_size=16, shuffle=False
    )

    if Model=="MeanPoolClassifier":
        model = MeanPoolClassifier(dim=dim, hidden=hidden, dropout=dropout).to(device)
    elif Model=="AttnPoolClassifier":
        model = AttnPoolClassifier(dim=dim, hidden=hidden, dropout=dropout).to(device)
    elif Model=="AttnMLPPoolClassifier":
        model = AttnMLPPoolClassifier(dim=dim, attn_hidden=attn_hidden, hidden=hidden, dropout=dropout).to(device)

    model = train_with_early_stopping(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=50,
        patience=7,
        lr=1e-3,
        weight_decay=1e-2,
        save_path=f"best_model_{return_period}r.pt",
    )

    test_logits,test_loss, test_auc = eval_loop_auc(model, test_loader, device)

    test_auc_ci,test_se = bootstrap_auc_se(
        y_true=[y for _, y in test_loader.dataset],
        y_scores=test_logits,
        n_bootstraps=1000,
        random_seed=42,
    )

    return model, test_loss, test_auc, test_auc_ci,test_se

def call_model_fin(Model="AttnPoolTwoTower",dim=768, fin_dim=4, hidden=256, dropout=0.2,return_period=1):
    """
    trains the specifed model at the given return period and returns the trained model, test loss and test AUC.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = load_cached_finbert_fin_dataset(
        cache_dir=CACHE_DIR, split="train",return_days=return_period ,batch_size=16, shuffle=True
    )
    val_loader = load_cached_finbert_fin_dataset(
        cache_dir=CACHE_DIR, split="val",return_days=return_period,batch_size=16, shuffle=False
    )
    test_loader = load_cached_finbert_fin_dataset(
        cache_dir=CACHE_DIR, split="test",return_days=return_period,batch_size=16, shuffle=False
    )
    if Model=="AttnPoolTwoTower":
        model = AttnPoolTwoTower(dim=768, fin_dim=4, hidden=256, dropout=0.2).to(device)
    
    model = train_with_early_stopping_fin(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=50,
        patience=7,
        lr=1e-3,
        weight_decay=1e-2,
        save_path=f"best_model_fin_{return_period}r.pt",
    )

    test_logits,test_loss, test_auc = eval_loop_auc_fin(model, test_loader, device)

    test_auc_ci,test_se = bootstrap_auc_se(
        y_true=[y for _, _, y in test_loader.dataset],
        y_scores=test_logits,
        n_bootstraps=1000,
        random_seed=42,
    )

    return model, test_loss, test_auc, test_auc_ci,test_se

@torch.no_grad()
def extract_meanpooled_vectors(loader, device):
    """
    Extracts mean-pooled transcript vectors from a loader that yields (Z_pad, mask, y).
    Returns:
        X: (N, dim) numpy array
        y: (N,) numpy array
    """
    X_list, y_list = [], []

    for Z, mask, y in loader:
        Z = Z.to(device, non_blocking=True)           # (B, C, dim)
        mask = mask.to(device, non_blocking=True)     # (B, C)

        mask3 = mask.unsqueeze(-1).float()            # (B, C, 1)
        denom = mask3.sum(dim=1).clamp(min=1e-9)      # (B, 1)
        doc = (Z * mask3).sum(dim=1) / denom          # (B, dim)

        X_list.append(doc.cpu().numpy())
        y_list.append(y.cpu().numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

@torch.no_grad()
def extract_meanpooled_vectors_with_fin(loader, device):
    """
    Extracts mean-pooled transcript vectors and concatenates financial features.
    Loader yields (Z_pad, mask, fin, y).
    Returns:
        X: (N, dim + fin_dim)
        y: (N,)
    """
    X_list, y_list = [], []

    for Z, mask, fin, y in loader:
        Z = Z.to(device, non_blocking=True)           # (B, C, dim)
        mask = mask.to(device, non_blocking=True)     # (B, C)
        fin = fin.to(device, non_blocking=True)       # (B, fin_dim)

        mask3 = mask.unsqueeze(-1).float()
        denom = mask3.sum(dim=1).clamp(min=1e-9)
        doc = (Z * mask3).sum(dim=1) / denom          # (B, dim)

        feats = torch.cat([doc, fin], dim=1)          # (B, dim+fin_dim)

        X_list.append(feats.cpu().numpy())
        y_list.append(y.cpu().numpy())

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

def fit_svm_with_val_selection(X_train, y_train, X_val, y_val, C_grid=None, class_weight="balanced"):
    """
    Fits a Linear SVM with feature scaling and selects C using validation AUC.

    Returns:
        best_pipe: fitted sklearn Pipeline
        best_C: chosen C
        best_val_auc: validation AUC at best C
    """
    if C_grid is None:
        C_grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    best_pipe, best_C, best_auc = None, None, -np.inf

    for C in C_grid:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(C=C, class_weight=class_weight, dual="auto", max_iter=20000)),
        ])
        pipe.fit(X_train, y_train)

        val_scores = pipe.decision_function(X_val)
        val_auc = roc_auc_score(y_val, val_scores)

        if val_auc > best_auc:
            best_auc = val_auc
            best_C = C
            best_pipe = pipe

    return best_pipe, best_C, best_auc

def eval_svm_auc(pipe, X, y):
    """
    Evaluates fitted pipeline on data, returns (scores, auc).
    """
    scores = pipe.decision_function(X)
    auc = roc_auc_score(y, scores)
    return scores, auc

def meanpooling_withsvm(
    return_period=1,
    cache_dir=CACHE_DIR,
    batch_size=64,
    C_grid=None,
    class_weight="balanced",
):
    """
    Mean-pool chunk CLS embeddings -> train Linear SVM on transcript vectors.
    Uses validation AUC to select C. Reports test AUC + bootstrap CI/SE.

    Returns:
        dict with:
            best_C, val_auc, test_auc, test_auc_ci, test_se, test_scores, svm_pipe
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = load_cached_finbert_dataset(
        split="train", return_days=return_period, cache_dir=cache_dir, batch_size=batch_size, shuffle=False
    )
    val_loader = load_cached_finbert_dataset(
        split="val", return_days=return_period, cache_dir=cache_dir, batch_size=batch_size, shuffle=False
    )
    test_loader = load_cached_finbert_dataset(
        split="test", return_days=return_period, cache_dir=cache_dir, batch_size=batch_size, shuffle=False
    )

    # 1) Extract features
    X_train, y_train = extract_meanpooled_vectors(train_loader, device)
    X_val, y_val     = extract_meanpooled_vectors(val_loader, device)
    X_test, y_test   = extract_meanpooled_vectors(test_loader, device)

    # 2) Fit + select C on val
    pipe, best_C, val_auc = fit_svm_with_val_selection(
        X_train, y_train, X_val, y_val, C_grid=C_grid, class_weight=class_weight
    )

    # 3) Test eval
    test_scores, test_auc = eval_svm_auc(pipe, X_test, y_test)

    # 4) Bootstrap CI/SE (use your existing function)
    test_auc_ci, test_se = bootstrap_auc_se(
        y_true=y_test,
        y_scores=test_scores,
        n_bootstraps=1000,
        random_seed=42,
    )

    return {
        "best_C": best_C,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "test_auc_ci": test_auc_ci,
        "test_se": test_se,
        "test_scores": test_scores,
        "svm_pipe": pipe,
    }

def meanpooling_withsvm_fin(
    return_period=1,
    cache_dir=CACHE_DIR,
    batch_size=64,
    C_grid=None,
    class_weight="balanced",
):
    """
    Mean-pool chunk CLS embeddings, concatenate fin_features, then train Linear SVM.
    Returns dict similar to meanpooling_withsvm.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = load_cached_finbert_fin_dataset(
        split="train", return_days=return_period, cache_dir=cache_dir, batch_size=batch_size, shuffle=False
    )
    val_loader = load_cached_finbert_fin_dataset(
        split="val", return_days=return_period, cache_dir=cache_dir, batch_size=batch_size, shuffle=False
    )
    test_loader = load_cached_finbert_fin_dataset(
        split="test", return_days=return_period, cache_dir=cache_dir, batch_size=batch_size, shuffle=False
    )

    X_train, y_train = extract_meanpooled_vectors_with_fin(train_loader, device)
    X_val, y_val     = extract_meanpooled_vectors_with_fin(val_loader, device)
    X_test, y_test   = extract_meanpooled_vectors_with_fin(test_loader, device)

    pipe, best_C, val_auc = fit_svm_with_val_selection(
        X_train, y_train, X_val, y_val, C_grid=C_grid, class_weight=class_weight
    )

    test_scores, test_auc = eval_svm_auc(pipe, X_test, y_test)

    test_auc_ci, test_se = bootstrap_auc_se(
        y_true=y_test,
        y_scores=test_scores,
        n_bootstraps=1000,
        random_seed=42,
    )

    return {
        "best_C": best_C,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "test_auc_ci": test_auc_ci,
        "test_se": test_se,
        "test_scores": test_scores,
        "svm_pipe": pipe,
    }