import numpy as np
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch

import numpy as np
from config import CACHE_DIR
from finbert_models_utils import load_cached_finbert_dataset, load_cached_finbert_fin_dataset, bootstrap_auc_se

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


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

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

# results = meanpooling_withsvm(return_period=1, batch_size=128)
# print(results["best_C"], results["val_auc"], results["test_auc"], results["test_auc_ci"], results["test_se"])