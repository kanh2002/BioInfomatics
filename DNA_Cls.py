import warnings
from pathlib import Path
import random
import json
import re
import joblib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from Bio import SeqIO
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import NuSVC, LinearSVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
FIG_DIR = BASE_DIR / "figures"
MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.2

DNA_TO_INDEX = {"A": 1, "T": 2, "C": 3, "G": 4}
INDEX_TO_DNA = {v: k for k, v in DNA_TO_INDEX.items()}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# Data
# =========================
def read_fasta(file_path, label):
    sequences = []
    ids = []

    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq).upper().replace("\n", "").replace(" ", "")
        if len(seq) > 0 and all(base in "ATCG" for base in seq):
            sequences.append(seq)
            ids.append(record.id)

    return pd.DataFrame({
        "id": ids,
        "sequence": sequences,
        "label": label,
        "length": [len(s) for s in sequences]
    })


def load_dataset():
    diabetic_file = DATASET_DIR / "DMT2_1296.fasta"
    non_diabetic_file = DATASET_DIR / "NONDM.fasta"

    df_diabetic = read_fasta(diabetic_file, 1)
    df_non = read_fasta(non_diabetic_file, 0)

    df = pd.concat([df_diabetic, df_non], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print("\n===== DATASET SUMMARY =====")
    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts().rename({0: "Non-diabetic", 1: "Diabetic"}))
    print(df.groupby("label")["length"].describe())

    return df


# =========================
# Classical metrics
# =========================
def evaluate_model(y_true, y_pred, y_prob, model_name):
    y_prob_pos = y_prob[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob_pos),
        "log_loss": log_loss(y_true, y_prob),
    }

    print(f"\n=== {model_name} ===")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k:10s}: {v:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return metrics


def probability_metrics(y_true, y_prob, model_name):
    y_pred = np.argmax(y_prob, axis=1)
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob[:, 1]),
        "log_loss": log_loss(y_true, y_prob),
    }


def build_similarity_groups(sequences, threshold=0.92, neighbor_k=25):
    similarity_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(4, 6),
        min_df=1,
        norm="l2",
    )
    X_sim = similarity_vectorizer.fit_transform(sequences)
    if X_sim.shape[0] == 0:
        return np.array([], dtype=int), pd.DataFrame()

    n_neighbors = min(neighbor_k + 1, X_sim.shape[0])
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    nn.fit(X_sim)
    distances, indices = nn.kneighbors(X_sim)

    parent = list(range(X_sim.shape[0]))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for row_idx in range(X_sim.shape[0]):
        for dist, neighbor_idx in zip(distances[row_idx], indices[row_idx]):
            if neighbor_idx == row_idx:
                continue
            similarity = 1.0 - float(dist)
            if similarity >= threshold:
                union(row_idx, int(neighbor_idx))

    root_to_group = {}
    group_ids = np.empty(X_sim.shape[0], dtype=int)
    next_group_id = 0
    for idx in range(X_sim.shape[0]):
        root = find(idx)
        if root not in root_to_group:
            root_to_group[root] = next_group_id
            next_group_id += 1
        group_ids[idx] = root_to_group[root]

    return group_ids, pd.DataFrame({"group_id": group_ids})


def similarity_split_indices(sequences, labels, test_size=0.2, val_size=0.15, threshold=0.92):
    group_ids, _ = build_similarity_groups(sequences, threshold=threshold)
    if len(group_ids) == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty

    group_df = pd.DataFrame({"group_id": group_ids, "label": labels})
    group_summary = (
        group_df.groupby("group_id")
        .agg(size=("label", "size"), diabetic_ratio=("label", "mean"))
        .reset_index()
    )
    group_summary["majority_label"] = (group_summary["diabetic_ratio"] >= 0.5).astype(int)

    train_groups, test_groups = train_test_split(
        group_summary["group_id"],
        test_size=test_size,
        random_state=SEED,
        stratify=group_summary["majority_label"],
    )

    train_group_table = (
        group_summary[group_summary["group_id"].isin(train_groups)]
        .sort_values("group_id")
        .reset_index(drop=True)
    )

    train_groups, val_groups = train_test_split(
        train_group_table["group_id"].to_numpy(),
        test_size=val_size,
        random_state=SEED,
        stratify=train_group_table["majority_label"].to_numpy(),
    )

    train_idx = np.where(np.isin(group_ids, train_groups))[0]
    val_idx = np.where(np.isin(group_ids, val_groups))[0]
    test_idx = np.where(np.isin(group_ids, test_groups))[0]

    split_frame = pd.DataFrame({
        "group_id": group_ids,
        "label": labels,
    })
    split_frame["split"] = "train"
    split_frame.loc[np.isin(group_ids, val_groups), "split"] = "val"
    split_frame.loc[np.isin(group_ids, test_groups), "split"] = "test"

    split_summary = (
        split_frame.groupby("split")
        .agg(samples=("label", "size"), diabetic_ratio=("label", "mean"))
        .reset_index()
    )
    split_summary.to_csv(MODEL_DIR / "similarity_split_summary.csv", index=False)

    group_summary.to_csv(MODEL_DIR / "similarity_groups.csv", index=False)

    return train_idx, val_idx, test_idx


def leakage_check(sequences, train_idx, val_idx, test_idx, threshold=0.95):
    cleaned = np.array(sequences)
    tfidf_leak = TfidfVectorizer(analyzer="char", ngram_range=(4, 6), min_df=1, norm="l2")
    X_all = tfidf_leak.fit_transform(cleaned)

    split_map = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    rows = []
    for left_name, right_name in [("train", "val"), ("train", "test"), ("val", "test")]:
        left_idx = split_map[left_name]
        right_idx = split_map[right_name]
        if len(left_idx) == 0 or len(right_idx) == 0:
            continue

        exact_overlap = len(set(cleaned[left_idx]).intersection(set(cleaned[right_idx])))
        sim_matrix = X_all[left_idx] @ X_all[right_idx].T
        max_similarity = float(sim_matrix.max()) if sim_matrix.nnz else 0.0
        high_similarity_pairs = int((sim_matrix >= threshold).nnz)

        rows.append({
            "left_split": left_name,
            "right_split": right_name,
            "exact_duplicate_overlap": exact_overlap,
            "high_similarity_pairs": high_similarity_pairs,
            "max_cosine_similarity": max_similarity,
        })

    leakage_df = pd.DataFrame(rows)
    leakage_df.to_csv(MODEL_DIR / "leakage_report.csv", index=False)
    print("\n===== LEAKAGE CHECK =====")
    print(leakage_df)

    return leakage_df


def weighted_soft_vote(prob_store, selected_names):
    weights = np.array(
        [prob_store[name]["val_metrics"]["roc_auc"] for name in selected_names],
        dtype=float,
    )
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(selected_names), dtype=float)
    weights = weights / weights.sum()

    blended_val = np.zeros_like(prob_store[selected_names[0]]["val_prob"], dtype=float)
    blended_test = np.zeros_like(prob_store[selected_names[0]]["test_prob"], dtype=float)

    for weight, name in zip(weights, selected_names):
        blended_val += weight * prob_store[name]["val_prob"]
        blended_test += weight * prob_store[name]["test_prob"]

    return blended_val, blended_test, weights


def evaluate_from_probabilities(y_true, y_prob, model_name):
    y_pred = np.argmax(y_prob, axis=1)
    return evaluate_model(y_true, y_pred, y_prob, model_name)


def build_hybrid_fusion_report(prob_store, y_test):
    ranked_names = sorted(
        prob_store.keys(),
        key=lambda name: prob_store[name]["val_metrics"]["roc_auc"],
        reverse=True,
    )

    fusion_candidates = {
        "HybridFusion_Top3": ranked_names[:3],
        "HybridFusion_DeepOnly": [
            name for name in ranked_names if prob_store[name]["family"] == "deep"
        ][:2],
        "HybridFusion_ClassicalOnly": [
            name for name in ranked_names if prob_store[name]["family"] == "classical"
        ][:3],
        "HybridFusion_NoXGBoost": [name for name in ranked_names if name != "XGBoost"][:3],
    }

    rows = []
    for fusion_name, model_names in fusion_candidates.items():
        if not model_names:
            continue

        _, test_prob, weights = weighted_soft_vote(prob_store, model_names)
        metrics = probability_metrics(y_test, test_prob, fusion_name)
        metrics["components"] = "|".join(model_names)
        metrics["weights"] = "|".join(f"{weight:.3f}" for weight in weights)
        rows.append(metrics)

    fusion_df = pd.DataFrame(rows).sort_values(by=["roc_auc", "f1"], ascending=False)
    fusion_df.to_csv(MODEL_DIR / "hybrid_fusion_results.csv", index=False)
    return fusion_df


def build_ablation_report(prob_store, y_test):
    ranked_names = sorted(
        prob_store.keys(),
        key=lambda name: prob_store[name]["val_metrics"]["roc_auc"],
        reverse=True,
    )

    ablation_sets = {
        "BestSingle": [ranked_names[0]] if ranked_names else [],
        "Top3_All": ranked_names[:3],
        "Top3_ClassicalOnly": [
            name for name in ranked_names if prob_store[name]["family"] == "classical"
        ][:3],
        "Top3_DeepOnly": [
            name for name in ranked_names if prob_store[name]["family"] == "deep"
        ][:2],
        "Top3_NoXGBoost": [name for name in ranked_names if name != "XGBoost"][:3],
    }

    rows = []
    for ablation_name, model_names in ablation_sets.items():
        if not model_names:
            continue

        _, test_prob, weights = weighted_soft_vote(prob_store, model_names)
        metrics = probability_metrics(y_test, test_prob, ablation_name)
        metrics["components"] = "|".join(model_names)
        metrics["weights"] = "|".join(f"{weight:.3f}" for weight in weights)
        rows.append(metrics)

    ablation_df = pd.DataFrame(rows).sort_values(by=["roc_auc", "f1"], ascending=False)
    ablation_df.to_csv(MODEL_DIR / "ablation_results.csv", index=False)
    return ablation_df


# =========================
# Plot functions
# =========================
def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def _save_plot(fig, stem):
    safe_stem = _safe_name(stem)
    png_path = FIG_DIR / f"{safe_stem}.png"

    try:
        fig.write_image(str(png_path), width=1400, height=900, scale=2)
    except Exception as exc:
        print(f"Plot PNG export failed for {safe_stem}: {exc}")


def plot_metric_bar(benchmark_df):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"]

    for metric in metrics:
        sorted_df = benchmark_df.sort_values(metric, ascending=(metric == "log_loss"))
        fig = px.bar(
            sorted_df,
            x="model",
            y=metric,
            title=f"Model comparison - {metric}",
        )
        fig.update_xaxes(tickangle=35)
        _save_plot(fig, f"bar_{metric}")


def plot_roc_curves(curve_store):
    fig = go.Figure()
    for name, data in curve_store.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"][:, 1])
        score = roc_auc_score(data["y_true"], data["y_prob"][:, 1])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={score:.3f})"))

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line={"dash": "dash"},
        )
    )
    fig.update_layout(title="ROC curves", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    _save_plot(fig, "roc_curves")


def plot_pr_curves(curve_store):
    fig = go.Figure()
    for name, data in curve_store.items():
        precision, recall, _ = precision_recall_curve(
            data["y_true"], data["y_prob"][:, 1]
        )
        pr_auc = auc(recall, precision)
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode="lines", name=f"{name} (AUC={pr_auc:.3f})")
        )

    fig.update_layout(title="Precision-Recall curves", xaxis_title="Recall", yaxis_title="Precision")
    _save_plot(fig, "precision_recall_curves")


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Non-DM", "DM"]
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        title=f"Confusion matrix - {model_name}",
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="True")
    _save_plot(fig, f"confusion_{model_name}")


def plot_training_history(history, model_name):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines", name="Train loss"))
    fig_loss.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines", name="Validation loss"))
    fig_loss.update_layout(
        title=f"Training curve - {model_name}",
        xaxis_title="Epoch",
        yaxis_title="Loss",
    )
    _save_plot(fig_loss, f"training_curve_{model_name}")

    fig_score = go.Figure()
    fig_score.add_trace(go.Scatter(x=epochs, y=history["val_auc"], mode="lines", name="Validation AUC"))
    fig_score.add_trace(go.Scatter(x=epochs, y=history["val_f1"], mode="lines", name="Validation F1"))
    fig_score.update_layout(
        title=f"Validation scores - {model_name}",
        xaxis_title="Epoch",
        yaxis_title="Score",
    )
    _save_plot(fig_score, f"val_scores_{model_name}")


def plot_top_kmers(vectorizer, model, top_n=25):
    if not hasattr(model, "feature_importances_"):
        return

    names = np.array(vectorizer.get_feature_names_out())
    importances = model.feature_importances_

    top_idx = np.argsort(importances)[-top_n:]
    top_names = names[top_idx]
    top_values = importances[top_idx]

    top_df = pd.DataFrame({"kmer": top_names, "importance": top_values}).sort_values("importance")

    fig = px.bar(
        top_df,
        x="importance",
        y="kmer",
        orientation="h",
        title="Top k-mer features from XGBoost",
    )
    _save_plot(fig, "top_kmers_xgboost")

    pd.DataFrame({
        "kmer": top_names[::-1],
        "importance": top_values[::-1]
    }).to_csv(MODEL_DIR / "top_kmers_xgboost.csv", index=False)


# =========================
# Deep learning dataset
# =========================
def encode_sequences(sequences, max_len):
    encoded = np.zeros((len(sequences), max_len), dtype=np.int64)

    for i, seq in enumerate(sequences):
        seq_ids = [DNA_TO_INDEX.get(ch, 0) for ch in seq[:max_len]]
        encoded[i, :len(seq_ids)] = seq_ids

    return encoded


class DNADataset(Dataset):
    def __init__(self, sequences, labels, max_len):
        self.x = encode_sequences(sequences, max_len)
        self.y = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# =========================
# Deep models
# =========================
class DeepTextCNN(nn.Module):
    def __init__(
        self,
        vocab_size=5,
        embed_dim=64,
        num_filters=128,
        kernel_sizes=(3, 5, 7, 11),
        dropout=0.4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        self.bn = nn.BatchNorm1d(num_filters * len(kernel_sizes))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)

        pooled_features = []
        for conv in self.convs:
            h = torch.relu(conv(x))
            max_pool = torch.max(h, dim=2).values
            avg_pool = torch.mean(h, dim=2)
            pooled_features.append(0.5 * max_pool + 0.5 * avg_pool)

        features = torch.cat(pooled_features, dim=1)
        features = self.bn(features)
        features = self.dropout(features)

        return self.classifier(features)


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size=5,
        embed_dim=64,
        num_filters=96,
        hidden_dim=96,
        dropout=0.4,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, num_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)

        out, _ = self.lstm(x)
        pooled = torch.max(out, dim=1).values

        return self.classifier(pooled)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits, targets, reduction="none", weight=self.alpha
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def train_torch_model(
    model,
    train_sequences,
    train_labels,
    val_sequences,
    val_labels,
    test_sequences,
    test_labels,
    model_name,
    epochs=35,
    batch_size=64,
    lr=1e-3,
    patience=7,
):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lengths = np.array([len(s) for s in train_sequences])
    max_len = int(np.percentile(lengths, 95))
    max_len = max(128, min(max_len, 1200))

    train_ds = DNADataset(train_sequences, train_labels, max_len)
    val_ds = DNADataset(val_sequences, val_labels, max_len)
    test_ds = DNADataset(test_sequences, test_labels, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = model.to(device)

    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor(
        [len(train_labels) / max(class_counts[0], 1),
         len(train_labels) / max(class_counts[1], 1)],
        dtype=torch.float32,
        device=device,
    )

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_auc = -1
    best_state = None
    bad_epochs = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_f1": [],
    }

    def run_eval(loader):
        model.eval()
        losses = []
        probs_all = []
        labels_all = []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)
                probs = torch.softmax(logits, dim=1)

                losses.append(loss.item())
                probs_all.append(probs.cpu().numpy())
                labels_all.append(yb.cpu().numpy())

        probs_all = np.concatenate(probs_all, axis=0)
        labels_all = np.concatenate(labels_all, axis=0)
        preds = probs_all.argmax(axis=1)

        return {
            "loss": float(np.mean(losses)),
            "probs": probs_all,
            "preds": preds,
            "labels": labels_all,
            "auc": roc_auc_score(labels_all, probs_all[:, 1]),
            "f1": f1_score(labels_all, preds),
        }

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses.append(loss.item())

        val_out = run_eval(val_loader)
        train_loss = float(np.mean(train_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_out["loss"])
        history["val_auc"].append(val_out["auc"])
        history["val_f1"].append(val_out["f1"])

        scheduler.step(val_out["loss"])

        print(
            f"{model_name} epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_out['loss']:.4f} | "
            f"val_auc={val_out['auc']:.4f} | val_f1={val_out['f1']:.4f}"
        )

        if val_out["auc"] > best_val_auc:
            best_val_auc = val_out["auc"]
            best_state = {
                "state_dict": model.state_dict(),
                "max_len": max_len,
                "model_name": model_name,
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping {model_name} at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state["state_dict"])

    val_out = run_eval(val_loader)
    test_out = run_eval(test_loader)
    plot_training_history(history, model_name)

    return val_out["preds"], val_out["probs"], test_out["preds"], test_out["probs"], best_state, history


# =========================
# Cross validation for classical models
# =========================
def classical_cv_report(models, X, y):
    rows = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    print("\n===== CLASSICAL MODEL CROSS-VALIDATION =====")

    for name, model in models.items():
        fold_metrics = []

        for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
            X_tr, X_va = X[tr], X[va]
            y_tr, y_va = y[tr], y[va]

            sampler = SMOTE(random_state=SEED)
            X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)

            clf = clone(model)
            clf.fit(X_tr_res, y_tr_res)

            pred = clf.predict(X_va)
            prob = clf.predict_proba(X_va)

            fold_metrics.append({
                "model": name,
                "fold": fold,
                "accuracy": accuracy_score(y_va, pred),
                "precision": precision_score(y_va, pred, zero_division=0),
                "recall": recall_score(y_va, pred, zero_division=0),
                "f1": f1_score(y_va, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_va, prob[:, 1]),
                "log_loss": log_loss(y_va, prob),
            })

        fold_df = pd.DataFrame(fold_metrics)
        rows.append(fold_df)
        print(name)
        print(fold_df.drop(columns=["model", "fold"]).mean().round(4))

    cv_df = pd.concat(rows, ignore_index=True)
    cv_df.to_csv(MODEL_DIR / "cv_results.csv", index=False)

    summary = cv_df.groupby("model").agg(["mean", "std"])
    summary.to_csv(MODEL_DIR / "cv_summary.csv")

    return cv_df


# =========================
# Main
# =========================
def main():
    set_seed(SEED)

    df = load_dataset()
    sequences = df["sequence"].tolist()
    y = df["label"].values

    train_idx, val_idx, test_idx = similarity_split_indices(
        sequences,
        y,
        test_size=TEST_SIZE,
        val_size=0.15,
        threshold=0.92,
    )

    leakage_df = leakage_check(sequences, train_idx, val_idx, test_idx)

    seq_train = [sequences[i] for i in train_idx]
    seq_val = [sequences[i] for i in val_idx]
    seq_test = [sequences[i] for i in test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # -------------------------
    # Improved feature extraction
    # -------------------------
    # TF-IDF char 3-6-mer, fit only on train to avoid leakage.
    tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 6),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )

    X_train = tfidf.fit_transform(seq_train)
    X_val = tfidf.transform(seq_val)
    X_test = tfidf.transform(seq_test)

    print(f"\nTF-IDF feature shape: {X_train.shape}")

    # -------------------------
    # Resampling only on train
    # -------------------------
    sampler = ADASYN(random_state=SEED)
    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

    print(f"Train original: {X_train.shape[0]}")
    print(f"Train after ADASYN: {X_train_res.shape[0]}")

    # -------------------------
    # Classical models
    # -------------------------
    classical_models = {
        "LinearSVC_Calibrated": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
            method="sigmoid",
            cv=3,
        ),
        "NuSVC_RBF": NuSVC(
            nu=0.5,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.2,
            min_child_weight=2,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
        "MLP_SVD": Pipeline([
            ("svd", TruncatedSVD(n_components=256, random_state=SEED)),
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(256, 64),
                activation="relu",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=300,
                early_stopping=True,
                random_state=SEED,
            ))
        ]),
    }

    # CV để báo cáo chắc hơn
    cv_df = classical_cv_report(classical_models, X_train, y_train)

    all_metrics = []
    fitted_models = {}
    curve_store = {}
    prob_store = {}

    for name, model in classical_models.items():
        print(f"\n>>> Training final {name}...")
        model.fit(X_train_res, y_train_res)

        val_prob = model.predict_proba(X_val)
        test_prob = model.predict_proba(X_test)

        val_metrics = probability_metrics(y_val, val_prob, f"{name}_VAL")
        test_metrics = evaluate_model(y_test, np.argmax(test_prob, axis=1), test_prob, name)

        all_metrics.append(test_metrics)
        fitted_models[name] = model

        curve_store[name] = {"y_true": y_test, "y_prob": test_prob}
        plot_confusion_matrix(y_test, np.argmax(test_prob, axis=1), name)

        prob_store[name] = {
            "family": "classical",
            "val_prob": val_prob,
            "test_prob": test_prob,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

    # -------------------------
    # Deep learning models
    # -------------------------
    print("\n>>> Training DeepTextCNN improved...")
    cnn_val_pred, cnn_val_prob, cnn_pred, cnn_prob, cnn_ckpt, cnn_history = train_torch_model(
        DeepTextCNN(),
        seq_train,
        y_train,
        seq_val,
        y_val,
        seq_test,
        y_test,
        model_name="DeepTextCNN_Improved",
        epochs=35,
        batch_size=64,
        lr=1e-3,
        patience=7,
    )

    cnn_val_metrics = probability_metrics(y_val, cnn_val_prob, "DeepTextCNN_Improved_VAL")
    cnn_test_metrics = evaluate_model(y_test, cnn_pred, cnn_prob, "DeepTextCNN_Improved")

    all_metrics.append(cnn_test_metrics)
    curve_store["DeepTextCNN_Improved"] = {"y_true": y_test, "y_prob": cnn_prob}
    plot_confusion_matrix(y_test, cnn_pred, "DeepTextCNN_Improved")

    prob_store["DeepTextCNN_Improved"] = {
        "family": "deep",
        "val_prob": cnn_val_prob,
        "test_prob": cnn_prob,
        "val_metrics": cnn_val_metrics,
        "test_metrics": cnn_test_metrics,
    }

    print("\n>>> Training CNNBiLSTM...")
    bilstm_val_pred, bilstm_val_prob, bilstm_pred, bilstm_prob, bilstm_ckpt, bilstm_history = train_torch_model(
        CNNBiLSTM(),
        seq_train,
        y_train,
        seq_val,
        y_val,
        seq_test,
        y_test,
        model_name="CNNBiLSTM",
        epochs=35,
        batch_size=64,
        lr=8e-4,
        patience=7,
    )

    bilstm_val_metrics = probability_metrics(y_val, bilstm_val_prob, "CNNBiLSTM_VAL")
    bilstm_test_metrics = evaluate_model(y_test, bilstm_pred, bilstm_prob, "CNNBiLSTM")

    all_metrics.append(bilstm_test_metrics)
    curve_store["CNNBiLSTM"] = {"y_true": y_test, "y_prob": bilstm_prob}
    plot_confusion_matrix(y_test, bilstm_pred, "CNNBiLSTM")

    prob_store["CNNBiLSTM"] = {
        "family": "deep",
        "val_prob": bilstm_val_prob,
        "test_prob": bilstm_prob,
        "val_metrics": bilstm_val_metrics,
        "test_metrics": bilstm_test_metrics,
    }

    # -------------------------
    # Hybrid fusion + ablation
    # -------------------------
    fusion_df = build_hybrid_fusion_report(prob_store, y_test)
    ablation_df = build_ablation_report(prob_store, y_test)

    best_single_name = max(prob_store, key=lambda name: prob_store[name]["val_metrics"]["roc_auc"])
    best_single_metrics = probability_metrics(
        y_test,
        prob_store[best_single_name]["test_prob"],
        f"BestSingle_{best_single_name}",
    )
    best_single_metrics["components"] = best_single_name
    best_single_metrics["weights"] = "1.000"
    all_metrics.append(best_single_metrics)

    best_fusion_row = fusion_df.iloc[0].to_dict() if len(fusion_df) > 0 else None
    if best_fusion_row is not None:
        print("\n===== HYBRID FUSION =====")
        print(fusion_df.round(4))
        all_metrics.append({
            "model": best_fusion_row["model"],
            "accuracy": best_fusion_row["accuracy"],
            "precision": best_fusion_row["precision"],
            "recall": best_fusion_row["recall"],
            "f1": best_fusion_row["f1"],
            "roc_auc": best_fusion_row["roc_auc"],
            "log_loss": best_fusion_row["log_loss"],
        })

    print("\n===== ABLATION STUDY =====")
    print(ablation_df.round(4))

    # -------------------------
    # Save benchmark + plots
    # -------------------------
    benchmark_df = pd.DataFrame(all_metrics)
    benchmark_df = benchmark_df.sort_values(
        by=["roc_auc", "f1", "recall"],
        ascending=False
    )

    benchmark_path = MODEL_DIR / "benchmark_results.csv"
    benchmark_path_improved = MODEL_DIR / "benchmark_results_improved.csv"
    benchmark_df.to_csv(benchmark_path, index=False)
    benchmark_df.to_csv(benchmark_path_improved, index=False)

    print("\n===== FINAL BENCHMARK =====")
    print(benchmark_df.round(4))
    print(f"\nSaved benchmark to: {benchmark_path}")
    print(f"Saved benchmark copy to: {benchmark_path_improved}")

    plot_metric_bar(benchmark_df)
    plot_roc_curves(curve_store)
    plot_pr_curves(curve_store)

    # Top k-mers for biological interpretation
    if "XGBoost" in fitted_models:
        plot_top_kmers(tfidf, fitted_models["XGBoost"], top_n=30)

    # -------------------------
    # Save artifacts
    # -------------------------
    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer_improved.joblib")
    joblib.dump(tfidf, MODEL_DIR / "tfidf_vectorizer.joblib")

    for name, model in fitted_models.items():
        safe_name = name.replace("/", "_")
        joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")

    if best_single_name in prob_store:
        joblib.dump(fitted_models[best_single_name], MODEL_DIR / "best_single_model.joblib")

    torch.save(cnn_ckpt, MODEL_DIR / "deep_textcnn_improved.pt")
    torch.save(bilstm_ckpt, MODEL_DIR / "cnn_bilstm.pt")

    with open(MODEL_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED,
            "n_splits": N_SPLITS,
            "test_size": TEST_SIZE,
            "val_size": 0.15,
            "similarity_threshold": 0.92,
            "tfidf_ngram_range": [3, 6],
            "deep_models": ["DeepTextCNN_Improved", "CNNBiLSTM"],
            "fusion_models": fusion_df["components"].tolist() if len(fusion_df) > 0 else [],
            "ablation_models": ablation_df["components"].tolist() if len(ablation_df) > 0 else [],
            "figures_dir": str(FIG_DIR),
        }, f, indent=2)

    print("\nSaved all models to model/")
    print("Saved all figures to figures/")
    print("\nImportant output figures:")
    print("- figures/roc_curves.png")
    print("- figures/precision_recall_curves.png")
    print("- figures/bar_accuracy.png, bar_f1.png, bar_roc_auc.png")
    print("- figures/top_kmers_xgboost.png")
    print("- figures/training_curve_DeepTextCNN_Improved.png")
    print("- figures/training_curve_CNNBiLSTM.png")


if __name__ == "__main__":
    main()