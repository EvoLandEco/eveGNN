#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RFF → multi-h stacked CNN classifier for PD/ED/NND per view (TES, TAS),
with richer analysis and figures.

Enhancements over the base script:
- Save softmax probabilities, predictions, and validation indices per run.
- Confusion matrices (raw & row-normalized) per view × undersample rate.
- Learning curves (train/val loss vs epoch).
- Per-class precision/recall/F1 (classification_report.csv) and Matthews CC.
- Aggregate tables across runs + summary CSV.
- Merge `summary.csv` params → per-parameter and per-metric performance crossed
  by TES/TAS (numeric params binned by quantiles). Figures: 2x3 grids
  (rows=TES/TAS, cols=PD/ED/NND) showing macro-F1 by parameter bin.
- Consistent, informative filenames.

Inputs
------
- spectral_out/series_rff.csv
  Required cols: tree_id, metric, view, lap, h, lambda, density, t_norm (or height)
- spectral_out/summary.csv
  Used to attach parameters per tree_id for grouped performance.

Outputs (cnn_rff_results/)
--------------------------
- results_summary.csv                      (one row per {view, rate})
- per_run/report_{view}_rate{r}.csv        (classification report dict)
- per_run/preds_{view}_rate{r}.csv         (val rows: tree_id, metric, y_true, y_pred, probs...)
- confusion_matrices/cm_{view}_rate{r}.png           (raw)
- confusion_matrices/cm_norm_{view}_rate{r}.png      (row-normalized)
- learning_curves/loss_{view}_rate{r}.png
- params/perf_param_{param}.csv            (grouped tables)
- params/fig_param_{param}_F1.png          (2x3 grids F1 by bins)
- meta_h_channels_{view}.json
- logs.txt
"""

# ============================== CONFIG =======================================

INPUT_DIR        = "spectral_out"
SERIES_RFF_FILE  = "series_rff.csv"
SUMMARY_FILE     = "summary.csv"
OUTPUT_DIR       = "cnn_rff_results"

LAP_FILTER       = "nMGL"    # Laplacian to use
IMAGE_H          = 128        # λ (rows)
IMAGE_W          = 96        # time (cols)

# Multi-h channel selection
H_CHANNELS_MODE  = "all"     # "all" | "topk"
H_CHANNELS_TOPK  = 6
ROUND_H_DIGITS   = 6

# Train/eval
BATCH_SIZE       = 32
EPOCHS           = 60
LEARNING_RATE    = 1e-4
WEIGHT_DECAY     = 1e-2
VAL_SIZE         = 0.20
RANDOM_STATE     = 1337
UNDERSAMPLE_RATES = [0.6, 0.7, 0.8, 0.9, 1.0]  # relative to max class count

# Grouped-performance settings
PARAM_COLS_CANON = ["lambda", "mu", "beta_n", "beta_phi", "gamma_n", "gamma_phi"]
N_BINS_PER_PARAM = 2          # quantile bins for numeric params
MIN_GROUP_SIZE   = 6          # ignore tiny groups

# =============================================================================

import os
import json
import math
import random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------- Utils -----------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def seed_all(seed=RANDOM_STATE):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def nord_cmap():
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "nord", ["#2E3440","#3B4252","#434C5E","#4C566A","#5E81AC","#81A1C1","#88C0D0","#E5E9F0"], N=256
    )

def load_series_rff(path):
    df = pd.read_csv(path)
    required = {"tree_id", "metric", "view", "lap", "lambda", "density"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if "h" not in df.columns:
        raise ValueError("series_rff.csv must include column 'h' (bandwidth).")
    if ("t_norm" not in df.columns) and ("height" not in df.columns):
        raise ValueError("series_rff.csv requires either 't_norm' or 'height'.")
    return df

def pick_time_col(df):
    return "t_norm" if "t_norm" in df.columns else "height"

def normalize_time_col(df):
    # If only 'height' exists, min-max normalize per tree/view/lap/h for comparability
    if "t_norm" in df.columns:
        df["t_use"] = df["t_norm"].astype(float)
    else:
        df["height"] = df["height"].astype(float)
        df["t_use"] = df.groupby(["tree_id", "metric", "view", "lap", "h"])["height"]\
                        .transform(lambda x: (x - x.min()) / max(1e-12, (x.max() - x.min())))
    return df

def build_global_bins(df_view, n_rows=IMAGE_H, n_cols=IMAGE_W):
    lam = df_view["lambda"].astype(float)
    t   = df_view["t_use"].astype(float)
    lam_min, lam_max = float(lam.min()), float(lam.max())
    t_min, t_max     = float(max(0.0, t.min())), float(t.max())
    lam_edges = np.linspace(lam_min, lam_max + 1e-12, n_rows + 1)
    t_edges   = np.linspace(t_min, t_max + 1e-12, n_cols + 1)
    return lam_edges, t_edges

def image_from_group(g, lam_edges, t_edges):
    """Bin-average densities for one (tree_id, metric, h, view) group into [rows=λ, cols=time]."""
    lam = g["lambda"].astype(float).to_numpy()
    t   = g["t_use"].astype(float).to_numpy()
    d   = g["density"].astype(float).to_numpy()
    j = np.clip(np.digitize(lam, lam_edges) - 1, 0, len(lam_edges) - 2)  # rows (λ)
    i = np.clip(np.digitize(t,   t_edges)   - 1, 0, len(t_edges)   - 2)  # cols (t)
    H, W = len(lam_edges) - 1, len(t_edges) - 1
    S = np.zeros((H, W), dtype=np.float32)
    C = np.zeros((H, W), dtype=np.int32)
    np.add.at(S, (j, i), d)
    np.add.at(C, (j, i), 1)
    C = np.maximum(C, 1)
    return S / C


# -------------------- Dataset builder (multi-h stacked) -----------------------

def choose_h_channels(df_view, mode="all", topk=H_CHANNELS_TOPK, round_digits=ROUND_H_DIGITS):
    # round 'h' for stability and count frequency
    hs = df_view["h"].round(round_digits)
    counts = hs.value_counts()
    if mode == "topk":
        chosen = counts.index[:max(1, min(topk, len(counts)))].tolist()
    else:
        chosen = counts.sort_index().index.tolist()
    return [float(h) for h in chosen]

def build_dataset_multi_h_for_view(df_all, view, lap=LAP_FILTER):
    """
    Return:
      X : (N, C, H, W) float32 images (C = number of selected h channels)
      y : (N,) int64 labels
      labels : class names list (['pd','ed','nnd'] subset present)
      meta  : dict(lam_edges, t_edges, h_channels, metas_df)
    """
    df = df_all[(df_all["view"] == view) & (df_all["lap"] == lap)].copy()
    if df.empty:
        return None, None, None, None
    df = normalize_time_col(df)
    df["h"] = df["h"].astype(float).round(ROUND_H_DIGITS)

    # Choose h channels for this view
    h_channels = choose_h_channels(df, mode=H_CHANNELS_MODE, topk=H_CHANNELS_TOPK, round_digits=ROUND_H_DIGITS)
    if not h_channels:
        return None, None, None, None

    # Build global bins across all selected h
    df_sel = df[df["h"].isin(h_channels)]
    lam_edges, t_edges = build_global_bins(df_sel, IMAGE_H, IMAGE_W)

    # Build one sample per (tree_id, metric); stack per-h images as channels
    samples = []
    for (tree_id, metric), g_tm in df_sel.groupby(["tree_id", "metric"], sort=False):
        chans = []
        for hval in h_channels:
            g = g_tm[g_tm["h"] == hval]
            if g.empty:
                chans.append(np.zeros((IMAGE_H, IMAGE_W), dtype=np.float32))
            else:
                chans.append(image_from_group(g, lam_edges, t_edges))
        img = np.stack(chans, axis=0)  # (C,H,W)
        samples.append((tree_id, metric, img))

    if not samples:
        return None, None, None, None

    labels = sorted({m for _, m, _ in samples})
    label_to_idx = {m: i for i, m in enumerate(labels)}

    X = np.stack([im for _, _, im in samples], axis=0).astype(np.float32)  # (N,C,H,W)
    y = np.array([label_to_idx[m] for _, m, _ in samples], dtype=np.int64)

    metas = pd.DataFrame({"tree_id": [tid for tid, _, _ in samples],
                          "metric":  [m for _, m, _ in samples]})
    meta = dict(lam_edges=lam_edges, t_edges=t_edges, h_channels=h_channels, metas=metas)
    return X, y, labels, meta


# ----------------------------- Torch bits ------------------------------------

class RFFSpectraDS(Dataset):
    def __init__(self, X, y, mean=None, std=None):
        self.X = X
        self.y = y
        if mean is None or std is None:
            mean = X.mean(axis=(0, 2, 3)).astype(np.float32)                 # (C,)
            std  = (X.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)         # (C,)
        self.mean = mean
        self.std  = std

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]                                                      # (C,H,W)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]         # channelwise norm
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

class SmallSpecCNN(nn.Module):
    def __init__(self, in_ch, n_classes=3):
        super().__init__()
        def block(cin, cout, k=3, p=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=k, padding=p, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
        self.feat = nn.Sequential(
            block(in_ch, 16, k=5, p=2),
            block(16, 32, k=3, p=1),
            block(32, 64, k=3, p=1)
        )
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.feat(x)                     # (B,64,H',W')
        x = F.adaptive_avg_pool2d(x, 1)      # (B,64,1,1)
        x = torch.flatten(x, 1)              # (B,64)
        x = self.dropout(x)
        return self.head(x)


# ------------------------- training / eval -----------------------------------

def undersample_indices(y, rate, rng):
    """
    Random undersample majority classes on index array y to keep at most
    floor(rate * max_count) for any class. Minority classes remain unchanged.
    """
    idx_all = np.arange(len(y))
    counts = Counter(y)
    max_c = max(counts.values())
    cap = int(math.floor(rate * max_c))
    keep = []
    for cls, n in counts.items():
        cls_idx = idx_all[y == cls]
        if n > cap:
            keep_idx = rng.choice(cls_idx, size=cap, replace=False)
        else:
            keep_idx = cls_idx
        keep.append(keep_idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return keep

def plot_confmat(cm, labels, title, out_png, normalize=None, cmap="Blues"):
    if normalize == "true":  # row-normalize
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm[np.isnan(cm)] = 0.0
    fig = plt.figure(figsize=(5.2, 4.6), dpi=220)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i,j]:.2f}" if normalize == "true" else f"{int(cm[i,j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def train_one(view, X, y, labels, meta, out_dir, rate):
    """
    Train once (one view, one undersample rate); return result dict and save per-run artifacts.
    """
    ensure_dir(os.path.join(out_dir, "per_run"))
    ensure_dir(os.path.join(out_dir, "confusion_matrices"))
    ensure_dir(os.path.join(out_dir, "learning_curves"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    train_idx, val_idx = next(splitter.split(X, y))

    # undersample on training split
    rng = np.random.default_rng(RANDOM_STATE)
    us_idx_rel = undersample_indices(y[train_idx], rate=rate, rng=rng)
    train_idx = train_idx[us_idx_rel]

    # per-channel normalization from train split
    mean = X[train_idx].mean(axis=(0, 2, 3)).astype(np.float32)           # (C,)
    std  = (X[train_idx].std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)   # (C,)

    ds_train = RFFSpectraDS(X[train_idx], y[train_idx], mean, std)
    ds_val   = RFFSpectraDS(X[val_idx],   y[val_idx],   mean, std)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = SmallSpecCNN(in_ch=X.shape[1], n_classes=len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, verbose=False)

    best = {"loss": float("inf"), "state": None}
    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS+1):
        # train
        model.train()
        run_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            run_loss += float(loss.item()) * xb.size(0)
        train_loss = run_loss / max(1, len(ds_train))

        # validate
        model.eval()
        val_loss = 0.0
        preds, gts, probs = [], [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_loss += float(loss.item()) * xb.size(0)
                preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                gts.append(yb.cpu().numpy())
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
        val_loss /= max(1, len(ds_val))
        scheduler.step(val_loss)  # ReduceLROnPlateau on val loss

        history["epoch"].append(epoch); history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)

        if val_loss < best["loss"]:
            best["loss"] = val_loss
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"[{view}|rate={rate:.1f}] epoch {epoch:02d}/{EPOCHS}  train {train_loss:.4f}  val {val_loss:.4f}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    # final eval on val split
    model.eval()
    preds, gts, probs = [], [], []
    with torch.no_grad():
        for xb, yb in dl_val:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            gts.append(yb.numpy())
    y_true = np.concatenate(gts); y_pred = np.concatenate(preds); y_prob = np.concatenate(probs)

    # metrics
    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro")
    recm = recall_score(y_true, y_pred, average="macro")
    prem = precision_score(y_true, y_pred, average="macro")
    mcc  = matthews_corrcoef(y_true, y_pred)

    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    rep_df = pd.DataFrame(report).transpose()
    rep_df.to_csv(os.path.join(out_dir, "per_run", f"report_{view}_rate{rate:.1f}.csv"))

    # confusion matrices
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    plot_confmat(cm, labels, f"{view.upper()}  rate={rate:.1f}  (raw)",
                 os.path.join(out_dir, "confusion_matrices", f"cm_{view}_rate{rate:.1f}.png"), normalize=None)
    plot_confmat(cm, labels, f"{view.upper()}  rate={rate:.1f}  (row-normalized)",
                 os.path.join(out_dir, "confusion_matrices", f"cm_norm_{view}_rate{rate:.1f}.png"),
                 normalize="true")

    # learning curve
    fig = plt.figure(figsize=(5.0, 4.0), dpi=220)
    ax = fig.add_subplot(111)
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title(f"{view.upper()}  rate={rate:.1f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curves", f"loss_{view}_rate{rate:.1f}.png"), bbox_inches="tight")
    plt.close(fig)

    # save per-row preds for later aggregation/param analysis
    metas = meta["metas"].iloc[val_idx].reset_index(drop=True).copy()
    probs_df = pd.DataFrame(y_prob, columns=[f"p_{lab}" for lab in labels])
    preds_df = pd.DataFrame({
        "view": view,
        "rate": rate,
        "y_true": [labels[i] for i in y_true],
        "y_pred": [labels[i] for i in y_pred]
    })
    out_df = pd.concat([metas.reset_index(drop=True), preds_df, probs_df], axis=1)
    out_df.to_csv(os.path.join(out_dir, "per_run", f"preds_{view}_rate{rate:.1f}.csv"), index=False)

    return {
        "view": view, "rate": rate,
        "acc": acc, "macro_f1": f1m, "macro_rec": recm, "macro_prec": prem, "mcc": mcc,
        "labels": labels, "val_idx": val_idx, "history": history
    }


# -------------------- Per-parameter performance (with summary.csv) -----------

def attach_params(summary_df):
    """Return a (tree_id → params) table with *_par names to avoid collisions."""
    df = summary_df.copy()
    cols = ["tree_id", "metric", "view"] + [c for c in PARAM_COLS_CANON if c in df.columns]
    df = df[cols]
    rename = {c: f"{c}_par" for c in PARAM_COLS_CANON if c in df.columns}
    return df.rename(columns=rename)

def bin_numeric_series(s: pd.Series, k=N_BINS_PER_PARAM):
    s = pd.to_numeric(s, errors="coerce")
    try:
        binned = pd.qcut(s, q=np.linspace(0,1,k+1), duplicates="drop")
    except Exception:
        # fall back: unique values too few; just return the raw value
        return s.astype(str)
    # compact labels
    return binned.astype(str)

def grouped_performance(preds_csvs, summary_df, out_dir):
    """Merge per-run preds with params, compute grouped perf tables & figures."""
    ensure_dir(os.path.join(out_dir, "params"))

    # concat all preds
    preds = pd.concat([pd.read_csv(p) for p in preds_csvs], ignore_index=True)
    params = attach_params(summary_df)

    # join params on (tree_id, metric, view)
    df = preds.merge(params, on=["tree_id", "metric", "view"], how="left")

    # existing labels
    metrics_present = sorted(df["metric"].unique().tolist())
    views_present   = sorted(df["view"].unique().tolist())

    # compute per-parameter grouped perf
    for raw in PARAM_COLS_CANON:
        col = f"{raw}_par"
        if col not in df.columns:
            continue

        # bin numeric parameters
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype.kind in "if":
            df[f"{col}_bin"] = bin_numeric_series(df[col], k=N_BINS_PER_PARAM)
            gkey = f"{col}_bin"
        else:
            gkey = col

        rows = []
        for v in views_present:
            for m in metrics_present:
                sub = df[(df["view"]==v) & (df["metric"]==m)].copy()
                if sub.empty: continue
                for g, gg in sub.groupby(gkey):
                    if len(gg) < MIN_GROUP_SIZE:   # ignore tiny groups
                        continue
                    y_true = gg["y_true"].to_numpy()
                    y_pred = gg["y_pred"].to_numpy()
                    rows.append({
                        "param": col, "group": str(g), "view": v, "metric": m,
                        "n": len(gg),
                        "accuracy": accuracy_score(y_true, y_pred),
                        "macro_f1": f1_score(y_true, y_pred, average="macro"),
                        "macro_rec": recall_score(y_true, y_pred, average="macro"),
                        "macro_prec": precision_score(y_true, y_pred, average="macro"),
                        "mcc": matthews_corrcoef(y_true, y_pred)
                    })
        if not rows:
            continue

        res = pd.DataFrame(rows)
        res.sort_values(["param","group","view","metric"], inplace=True)
        res.to_csv(os.path.join(out_dir, "params", f"perf_param_{raw}.csv"), index=False)

        # ---- Figure: 2 rows (TES/TAS) × 3 cols (PD/ED/NND) bar of macro-F1 by bin ----
        # Collect bins in order of appearance
        bins_order = list(dict.fromkeys(res["group"].tolist()))
        ncols = max(1, len(metrics_present))
        fig_w = 4.8 * ncols; fig_h = 3.6 * max(1, len(views_present))
        fig, axes = plt.subplots(nrows=len(views_present), ncols=ncols,
                                 figsize=(fig_w, fig_h), dpi=220, squeeze=False)
        for ri, v in enumerate(views_present):
            for ci, m in enumerate(metrics_present):
                ax = axes[ri, ci]
                sub = res[(res["view"]==v) & (res["metric"]==m)]
                if sub.empty:
                    ax.axis("off"); continue
                # align to same bin order
                vals = []
                for b in bins_order:
                    row = sub[sub["group"]==b]
                    vals.append(row["macro_f1"].iloc[0] if not row.empty else np.nan)
                ax.bar(np.arange(len(bins_order)), vals)
                ax.set_xticks(np.arange(len(bins_order)))
                ax.set_xticklabels(bins_order, rotation=30, ha="right", fontsize=8)
                ax.set_ylim(0, 1.0)
                ax.set_ylabel("macro-F1" if ci==0 else "")
                ax.set_title(f"{v.upper()} — {m}")
                ax.grid(alpha=0.2, linewidth=0.5)
        fig.suptitle(f"Macro-F1 by {raw} bins", y=0.98, fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "params", f"fig_param_{raw}_F1.png"), bbox_inches="tight")
        plt.close(fig)


# ================================= MAIN ======================================

def main():
    seed_all()
    ensure_dir(OUTPUT_DIR)
    ensure_dir(os.path.join(OUTPUT_DIR, "per_run"))
    logs_path = os.path.join(OUTPUT_DIR, "logs.txt")

    # Load data
    series_rff = load_series_rff(os.path.join(INPUT_DIR, SERIES_RFF_FILE))
    Xy_by_view = {}
    meta_by_view = {}

    for view in ["tes", "tas"]:
        X, y, labels, meta = build_dataset_multi_h_for_view(series_rff, view=view, lap=LAP_FILTER)
        if X is None:
            print(f"[WARN] No data for view={view}")
            continue
        Xy_by_view[view] = (X, y, labels)
        meta_by_view[view] = meta
        # Save meta channels for reproducibility
        ensure_dir(OUTPUT_DIR)
        with open(os.path.join(OUTPUT_DIR, f"meta_h_channels_{view}.json"), "w") as f:
            json.dump({"h_channels": meta["h_channels"]}, f, indent=2)

    if not Xy_by_view:
        raise RuntimeError("No datasets built; check inputs and filters.")

    # Sweep rates and train/eval per view
    rows = []
    per_run_pred_paths = []
    for view, (X, y, labels) in Xy_by_view.items():
        for rate in UNDERSAMPLE_RATES:
            res = train_one(view, X, y, labels, meta_by_view[view], out_dir=OUTPUT_DIR, rate=rate)
            rows.append(res)
            per_run_pred_paths.append(os.path.join(OUTPUT_DIR, "per_run", f"preds_{view}_rate{rate:.1f}.csv"))

    # Summary CSV
    summary = pd.DataFrame(rows, columns=["view","rate","acc","macro_f1","macro_rec","macro_prec","mcc"])
    summary.sort_values(["view","rate"], inplace=True)
    summary.to_csv(os.path.join(OUTPUT_DIR, "results_summary.csv"), index=False)

    # Per-parameter / per-metric performance crossed with TES/TAS
    summary_df = pd.read_csv(os.path.join(INPUT_DIR, SUMMARY_FILE))
    grouped_performance(per_run_pred_paths, summary_df, OUTPUT_DIR)

    # Write text log
    with open(logs_path, "w") as f:
        f.write("Training runs:\n")
        for r in rows:
            f.write(f"{r['view']} rate={r['rate']:.1f}: acc={r['acc']:.4f}, F1m={r['macro_f1']:.4f}, MCC={r['mcc']:.4f}\n")
    print(f"Done. Outputs in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
