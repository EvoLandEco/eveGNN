#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ======================================================
# 0) EDIT ME: Confusion matrix (raw counts; rows=true)
# ======================================================
classes = ["PD", "ED", "NND"]
CM_raw = pd.DataFrame(
    [[1673, 322, 1005],
     [ 752, 654, 1594],
     [ 546, 302, 2152]],
    index=classes, columns=classes
)

# ======================================================
# 1) Build "proper" CM variants (use all of them)
# ======================================================
# Row-normalized: P(pred | true)  -> per-class recall/error view
row_sums = CM_raw.sum(axis=1).replace(0, np.nan)
CM_true = CM_raw.div(row_sums, axis=0).fillna(0.0)

# Column-normalized: P(true | pred) -> per-class precision view
col_sums = CM_raw.sum(axis=0).replace(0, np.nan)
CM_pred = CM_raw.div(col_sums, axis=1).fillna(0.0)

# Global-normalized: fraction of all samples
CM_all = CM_raw / CM_raw.values.sum()

# Symmetric pairwise misclassification rate (distance-like), diag=0
# M[i,j] = (C[i,j] + C[j,i]) / (n_i + n_j), where n_i is row-sum for true class i
n_per_class = CM_raw.sum(axis=1)
M = pd.DataFrame(0.0, index=CM_raw.index, columns=CM_raw.columns)
for i in CM_raw.index:
    for j in CM_raw.columns:
        if i == j:
            continue
        num = float(CM_raw.loc[i, j] + CM_raw.loc[j, i])
        den = float(n_per_class[i] + n_per_class[j])
        M.loc[i, j] = (num / den) if den > 0 else 0.0

# We’ll compare M to JSD and (1-Overlap).

# ======================================================
# 2) Read JSD & Overlap from pairwise CSV (TES)
#    File: /spectral_out_plots/cmp_pairwise_stats_TES.csv
#    Columns: pair, type, stat, value
# ======================================================
PAIRWISE_CSV = "spectral_out_plots/cmp_pairwise_stats_TES.csv"
PAIR_TYPE = "all"  # or "matched"

if not os.path.isfile(PAIRWISE_CSV):
    raise FileNotFoundError(f"Cannot find: {PAIRWISE_CSV}")
dfp = pd.read_csv(PAIRWISE_CSV)

# normalize "PD–ED" / "PD - ED" / "PD — ED"
dash = re.compile(r"\s*[-–—]\s*")
def split_pair(s):
    a, b = dash.split(str(s).strip())
    return a.upper(), b.upper()

# keep requested type if present
if (dfp["type"] == PAIR_TYPE).any():
    dfp = dfp[dfp["type"] == PAIR_TYPE].copy()

# mean value per pair/stat (if multiple rows)
g = (dfp.groupby(["pair", "stat"], dropna=False)["value"]
        .mean().reset_index())

# Build square matrices in the exact CM order (uppercase labels)
C = [c.upper() for c in CM_raw.index]
def pairmeans_to_matrix(gsub: pd.DataFrame, diag_fill: float, classes=C) -> pd.DataFrame:
    mat = pd.DataFrame(np.full((len(classes), len(classes)), np.nan, float),
                       index=classes, columns=classes)
    for _, row in gsub.iterrows():
        a, b = split_pair(row["pair"])
        if a in classes and b in classes and a != b:
            mat.loc[a, b] = row["value"]
            mat.loc[b, a] = row["value"]
    np.fill_diagonal(mat.values, diag_fill)
    return mat

JSD = pairmeans_to_matrix(g[g["stat"].str.upper() == "JSD"],      diag_fill=0.0, classes=C)
OVL = pairmeans_to_matrix(g[g["stat"].str.capitalize() == "Overlap"], diag_fill=1.0, classes=C)

# Align M to uppercase class labels as well
M.index = [c.upper() for c in M.index]
M.columns = [c.upper() for c in M.columns]

# Ensure identical ordering
JSD = JSD.loc[C, C]
OVL = OVL.loc[C, C]
M   = M.loc[C, C]

# Distances for comparison
D_jsd = JSD.copy()        # divergence (0..1)
D_ovl = (1.0 - OVL).copy()  # convert similarity to distance (0..1)

# ======================================================
# 3) Mantel test (Spearman) with a fixed finite mask
#    - Compare off-diagonals of two distance matrices
# ======================================================
def _lower_tri_with_mask(A: pd.DataFrame, B: pd.DataFrame):
    a = A.values.astype(float); b = B.values.astype(float)
    tri = np.tril_indices_from(a, k=-1)
    av, bv = a[tri], b[tri]
    mask = np.isfinite(av) & np.isfinite(bv)
    return av[mask], bv[mask], tri, mask

def mantel_spearman(A: pd.DataFrame, B: pd.DataFrame, n_perm: int = 4999, seed: int = 0):
    rng = np.random.default_rng(seed)
    a, b, tri, mask = _lower_tri_with_mask(A, B)
    if a.size == 0:  # nothing comparable
        return np.nan, np.nan
    r_obs = spearmanr(a, b, nan_policy="omit").correlation

    labels = np.arange(B.shape[0])
    perm_ge = 0
    for _ in range(n_perm):
        p = rng.permutation(labels)
        Bp = B.values[p][:, p]
        bp = Bp[tri][mask]  # apply the SAME mask positions
        r_perm = spearmanr(a, bp, nan_policy="omit").correlation
        if np.isnan(r_perm):
            continue
        if r_perm >= r_obs:
            perm_ge += 1
    pval = (perm_ge + 1) / (n_perm + 1)
    return r_obs, pval

rM_jsd, pM_jsd = mantel_spearman(M, D_jsd, n_perm=4999, seed=42)
rM_ovl, pM_ovl = mantel_spearman(M, D_ovl, n_perm=4999, seed=43)

print(f"Mantel (Spearman)  M vs JSD   : r={rM_jsd:.3f}, p={pM_jsd:.4g}")
print(f"Mantel (Spearman)  M vs 1-OVL : r={rM_ovl:.3f}, p={pM_ovl:.4g}")

# quick off-diagonal Spearman (sanity check, same mask)
def offdiag_spearman(A: pd.DataFrame, B: pd.DataFrame):
    a, b, _, _ = _lower_tri_with_mask(A, B)
    return spearmanr(a, b, nan_policy="omit")
rho_jsd = offdiag_spearman(M, D_jsd)
rho_ovl = offdiag_spearman(M, D_ovl)
print(f"Off-diag Spearman  M vs JSD   : rho={rho_jsd.correlation:.3f}, p={rho_jsd.pvalue:.4g}")
print(f"Off-diag Spearman  M vs 1-OVL : rho={rho_ovl.correlation:.3f}, p={rho_ovl.pvalue:.4g}")

# ======================================================
# 4) Visuals
#    A) CM variants: CM_true, CM_pred, M (pairwise error)
#    B) Spectral distances: JSD, 1-Overlap
#    C) Scatter: M vs JSD / 1-OVL
# ======================================================
# A) CM variants
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=170)
for ax, mat, title in zip(
    axes, [CM_true, CM_pred, M],
    ["P(pred|true) (row-normalized)", "P(true|pred) (col-normalized)", "Pair misclassification rate (M)"]
):
    im = ax.imshow(mat.values, vmin=0, vmax=np.nanmax(mat.values), cmap="viridis")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout(); plt.show()

# B) Spectral distances
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=170)
for ax, mat, title in zip(
    axes, [D_jsd.loc[C, C], D_ovl.loc[C, C]],
    ["JSD (distance)", "1 - Overlap (distance)"]
):
    vmax = max(1e-9, np.nanmax(mat.values))
    im = ax.imshow(mat.values, vmin=0, vmax=vmax, cmap="viridis")
    ax.set_xticks(range(len(C))); ax.set_yticks(range(len(C)))
    ax.set_xticklabels(C, rotation=45, ha="right"); ax.set_yticklabels(C)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout(); plt.show()

# C) Scatter: M vs JSD / 1-OVL
def _lower_tri_flat(df):
    a = df.values; tri = np.tril_indices_from(a, k=-1); return a[tri]
xM = _lower_tri_flat(M.loc[C, C])
xJ = _lower_tri_flat(D_jsd.loc[C, C])
xO = _lower_tri_flat(D_ovl.loc[C, C])
mask = np.isfinite(xM) & np.isfinite(xJ)  # use same finite mask per pair
mask2 = np.isfinite(xM) & np.isfinite(xO)

fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), dpi=170)
axes[0].scatter(xJ[mask], xM[mask], s=30, alpha=0.85)
axes[0].set_xlabel("JSD (off-diagonal)"); axes[0].set_ylabel("Pair misclassification rate")
axes[0].set_title(f"M vs JSD  (ρ={rho_jsd.correlation:.2f}, p={rho_jsd.pvalue:.2g})")

axes[1].scatter(xO[mask2], xM[mask2], s=30, alpha=0.85)
axes[1].set_xlabel("1 - Overlap (off-diagonal)"); axes[1].set_ylabel("Pair misclassification rate")
axes[1].set_title(f"M vs 1-OVL (ρ={rho_ovl.correlation:.2f}, p={rho_ovl.pvalue:.2g})")

fig.tight_layout(); plt.show()
