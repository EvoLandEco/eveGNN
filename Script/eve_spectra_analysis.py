#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-run unrolled spectra analysis (Matplotlib only):

- Builds surfaces from series_gauss + summary (RFF optional as before).
- Grid comparison figures have unified "cmp_" filename prefix.
- 3D grids (Matplotlib) unify x/y/z ranges across panels.
- Heatmap grids unify color scale per figure; use Nord palette and export linear, sqrt, log2 versions.
- Per-parameter merged comparisons:
  * Variant 1 (overlay): in each panel, overlay two parameter levels; layout = 2 rows (TES/TAS) x 3 cols (PD/ED/NND).
    For heatmaps, Variant 1 shows a *difference* heatmap (levelB - levelA) with symmetric color limits.
  * Variant 2 (per-level): split into several figures; each level gets its own 2x3 grid (TES/TAS x PD/ED/NND),
    for both 3D and heatmaps.

Edit CONFIG below to match your paths/preferences, then run directly.
"""

# ============================== CONFIG =======================================

INPUT_DIR      = "spectral_out"          # where the CSVs live
OUTPUT_DIR     = "spectral_out_plots"    # where figures/tables are saved

GAUSS_FILE     = "series_gauss.csv"      # expected under INPUT_DIR (Gaussian unrolled)
RFF_FILE       = "series_rff.csv"        # optional, processed if present (unchanged behavior)
SUMMARY_FILE   = "summary.csv"           # required

LAP            = "nMGL"                  # Laplacian to visualize

# Comparison figure unified prefix (so Finder groups them together)
CMP_PREFIX     = "cmp_"

# Parameter columns (will exist as *_par after merge) to consider
PARAM_GROUP    = ["lambda_par", "mu_par", "beta_n", "beta_phi", "gamma_n", "gamma_phi"]

# Scalar stats to test/plot (present columns will be used)
DEFAULT_STATS  = [
    "rp_nrm_gap", "rp_nrm_princ", "rp_nrm_asym", "rp_nrm_peak",
    "nMGL_band_low", "nMGL_band_mid", "nMGL_band_high"
]

# Preferred PD/ED/NND column order
PREFERRED_METRIC_ORDER = ["pd", "ed", "nnd"]

# Layout & styling
HEATMAP_MAX_COLS = 5      # try to keep grids wide, not tall
HEATMAP_DPI      = 220
THREED_DPI       = 220
THREED_PANEL_W   = 5.0    # inches per panel
THREED_PANEL_H   = 4.3

# When too many parameter levels, limit Variant-2 to these representative ones (by quantiles)
PARAM_LEVELS_MAX = 4      # e.g., approx [min, 33%, 67%, max]

# =============================== IMPORTS =====================================

import os
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import kruskal
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn_extra.cluster import KMedoids

# =============================== IO & PREP ====================================

PARAM_COLS_CANON = ["lambda", "mu", "beta_n", "beta_phi", "gamma_n", "gamma_phi"]

def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _round_if_numeric_series(s: pd.Series, digits=3) -> pd.Series:
    return s.round(digits) if pd.api.types.is_numeric_dtype(s) else s

def load_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"tree_id", "metric", "view", "lap", "lambda", "density"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if "t_norm" not in df.columns and "height" not in df.columns:
        raise ValueError(f"{path} requires either 't_norm' or 'height'.")
    df["view"]   = df["view"].astype(str).str.strip().str.lower()
    df["metric"] = df["metric"].astype(str).str.strip().str.lower()
    df["lap"]    = df["lap"].astype(str).str.strip()
    return df

def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"tree_id", "metric", "view"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    df["view"]   = df["view"].astype(str).str.strip().str.lower()
    df["metric"] = df["metric"].astype(str).str.strip().str.lower()
    return df

def attach_params_no_collision(series_df: pd.DataFrame,
                               summary_df: pd.DataFrame,
                               extra_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Left-join summary params to series; always rename param names -> *_par.
    Optionally bring along extra columns from summary (e.g., size columns like 'n_nodes').
    """
    present_params = [c for c in PARAM_COLS_CANON if c in summary_df.columns]
    rename_map = {c: f"{c}_par" for c in present_params}

    keys = ["tree_id", "metric", "view"]
    include = present_params[:]

    # bring any requested extra columns (as-is)
    if extra_cols:
        include.extend([c for c in extra_cols if c in summary_df.columns])

    cols = [c for c in keys + include if c in summary_df.columns]
    params = summary_df.loc[:, cols].rename(columns=rename_map)
    return series_df.merge(params, on=["tree_id", "metric", "view"], how="left")


def pick_time_col(df: pd.DataFrame) -> str:
    return "t_norm" if "t_norm" in df.columns else "height"

def make_treatment_labels(df: pd.DataFrame, group_cols: List[str], digits: int = 3) -> pd.Series:
    cols = [c for c in group_cols if c in df.columns]
    if not cols:
        cols = ["metric"] if "metric" in df.columns else []
    if not cols:
        return pd.Series(["(all)"] * len(df), index=df.index)
    parts = []
    for g in cols:
        s = df[g]
        s = _round_if_numeric_series(s, digits).astype(str)
        parts.append(s)
    lab = parts[0]
    for i in range(1, len(parts)):
        lab = lab + " | " + parts[i]
    return lab.rename("treatment")


# ========================= SURFACE DATA CONSTRUCTION ==========================

def build_unrolled_surface_data(
    series_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    view: str,
    lap: str,
    group_cols: List[str],
    mode: str = "gauss",
    h: Optional[float] = None,
    digits: int = 6
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns dict[treatment] -> {"x": time_vals, "y": lambda_vals, "z": Z_matrix, "tvar": tcol}
    """
    df = series_df.copy()
    df = df[(df["view"] == view) & (df["lap"] == lap)].copy()
    if df.empty:
        return {}

    if mode.lower() == "rff":
        if "h" not in df.columns:
            raise ValueError("RFF mode requires 'h' column in the series.")
        if h is None:
            raise ValueError("RFF mode requires a bandwidth 'h'.")
        df = df[np.isclose(df["h"].astype(float), float(h))].copy()
        if df.empty:
            return {}

    df = attach_params_no_collision(df, summary_df)
    df["treatment"] = make_treatment_labels(df, group_cols)

    tcol = pick_time_col(df)
    df[tcol] = df[tcol].astype(float).round(digits)
    df["lambda"] = df["lambda"].astype(float).round(digits)
    df["density"] = df["density"].astype(float)

    g = df.groupby(["treatment", tcol, "lambda"], as_index=False)["density"].mean()

    surfaces: Dict[str, Dict[str, np.ndarray]] = {}
    for tr, sub in g.groupby("treatment"):
        sub = sub.sort_values(["lambda", tcol])
        pivot = sub.pivot(index="lambda", columns=tcol, values="density").sort_index()
        y_vals = pivot.index.to_numpy()                   # lambda axis
        x_vals = pivot.columns.to_numpy().astype(float)  # time axis
        Z = pivot.to_numpy()
        surfaces[str(tr)] = {"x": x_vals, "y": y_vals, "z": Z, "tvar": tcol}
    return surfaces

def build_surface_for_param_level(
    series_df: pd.DataFrame, summary_df: pd.DataFrame, *,
    view: str, lap: str, metric: str,
    param_name: str, level: float, tol: float = 1e-6
) -> Optional[Dict[str, np.ndarray]]:
    """One surface for a specific (view, metric, param==level)."""
    df = series_df[(series_df["view"] == view) & (series_df["lap"] == lap) & (series_df["metric"] == metric)].copy()
    if df.empty:
        return None
    df = attach_params_no_collision(df, summary_df)
    col = param_name if param_name.endswith("_par") else f"{param_name}_par"
    if col not in df.columns:
        return None
    mask = np.isclose(df[col].astype(float), float(level), rtol=1e-5, atol=tol)
    df = df[mask]
    if df.empty:
        return None
    # aggregate across trees onto a shared grid
    tcol = pick_time_col(df)
    df[tcol] = df[tcol].astype(float).round(6)
    df["lambda"] = df["lambda"].astype(float).round(6)
    df["density"] = df["density"].astype(float)
    g = df.groupby([tcol, "lambda"], as_index=False)["density"].mean()
    g = g.sort_values(["lambda", tcol])
    pivot = g.pivot(index="lambda", columns=tcol, values="density").sort_index()
    return {"x": pivot.columns.to_numpy().astype(float),
            "y": pivot.index.to_numpy().astype(float),
            "z": pivot.to_numpy(), "tvar": tcol}

# =============================== COLORMAPS ====================================

def nord_colormap() -> LinearSegmentedColormap:
    hex_list = ["#2E3440", "#3B4252", "#434C5E", "#4C566A",
                "#5E81AC", "#81A1C1", "#88C0D0", "#E5E9F0"]
    return LinearSegmentedColormap.from_list("nord_smooth", hex_list, N=256)

# =============================== HELPERS ======================================

def _mesh(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(x, y)
    return X, Y

def _transform_Z(Z: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    if mode == "sqrt":
        return np.sqrt(np.clip(Z, 0.0, None)), "sqrt density"
    if mode == "log2":
        return np.log2(np.clip(Z, 1e-12, None)), "log2 density"
    return Z, "density"

def _heatmap_vrange(surfaces: List[Dict[str, np.ndarray]], transform: str) -> Tuple[float,float]:
    vals = []
    for s in surfaces:
        if s is None or s["z"].size == 0: continue
        Z, _ = _transform_Z(s["z"].astype(float), transform)
        Z = Z[np.isfinite(Z)]
        if Z.size: vals.append((Z.min(), Z.max()))
    if not vals: return (0.0, 1.0)
    vmin = min(v[0] for v in vals); vmax = max(v[1] for v in vals)
    return float(vmin), float(vmax)

def _union_axes_and_regrid(a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Return union grids X,Y and regridded Za, Zb onto union."""
    xa, ya, Za = a["x"], a["y"], a["z"]
    xb, yb, Zb = b["x"], b["y"], b["z"]
    x_union = np.unique(np.concatenate([xa, xb]))
    y_union = np.unique(np.concatenate([ya, yb]))
    # regrid via pandas (index=y, columns=x)
    A = pd.DataFrame(Za, index=ya, columns=xa).reindex(index=y_union, columns=x_union)
    B = pd.DataFrame(Zb, index=yb, columns=xb).reindex(index=y_union, columns=x_union)
    return x_union, y_union, A.to_numpy(), B.to_numpy()

def _choose_param_levels(frame: pd.DataFrame, param_raw: str, max_levels: int = PARAM_LEVELS_MAX) -> List[float]:
    # accept either 'mu' or 'mu_par' (and same for others)
    candidates = [f"{param_raw}_par", param_raw]
    col = next((c for c in candidates if c in frame.columns), None)
    if col is None:
        print(f"[levels] Missing column for param='{param_raw}' (tried {candidates}).")
        return []
    s = pd.to_numeric(frame[col], errors="coerce").dropna()
    nunq = s.nunique(dropna=True)
    print(f"[levels] {col}: {nunq} unique values.")
    if nunq < 2:
        return []
    # If few levels, use them; else pick representative quantiles
    if nunq <= max_levels:
        return sorted(map(float, s.unique().tolist()))
    qs = np.linspace(0.0, 1.0, max_levels)
    return sorted({float(s.quantile(q)) for q in qs})

def _global_xyz_limits(surfaces):
    """Compute unified x/y/z limits over a list of surface dicts (each with x,y,z)."""
    xs, ys, zs = [], [], []
    for s in surfaces:
        if not s or "x" not in s or "y" not in s or "z" not in s:
            continue
        x = np.asarray(s["x"], dtype=float)
        y = np.asarray(s["y"], dtype=float)
        z = np.asarray(s["z"], dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        z = z[np.isfinite(z)]
        if x.size: xs.append((x.min(), x.max()))
        if y.size: ys.append((y.min(), y.max()))
        if z.size: zs.append((max(0.0, z.min()), z.max()))
    if not xs or not ys or not zs:
        # sensible fallback if nothing valid
        return (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
    xlim = (min(a for a, b in xs), max(b for a, b in xs))
    ylim = (min(a for a, b in ys), max(b for a, b in ys))
    zlim = (min(a for a, b in zs), max(b for a, b in zs))
    return xlim, ylim, zlim
def _metric_order(metrics: List[str]) -> List[str]:
    pref = [m for m in PREFERRED_METRIC_ORDER if m in metrics]
    rest = [m for m in sorted(metrics) if m not in pref]
    return pref + rest

def _nan_diagnostics_and_clean(summary_df: pd.DataFrame,
                               view: str,
                               stat_cols: List[str],
                               outdir: str,
                               prefix: str = "cmp_",
                               require_metrics: Tuple[str, ...] = ("pd","ed","nnd"),
                               min_rows_after_drop: int = 3
                               ) -> Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]]:
    """
    Build a clean feature table for PCA/Clustering:
      • restrict to one view and selected metrics
      • replace ±inf -> NaN
      • print column-wise NaN counts; save offending rows to CSV
      • drop all-NaN columns; drop zero-variance columns
      • drop rows with any NaN in remaining stats
    Returns (df_clean, X, y, used_stats) or None if not enough data remains.
    """
    dfv = summary_df[summary_df["view"] == view].copy()
    if require_metrics:
        dfv = dfv[dfv["metric"].isin(require_metrics)].copy()
    if dfv.empty:
        print(f"[PCA] {view.upper()} — no rows for required metrics.")
        return None

    # choose stats present
    used = [c for c in stat_cols if c in dfv.columns]
    if not used:
        print(f"[PCA] {view.upper()} — none of requested stat columns present.")
        return None

    # replace infs, then report NaNs per column
    dfv[used] = dfv[used].replace([np.inf, -np.inf], np.nan)
    na_counts = dfv[used].isna().sum().sort_values(ascending=False)
    has_na = na_counts[na_counts > 0]
    if not has_na.empty:
        print(f"[PCA] {view.upper()} NaN count by column:\n{has_na.to_string()}")
        bad = dfv.loc[dfv[used].isna().any(axis=1), ["tree_id","metric"] + used]
        bad_path = os.path.join(outdir, f"{prefix}nan_rows_{view.upper()}.csv")
        bad.to_csv(bad_path, index=False)
        print(f"[PCA] {view.upper()} — rows with NaNs saved → {bad_path}")

    # drop all-NaN columns outright
    all_na_cols = [c for c in used if dfv[c].isna().all()]
    if all_na_cols:
        print(f"[PCA] {view.upper()} — dropping all-NaN columns: {all_na_cols}")
    used = [c for c in used if c not in all_na_cols]

    # drop zero-variance columns (computed ignoring NaN)
    stds = dfv[used].std(skipna=True)
    zero_var_cols = stds[stds == 0].index.tolist()
    if zero_var_cols:
        print(f"[PCA] {view.upper()} — dropping zero-variance columns: {zero_var_cols}")
    used = [c for c in used if c not in zero_var_cols]

    if not used:
        print(f"[PCA] {view.upper()} — no usable stats after NA/variance filtering.")
        return None

    # drop rows with any NaN in the remaining stats
    before = len(dfv)
    dfv = dfv.dropna(subset=used + ["tree_id", "metric"]).copy()
    after = len(dfv)
    dropped = before - after
    print(f"[PCA] {view.upper()} — dropped {dropped} row(s) for NaN/∞; kept {after} rows; features={len(used)}")

    if after < min_rows_after_drop:
        print(f"[PCA] {view.upper()} — not enough rows after filtering (n={after}).")
        return None

    X = dfv[used].to_numpy(dtype=float)
    y = dfv["metric"].astype(str).to_numpy()
    # final sanity: ensure_all_finite (sklearn-like)
    if not np.isfinite(X).all():
        # find and print offending locations
        mask = ~np.isfinite(X)
        rows, cols = np.where(mask)
        bad_coords = list(zip(rows[:10], [used[j] for j in cols[:10]]))  # first few
        print(f"[PCA] {view.upper()} — non-finite values still present at (row, col): {bad_coords} (showing up to 10).")
        return None

    return dfv.loc[:, ["tree_id","metric"] + used], X, y, used

def run_pca_and_kmedoids(summary_df: pd.DataFrame,
                         outdir: str,
                         stats_cols: List[str],
                         prefix: str = "cmp_"):
    """
    PCA scatter (PC1–PC2) + K-medoids (k=3) with internal & external metrics, per view.
    Strict NA policy: drop rows/cols with NaN/∞; drop zero-variance features.
    """
    _ensure_dir(outdir)
    for view in ["tes", "tas"]:
        cleaned = _nan_diagnostics_and_clean(summary_df, view, stats_cols, outdir, prefix)
        if cleaned is None:
            print(f"[PCA] Skipping {view.upper()} — cleaning left no usable data.")
            continue
        dfv, X, y, used = cleaned

        # need at least k rows and ≥2 classes
        n_samples = X.shape[0]; n_classes = len(np.unique(y))
        if n_samples < 3:
            print(f"[PCA] Skipping {view.upper()} — only {n_samples} sample(s) after cleaning.")
            continue
        if n_classes < 2:
            print(f"[PCA] Skipping {view.upper()} — only {n_classes} class in y.")
            continue

        # Standardize (NaN-free, zero-variance-free now)
        scaler = StandardScaler()
        Xz = scaler.fit_transform(X)

        # PCA: keep up to 10 PCs; plot first two
        pca = PCA(n_components=min(10, Xz.shape[1]), random_state=42)
        pca.fit(Xz)
        _plot_pca_scatter(
            Xz, y, pca,
            out_png=os.path.join(outdir, f"{prefix}pca_scatter_{view.upper()}.png"),
            title=f"PCA of summary stats — {view.upper()} (features: {len(used)})"
        )

        # K-medoids (k=3)
        kmed = _fit_kmedoids(Xz, k=3, random_state=42)
        labels = kmed.labels_

        rep = _cluster_reports(Xz, y, labels)
        pd.DataFrame([rep]).to_csv(os.path.join(outdir, f"{prefix}kmedoids_metrics_{view.upper()}.csv"), index=False)
        _save_confusion_and_medoids(
            dfv=dfv.reset_index(drop=True),
            y_true=y, labels=labels, kmed=kmed,
            out_prefix=os.path.join(outdir, f"{prefix}kmedoids_{view.upper()}")
        )


# -------- helpers: detect a size column & filter one parameter setting --------
def _detect_size_col(summary_df: pd.DataFrame) -> str:
    # look up directly in summary_df (not a merged/augmented table)
    preferred = [
        "n_nodes", "num_nodes", "n_tips", "num_tips", "tips",
        "n_leaves", "nleaves", "n_taxa", "ntaxa",
        "size", "tree_size", "n"
    ]
    lower_map = {c.lower(): c for c in summary_df.columns}
    for name in preferred:
        if name in lower_map:
            return lower_map[name]
    # fallback: any column name containing these tokens
    for c in summary_df.columns:
        lc = c.lower()
        if any(tok in lc for tok in ["nodes", "tips", "leaves", "size"]):
            return c
    raise ValueError("No tree-size column found; add one (e.g., 'n_nodes').")

def _filter_param_combo(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # Keep a single parameter setting (all params fixed) for this metric.
    # Choose the **most frequent** combination.
    par_cols = [c for c in df.columns if c.endswith("_par")]
    sub = df[df["metric"] == metric].copy()
    if not par_cols:
        return sub
    grp = sub.groupby(par_cols, dropna=False, as_index=False).size().sort_values("size", ascending=False)
    top = grp.iloc[0][par_cols].to_dict()
    m = np.logical_and.reduce([np.isclose(sub[c].astype(float), float(top[c])) if pd.api.types.is_numeric_dtype(sub[c]) else (sub[c]==top[c]) for c in par_cols])
    return sub[m]

# -------- size-binned mean surfaces: small vs large, by metric & view --------
def _surfaces_by_size_bins(series_df,
                           summary_df,
                           metric: str,
                           view: str,
                           lap: str,
                           q_low: float = 0.20,
                           q_high: float = 0.80,
                           min_trees: int = 2,
                           digits: int = 6):
    """
    Build mean unrolled spectra surfaces for 'small' vs 'large' trees at a *fixed*
    parameter setting, holding (metric, view, lap) constant.

    Returns
    -------
    (surf_small, surf_large, (q_low_val, q_high_val), size_col)  where each surface is:
        {
          "x": <1D np.ndarray of time grid>,
          "y": <1D np.ndarray of lambda grid>,
          "z": <2D np.ndarray [len(y) x len(x)] of mean density>,
          "tvar": <"t_norm" or "height">
        }
    or None if not enough data to form two non-overlapping size bins.

    Notes
    -----
    * Parameter setting is chosen as the combo with the **largest number of unique trees**,
      not the most rows (more robust for binning by size).
    * Small/large bins are formed on per-tree size (disjoint tree_id sets).
    * Each surface equals the **mean across trees** of per-tree mean densities on the (t, λ) grid.
    """

    # ---- 0) basic filtering by view/lap/metric ----
    df = series_df.copy()
    df = df[(df["view"] == view) & (df["lap"] == lap) & (df["metric"] == metric)].copy()
    if df.empty:
        return None

    # ---- 1) detect a size column in summary_df ----
    pref_names = [
        "n_nodes", "num_nodes", "n_tips", "num_tips", "tips",
        "n_leaves", "nleaves", "n_taxa", "ntaxa", "size", "tree_size", "n"
    ]
    name_map = {c.lower(): c for c in summary_df.columns}
    size_col = None
    for nm in pref_names:
        if nm in name_map:
            size_col = name_map[nm]
            break
    if size_col is None:
        # fallback: any column containing size-like tokens
        for c in summary_df.columns:
            lc = c.lower()
            if any(tok in lc for tok in ["nodes", "tips", "leaves", "size", "ntaxa", "taxa"]):
                size_col = c
                break
    if size_col is None:
        return None  # no size info → cannot do size effect

    # ---- 2) merge parameters + size into the series (avoid 'lambda' collision) ----
    PARAM_CANON = ["lambda", "mu", "beta_n", "beta_phi", "gamma_n", "gamma_phi"]
    present_params = [c for c in PARAM_CANON if c in summary_df.columns]
    rename_map = {c: f"{c}_par" for c in present_params}
    # keep keys + params + size
    keys = ["tree_id", "metric", "view"]
    keep_cols = [c for c in keys + present_params + [size_col] if c in summary_df.columns]
    params = summary_df.loc[:, keep_cols].rename(columns=rename_map)

    df = df.merge(params, on=["tree_id", "metric", "view"], how="left")

    # ---- 3) fix to one parameter combination with MAX unique trees ----
    par_cols = [c for c in df.columns if c.endswith("_par")]
    if par_cols:
        # count unique trees per param combo
        counts = (df.groupby(par_cols)["tree_id"].nunique()
                    .sort_values(ascending=False))
        if len(counts) == 0:
            return None
        top_combo = counts.index[0]  # tuple of values or scalar if one col
        # build mask
        if isinstance(top_combo, tuple):
            mask = np.ones(len(df), dtype=bool)
            for col, val in zip(par_cols, top_combo):
                if pd.api.types.is_numeric_dtype(df[col]):
                    mask &= np.isclose(df[col].astype(float), float(val))
                else:
                    mask &= (df[col] == val)
        else:
            col = par_cols[0]; val = top_combo
            if pd.api.types.is_numeric_dtype(df[col]):
                mask = np.isclose(df[col].astype(float), float(val))
            else:
                mask = (df[col] == val)
        df = df[mask].copy()
        if df["tree_id"].nunique() < 2:
            return None  # not enough trees for size split

    # ---- 4) build disjoint small/large tree_id sets on per-tree sizes ----
    # per-tree size in this filtered subset
    per_tree = (df.groupby("tree_id", as_index=False)[size_col].first())
    if per_tree.empty or per_tree["tree_id"].nunique() < 2:
        return None

    ql_val = per_tree[size_col].quantile(q_low)
    qh_val = per_tree[size_col].quantile(q_high)

    # Disjoint sets by default (<= ql) vs (>= qh); if ql == qh, fall back to ranks
    if ql_val < qh_val:
        small_ids = per_tree.loc[per_tree[size_col] <= ql_val, "tree_id"].unique()
        large_ids = per_tree.loc[per_tree[size_col] >= qh_val, "tree_id"].unique()
    else:
        # fallback: strict bottom/top by rank to guarantee separation
        ordered = per_tree.sort_values(size_col)
        k = max(min_trees, int(np.floor(0.25 * len(ordered))))
        k = max(k, 1)
        small_ids = ordered.head(k)["tree_id"].unique()
        large_ids = ordered.tail(k)["tree_id"].unique()

    # ensure minimum counts and no overlap
    small_ids = np.setdiff1d(small_ids, large_ids, assume_unique=False)
    if len(small_ids) < min_trees or len(large_ids) < min_trees:
        return None

    small = df[df["tree_id"].isin(small_ids)].copy()
    large = df[df["tree_id"].isin(large_ids)].copy()
    if small.empty or large.empty:
        return None

    # ---- 5) helper: build mean surface with equal tree weight ----
    def _make_surface(block: pd.DataFrame):
        if block.empty:
            return None
        # choose time column
        tcol = "t_norm" if "t_norm" in block.columns else ("height" if "height" in block.columns else None)
        if tcol is None:
            return None

        # stabilize numeric keys
        block[tcol]   = block[tcol].astype(float).round(digits)
        block["lambda"]  = block["lambda"].astype(float).round(digits)
        block["density"] = block["density"].astype(float)

        # per-tree mean at each (t, λ) so each tree contributes equally
        g_tree = (block.groupby(["tree_id", tcol, "lambda"], as_index=False)["density"]
                        .mean())
        g_mean = (g_tree.groupby([tcol, "lambda"], as_index=False)["density"]
                        .mean()
                        .sort_values(["lambda", tcol]))

        pivot = g_mean.pivot(index="lambda", columns=tcol, values="density").sort_index()
        if pivot.size == 0:
            return None
        x = pivot.columns.to_numpy(dtype=float)   # time
        y = pivot.index.to_numpy(dtype=float)     # lambda
        Z = pivot.to_numpy(dtype=float)           # density
        return {"x": x, "y": y, "z": Z, "tvar": tcol}

    surf_small = _make_surface(small)
    surf_large = _make_surface(large)

    if surf_small is None or surf_large is None:
        return None

    return surf_small, surf_large, (float(ql_val), float(qh_val)), size_col

def _global_union_grid_for_view(series_df: pd.DataFrame, view: str, lap: str, digits: int = 6):
    df = series_df[(series_df["view"] == view) & (series_df["lap"] == lap)]
    if df.empty:
        return None, None, None
    tcol = pick_time_col(df)
    x_union = np.unique(df[tcol].astype(float).round(digits).to_numpy())
    y_union = np.unique(df["lambda"].astype(float).round(digits).to_numpy())
    return tcol, x_union, y_union

def _tree_surface_vector(df_metric_view: pd.DataFrame,
                         tree_id: str,
                         tcol: str,
                         x_union: np.ndarray,
                         y_union: np.ndarray,
                         digits: int = 6) -> Optional[np.ndarray]:
    """Return a normalized 1D vector (flattened Z) on union grid for one tree."""
    sub = df_metric_view[df_metric_view["tree_id"] == tree_id].copy()
    if sub.empty:
        return None
    sub[tcol]   = sub[tcol].astype(float).round(digits)
    sub["lambda"]  = sub["lambda"].astype(float).round(digits)
    sub["density"] = sub["density"].astype(float)

    # Per-tree mean at each (t, λ)
    g = (sub.groupby([tcol, "lambda"], as_index=False)["density"].mean()
              .sort_values(["lambda", tcol]))
    piv = g.pivot(index="lambda", columns=tcol, values="density")
    piv = piv.reindex(index=y_union, columns=x_union)  # regrid
    Z = np.nan_to_num(piv.to_numpy(dtype=float), nan=0.0)
    v = Z.ravel(order="C")
    s = v.sum()
    if s <= 0:
        return None
    return (v / s).astype(float)  # probability vector (for JSD/overlap)

def _collect_vectors_by_metric(series_df: pd.DataFrame, view: str, lap: str):
    """
    For a fixed (view, lap), build P(normalized) vectors per tree for each metric.
    Returns: dict metric -> (X: n_trees x n_cells, tree_ids: list[str])
    """
    tcol, x_union, y_union = _global_union_grid_for_view(series_df, view, lap)
    if tcol is None:
        return {}

    out = {}
    for m in _metric_order(sorted(series_df["metric"].unique().tolist())):
        dfm = series_df[(series_df["view"] == view) & (series_df["lap"] == lap) & (series_df["metric"] == m)]
        if dfm.empty:
            continue
        tids = sorted(dfm["tree_id"].unique().tolist())
        vecs = []
        keep_ids = []
        for tid in tids:
            v = _tree_surface_vector(dfm, tid, tcol, x_union, y_union)
            if v is not None:
                vecs.append(v); keep_ids.append(tid)
        if vecs:
            out[m] = (np.vstack(vecs), keep_ids, x_union, y_union, tcol)
    return out

# ---------- divergences / overlaps on probability vectors ----------
def _jsd_base2(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence in base 2 (0–1)."""
    p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    def _kl(x, y):
        return np.sum(x * (np.log2(x) - np.log2(y)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def _overlap_integral(p: np.ndarray, q: np.ndarray) -> float:
    """Integral of min(p, q) on the grid (0–1)."""
    s1 = p.sum(); s2 = q.sum()
    if s1 <= 0 or s2 <= 0:
        return np.nan
    return float(np.minimum(p / s1, q / s2).sum())

# -------- run all size-effect figures (overlay 3D + Δ heatmaps + boxplots) ---
def run_size_effects(series_df, summary_df, lap="nMGL", outdir="spectral_out_plots", prefix="cmp_"):
    _ensure_dir(outdir)
    metrics = _metric_order(sorted(series_df["metric"].unique().tolist()))
    print(f"[size] metrics found in series: {metrics}")

    # Build per-metric small/large surfaces ONCE (both TES and TAS)
    tes_pairs, tas_pairs = {}, {}
    for m in metrics:
        S_tes = _surfaces_by_size_bins(series_df, summary_df, m, "tes", lap)
        print(f"[size] TES {m}: {'ok' if S_tes else 'skip'}")
        S_tas = _surfaces_by_size_bins(series_df, summary_df, m, "tas", lap)
        print(f"[size] TAS {m}: {'ok' if S_tas else 'skip'}")
        if S_tes:
            s_small, s_large, (ql, qh), size_col = S_tes
            tes_pairs[m] = {"A": s_small, "B": s_large}
        if S_tas:
            s_small, s_large, (ql, qh), size_col = S_tas
            tas_pairs[m] = {"A": s_small, "B": s_large}

    print(f"[size] TES pairs: {list(tes_pairs.keys())}")
    print(f"[size] TAS pairs: {list(tas_pairs.keys())}")

    if not tes_pairs and not tas_pairs:
        print(
            "[size] No valid small/large surfaces to compare — likely too few trees or no size variation at the chosen parameter combo.")
        return  # nothing to plot

    la = "small (Q1)"; lb = "large (Q4)"

    # 3D overlay (both rows: TES & TAS)
    out3d = os.path.join(outdir, f"{prefix}size_overlay3D_TES_TAS.png")
    grid_3d_two_rows_overlay(levelA_name=la, levelB_name=lb,
                             tes_surfs=tes_pairs, tas_surfs=tas_pairs,
                             out_png=out3d,
                             main_title="Size effect — overlay (small vs large)")

    # Δ heatmaps (Large − Small), with per-figure symmetric color limits
    tes_pairmaps = {m:(p["A"], p["B"]) for m,p in tes_pairs.items() if p.get("A") and p.get("B")}
    tas_pairmaps = {m:(p["A"], p["B"]) for m,p in tas_pairs.items() if p.get("A") and p.get("B")}

    grid_heatmap_two_rows_diff(
        levelA_name=la, levelB_name=lb,
        tes_pairs=tes_pairmaps, tas_pairs=tas_pairmaps,
        out_linear=os.path.join(outdir, f"{prefix}size_diff_heatmap_linear_TES_TAS.png"),
        out_sqrt=  os.path.join(outdir, f"{prefix}size_diff_heatmap_sqrt_TES_TAS.png"),
        out_log2=  os.path.join(outdir, f"{prefix}size_diff_heatmap_log2_TES_TAS.png"),
        main_title="Size effect — Δ heatmaps (large − small)"
    )

from itertools import combinations  # (already imported earlier in your script per my last step)

def _filter_nonempty_for_violin(arrs, labels, min_n=2):
    """Keep only arrays with >= min_n finite points and nonzero variance."""
    kept_data, kept_labels = [], []
    for a, lab in zip(arrs, labels):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if a.size >= min_n and not np.allclose(a, a[0]):  # guard against constant arrays
            kept_data.append(a)
            kept_labels.append(lab)
    return kept_data, kept_labels

def _plot_violin_box(ax, data, labels, ylabel, title):
    """Violin+box plot that gracefully handles 'no data'."""
    if not data:  # nothing to draw
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return False
    parts = ax.violinplot(data, showextrema=False, showmeans=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.35)
    ax.boxplot(data, widths=0.15, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    return True

# =============================== PLOTTING =====================================

def _panel_3d_overlay(ax, surfA, surfB, colorA="C0", colorB="C3", labelA="A", labelB="B"):
    X, Y = _mesh(surfA["x"], surfA["y"])
    ZA = np.array(surfA["z"], dtype=float); ZA[~np.isfinite(ZA)] = np.nan
    ax.plot_surface(X, Y, ZA, color=colorA, alpha=0.85, linewidth=0, shade=False)
    Xb, Yb = _mesh(surfB["x"], surfB["y"])
    ZB = np.array(surfB["z"], dtype=float); ZB[~np.isfinite(ZB)] = np.nan
    ax.plot_surface(Xb, Yb, ZB, color=colorB, alpha=0.55, linewidth=0, shade=False)

def grid_3d_two_rows_overlay(levelA_name, levelB_name,
                             tes_surfs, tas_surfs,
                             out_png, main_title):
    """
    Draw overlay 3D grid: 2 rows (TES, TAS) x N metrics (columns).
    `tes_surfs` and `tas_surfs` must be dicts: metric -> {"A": surfA, "B": surfB},
    where each surf* is a dict with keys x,y,z,tvar (or None).
    """
    # Flatten all A/B surfaces to unify axis limits
    all_surfs = []
    for d in (tes_surfs, tas_surfs):
        for pair in d.values():
            if isinstance(pair, dict):
                for k in ("A", "B"):
                    s = pair.get(k)
                    if s is not None:
                        all_surfs.append(s)
    if not all_surfs:
        return

    xlim, ylim, zlim = _global_xyz_limits(all_surfs)

    # metric order
    metrics = _metric_order(list(set(list(tes_surfs.keys()) + list(tas_surfs.keys()))))
    if not metrics:
        return

    ncols = len(metrics)
    fig_w = THREED_PANEL_W * ncols
    fig_h = THREED_PANEL_H * 2.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=THREED_DPI)

    # Legend
    proxies = [Line2D([0], [0], color="C0", lw=6, alpha=0.85),
               Line2D([0], [0], color="C3", lw=6, alpha=0.55)]
    fig.legend(proxies, [levelA_name, levelB_name], loc="upper right")

    # Pick a tvar label from any available surface
    tvar = next((s.get("tvar", None) for s in all_surfs if s and "tvar" in s), "time")

    # Helper to overlay two surfaces on an axis
    def _overlay(ax, SA, SB):
        if SA is not None and SB is not None:
            _panel_3d_overlay(ax, SA, SB, labelA=levelA_name, labelB=levelB_name)
        elif SA is not None:
            X, Y = _mesh(SA["x"], SA["y"])
            ZA = np.array(SA["z"], dtype=float); ZA[~np.isfinite(ZA)] = np.nan
            ax.plot_surface(X, Y, ZA, color="C0", alpha=0.85, linewidth=0, shade=False)
        elif SB is not None:
            X, Y = _mesh(SB["x"], SB["y"])
            ZB = np.array(SB["z"], dtype=float); ZB[~np.isfinite(ZB)] = np.nan
            ax.plot_surface(X, Y, ZB, color="C3", alpha=0.55, linewidth=0, shade=False)
        else:
            ax.axis("off"); return False
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_xlabel(tvar); ax.set_ylabel(r"$\lambda$"); ax.set_zlabel("density")
        return True

    # TES row
    for j, m in enumerate(metrics, start=1):
        ax = fig.add_subplot(2, ncols, j, projection="3d")
        pair = tes_surfs.get(m, {})
        SA = pair.get("A"); SB = pair.get("B")
        if _overlay(ax, SA, SB):
            ax.set_title(f"{m} — TES")

    # TAS row
    for j, m in enumerate(metrics, start=1):
        ax = fig.add_subplot(2, ncols, ncols + j, projection="3d")
        pair = tas_surfs.get(m, {})
        SA = pair.get("A"); SB = pair.get("B")
        if _overlay(ax, SA, SB):
            ax.set_title(f"{m} — TAS")

    fig.suptitle(main_title, y=0.98, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def grid_heatmap_two_rows_diff(levelA_name: str, levelB_name: str,
                               tes_pairs: Dict[str, Tuple[Dict, Dict]],
                               tas_pairs: Dict[str, Tuple[Dict, Dict]],
                               out_linear: str, out_sqrt: str, out_log2: str,
                               main_title: str):
    """Difference heatmaps (B - A), with symmetric color limits per figure."""
    cmap = nord_colormap()
    metrics = _metric_order(list(set(list(tes_pairs.keys()) + list(tas_pairs.keys()))))
    # helper to draw one transform
    def _draw(transform: str, out_path: str):
        # collect diffs to get symmetric vmax
        diffs = []
        pairs_all = []
        for m in metrics:
            if m in tes_pairs: pairs_all.append(tes_pairs[m])
            if m in tas_pairs: pairs_all.append(tas_pairs[m])
        for A,B in pairs_all:
            if A is None or B is None: continue
            xu, yu, ZA, ZB = _union_axes_and_regrid(A, B)
            ZA_t, _ = _transform_Z(ZA, transform)
            ZB_t, _ = _transform_Z(ZB, transform)
            D = ZB_t - ZA_t
            diffs.append(D[np.isfinite(D)])
        # collect non-empty diffs only
        flat = np.concatenate([d for d in diffs if d.size]) if any(d.size for d in diffs) else np.array([])
        if flat.size:
            vmax = np.percentile(np.abs(flat), 98)  # robust symmetric scale
            vmax = max(vmax, 1e-9)
        else:
            vmax = 1.0
        vmin = -vmax
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

        ncols = min(len(metrics), HEATMAP_MAX_COLS) or 1
        nrows = 2
        # chunk pages if many metrics
        chunks = [metrics[i:i+ncols] for i in range(0, len(metrics), ncols)] or [[]]
        for pi, chunk in enumerate(chunks, start=1):
            fig_w = 4.6 * len(chunk); fig_h = 3.8 * nrows
            fig, axes = plt.subplots(nrows=nrows, ncols=len(chunk),
                                     figsize=(fig_w, fig_h), dpi=HEATMAP_DPI)
            if len(chunk) == 1:
                axes = np.array([[axes[0]], [axes[1]]])
            for ci, m in enumerate(chunk):
                # TES
                ax = axes[0, ci]
                A, B = tes_pairs.get(m, (None, None))
                if A is None or B is None:
                    ax.axis("off")
                else:
                    xu, yu, ZA, ZB = _union_axes_and_regrid(A, B)
                    ZA_t, lab = _transform_Z(ZA, transform)
                    ZB_t, _   = _transform_Z(ZB, transform)
                    D = ZB_t - ZA_t
                    im = ax.imshow(D, aspect="auto", origin="lower",
                                   extent=(xu.min(), xu.max(), yu.min(), yu.max()),
                                   cmap=cmap, norm=norm)
                    ax.set_title(f"{m} — TES ({levelB_name} - {levelA_name})")
                    ax.set_xlabel(A["tvar"]); ax.set_ylabel(r"$\lambda$")
                    ax.grid(True, color=(0,0,0,0.15), linewidth=0.3)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"Δ {lab}")
                # TAS
                ax = axes[1, ci]
                A, B = tas_pairs.get(m, (None, None))
                if A is None or B is None:
                    ax.axis("off")
                else:
                    xu, yu, ZA, ZB = _union_axes_and_regrid(A, B)
                    ZA_t, lab = _transform_Z(ZA, transform)
                    ZB_t, _   = _transform_Z(ZB, transform)
                    D = ZB_t - ZA_t
                    im = ax.imshow(D, aspect="auto", origin="lower",
                                   extent=(xu.min(), xu.max(), yu.min(), yu.max()),
                                   cmap=cmap, norm=norm)
                    ax.set_title(f"{m} — TAS ({levelB_name} - {levelA_name})")
                    ax.set_xlabel(A["tvar"]); ax.set_ylabel(r"$\lambda$")
                    ax.grid(True, color=(0,0,0,0.15), linewidth=0.3)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"Δ {lab}")
            fig.suptitle(main_title + f" — {transform}", y=0.995, fontsize=14)
            fig.tight_layout()
            suffix = "" if len(chunks) == 1 else f"_p{pi}"
            fig.savefig(out_path.replace(".png", f"{suffix}.png"), bbox_inches="tight")
            plt.close(fig)
    _draw("linear", out_linear)
    _draw("sqrt",   out_sqrt)
    _draw("log2",   out_log2)

def grid_3d_two_rows_singlelevel(level_name: str,
                                 tes_surfs: Dict[str, Dict], tas_surfs: Dict[str, Dict],
                                 out_png: str, main_title: str):
    all_surfs = [*tes_surfs.values(), *tas_surfs.values()]
    xlim, ylim, zlim = _global_xyz_limits([s for s in all_surfs if s is not None])
    metrics = _metric_order(list(set(list(tes_surfs.keys()) + list(tas_surfs.keys()))))
    ncols = max(1, len(metrics))
    fig_w = THREED_PANEL_W * ncols
    fig_h = THREED_PANEL_H * 2.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=THREED_DPI)

    # TES
    for j, m in enumerate(metrics, start=1):
        ax = fig.add_subplot(2, ncols, j, projection="3d")
        S = tes_surfs.get(m)
        if S is not None:
            X, Y = _mesh(S["x"], S["y"]); Z = np.array(S["z"], dtype=float); Z[~np.isfinite(Z)] = np.nan
            ax.plot_surface(X, Y, Z, color="C0", alpha=0.9, linewidth=0, shade=False)
        ax.set_title(f"{m} — TES ({level_name})")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_xlabel(S["tvar"] if S is not None else "time"); ax.set_ylabel(r"$\lambda$"); ax.set_zlabel("density")
    # TAS
    for j, m in enumerate(metrics, start=1):
        ax = fig.add_subplot(2, ncols, ncols + j, projection="3d")
        S = tas_surfs.get(m)
        if S is not None:
            X, Y = _mesh(S["x"], S["y"]); Z = np.array(S["z"], dtype=float); Z[~np.isfinite(Z)] = np.nan
            ax.plot_surface(X, Y, Z, color="C0", alpha=0.9, linewidth=0, shade=False)
        ax.set_title(f"{m} — TAS ({level_name})")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_xlabel(S["tvar"] if S is not None else "time"); ax.set_ylabel(r"$\lambda$"); ax.set_zlabel("density")

    fig.suptitle(main_title, y=0.98, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def grid_heatmap_two_rows_singlelevel(level_name: str,
                                      tes_surfs: Dict[str, Dict], tas_surfs: Dict[str, Dict],
                                      out_linear: str, out_sqrt: str, out_log2: str,
                                      main_title: str):
    cmap = nord_colormap()
    metrics = _metric_order(list(set(list(tes_surfs.keys()) + list(tas_surfs.keys()))))

    def _draw(transform: str, out_path: str):
        vmin, vmax = _heatmap_vrange([*tes_surfs.values(), *tas_surfs.values()], transform)
        ncols = min(len(metrics), HEATMAP_MAX_COLS) or 1
        chunks = [metrics[i:i+ncols] for i in range(0, len(metrics), ncols)] or [[]]
        for pi, chunk in enumerate(chunks, start=1):
            fig_w = 4.6 * len(chunk); fig_h = 3.8 * 2
            fig, axes = plt.subplots(nrows=2, ncols=len(chunk),
                                     figsize=(fig_w, fig_h), dpi=HEATMAP_DPI)
            if len(chunk) == 1:
                axes = np.array([[axes[0]], [axes[1]]])
            for ci, m in enumerate(chunk):
                # TES
                ax = axes[0, ci]; S = tes_surfs.get(m)
                if S is None:
                    ax.axis("off")
                else:
                    Zt, lab = _transform_Z(S["z"], transform)
                    im = ax.imshow(Zt, aspect="auto", origin="lower",
                                   extent=(S["x"].min(), S["x"].max(), S["y"].min(), S["y"].max()),
                                   cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(f"{m} — TES ({level_name})")
                    ax.set_xlabel(S["tvar"]); ax.set_ylabel(r"$\lambda$")
                    ax.grid(True, color=(0,0,0,0.15), linewidth=0.3)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)
                # TAS
                ax = axes[1, ci]; S = tas_surfs.get(m)
                if S is None:
                    ax.axis("off")
                else:
                    Zt, lab = _transform_Z(S["z"], transform)
                    im = ax.imshow(Zt, aspect="auto", origin="lower",
                                   extent=(S["x"].min(), S["x"].max(), S["y"].min(), S["y"].max()),
                                   cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(f"{m} — TAS ({level_name})")
                    ax.set_xlabel(S["tvar"]); ax.set_ylabel(r"$\lambda$")
                    ax.grid(True, color=(0,0,0,0.15), linewidth=0.3)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)
            fig.suptitle(main_title + f" — {transform}", y=0.995, fontsize=14)
            fig.tight_layout()
            suffix = "" if len(chunks) == 1 else f"_p{pi}"
            fig.savefig(out_path.replace(".png", f"{suffix}.png"), bbox_inches="tight")
            plt.close(fig)
    _draw("linear", out_linear)
    _draw("sqrt",   out_sqrt)
    _draw("log2",   out_log2)

def grid_heatmap_two_rows_twolevels(levelA_name: str, levelB_name: str,
                                    tes_pairmaps: Dict[str, Tuple[Dict, Dict]],
                                    tas_pairmaps: Dict[str, Tuple[Dict, Dict]],
                                    out_linear: str, out_sqrt: str, out_log2: str,
                                    main_title: str):
    """
    Two-level comparison heatmaps (side-by-side): rows = TES/TAS, columns = metrics × 2
    Uses a unified color scale (vmin/vmax) across all panels per figure.
    `tes_pairmaps`/`tas_pairmaps`: metric -> (surfA, surfB) where each surf has x,y,z,tvar.
    """
    cmap = nord_colormap()
    metrics = _metric_order(list(set(list(tes_pairmaps.keys()) + list(tas_pairmaps.keys()))))

    def _collect_surfaces_all():
        surfs = []
        for m in metrics:
            A, B = tes_pairmaps.get(m, (None, None))
            if A is not None: surfs.append(A)
            if B is not None: surfs.append(B)
            A, B = tas_pairmaps.get(m, (None, None))
            if A is not None: surfs.append(A)
            if B is not None: surfs.append(B)
        return surfs

    def _draw(transform: str, out_path: str):
        # global vmin/vmax across both levels and both views
        vmin, vmax = _heatmap_vrange(_collect_surfaces_all(), transform)
        ncols_metrics = max(1, min(len(metrics), HEATMAP_MAX_COLS))
        chunks = [metrics[i:i+ncols_metrics] for i in range(0, len(metrics), ncols_metrics)]
        for pi, chunk in enumerate(chunks, start=1):
            # each metric uses two subcolumns: [levelA | levelB]
            ncols = 2 * len(chunk)
            fig_w = 4.6 * ncols
            fig_h = 3.8 * 2
            fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(fig_w, fig_h), dpi=HEATMAP_DPI)

            # coerce axes to 2D array even for small grids
            if ncols == 1:
                axes = np.array([[axes[0]], [axes[1]]])

            for ci, m in enumerate(chunk):
                colA, colB = 2*ci, 2*ci + 1

                # TES row
                A, B = tes_pairmaps.get(m, (None, None))
                # A
                ax = axes[0, colA]
                if A is None:
                    ax.axis("off")
                else:
                    ZA_t, lab = _transform_Z(A["z"], transform)
                    im = ax.imshow(ZA_t, aspect="auto", origin="lower",
                                   extent=(A["x"].min(), A["x"].max(), A["y"].min(), A["y"].max()),
                                   cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(f"{m} — TES ({levelA_name})")
                    ax.set_xlabel(A["tvar"]); ax.set_ylabel(r"$\lambda$")
                    ax.grid(True, color=(0,0,0,0.15), linewidth=0.3)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)
                # B
                ax = axes[0, colB]
                if B is None:
                    ax.axis("off")
                else:
                    ZB_t, lab = _transform_Z(B["z"], transform)
                    im = ax.imshow(ZB_t, aspect="auto", origin="lower",
                                   extent=(B["x"].min(), B["x"].max(), B["y"].min(), B["y"].max()),
                                   cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(f"{m} — TES ({levelB_name})")
                    ax.set_xlabel(B["tvar"]); ax.set_ylabel(r"$\lambda$")
                    ax.grid(True, color=(0,0,0,0.15), linewidth=0.3)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)

                # TAS row
                A, B = tas_pairmaps.get(m, (None, None))
                # A
                ax = axes[1, colA]
                if A is None:
                    ax.axis("off")
                else:
                    ZA_t, lab = _transform_Z(A["z"], transform)
                    im = ax.imshow(ZA_t, aspect="auto", origin="lower",
                                   extent=(A["x"].min(), A["x"].max(), A["y"].min(), A["y"].max()),
                                   cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(f"{m} — TAS ({levelA_name})")
                    ax.set_xlabel(A["tvar"]); ax.set_ylabel(r"$\lambda$")
                    ax.grid(True, color=(0,0,0,0.15), linewidth=0.3)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)
                # B
                ax = axes[1, colB]
                if B is None:
                    ax.axis("off")
                else:
                    ZB_t, lab = _transform_Z(B["z"], transform)
                    im = ax.imshow(ZB_t, aspect="auto", origin="lower",
                                   extent=(B["x"].min(), B["x"].max(), B["y"].min(), B["y"].max()),
                                   cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(f"{m} — TAS ({levelB_name})")
                    ax.set_xlabel(B["tvar"]); ax.set_ylabel(r"$\lambda$")
                    ax.grid(True, color=(0,0,0,0.15), linewidth=0.3)
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)

            fig.suptitle(main_title + f" — {transform}", y=0.995, fontsize=14)
            fig.tight_layout()
            suffix = "" if len(chunks) == 1 else f"_p{pi}"
            fig.savefig(out_path.replace(".png", f"{suffix}.png"), bbox_inches="tight")
            plt.close(fig)

    _draw("linear", out_linear)
    _draw("sqrt",   out_sqrt)
    _draw("log2",   out_log2)

# =========================== STATS & VISUALIZATION ============================

def p_adjust_bh(pvals):
    p = np.asarray(pvals, dtype=float)
    m = p.size
    mask = np.isfinite(p)
    n = int(mask.sum())

    adj = np.full(m, np.nan, dtype=float)
    if n == 0:
        return adj.tolist()

    # sort only the finite p-values
    finite_idx = np.where(mask)[0]
    order = np.argsort(p[mask])
    ranked = p[mask][order]

    # reverse cumulative min of p_i * n / i
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev = min(prev, val)
        q[i] = min(prev, 1.0)

    # place adjusted p back to original positions
    adj_idx = finite_idx[order]
    adj[adj_idx] = q
    return adj.tolist()

def run_treatment_tests(summary_df: pd.DataFrame,
                        view: str,
                        treat_cols: List[str],
                        stats_cols: List[str],
                        out_prefix: str):
    df = summary_df.copy()
    df = df[df["view"] == view].copy()
    # rename params to *_par
    for c in PARAM_COLS_CANON:
        if c in df.columns:
            df.rename(columns={c: f"{c}_par"}, inplace=True)

    stats_cols = [c for c in stats_cols if c in df.columns]
    if not stats_cols:
        return

    df["treatment"] = make_treatment_labels(df, treat_cols)

    rows = []
    for stat in stats_cols:
        sub = df[["treatment", stat]].dropna()
        groups = [g[stat].to_numpy(dtype=float) for _, g in sub.groupby("treatment")]
        if len(groups) < 2 or any(len(g) == 0 for g in groups):
            rows.append({"stat": stat, "method": "kruskal", "p": np.nan})
            continue
        pval = kruskal(*groups).pvalue
        rows.append({"stat": stat, "method": "kruskal", "p": float(pval)})

        # violin + box + jitter
        cats = list(dict.fromkeys(sub["treatment"]))
        data = [sub.loc[sub["treatment"] == c, stat].astype(float).to_numpy() for c in cats]
        fig, ax = plt.subplots(figsize=(max(6.0, 1.6*len(cats)), 4.8), dpi=HEATMAP_DPI)
        parts = ax.violinplot(data, showextrema=False, showmeans=False)
        for pc in parts['bodies']:
            pc.set_alpha(0.4)
        ax.boxplot(data, widths=0.15, showfliers=False)
        rng = np.random.default_rng(123)
        for i, vals in enumerate(data, start=1):
            x = i + 0.12*(rng.random(len(vals)) - 0.5)
            ax.scatter(x, vals, s=12, alpha=0.6)
        ax.set_xticks(range(1, len(cats)+1))
        ax.set_xticklabels(cats, rotation=25, ha="right")
        ax.set_ylabel(stat)
        ax.set_title(f"{stat} by treatment — {view}")
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_{stat}_by_treatment_{view}.png", bbox_inches="tight")
        plt.close(fig)

    res = pd.DataFrame(rows)
    res["padj"] = p_adjust_bh(res["p"].tolist())
    res.to_csv(f"{out_prefix}_treatment_effects_{view}.csv", index=False)

def _pair_labels(metrics: List[str]):
    return [f"{a.upper()}–{b.upper()}" for a, b in combinations(metrics, 2)]

def quantify_metric_overlap_and_divergence(series_df: pd.DataFrame,
                                           view: str,
                                           lap: str,
                                           outdir: str,
                                           prefix: str = "cmp_class_"):
    """
    For a fixed (view, lap), quantify between-metric (PD/ED/NND) separability on unrolled spectra:
      - JSD distributions (matched by tree_id intersection and all-by-all)
      - Overlap (∑ min) distributions
      - Mean-JSD and Mean-Overlap heatmaps
      - Save raw tables for downstream tests
    """
    _ensure_dir(outdir)
    vecs = _collect_vectors_by_metric(series_df, view, lap)
    metrics = [m for m in _metric_order(list(vecs.keys())) if m in ("pd","ed","nnd")]
    if len(metrics) < 2:
        return

    rows = []
    # --- build per-pair stats ---
    for a, b in combinations(metrics, 2):
        Xa, ida, *_ = vecs[a]
        Xb, idb, *_ = vecs[b]
        ida = np.array(ida); idb = np.array(idb)
        # matched trees (same tree_id exists for both metrics)
        common = np.intersect1d(ida, idb)
        jsd_matched = []
        ovl_matched = []
        if common.size:
            # build aligned vectors by tree_id
            map_a = {tid:i for i,tid in enumerate(ida)}
            map_b = {tid:i for i,tid in enumerate(idb)}
            for tid in common:
                pa = Xa[map_a[tid]]; qb = Xb[map_b[tid]]
                jsd_matched.append(_jsd_base2(pa, qb))
                ovl_matched.append(_overlap_integral(pa, qb))
        # all-by-all (could be large; sample if needed)
        jsd_all = []
        ovl_all = []
        # keep it light but informative
        max_pairs = 10000
        cnt = 0
        for i in range(Xa.shape[0]):
            # random subset of B to bound cost (take up to K evenly)
            idx = np.linspace(0, Xb.shape[0]-1, num=min(Xb.shape[0], 64), dtype=int)
            for j in idx:
                jsd_all.append(_jsd_base2(Xa[i], Xb[j]))
                ovl_all.append(_overlap_integral(Xa[i], Xb[j]))
                cnt += 1
                if cnt >= max_pairs:
                    break
            if cnt >= max_pairs:
                break

        lab = f"{a.upper()}–{b.upper()}"
        for v in jsd_matched: rows.append({"pair": lab, "type": "matched", "stat": "JSD", "value": v})
        for v in jsd_all:     rows.append({"pair": lab, "type": "all",     "stat": "JSD", "value": v})
        for v in ovl_matched: rows.append({"pair": lab, "type": "matched", "stat": "Overlap", "value": v})
        for v in ovl_all:     rows.append({"pair": lab, "type": "all",     "stat": "Overlap", "value": v})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, f"{prefix}pairwise_stats_{view.upper()}.csv"), index=False)

    # ---- violin/box for JSD & Overlap (matched + all) ----
    for stat in ["JSD", "Overlap"]:
        sub = df[df["stat"] == stat]
        if sub.empty:
            continue
        pairs = _pair_labels(metrics)  # e.g., ['PD–ED','PD–NND','ED–NND']
        types = ["matched", "all"]
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.2, 8.4), dpi=HEATMAP_DPI, sharex=True)
        drew_any = False
        for r, t in enumerate(types):
            ax = axes[r]
            raw = [sub[(sub["pair"] == lab) & (sub["type"] == t)]["value"].to_numpy() for lab in pairs]
            data, labs = _filter_nonempty_for_violin(raw, pairs, min_n=2)
            ok = _plot_violin_box(ax, data, labs, ylabel=stat, title=f"{stat} — {t}")
            drew_any = drew_any or ok
        if drew_any:
            fig.suptitle(f"Between-metric {stat} on unrolled spectra — {view.upper()}", y=0.995)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{prefix}{stat.lower()}_violin_{view.upper()}.png"),
                        bbox_inches="tight")
        plt.close(fig)

    # ---- mean-heatmaps (3x3, symmetric pairs filled with means) ----
    pairs = list(combinations(metrics, 2))
    # mean JSD
    M_jsd = {f"{a.upper()}–{b.upper()}": df[(df["pair"]==f"{a.upper()}–{b.upper()}") & (df["stat"]=="JSD")]["value"].mean()
             for a,b in pairs}
    # mean Overlap
    M_ovl = {f"{a.upper()}–{b.upper()}": df[(df["pair"]==f"{a.upper()}–{b.upper()}") & (df["stat"]=="Overlap")]["value"].mean()
             for a,b in pairs}

    def _matrix_from_pairmeans(metrics, D, fill_diag):
        n = len(metrics)
        M = np.full((n,n), np.nan, float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    M[i,j] = fill_diag
                else:
                    key = f"{metrics[min(i,j)].upper()}–{metrics[max(i,j)].upper()}"
                    M[i,j] = D.get(key, np.nan)
        return M

    metsU = [m.upper() for m in metrics]
    H_jsd = _matrix_from_pairmeans(metrics, M_jsd, 0.0)
    H_ovl = _matrix_from_pairmeans(metrics, M_ovl, 1.0)

    cmap = nord_colormap()
    # JSD heatmap (0..1, larger = more different)
    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=HEATMAP_DPI)
    im = ax.imshow(H_jsd, vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(len(metsU))); ax.set_yticks(range(len(metsU)))
    ax.set_xticklabels(metsU); ax.set_yticklabels(metsU)
    ax.set_title(f"Mean JSD (base-2) — {view.upper()}")
    for i in range(H_jsd.shape[0]):
        for j in range(H_jsd.shape[1]):
            ax.text(j, i, f"{H_jsd[i,j]:.2f}", ha="center", va="center", fontsize=9, color="w")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="JSD")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{prefix}mean_jsd_heatmap_{view.upper()}.png"), bbox_inches="tight")
    plt.close(fig)

    # Overlap heatmap (0..1, larger = more similar)
    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=HEATMAP_DPI)
    im = ax.imshow(H_ovl, vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_xticks(range(len(metsU))); ax.set_yticks(range(len(metsU)))
    ax.set_xticklabels(metsU); ax.set_yticklabels(metsU)
    ax.set_title(f"Mean overlap (∑ min) — {view.upper()}")
    for i in range(H_ovl.shape[0]):
        for j in range(H_ovl.shape[1]):
            ax.text(j, i, f"{H_ovl[i,j]:.2f}", ha="center", va="center", fontsize=9, color="w")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="overlap")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{prefix}mean_overlap_heatmap_{view.upper()}.png"), bbox_inches="tight")
    plt.close(fig)

def _feature_table_from_summary(summary_df: pd.DataFrame,
                                view: str,
                                stat_cols: List[str],
                                require_metrics: Tuple[str, ...] = ("pd","ed","nnd"),
                                min_complete: int = 1) -> Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]]:
    """
    Build (dfv, X, y, used_stats) for a given view.
    - Restricts to PD/ED/NND by default (change require_metrics if needed).
    - If none of the requested stat_cols exist, falls back to all numeric, non-param stats.
    - If no complete cases, median-impute columns and proceed (instead of returning 0 rows).
    Returns None if we still have no samples after fallback.
    """
    dfv = summary_df[summary_df["view"] == view].copy()
    if require_metrics:
        dfv = dfv[dfv["metric"].isin(require_metrics)].copy()
    if dfv.empty:
        return None

    # pick usable stat columns
    used = [c for c in stat_cols if c in dfv.columns]
    if not used:
        exclude = set(["tree_id", "metric", "view"]) | {c for c in dfv.columns if c.endswith("_par")}
        used = [c for c in dfv.columns
                if c not in exclude and pd.api.types.is_numeric_dtype(dfv[c])]
    if not used:
        return None

    # try complete cases first
    complete = dfv.dropna(subset=used + ["tree_id", "metric"])
    if len(complete) >= min_complete:
        X = complete[used].to_numpy(dtype=float)
        y = complete["metric"].astype(str).to_numpy()
        return complete.loc[:, ["tree_id", "metric"] + used], X, y, used

    # fallback: median impute per-column and keep all rows that have tree_id+metric
    dfv = dfv.dropna(subset=["tree_id", "metric"]).copy()
    if dfv.empty:
        return None
    med = dfv[used].median(numeric_only=True)
    dfv[used] = dfv[used].fillna(med)
    X = dfv[used].to_numpy(dtype=float)
    y = dfv["metric"].astype(str).to_numpy()
    if X.shape[0] == 0:
        return None
    return dfv.loc[:, ["tree_id", "metric"] + used], X, y, used

def _plot_pca_scatter(Xz: np.ndarray, y: np.ndarray, pca: PCA, out_png: str, title: str):
    """2D PCA scatter colored by metric, with explained variance in title."""
    mets = ["pd","ed","nnd"]
    colors = {"pd":"C0", "ed":"C1", "nnd":"C3"}
    Z = pca.transform(Xz)[:, :2]  # already standardized
    fig, ax = plt.subplots(figsize=(7.8, 6.4), dpi=HEATMAP_DPI)
    for m in mets:
        sel = (y == m)
        if sel.any():
            ax.scatter(Z[sel,0], Z[sel,1], s=18, alpha=0.75, label=m.upper(), c=colors.get(m, "C7"))
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.35, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def _plot_confusion_heatmaps(tab: pd.DataFrame, out_prefix: str, title_base: str = "K-medoids confusion"):
    """
    Save two heatmaps:
      • raw counts     → f"{out_prefix}_confusion_raw.png"
      • row-normalized → f"{out_prefix}_confusion_norm.png"
    Also saves a normalized CSV alongside the existing raw CSV.
    """
    cmap = nord_colormap()
    # Make sure index/columns are strings for labeling
    row_labels = [str(i) for i in tab.index.tolist()]
    col_labels = [str(c) for c in tab.columns.tolist()]

    # --- RAW ---
    A = tab.to_numpy(dtype=float)
    vmax_raw = max(1.0, np.nanmax(A))
    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=HEATMAP_DPI)
    im = ax.imshow(A, vmin=0.0, vmax=vmax_raw, cmap=cmap, origin="upper", aspect="auto")
    ax.set_xticks(range(A.shape[1])); ax.set_yticks(range(A.shape[0]))
    ax.set_xticklabels(col_labels);   ax.set_yticklabels(row_labels)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Metric (true)")
    ax.set_title(f"{title_base} — raw counts")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{int(A[i,j])}", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_confusion_raw.png", bbox_inches="tight")
    plt.close(fig)

    # --- ROW-NORMALIZED (percent) ---
    row_sums = A.sum(axis=1, keepdims=True)
    N = np.divide(A, row_sums, out=np.zeros_like(A), where=row_sums > 0)
    vmax_norm = 1.0
    fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=HEATMAP_DPI)
    im = ax.imshow(N, vmin=0.0, vmax=vmax_norm, cmap=cmap, origin="upper", aspect="auto")
    ax.set_xticks(range(N.shape[1])); ax.set_yticks(range(N.shape[0]))
    ax.set_xticklabels(col_labels);   ax.set_yticklabels(row_labels)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Metric (true)")
    ax.set_title(f"{title_base} — row-normalized")
    for i in range(N.shape[0]):
        for j in range(N.shape[1]):
            ax.text(j, i, f"{100*N[i,j]:.0f}%", ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row %")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_confusion_norm.png", bbox_inches="tight")
    plt.close(fig)

    # Save normalized CSV too
    norm_df = pd.DataFrame(N, index=tab.index, columns=tab.columns)
    norm_df.to_csv(f"{out_prefix}_confusion_norm.csv")


def _fit_kmedoids(Xz: np.ndarray, k: int = 3, random_state: int = 42) -> KMedoids:
    # 'k-medoids++' init gives better separated starts; pam/alternate available too. :contentReference[oaicite:5]{index=5}
    return KMedoids(n_clusters=k, init="k-medoids++", metric="euclidean", random_state=random_state).fit(Xz)

def _cluster_reports(Xz: np.ndarray, y_true: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    # Internal metrics (unsupervised) and external (vs PD/ED/NND)
    out = {}
    out["silhouette"] = float(silhouette_score(Xz, labels))              # [-1,1]; higher better. :contentReference[oaicite:6]{index=6}
    out["calinski_harabasz"] = float(calinski_harabasz_score(Xz, labels)) # higher better. :contentReference[oaicite:7]{index=7}
    out["davies_bouldin"] = float(davies_bouldin_score(Xz, labels))       # lower better. :contentReference[oaicite:8]{index=8}
    out["ARI"] = float(adjusted_rand_score(y_true, labels))                # 0≈chance, 1=perfect. :contentReference[oaicite:9]{index=9}
    out["NMI"] = float(normalized_mutual_info_score(y_true, labels))       # 0..1
    return out

def _save_confusion_and_medoids(dfv: pd.DataFrame, y_true: np.ndarray, labels: np.ndarray,
                                kmed: KMedoids, out_prefix: str):
    # Confusion table (rows = true metric, cols = cluster)
    mets = ["pd","ed","nnd"]
    tab = pd.crosstab(pd.Categorical(y_true, categories=mets),
                      labels,
                      rownames=["metric"], colnames=["cluster"], dropna=False)
    tab.to_csv(f"{out_prefix}_confusion.csv")

    # Figures: raw + row-normalized
    _plot_confusion_heatmaps(tab, out_prefix, title_base="K-medoids confusion")

    # Medoid exemplars
    medoid_rows = dfv.iloc[kmed.medoid_indices_][["tree_id","metric"]].copy()
    medoid_rows["cluster"] = range(kmed.n_clusters)
    medoid_rows.to_csv(f"{out_prefix}_medoids.csv", index=False)

def _choose_param_levels_flexible(df: pd.DataFrame, param_raw: str, max_levels: int = PARAM_LEVELS_MAX):
    """
    Like _choose_param_levels, but accepts either '<p>' or '<p>_par' in df.
    Returns representative levels (unique values if small; otherwise quantiles).
    """
    for col in (param_raw, f"{param_raw}_par"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.nunique() < 2:
                return []
            if s.nunique() <= max_levels:
                return sorted(map(float, s.unique().tolist()))
            qs = np.linspace(0.0, 1.0, max_levels)
            return sorted({float(s.quantile(q)) for q in qs})
    return []

def _plot_metric_heatmap(mat_df: pd.DataFrame, *, vmin: float, vmax: float,
                         title: str, out_png: str, value_fmt: str = ".2f"):
    """Generic heatmap with value annotations (rows = levels, cols = TES/TAS)."""
    cmap = nord_colormap()
    A = mat_df.to_numpy(dtype=float)
    fig_h = 0.8 + 0.5 * max(1, A.shape[0])
    fig, ax = plt.subplots(figsize=(5.6, fig_h), dpi=HEATMAP_DPI)
    im = ax.imshow(A, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto", origin="upper")
    ax.set_yticks(range(A.shape[0])); ax.set_yticklabels([f"{v:g}" for v in mat_df.index])
    ax.set_xticks(range(A.shape[1])); ax.set_xticklabels([str(c).upper() for c in mat_df.columns])
    ax.set_xlabel("view"); ax.set_ylabel("parameter level")
    ax.set_title(title)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            txt = "–" if not np.isfinite(A[i, j]) else format(A[i, j], value_fmt)
            ax.text(j, i, txt, ha="center", va="center", color="w", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def run_pca_kmedoids_by_params(summary_df: pd.DataFrame,
                               outdir: str,
                               stats_cols: List[str],
                               params: List[str],
                               max_levels: int = PARAM_LEVELS_MAX,
                               prefix: str = "cmp_",
                               min_samples: int = 3):
    """
    For each parameter and (TES/TAS), run PCA + K-medoids at each selected level.
    Saves:
      * PCA scatter per (param, level, view)
      * Confusion heatmaps per (param, level, view)
      * CSV of metrics per (param, level, view)
      * Heatmaps of ARI/NMI/silhouette across levels (rows) and view (columns)
    """
    _ensure_dir(outdir)
    # normalize lower-case metrics just like load_summary()
    df = summary_df.copy()
    df["view"]   = df["view"].astype(str).str.strip().str.lower()
    df["metric"] = df["metric"].astype(str).str.strip().str.lower()

    for param in params:
        # pick representative levels for this param
        levels = _choose_param_levels_flexible(df, param, max_levels=max_levels)
        if len(levels) < 2:
            print(f"[PCA/param] {param}: not enough distinct levels; skipping.")
            continue

        all_rows = []  # collect quality metrics for the summary heatmaps
        for view in ["tes", "tas"]:
            for lev in levels:
                # subset by param==level (accept either raw or *_par)
                mask = pd.Series(True, index=df.index)
                if param in df.columns:
                    mask &= np.isclose(pd.to_numeric(df[param], errors="coerce"), float(lev), atol=1e-12)
                elif f"{param}_par" in df.columns:
                    mask &= np.isclose(pd.to_numeric(df[f"{param}_par"], errors="coerce"), float(lev), atol=1e-12)
                else:
                    continue
                df_sub = df[mask].copy()

                tag = f"{param}_{lev:g}_{view.upper()}"
                cleaned = _nan_diagnostics_and_clean(df_sub, view, stats_cols, outdir,
                                                     prefix=f"{prefix}param_{tag}_")
                if cleaned is None:
                    print(f"[PCA/param] {tag}: no usable rows after cleaning.")
                    continue

                dfv, X, y, used = cleaned
                if X.shape[0] < min_samples or np.unique(y).size < 2:
                    print(f"[PCA/param] {tag}: too few samples/classes (n={X.shape[0]}).")
                    continue

                # Standardize → PCA → K-medoids
                scaler = StandardScaler()
                Xz = scaler.fit_transform(X)

                pca = PCA(n_components=min(10, Xz.shape[1]), random_state=42).fit(Xz)
                _plot_pca_scatter(
                    Xz, y, pca,
                    out_png=os.path.join(outdir, f"{prefix}pca_scatter_{tag}.png"),
                    title=f"PCA — {view.upper()} | {param}={lev:g} (features: {len(used)})"
                )

                kmed = _fit_kmedoids(Xz, k=3, random_state=42)
                labels = kmed.labels_
                rep = _cluster_reports(Xz, y, labels)
                rep.update({"param": param, "level": float(lev), "view": view, "n": int(X.shape[0])})
                all_rows.append(rep)

                _save_confusion_and_medoids(
                    dfv=dfv.reset_index(drop=True),
                    y_true=y, labels=labels, kmed=kmed,
                    out_prefix=os.path.join(outdir, f"{prefix}kmedoids_{tag}")
                )

        # ----- summary heatmaps over levels (rows) × view (cols) -----
        if not all_rows:
            continue
        metr = pd.DataFrame(all_rows)
        metr.sort_values(["level", "view"], inplace=True)
        out_csv = os.path.join(outdir, f"{prefix}kmedoids_metrics_param_sweep_{param}.csv")
        metr.to_csv(out_csv, index=False)

        # Dense matrices for a few headline metrics
        for name, (vmin, vmax) in {
            "ARI": (0.0, 1.0),
            "NMI": (0.0, 1.0),
            "silhouette": (-1.0, 1.0),
        }.items():
            M = (metr.pivot_table(index="level", columns="view", values=name, aggfunc="mean")
                      .sort_index())
            _plot_metric_heatmap(
                M, vmin=vmin, vmax=vmax,
                title=f"{param.upper()} — {name} (K-medoids, PD/ED/NND)",
                out_png=os.path.join(outdir, f"{prefix}kmedoids_{param}_{name}_heatmap.png"),
                value_fmt=".2f"
            )


# ================================ MAIN ========================================

def main():
    _ensure_dir(OUTPUT_DIR)

    # Load inputs
    summary_df = load_summary(os.path.join(INPUT_DIR, SUMMARY_FILE))
    gauss_path = os.path.join(INPUT_DIR, GAUSS_FILE)
    have_gauss = os.path.isfile(gauss_path)

    # Stats — by metric and by metric+params (TES/TAS)
    run_treatment_tests(summary_df, view="tes",
                        treat_cols=["metric"], stats_cols=DEFAULT_STATS,
                        out_prefix=os.path.join(OUTPUT_DIR, f"{CMP_PREFIX}stats_metric"))
    run_treatment_tests(summary_df, view="tas",
                        treat_cols=["metric"], stats_cols=DEFAULT_STATS,
                        out_prefix=os.path.join(OUTPUT_DIR, f"{CMP_PREFIX}stats_metric"))
    run_treatment_tests(summary_df, view="tes",
                        treat_cols=["metric"] + PARAM_GROUP, stats_cols=DEFAULT_STATS,
                        out_prefix=os.path.join(OUTPUT_DIR, f"{CMP_PREFIX}stats_params"))
    run_treatment_tests(summary_df, view="tas",
                        treat_cols=["metric"] + PARAM_GROUP, stats_cols=DEFAULT_STATS,
                        out_prefix=os.path.join(OUTPUT_DIR, f"{CMP_PREFIX}stats_params"))

    if not have_gauss:
        print("No Gaussian series found; analysis finished (stats only).")
        return

    series_g = load_series(gauss_path)

    # ======== PCA + K-medoids on summary spectral stats (separability of PD/ED/NND) ========
    run_pca_and_kmedoids(
        summary_df=summary_df,
        outdir=OUTPUT_DIR,
        stats_cols=DEFAULT_STATS,
        prefix=CMP_PREFIX
    )

    # ======== PCA + K-medoids by parameter levels (compare separability across settings) ========
    run_pca_kmedoids_by_params(
        summary_df=summary_df,
        outdir=OUTPUT_DIR,
        stats_cols=DEFAULT_STATS,
        params=["lambda", "mu", "beta_n", "beta_phi", "gamma_n", "gamma_phi"],
        max_levels=PARAM_LEVELS_MAX,
        prefix=CMP_PREFIX
    )

    # ======== Between-metric similarities/differences (JSD & Overlap) ========
    for vw in ["tes", "tas"]:
        quantify_metric_overlap_and_divergence(series_g, view=vw, lap=LAP,
                                               outdir=OUTPUT_DIR, prefix=f"{CMP_PREFIX}")

    # ======== Size effects (Gaussian only) ========
    run_size_effects(series_g, summary_df, lap=LAP, outdir=OUTPUT_DIR, prefix=CMP_PREFIX)

    # ======== Per-parameter merged comparisons (Gaussian only) ========
    metrics_all = _metric_order(sorted(set(series_g["metric"].unique())))
    for param in PARAM_GROUP:
        raw_name = param.replace("_par", "")
        # Determine representative levels from summary
        levels = _choose_param_levels(attach_params_no_collision(series_g, summary_df), raw_name, PARAM_LEVELS_MAX)
        if len(levels) < 2:
            # nothing to compare
            continue

        # ------- Variant 1: overlay two extremes (min vs max) -------
        levelA, levelB = float(min(levels)), float(max(levels))
        tes_pairs: Dict[str, Dict[str, Dict]] = {}  # metric -> {"A": surf, "B": surf}
        tas_pairs: Dict[str, Dict[str, Dict]] = {}
        for m in metrics_all:
            SA_tes = build_surface_for_param_level(series_g, summary_df, view="tes", lap=LAP,
                                                   metric=m, param_name=raw_name, level=levelA)
            SB_tes = build_surface_for_param_level(series_g, summary_df, view="tes", lap=LAP,
                                                   metric=m, param_name=raw_name, level=levelB)
            SA_tas = build_surface_for_param_level(series_g, summary_df, view="tas", lap=LAP,
                                                   metric=m, param_name=raw_name, level=levelA)
            SB_tas = build_surface_for_param_level(series_g, summary_df, view="tas", lap=LAP,
                                                   metric=m, param_name=raw_name, level=levelB)
            if SA_tes is not None or SB_tes is not None:
                tes_pairs[m] = {"A": SA_tes, "B": SB_tes}
            if SA_tas is not None or SB_tas is not None:
                tas_pairs[m] = {"A": SA_tas, "B": SB_tas}

        # 3D overlay grid (TES row, TAS row; PD/ED/NND columns)
        grid_3d_two_rows_overlay(levelA_name=f"{raw_name}={levelA:g}",
                                 levelB_name=f"{raw_name}={levelB:g}",
                                 tes_surfs=tes_pairs, tas_surfs=tas_pairs,
                                 out_png=os.path.join(OUTPUT_DIR, f"{CMP_PREFIX}gauss_{raw_name}_overlay3D.png"),
                                 main_title=f"Gaussian unrolled — Overlay ({raw_name}: {levelA:g} vs {levelB:g})")

        # Heatmap difference grids (Δ level)
        # Build pair maps metric -> (A,B) for TES/TAS
        tes_pairmaps = {m: (tes_pairs[m]["A"], tes_pairs[m]["B"]) for m in tes_pairs if "A" in tes_pairs[m] and "B" in tes_pairs[m]}
        tas_pairmaps = {m: (tas_pairs[m]["A"], tas_pairs[m]["B"]) for m in tas_pairs if "A" in tas_pairs[m] and "B" in tas_pairs[m]}
        grid_heatmap_two_rows_diff(levelA_name=f"{raw_name}={levelA:g}",
                                   levelB_name=f"{raw_name}={levelB:g}",
                                   tes_pairs=tes_pairmaps, tas_pairs=tas_pairmaps,
                                   out_linear=os.path.join(OUTPUT_DIR, f"{CMP_PREFIX}gauss_{raw_name}_overlay_heatmap_linear.png"),
                                   out_sqrt=os.path.join(OUTPUT_DIR,   f"{CMP_PREFIX}gauss_{raw_name}_overlay_heatmap_sqrt.png"),
                                   out_log2=os.path.join(OUTPUT_DIR,   f"{CMP_PREFIX}gauss_{raw_name}_overlay_heatmap_log2.png"),
                                   main_title=f"Gaussian Δ heatmaps — ({raw_name}: {levelB:g} - {levelA:g})")
        # Side-by-side (two-level) heatmaps: A and B in adjacent columns for each metric
        grid_heatmap_two_rows_twolevels(
            levelA_name=f"{raw_name}={levelA:g}",
            levelB_name=f"{raw_name}={levelB:g}",
            tes_pairmaps=tes_pairmaps,
            tas_pairmaps=tas_pairmaps,
            out_linear=os.path.join(OUTPUT_DIR, f"{CMP_PREFIX}gauss_{raw_name}_twolevel_heatmap_linear.png"),
            out_sqrt=os.path.join(OUTPUT_DIR, f"{CMP_PREFIX}gauss_{raw_name}_twolevel_heatmap_sqrt.png"),
            out_log2=os.path.join(OUTPUT_DIR, f"{CMP_PREFIX}gauss_{raw_name}_twolevel_heatmap_log2.png"),
            main_title=f"Gaussian two-level heatmaps — {raw_name}: {levelA:g} vs {levelB:g}"
        )

        # ------- Variant 2: per-level subpanels (split into several figures) -------
        # Limit to representative levels to avoid explosion
        rep_levels = levels if len(levels) <= PARAM_LEVELS_MAX else [levels[0], levels[len(levels)//2], levels[-1]]
        for lev in rep_levels:
            tes_surfs = {}
            tas_surfs = {}
            for m in metrics_all:
                S_tes = build_surface_for_param_level(series_g, summary_df, view="tes", lap=LAP,
                                                      metric=m, param_name=raw_name, level=lev)
                S_tas = build_surface_for_param_level(series_g, summary_df, view="tas", lap=LAP,
                                                      metric=m, param_name=raw_name, level=lev)
                if S_tes is not None: tes_surfs[m] = S_tes
                if S_tas is not None: tas_surfs[m] = S_tas

            tag = f"{raw_name}_{lev:g}"
            # 3D single-level grid
            grid_3d_two_rows_singlelevel(level_name=f"{raw_name}={lev:g}",
                                         tes_surfs=tes_surfs, tas_surfs=tas_surfs,
                                         out_png=os.path.join(OUTPUT_DIR, f"{CMP_PREFIX}gauss_{tag}_singlelevel3D.png"),
                                         main_title=f"Gaussian unrolled — {raw_name}={lev:g}")
            # Heatmap single-level grids (linear/sqrt/log2), unified vmin/vmax per figure
            grid_heatmap_two_rows_singlelevel(level_name=f"{raw_name}={lev:g}",
                                              tes_surfs=tes_surfs, tas_surfs=tas_surfs,
                                              out_linear=os.path.join(OUTPUT_DIR, f"{CMP_PREFIX}gauss_{tag}_singlelevel_heatmap_linear.png"),
                                              out_sqrt=os.path.join(OUTPUT_DIR,   f"{CMP_PREFIX}gauss_{tag}_singlelevel_heatmap_sqrt.png"),
                                              out_log2=os.path.join(OUTPUT_DIR,   f"{CMP_PREFIX}gauss_{tag}_singlelevel_heatmap_log2.png"),
                                              main_title=f"Gaussian heatmaps — {raw_name}={lev:g}")

    print(f"All outputs written to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
