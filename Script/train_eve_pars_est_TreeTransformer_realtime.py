#!/usr/bin/env python3
"""
Tree-aware graph transformer for eve / BD-ED-NND phylogenetic-tree inference.

This script is designed to be a drop-in companion to the existing
train_eve_pars_est_DiffPool.py workflow. It reads the same exported RDS tree
folders:

    <name>/<task_type>/GNN/tree/tree_*.rds
    <name>/<task_type>/GNN/tree/EL/EL_*.rds
    <name>/<task_type>/GNN/tree/BT/BT_*.rds

and writes compatible STBO outputs:

    <name>/<task_type>/STBO/<task_type>_tree_transformer_<run_id>.rds
    <name>/<task_type>/STBO/<task_type>_final_tree_transformer_<run_id>.rds

If pyreadr is unavailable, CSV fallbacks are written instead. In the intended
project environment, pyreadr should be installed so the RDS files are produced.

Model idea
----------
The model uses a transformer encoder over tree nodes, but attention scores are
not purely sequence-position based. Each attention logit receives a learned bias
from pairwise tree features:

  * topological path distance between nodes;
  * patristic path distance between nodes;
  * ancestor/descendant direction;
  * absolute node-age difference;
  * direct-edge indicator;
  * self-pair indicator.

Each node also receives tree-specific positional features: root/tip/internal
flags, degree, depth from root, node age, parent-edge length, and subtree size.

The regression head predicts the first n_predicted_values parameters, while the
classification head predicts the scenario class.
"""

from __future__ import annotations

import argparse
import builtins
import json
import math
import os
import random
import signal
import sys
import time
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

print = partial(builtins.print, flush=True)
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except AttributeError:
    pass


def _handle_signal(signum, frame):
    print(f"Received signal {signum}; terminating.", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    raise SystemExit(128 + int(signum))

for _sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGUSR1", None)):
    if _sig is not None:
        signal.signal(_sig, _handle_signal)
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ----------------------------- configuration ----------------------------- #

DEFAULT_CONFIG = {
    # old-workflow-compatible loss and target settings
    "alpha": 0.5,
    "beta": 0.5,
    "huber_delta": 1.0,
    "n_predicted_values": 2,
    "n_classes": 3,
    "class_names": ["bd", "ed", "nnd"],
    # For old plotting code that expects pd_prob/ed_prob/nnd_prob, keep pd_prob
    # as the first probability column even when the first class is BD.
    "class_probability_columns": ["pd_prob", "ed_prob", "nnd_prob"],
    "metric_aliases": {"pd": "bd", "BD": "bd", "PD": "bd", "ED": "ed", "NND": "nnd"},

    # data and split settings
    "seed": 12345,
    "train_fraction": 0.9,
    "shuffle_data": True,
    "max_nodes_limit": 2500,
    "min_nodes": 4,
    "normalize_edge_length": True,
    "target_standardization": True,

    # transformer settings
    "epoch_number_transformer": 51,
    "train_batch_size": 8,
    "test_batch_size": 8,
    "learning_rate": 0.0005,
    "weight_decay": 0.01,
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 4,
    "dim_feedforward": 256,
    "dropout_ratio": 0.15,
    "attention_radius_edges": 0,  # 0 means global/full attention
    "gradient_clip_norm": 1.0,
    "num_workers": 0,
    "early_stopping_patience": 0,  # 0 disables early stopping

    # output settings
    "output_tag": "tree_transformer",
    "write_csv_fallback": True,
    "save_every_epoch_predictions": False,
    "load_progress_every": 100,
    "train_progress_every": 10,
    "eval_progress_every": 10,
}

PARAM_COLUMNS = ["lambda", "mu", "beta_n", "beta_phi", "gamma_n", "gamma_phi"]
DIFF_COLUMNS = [f"{c}_diff" for c in PARAM_COLUMNS]
PRED_COLUMNS = [f"{c}_pred" for c in PARAM_COLUMNS]


def deep_update(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            tmp = dict(out[key])
            tmp.update(value)
            out[key] = tmp
        else:
            out[key] = value
    return out


def resolve_config_path(path: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if path:
        candidates.append(Path(path))
    candidates.extend([
        Path("../Config/eve_train_tree_transformer.yaml"),
        Path("./eve_train_tree_transformer.yaml"),
        Path(__file__).resolve().with_name("eve_train_tree_transformer.yaml"),
        Path("/mnt/data/eve_train_tree_transformer.yaml"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_config(path: Optional[str]) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    config_path = resolve_config_path(path)
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        cfg = deep_update(cfg, loaded)
        cfg["_config_path"] = str(config_path)
    else:
        cfg["_config_path"] = None

    cfg["class_names"] = [str(x).lower() for x in cfg["class_names"]]
    cfg["n_classes"] = int(cfg.get("n_classes", len(cfg["class_names"])))
    if cfg["n_classes"] != len(cfg["class_names"]):
        raise ValueError("n_classes must equal len(class_names).")

    if "class_probability_columns" not in cfg or cfg["class_probability_columns"] is None:
        cfg["class_probability_columns"] = [f"{x}_prob" for x in cfg["class_names"]]
    if len(cfg["class_probability_columns"]) != cfg["n_classes"]:
        raise ValueError("class_probability_columns must have one name per class.")

    # Normalize aliases to lowercase keys/values.
    aliases = cfg.get("metric_aliases", {}) or {}
    cfg["metric_aliases"] = {str(k).lower(): str(v).lower() for k, v in aliases.items()}
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------- RDS utilities ----------------------------- #


def _require_pyreadr():
    try:
        import pyreadr  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyreadr is required to read/write the RDS files exported by the R workflow. "
            "Install it in the project Python environment with `pip install pyreadr`, "
            "or run --self-test to check the model without RDS I/O."
        ) from exc
    return pyreadr


def read_rds_array(path: Path, dtype=None) -> np.ndarray:
    pyreadr = _require_pyreadr()
    result = pyreadr.read_r(str(path))
    if None in result:
        obj = result[None]
    else:
        obj = next(iter(result.values()))
    arr = np.asarray(obj)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def write_rds_or_csv(df: pd.DataFrame, rds_path: Path, write_csv_fallback: bool = True) -> None:
    try:
        pyreadr = _require_pyreadr()
        pyreadr.write_rds(str(rds_path), df.astype(object))
    except Exception as exc:
        if not write_csv_fallback:
            raise
        csv_path = rds_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"[warning] Could not write {rds_path.name} as RDS ({exc}). Wrote CSV fallback: {csv_path}")


# ----------------------------- data parsing ------------------------------ #


@dataclass
class TreeExample:
    key: str
    edge_index: np.ndarray          # [2, E], directed parent -> child, 0-based
    el_matrix: np.ndarray           # [N, 3], old neighbor-length node features
    brts: np.ndarray                # branching times, variable length
    params_full: np.ndarray         # always length 6, missing effects padded with zero
    y_re: np.ndarray                # first n_predicted_values parameters
    y_cl: int
    metric: str
    num_nodes: int
    node_features: np.ndarray       # [N, F]
    root_index: int
    depth_time: np.ndarray          # [N]
    depth_edges: np.ndarray         # [N]
    edge_len_to_parent: np.ndarray  # [N]
    parent: np.ndarray              # [N]
    children: List[List[int]]
    brts_summary: np.ndarray        # [BRT_FEATURE_DIM]


BRT_FEATURE_DIM = 10
PAIR_FEATURE_DIM = 6


def normalize_metric_token(token: str, cfg: dict) -> str:
    low = str(token).lower()
    if low in cfg["metric_aliases"]:
        return cfg["metric_aliases"][low]
    return low


def parse_eve_filename(filename: str, prefix: str, cfg: dict) -> Tuple[str, np.ndarray]:
    """Return (metric, full six-parameter vector) parsed from exported filenames.

    Expected eve filename body:
      <lambda>_<mu>_<beta_n>_<beta_phi>_<gamma_n>_<gamma_phi>_<age>_<metric>_<i>

    The parser is deliberately permissive: it finds the metric token among the
    configured class names/aliases and extracts numeric fields. If only lambda
    and mu are available, the four effect parameters are padded with zero.
    """
    stem = Path(filename).stem
    expected = prefix + "_"
    if not stem.startswith(expected):
        raise ValueError(f"Unexpected filename prefix for {filename}; expected {expected!r}.")

    body = stem[len(expected):]
    parts = body.split("_")

    valid_tokens = set(cfg["class_names"])
    valid_tokens.update(str(k).lower() for k in cfg.get("metric_aliases", {}).keys())
    valid_tokens.update(str(v).lower() for v in cfg.get("metric_aliases", {}).values())
    valid_tokens.update(["bd", "pd", "ed", "nnd"])

    metric: Optional[str] = None
    for part in parts:
        low = part.lower()
        if low in valid_tokens:
            metric = normalize_metric_token(low, cfg)
            break
    if metric is None:
        raise ValueError(
            f"Could not find a class/metric token in {filename}. "
            f"Known class names are {cfg['class_names']} and aliases {cfg.get('metric_aliases', {})}."
        )

    numeric_values: List[float] = []
    for part in parts:
        try:
            numeric_values.append(float(part))
        except ValueError:
            continue

    if len(numeric_values) >= 6:
        params = np.array(numeric_values[:6], dtype=np.float32)
    elif len(numeric_values) >= 2:
        params = np.zeros(6, dtype=np.float32)
        params[0:2] = np.array(numeric_values[:2], dtype=np.float32)
    else:
        raise ValueError(f"Could not extract at least lambda and mu from {filename}.")
    return metric, params


def file_key(path: Path, prefix: str) -> str:
    stem = path.stem
    expected = prefix + "_"
    if not stem.startswith(expected):
        raise ValueError(f"Unexpected file prefix for {path.name}; expected {expected!r}.")
    return stem[len(expected):]


def collect_rds_triplets(task_dir: Path) -> List[Tuple[Path, Path, Path]]:
    tree_dir = task_dir / "GNN" / "tree"
    el_dir = tree_dir / "EL"
    bt_dir = tree_dir / "BT"
    if not tree_dir.exists() or not el_dir.exists() or not bt_dir.exists():
        raise FileNotFoundError(
            f"Expected exported folders under {task_dir}: GNN/tree, GNN/tree/EL, and GNN/tree/BT."
        )

    tree_files = sorted(tree_dir.glob("tree_*.rds"))
    el_files = sorted(el_dir.glob("EL_*.rds"))
    bt_files = sorted(bt_dir.glob("BT_*.rds"))

    tree_map = {file_key(p, "tree"): p for p in tree_files}
    el_map = {file_key(p, "EL"): p for p in el_files}
    bt_map = {file_key(p, "BT"): p for p in bt_files}

    common_keys = sorted(set(tree_map) & set(el_map) & set(bt_map))
    missing = (set(tree_map) ^ set(el_map)) | (set(tree_map) ^ set(bt_map))
    if missing:
        print(f"[warning] Ignoring {len(missing)} keys that are not present in all tree/EL/BT folders.")
    if not common_keys:
        raise ValueError(f"No matching tree/EL/BT RDS triplets found in {task_dir}.")

    return [(tree_map[k], el_map[k], bt_map[k]) for k in common_keys]


# -------------------------- tree feature extraction ----------------------- #


def infer_parent_edge_lengths(edge_index: np.ndarray, el_matrix: np.ndarray, num_nodes: int) -> np.ndarray:
    """Infer the branch length from each node to its parent.

    The R exporter writes for every node a padded vector of edge lengths to its
    neighbors but drops the neighbor IDs. In ape's usual edge ordering, the edge
    from a node's parent is encountered before the node's child edges. Therefore
    the first non-zero entry in the child's row is a good reconstruction of the
    parent-edge length. If this assumption fails for a row, we fall back to the
    row mean or a global median of positive lengths.
    """
    positive = el_matrix[np.isfinite(el_matrix) & (el_matrix > 0)]
    fallback = float(np.median(positive)) if positive.size else 1.0
    edge_len = np.zeros(num_nodes, dtype=np.float32)

    dst_nodes = edge_index[1] if edge_index.size else np.array([], dtype=np.int64)
    for child in dst_nodes:
        child = int(child)
        row = el_matrix[child] if child < el_matrix.shape[0] else np.array([])
        nz = row[np.isfinite(row) & (row > 0)]
        if nz.size:
            edge_len[child] = float(nz[0])
        else:
            edge_len[child] = fallback
    return edge_len


def build_parent_children(edge_index: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, List[List[int]], int]:
    parent = np.full(num_nodes, -1, dtype=np.int64)
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    indeg = np.zeros(num_nodes, dtype=np.int64)
    outdeg = np.zeros(num_nodes, dtype=np.int64)

    for src, dst in edge_index.T:
        src_i = int(src)
        dst_i = int(dst)
        if src_i < 0 or dst_i < 0 or src_i >= num_nodes or dst_i >= num_nodes:
            continue
        children[src_i].append(dst_i)
        # Phylogenetic trees should have one parent per non-root node. If a file
        # is undirected, the first incoming edge is retained and disconnected
        # handling below will still keep the script from crashing.
        if parent[dst_i] == -1:
            parent[dst_i] = src_i
        indeg[dst_i] += 1
        outdeg[src_i] += 1

    roots = np.where((indeg == 0) & (outdeg > 0))[0]
    if roots.size == 0:
        roots = np.where(indeg == 0)[0]
    root = int(roots[0]) if roots.size else 0
    return parent, children, root


def compute_depths(
    children: List[List[int]],
    root: int,
    edge_len_to_parent: np.ndarray,
    num_nodes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    depth_edges = np.full(num_nodes, -1, dtype=np.float32)
    depth_time = np.full(num_nodes, np.nan, dtype=np.float32)
    depth_edges[root] = 0.0
    depth_time[root] = 0.0
    stack = [root]
    while stack:
        node = stack.pop()
        for child in children[node]:
            if depth_edges[child] >= 0:
                continue
            depth_edges[child] = depth_edges[node] + 1.0
            depth_time[child] = depth_time[node] + float(edge_len_to_parent[child])
            stack.append(child)

    # Graceful fallback for rare malformed/undirected exports.
    missing = np.where(depth_edges < 0)[0]
    if missing.size:
        for node in missing:
            depth_edges[node] = 0.0
            depth_time[node] = 0.0
    depth_time = np.nan_to_num(depth_time, nan=0.0, posinf=0.0, neginf=0.0)
    return depth_edges, depth_time


def compute_subtree_counts(children: List[List[int]], depth_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(children)
    tip_count = np.zeros(n, dtype=np.float32)
    node_count = np.ones(n, dtype=np.float32)
    order = np.argsort(-depth_edges)  # deepest nodes first
    for node in order:
        if len(children[int(node)]) == 0:
            tip_count[int(node)] = 1.0
        else:
            tip_count[int(node)] = sum(tip_count[ch] for ch in children[int(node)])
        if node != order[-1]:
            pass
        for ch in children[int(node)]:
            node_count[int(node)] += node_count[ch]
    return tip_count, node_count


def brts_summary_features(brts: np.ndarray, tree_height: float) -> np.ndarray:
    brts = np.asarray(brts, dtype=np.float32).reshape(-1)
    brts = brts[np.isfinite(brts) & (brts > 0)]
    if brts.size == 0:
        return np.zeros(BRT_FEATURE_DIM, dtype=np.float32)
    scale = max(float(tree_height), float(np.max(brts)), 1e-6)
    z = brts / scale
    out = np.array([
        float(brts.size),
        float(np.mean(z)),
        float(np.std(z)),
        float(np.min(z)),
        float(np.max(z)),
        float(np.quantile(z, 0.10)),
        float(np.quantile(z, 0.25)),
        float(np.quantile(z, 0.50)),
        float(np.quantile(z, 0.75)),
        float(np.quantile(z, 0.90)),
    ], dtype=np.float32)
    # Count can be large; log-scale it to make it commensurate with the other summaries.
    out[0] = math.log1p(out[0])
    return out


def build_node_features(edge_index: np.ndarray, el_matrix: np.ndarray, brts: np.ndarray, cfg: dict):
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2:
        raise ValueError("edge_index RDS must be a 2D matrix.")
    # R exporter stores [E,2]; Python model uses [2,E].
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2,E] or [E,2], got {edge_index.shape}.")

    el_matrix = np.asarray(el_matrix, dtype=np.float32)
    if el_matrix.ndim == 1:
        el_matrix = el_matrix.reshape(-1, 1)
    if el_matrix.shape[1] < 3:
        pad = np.zeros((el_matrix.shape[0], 3 - el_matrix.shape[1]), dtype=np.float32)
        el_matrix = np.concatenate([el_matrix, pad], axis=1)
    elif el_matrix.shape[1] > 3:
        el_matrix = el_matrix[:, :3]

    num_nodes = int(max(edge_index.max(initial=0) + 1, el_matrix.shape[0]))
    if el_matrix.shape[0] < num_nodes:
        el_matrix = np.pad(el_matrix, ((0, num_nodes - el_matrix.shape[0]), (0, 0)), mode="constant")
    elif el_matrix.shape[0] > num_nodes:
        num_nodes = el_matrix.shape[0]

    parent, children, root = build_parent_children(edge_index, num_nodes)
    edge_len_to_parent = infer_parent_edge_lengths(edge_index, el_matrix, num_nodes)
    depth_edges, depth_time = compute_depths(children, root, edge_len_to_parent, num_nodes)

    outdeg = np.array([len(c) for c in children], dtype=np.float32)
    indeg = (parent >= 0).astype(np.float32)
    degree = indeg + outdeg
    is_root = np.zeros(num_nodes, dtype=np.float32)
    is_root[root] = 1.0
    is_tip = (outdeg == 0).astype(np.float32)
    is_internal = ((outdeg > 0) & (is_root == 0)).astype(np.float32)

    max_depth_edges = max(float(np.max(depth_edges)), 1.0)
    tip_depths = depth_time[is_tip > 0]
    tree_height = float(np.max(tip_depths)) if tip_depths.size else float(np.max(depth_time))
    tree_height = max(tree_height, 1e-6)
    node_age = np.maximum(tree_height - depth_time, 0.0)

    tip_count, node_count = compute_subtree_counts(children, depth_edges)
    n_tips = max(float(np.sum(is_tip)), 1.0)

    if cfg.get("normalize_edge_length", True):
        el_scaled = el_matrix / tree_height
        parent_len_scaled = edge_len_to_parent / tree_height
    else:
        el_scaled = el_matrix.copy()
        parent_len_scaled = edge_len_to_parent.copy()

    node_features = np.column_stack([
        el_scaled[:, 0],
        el_scaled[:, 1],
        el_scaled[:, 2],
        degree / 3.0,
        indeg,
        outdeg / 2.0,
        is_root,
        is_tip,
        is_internal,
        depth_edges / max_depth_edges,
        depth_time / tree_height,
        node_age / tree_height,
        parent_len_scaled,
        tip_count / n_tips,
        node_count / max(float(num_nodes), 1.0),
    ]).astype(np.float32)

    brts_summary = brts_summary_features(brts, tree_height)
    return node_features, root, depth_time.astype(np.float32), depth_edges.astype(np.float32), edge_len_to_parent, parent, children, brts_summary


def build_adjacency_lists(children: List[List[int]], edge_len_to_parent: np.ndarray) -> List[List[Tuple[int, float]]]:
    n = len(children)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for parent, child_list in enumerate(children):
        for child in child_list:
            w = float(edge_len_to_parent[child])
            if not math.isfinite(w) or w <= 0:
                w = 1.0
            adj[parent].append((child, w))
            adj[child].append((parent, w))
    return adj


def all_pairs_tree_distances(children: List[List[int]], edge_len_to_parent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(children)
    adj = build_adjacency_lists(children, edge_len_to_parent)
    topo = np.zeros((n, n), dtype=np.float32)
    patristic = np.zeros((n, n), dtype=np.float32)
    for source in range(n):
        stack = [(source, -1, 0.0, 0.0)]
        while stack:
            node, parent, d_edges, d_time = stack.pop()
            topo[source, node] = d_edges
            patristic[source, node] = d_time
            for nbr, weight in adj[node]:
                if nbr == parent:
                    continue
                stack.append((nbr, node, d_edges + 1.0, d_time + weight))
    return topo, patristic


def ancestor_matrix(children: List[List[int]]) -> np.ndarray:
    n = len(children)
    anc = np.zeros((n, n), dtype=np.float32)
    for source in range(n):
        stack = list(children[source])
        while stack:
            node = stack.pop()
            anc[source, node] = 1.0
            stack.extend(children[node])
    return anc


def pair_features_for_example(ex: TreeExample, attention_radius_edges: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    topo, patristic = all_pairs_tree_distances(ex.children, ex.edge_len_to_parent)
    anc = ancestor_matrix(ex.children)
    n = ex.num_nodes

    max_topo = max(float(np.max(topo)), 1.0)
    max_pat = max(float(np.max(patristic)), 1e-6)
    max_age = max(float(np.max(ex.depth_time)), 1e-6)
    depth_diff = np.abs(ex.depth_time[:, None] - ex.depth_time[None, :]) / max_age
    direct = (topo == 1).astype(np.float32)
    self_pair = np.eye(n, dtype=np.float32)
    ancestor_direction = anc - anc.T

    pair = np.stack([
        np.log1p(topo) / math.log1p(max_topo),
        patristic / max_pat,
        ancestor_direction,
        depth_diff.astype(np.float32),
        direct,
        self_pair,
    ], axis=-1).astype(np.float32)

    local_mask = None
    if attention_radius_edges and attention_radius_edges > 0:
        local_mask = (topo <= float(attention_radius_edges)) | (self_pair > 0)
    return pair, local_mask


class EveTreeDataset(Dataset):
    def __init__(self, examples: Sequence[TreeExample]):
        self.examples = list(examples)
        if not self.examples:
            raise ValueError("Dataset is empty after filtering.")
        self.node_feature_dim = int(self.examples[0].node_features.shape[1])

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TreeExample:
        return self.examples[idx]


def load_eve_examples(task_dir: Path, cfg: dict) -> List[TreeExample]:
    triplets = collect_rds_triplets(task_dir)
    class_to_idx = {name: i for i, name in enumerate(cfg["class_names"])}
    n_pred = int(cfg["n_predicted_values"])
    examples: List[TreeExample] = []

    total_triplets = len(triplets)
    load_progress_every = int(cfg.get("load_progress_every", 0) or 0)
    load_start = time.time()
    print(f"Found {total_triplets} matching tree/EL/BT triplets.")
    for triplet_idx, (tree_path, el_path, bt_path) in enumerate(triplets, start=1):
        if load_progress_every > 0 and (triplet_idx == 1 or triplet_idx % load_progress_every == 0 or triplet_idx == total_triplets):
            elapsed = time.time() - load_start
            print(f"Loading trees: {triplet_idx}/{total_triplets} elapsed={elapsed:.1f}s")
        metric, params = parse_eve_filename(tree_path.name, "tree", cfg)
        if metric not in class_to_idx:
            raise ValueError(
                f"Metric {metric!r} from {tree_path.name} is not in class_names={cfg['class_names']}. "
                "Add it to class_names or metric_aliases."
            )

        edge_index = read_rds_array(tree_path, dtype=np.int64)
        el_matrix = read_rds_array(el_path, dtype=np.float32)
        brts = read_rds_array(bt_path, dtype=np.float32).reshape(-1)

        node_features, root, depth_time, depth_edges, edge_len_parent, parent, children, brts_summary = build_node_features(
            edge_index=edge_index,
            el_matrix=el_matrix,
            brts=brts,
            cfg=cfg,
        )
        num_nodes = int(node_features.shape[0])
        if num_nodes < int(cfg["min_nodes"]):
            continue
        if num_nodes > int(cfg["max_nodes_limit"]):
            continue
        if n_pred > 6:
            raise ValueError("n_predicted_values cannot exceed 6 for the current eve parameter vector.")

        ex = TreeExample(
            key=file_key(tree_path, "tree"),
            edge_index=edge_index.T if edge_index.shape[0] != 2 else edge_index,
            el_matrix=el_matrix,
            brts=brts,
            params_full=params.astype(np.float32),
            y_re=params[:n_pred].astype(np.float32),
            y_cl=class_to_idx[metric],
            metric=metric,
            num_nodes=num_nodes,
            node_features=node_features,
            root_index=root,
            depth_time=depth_time,
            depth_edges=depth_edges,
            edge_len_to_parent=edge_len_parent,
            parent=parent,
            children=children,
            brts_summary=brts_summary,
        )
        examples.append(ex)

    if not examples:
        raise ValueError("No usable examples remained after node-count filtering.")
    return examples


class TreeBatchCollator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.attention_radius_edges = int(cfg.get("attention_radius_edges", 0) or 0)

    def __call__(self, batch: Sequence[TreeExample]) -> dict:
        batch_size = len(batch)
        max_nodes = max(ex.num_nodes for ex in batch)
        feat_dim = batch[0].node_features.shape[1]
        n_pred = len(batch[0].y_re)

        x = torch.zeros((batch_size, max_nodes, feat_dim), dtype=torch.float32)
        pair = torch.zeros((batch_size, max_nodes, max_nodes, PAIR_FEATURE_DIM), dtype=torch.float32)
        node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
        local_mask = torch.ones((batch_size, max_nodes, max_nodes), dtype=torch.bool)
        brts_summary = torch.zeros((batch_size, BRT_FEATURE_DIM), dtype=torch.float32)
        y_re = torch.zeros((batch_size, n_pred), dtype=torch.float32)
        y_full = torch.zeros((batch_size, 6), dtype=torch.float32)
        y_cl = torch.zeros((batch_size,), dtype=torch.long)
        num_nodes = torch.zeros((batch_size,), dtype=torch.long)
        root_index = torch.zeros((batch_size,), dtype=torch.long)
        keys: List[str] = []
        metrics: List[str] = []

        for i, ex in enumerate(batch):
            n = ex.num_nodes
            pair_i, local_mask_i = pair_features_for_example(ex, self.attention_radius_edges)
            x[i, :n, :] = torch.from_numpy(ex.node_features)
            pair[i, :n, :n, :] = torch.from_numpy(pair_i)
            node_mask[i, :n] = True
            if local_mask_i is not None:
                local_mask[i, :, :] = False
                local_mask[i, :n, :n] = torch.from_numpy(local_mask_i.astype(bool))
            brts_summary[i] = torch.from_numpy(ex.brts_summary)
            y_re[i] = torch.from_numpy(ex.y_re)
            y_full[i] = torch.from_numpy(ex.params_full)
            y_cl[i] = int(ex.y_cl)
            num_nodes[i] = n
            root_index[i] = int(ex.root_index)
            keys.append(ex.key)
            metrics.append(ex.metric)

        return {
            "x": x,
            "pair": pair,
            "node_mask": node_mask,
            "local_mask": local_mask,
            "brts_summary": brts_summary,
            "y_re": y_re,
            "y_full": y_full,
            "y_cl": y_cl,
            "num_nodes": num_nodes,
            "root_index": root_index,
            "keys": keys,
            "metrics": metrics,
        }


# ------------------------------- model ----------------------------------- #


class TreeBiasedMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, pair_dim: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.pair_bias = nn.Sequential(
            nn.Linear(pair_dim, max(16, num_heads * 4)),
            nn.GELU(),
            nn.Linear(max(16, num_heads * 4), num_heads),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pair: torch.Tensor,
        node_mask: torch.Tensor,
        local_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, n_nodes, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, n_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = self.pair_bias(pair).permute(0, 3, 1, 2)  # [B,H,N,N]
        scores = scores + bias

        key_mask = node_mask[:, None, None, :]  # [B,1,1,N]
        scores = scores.masked_fill(~key_mask, -1e9)
        if local_mask is not None:
            scores = scores.masked_fill(~local_mask[:, None, :, :], -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # [B,H,N,Dh]
        out = out.transpose(1, 2).contiguous().view(bsz, n_nodes, self.d_model)
        out = self.out_proj(out)
        out = out * node_mask[:, :, None].to(out.dtype)
        return out


class TreeTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, pair_dim: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TreeBiasedMultiheadAttention(d_model, num_heads, pair_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pair: torch.Tensor, node_mask: torch.Tensor, local_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.dropout1(self.attn(h, pair, node_mask, local_mask))
        h = self.norm2(x)
        x = x + self.dropout2(self.ff(h))
        return x * node_mask[:, :, None].to(x.dtype)


class TreeAwareGraphTransformer(nn.Module):
    def __init__(self, node_feature_dim: int, cfg: dict):
        super().__init__()
        self.cfg = cfg
        d_model = int(cfg["d_model"])
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.layers = nn.ModuleList([
            TreeTransformerLayer(
                d_model=d_model,
                num_heads=int(cfg["num_heads"]),
                pair_dim=PAIR_FEATURE_DIM,
                dim_feedforward=int(cfg["dim_feedforward"]),
                dropout=float(cfg["dropout_ratio"]),
            )
            for _ in range(int(cfg["num_layers"]))
        ])
        graph_dim = d_model * 3 + BRT_FEATURE_DIM  # mean, max, root, brts summary
        hidden = int(cfg["dim_feedforward"])
        dropout = float(cfg["dropout_ratio"])
        self.fusion = nn.Sequential(
            nn.Linear(graph_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reg_head = nn.Linear(hidden, int(cfg["n_predicted_values"]))
        self.class_head = nn.Linear(hidden, int(cfg["n_classes"]))

    def forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(batch["x"])
        node_mask = batch["node_mask"]
        for layer in self.layers:
            x = layer(x, batch["pair"], node_mask, batch.get("local_mask"))

        mask_f = node_mask[:, :, None].to(x.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pool = (x * mask_f).sum(dim=1) / denom
        x_for_max = x.masked_fill(~node_mask[:, :, None], -1e9)
        max_pool = x_for_max.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        root_idx = batch["root_index"].clamp(min=0, max=x.shape[1] - 1)
        root_pool = x[torch.arange(x.shape[0], device=x.device), root_idx]
        graph = torch.cat([mean_pool, max_pool, root_pool, batch["brts_summary"]], dim=-1)
        h = self.fusion(graph)
        return self.reg_head(h), self.class_head(h)


# ---------------------------- training utils ----------------------------- #


@dataclass
class TargetScaler:
    mean: np.ndarray
    std: np.ndarray
    enabled: bool = True

    @classmethod
    def fit(cls, y: np.ndarray, enabled: bool = True) -> "TargetScaler":
        if not enabled:
            return cls(mean=np.zeros(y.shape[1], dtype=np.float32), std=np.ones(y.shape[1], dtype=np.float32), enabled=False)
        mean = y.mean(axis=0).astype(np.float32)
        std = y.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        return cls(mean=mean, std=std, enabled=True)

    def transform_tensor(self, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return y
        mean = torch.as_tensor(self.mean, dtype=y.dtype, device=y.device)
        std = torch.as_tensor(self.std, dtype=y.dtype, device=y.device)
        return (y - mean) / std

    def inverse_tensor(self, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return y
        mean = torch.as_tensor(self.mean, dtype=y.dtype, device=y.device)
        std = torch.as_tensor(self.std, dtype=y.dtype, device=y.device)
        return y * std + mean

    def to_json(self) -> dict:
        return {"enabled": self.enabled, "mean": self.mean.tolist(), "std": self.std.tolist()}


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def combined_loss(
    pred_re_scaled: torch.Tensor,
    target_re: torch.Tensor,
    logits: torch.Tensor,
    target_cl: torch.Tensor,
    scaler: TargetScaler,
    cfg: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_scaled = scaler.transform_tensor(target_re)
    reg_loss = F.huber_loss(pred_re_scaled, target_scaled, delta=float(cfg["huber_delta"]))
    cls_loss = F.cross_entropy(logits, target_cl)
    loss = float(cfg["alpha"]) * reg_loss + float(cfg["beta"]) * cls_loss
    return loss, reg_loss, cls_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    scaler: TargetScaler,
    device: torch.device,
    cfg: dict,
    epoch: Optional[int] = None,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_reg = 0.0
    total_cls = 0.0
    total_n = 0

    pred_list = []
    target_re_list = []
    target_full_list = []
    prob_list = []
    label_list = []
    node_list = []
    key_list = []
    metric_list = []

    n_classes = int(cfg["n_classes"])
    correct = 0
    per_class_total = np.zeros(n_classes, dtype=np.int64)
    per_class_correct = np.zeros(n_classes, dtype=np.int64)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

    eval_progress_every = int(cfg.get("eval_progress_every", 0) or 0)
    eval_start = time.time()
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        if eval_progress_every > 0 and (batch_idx == 1 or batch_idx % eval_progress_every == 0 or batch_idx == total_batches):
            elapsed = time.time() - eval_start
            prefix = f"Epoch {epoch:03d} " if epoch is not None else ""
            print(f"{prefix}eval batch {batch_idx}/{total_batches} elapsed={elapsed:.1f}s")
        batch = move_batch(batch, device)
        pred_scaled, logits = model(batch)
        loss, reg_loss, cls_loss = combined_loss(pred_scaled, batch["y_re"], logits, batch["y_cl"], scaler, cfg)
        bsz = batch["y_cl"].shape[0]
        total_loss += float(loss.item()) * bsz
        total_reg += float(reg_loss.item()) * bsz
        total_cls += float(cls_loss.item()) * bsz
        total_n += bsz

        pred = scaler.inverse_tensor(pred_scaled)
        probs = torch.softmax(logits, dim=-1)
        labels = torch.argmax(probs, dim=-1)
        correct += int((labels == batch["y_cl"]).sum().item())
        for true_i, pred_i in zip(batch["y_cl"].detach().cpu().numpy(), labels.detach().cpu().numpy()):
            per_class_total[int(true_i)] += 1
            confusion[int(true_i), int(pred_i)] += 1
            if int(true_i) == int(pred_i):
                per_class_correct[int(true_i)] += 1

        pred_list.append(pred.detach().cpu().numpy())
        target_re_list.append(batch["y_re"].detach().cpu().numpy())
        target_full_list.append(batch["y_full"].detach().cpu().numpy())
        prob_list.append(probs.detach().cpu().numpy())
        label_list.append(batch["y_cl"].detach().cpu().numpy())
        node_list.append(batch["num_nodes"].detach().cpu().numpy())
        key_list.extend(batch["keys"])
        metric_list.extend(batch["metrics"])

    predictions = np.concatenate(pred_list, axis=0)
    targets_re = np.concatenate(target_re_list, axis=0)
    targets_full = np.concatenate(target_full_list, axis=0)
    probs = np.concatenate(prob_list, axis=0)
    labels_true = np.concatenate(label_list, axis=0)
    nodes = np.concatenate(node_list, axis=0)

    n_pred = predictions.shape[1]
    diffs_pred = np.abs(predictions - targets_re)
    diffs_full = np.zeros((predictions.shape[0], 6), dtype=np.float32)
    preds_full = np.zeros((predictions.shape[0], 6), dtype=np.float32)
    diffs_full[:, :n_pred] = diffs_pred
    preds_full[:, :n_pred] = predictions

    per_class_acc = np.full(n_classes, np.nan, dtype=np.float32)
    present = per_class_total > 0
    per_class_acc[present] = per_class_correct[present] / per_class_total[present]

    return {
        "loss_all": total_loss / max(total_n, 1),
        "loss_reg": total_reg / max(total_n, 1),
        "loss_cls": total_cls / max(total_n, 1),
        "accuracy": correct / max(total_n, 1),
        "per_class_accuracy": per_class_acc,
        "confusion": confusion,
        "diffs_full": diffs_full,
        "preds_full": preds_full,
        "targets_full": targets_full,
        "probs": probs,
        "labels_true": labels_true,
        "nodes": nodes,
        "keys": key_list,
        "metrics": metric_list,
        "mean_diffs": diffs_full.mean(axis=0),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: TargetScaler,
    device: torch.device,
    cfg: dict,
    epoch: Optional[int] = None,
) -> dict:
    model.train()
    total_loss = 0.0
    total_reg = 0.0
    total_cls = 0.0
    total_n = 0
    train_progress_every = int(cfg.get("train_progress_every", 0) or 0)
    train_start = time.time()
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pred_scaled, logits = model(batch)
        loss, reg_loss, cls_loss = combined_loss(pred_scaled, batch["y_re"], logits, batch["y_cl"], scaler, cfg)
        loss.backward()
        clip = float(cfg.get("gradient_clip_norm", 0.0) or 0.0)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        bsz = batch["y_cl"].shape[0]
        if train_progress_every > 0 and (batch_idx == 1 or batch_idx % train_progress_every == 0 or batch_idx == total_batches):
            elapsed = time.time() - train_start
            prefix = f"Epoch {epoch:03d} " if epoch is not None else ""
            print(f"{prefix}train batch {batch_idx}/{total_batches} loss={float(loss.item()):.4f} reg={float(reg_loss.item()):.4f} cls={float(cls_loss.item()):.4f} elapsed={elapsed:.1f}s")
        total_loss += float(loss.item()) * bsz
        total_reg += float(reg_loss.item()) * bsz
        total_cls += float(cls_loss.item()) * bsz
        total_n += bsz
    return {
        "loss_all": total_loss / max(total_n, 1),
        "loss_reg": total_reg / max(total_n, 1),
        "loss_cls": total_cls / max(total_n, 1),
    }


def split_examples(examples: List[TreeExample], cfg: dict) -> Tuple[List[TreeExample], List[TreeExample]]:
    examples = list(examples)
    if cfg.get("shuffle_data", True):
        rng = random.Random(int(cfg["seed"]))
        rng.shuffle(examples)
    split = int(len(examples) * float(cfg["train_fraction"]))
    split = min(max(split, 1), len(examples) - 1)
    return examples[:split], examples[split:]


def make_performance_frame(history: List[dict], cfg: dict) -> pd.DataFrame:
    rows = []
    for h in history:
        row = {
            "Epoch": h["epoch"],
            "Train_Loss_ALL": h["train"]["loss_all"],
            "Train_Loss_Regression": h["train"]["loss_reg"],
            "Train_Loss_Classification": h["train"]["loss_cls"],
            "Test_Loss_ALL": h["test"]["loss_all"],
            "Test_Loss_Regression": h["test"]["loss_reg"],
            "Test_Loss_Classification": h["test"]["loss_cls"],
            "Test_Overall_Accuracy": h["test"]["accuracy"],
        }
        for idx, col in enumerate(DIFF_COLUMNS):
            row[col] = float(h["test"]["mean_diffs"][idx])
        for idx, class_name in enumerate(cfg["class_names"]):
            row[f"Test_Accuracy_{class_name}"] = float(h["test"]["per_class_accuracy"][idx])
        rows.append(row)
    # Put old diff columns first, then epoch/loss/accuracy columns.
    df = pd.DataFrame(rows)
    ordered = DIFF_COLUMNS + [c for c in df.columns if c not in DIFF_COLUMNS]
    return df[ordered]


def make_final_frame(eval_result: dict, cfg: dict) -> pd.DataFrame:
    final_differences = pd.DataFrame(eval_result["diffs_full"], columns=DIFF_COLUMNS)
    final_predictions = pd.DataFrame(eval_result["preds_full"], columns=PRED_COLUMNS)
    final_y = pd.DataFrame(eval_result["targets_full"], columns=PARAM_COLUMNS)
    final_nodes = pd.DataFrame({"num_nodes": eval_result["nodes"].astype(int)})
    prob_cols = list(cfg["class_probability_columns"])
    final_label_prob = pd.DataFrame(eval_result["probs"], columns=prob_cols)
    final_label_true = pd.DataFrame({"true_class": eval_result["labels_true"].astype(int)})
    final_meta = pd.DataFrame({"key": eval_result["keys"], "metric": eval_result["metrics"]})
    return pd.concat([
        final_differences,
        final_predictions,
        final_y,
        final_nodes,
        final_label_prob,
        final_label_true,
        final_meta,
    ], axis=1)


def save_outputs(
    model: nn.Module,
    scaler: TargetScaler,
    cfg: dict,
    output_dir: Path,
    task_type: str,
    run_id: str,
    history: List[dict],
    final_eval: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = str(cfg.get("output_tag", "tree_transformer"))
    model_path = output_dir / f"{task_type}_model_{tag}_{run_id}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": cfg,
        "target_scaler": scaler.to_json(),
    }, model_path)

    perf_df = make_performance_frame(history, cfg)
    final_df = make_final_frame(final_eval, cfg)
    perf_path = output_dir / f"{task_type}_{tag}_{run_id}.rds"
    final_path = output_dir / f"{task_type}_final_{tag}_{run_id}.rds"
    write_rds_or_csv(perf_df, perf_path, bool(cfg.get("write_csv_fallback", True)))
    write_rds_or_csv(final_df, final_path, bool(cfg.get("write_csv_fallback", True)))

    # Extra diagnostics, useful for the paper tables but not required by the old workflow.
    perf_df.to_csv(output_dir / f"{task_type}_{tag}_{run_id}.csv", index=False)
    final_df.to_csv(output_dir / f"{task_type}_final_{tag}_{run_id}.csv", index=False)
    pd.DataFrame(
        final_eval["confusion"],
        index=[f"true_{x}" for x in cfg["class_names"]],
        columns=[f"pred_{x}" for x in cfg["class_names"]],
    ).to_csv(output_dir / f"{task_type}_{tag}_{run_id}_confusion_matrix.csv")
    with open(output_dir / f"{task_type}_{tag}_{run_id}_scaler_config.json", "w", encoding="utf-8") as handle:
        json.dump({"target_scaler": scaler.to_json(), "config": cfg}, handle, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved compatible performance output: {perf_path}")
    print(f"Saved compatible final output: {final_path}")


# ------------------------------- self-test -------------------------------- #


def make_toy_tree(key: str, metric: str, params: Sequence[float], cfg: dict) -> TreeExample:
    # Small rooted binary tree with 5 nodes: 0 root -> 1,2; 1 -> 3,4.
    edge_index = np.array([[0, 0, 1, 1], [1, 2, 3, 4]], dtype=np.int64)
    lengths = {1: 0.6, 2: 1.0, 3: 0.4, 4: 0.4}
    el = np.array([
        [0.6, 1.0, 0.0],
        [0.6, 0.4, 0.4],
        [1.0, 0.0, 0.0],
        [0.4, 0.0, 0.0],
        [0.4, 0.0, 0.0],
    ], dtype=np.float32)
    brts = np.array([1.0, 0.4], dtype=np.float32)
    node_features, root, depth_time, depth_edges, edge_len_parent, parent, children, brts_summary = build_node_features(
        edge_index, el, brts, cfg
    )
    params_full = np.zeros(6, dtype=np.float32)
    params_full[:len(params)] = np.array(params, dtype=np.float32)
    class_to_idx = {name: i for i, name in enumerate(cfg["class_names"])}
    metric_norm = normalize_metric_token(metric, cfg)
    n_pred = int(cfg["n_predicted_values"])
    return TreeExample(
        key=key,
        edge_index=edge_index,
        el_matrix=el,
        brts=brts,
        params_full=params_full,
        y_re=params_full[:n_pred],
        y_cl=class_to_idx[metric_norm],
        metric=metric_norm,
        num_nodes=node_features.shape[0],
        node_features=node_features,
        root_index=root,
        depth_time=depth_time,
        depth_edges=depth_edges,
        edge_len_to_parent=edge_len_parent,
        parent=parent,
        children=children,
        brts_summary=brts_summary,
    )


def run_self_test(cfg: dict) -> None:
    print("Running synthetic self-test; this does not use RDS files or evesim.")
    cfg = dict(cfg)
    cfg["epoch_number_transformer"] = 3
    cfg["train_batch_size"] = 2
    cfg["test_batch_size"] = 2
    cfg["d_model"] = 32
    cfg["num_heads"] = 4
    cfg["num_layers"] = 2
    cfg["dim_feedforward"] = 64
    examples = [
        make_toy_tree("toy_bd_1", "bd", [0.3, 0.05], cfg),
        make_toy_tree("toy_ed_1", "ed", [0.4, 0.10], cfg),
        make_toy_tree("toy_nnd_1", "nnd", [0.5, 0.12], cfg),
        make_toy_tree("toy_bd_2", "bd", [0.35, 0.08], cfg),
        make_toy_tree("toy_ed_2", "ed", [0.45, 0.11], cfg),
        make_toy_tree("toy_nnd_2", "nnd", [0.55, 0.15], cfg),
    ]
    train_examples, test_examples = examples[:4], examples[4:]
    train_dataset = EveTreeDataset(train_examples)
    test_dataset = EveTreeDataset(test_examples)
    collate = TreeBatchCollator(cfg)
    train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate)
    y_train = np.stack([ex.y_re for ex in train_examples])
    scaler = TargetScaler.fit(y_train, enabled=bool(cfg.get("target_standardization", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TreeAwareGraphTransformer(train_dataset.node_feature_dim, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    for epoch in range(1, 3):
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch=epoch)
        eval_stats = evaluate(model, test_loader, scaler, device, cfg, epoch=epoch)
        print(f"self-test epoch {epoch}: train_loss={train_stats['loss_all']:.4f}, test_acc={eval_stats['accuracy']:.4f}")
    print("Self-test completed.")


# -------------------------------- main ------------------------------------ #


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a tree-aware graph transformer for eve tree inference.")
    parser.add_argument("name", nargs="?", help="Base output/data directory, same as the old workflow.")
    parser.add_argument("task_type", nargs="?", help="Task folder, e.g. EVE_FREE_TES.")
    parser.add_argument("run_id", nargs="?", default="1", help="Identifier used in output filenames.")
    parser.add_argument("--config", default=None, help="Path to eve_train_tree_transformer.yaml.")
    parser.add_argument("--self-test", action="store_true", help="Run a small synthetic forward/training test without RDS files.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu or cuda.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    if args.self_test:
        run_self_test(cfg)
        return 0

    if not args.name or not args.task_type:
        parser.error("name and task_type are required unless --self-test is used.")

    name = Path(args.name)
    task_type = str(args.task_type)
    task_dir = name / task_type
    output_dir = task_dir / "STBO"

    print(f"Config path: {cfg.get('_config_path')}")
    print(f"Data directory: {task_dir}")
    print(f"Class names: {cfg['class_names']}")
    print(f"Regression target dimension: {cfg['n_predicted_values']}")

    examples = load_eve_examples(task_dir, cfg)
    print(f"Loaded {len(examples)} usable trees after filtering.")
    counts = pd.Series([ex.metric for ex in examples]).value_counts().to_dict()
    print(f"Class counts: {counts}")
    node_sizes = np.array([ex.num_nodes for ex in examples])
    print(f"Node counts: min={node_sizes.min()}, median={np.median(node_sizes):.1f}, max={node_sizes.max()}")

    train_examples, test_examples = split_examples(examples, cfg)
    print(f"Training dataset length: {len(train_examples)}")
    print(f"Testing dataset length: {len(test_examples)}")

    train_dataset = EveTreeDataset(train_examples)
    test_dataset = EveTreeDataset(test_examples)
    collate = TreeBatchCollator(cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train_batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(cfg["test_batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    y_train = np.stack([ex.y_re for ex in train_examples])
    scaler = TargetScaler.fit(y_train, enabled=bool(cfg.get("target_standardization", True)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training using {device}")

    model = TreeAwareGraphTransformer(train_dataset.node_feature_dim, cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    print(model)

    history: List[dict] = []
    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    patience = int(cfg.get("early_stopping_patience", 0) or 0)
    n_epochs = int(cfg["epoch_number_transformer"])

    for epoch in range(1, n_epochs):
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch=epoch)
        test_stats = evaluate(model, test_loader, scaler, device, cfg, epoch=epoch)
        history.append({"epoch": epoch, "train": train_stats, "test": test_stats})
        per_class = test_stats["per_class_accuracy"]
        per_class_text = ", ".join(
            f"{name}={acc:.4f}" if np.isfinite(acc) else f"{name}=nan"
            for name, acc in zip(cfg["class_names"], per_class)
        )
        mean_diff_text = ", ".join(
            f"{col}={val:.4g}" for col, val in zip(DIFF_COLUMNS[:int(cfg['n_predicted_values'])], test_stats["mean_diffs"][:int(cfg['n_predicted_values'])])
        )
        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_stats['loss_all']:.4f} "
            f"test_loss={test_stats['loss_all']:.4f} "
            f"test_reg={test_stats['loss_reg']:.4f} "
            f"test_cls={test_stats['loss_cls']:.4f} "
            f"acc={test_stats['accuracy']:.4f} "
            f"[{per_class_text}] "
            f"[{mean_diff_text}]"
        )

        if test_stats["loss_all"] < best_loss:
            best_loss = test_stats["loss_all"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}; best test loss={best_loss:.4f}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_eval = evaluate(model, test_loader, scaler, device, cfg)
    print("Final mean absolute differences:")
    for col, value in zip(DIFF_COLUMNS, final_eval["mean_diffs"]):
        print(f"  {col}: {value:.6g}")
    print(f"Final overall accuracy: {final_eval['accuracy']:.6g}")
    print(f"Final per-class accuracy: {final_eval['per_class_accuracy']}")
    print("Confusion matrix, rows=true and columns=pred:")
    print(final_eval["confusion"])

    save_outputs(model, scaler, cfg, output_dir, task_type, str(args.run_id), history, final_eval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
