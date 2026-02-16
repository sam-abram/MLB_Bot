#!/usr/bin/env python3
"""
train_model2.py

Train a PyTorch multiclass model using outputs from preprocessing2.py.

Runnable:
    python train_model2.py
    python train_model2.py /path/to/preprocessed_dir

Assumptions / contract (preprocessing2):
- Reads <PREPROCESSED_DIR>/metadata.json
- Reads train/val/test from <PREPROCESSED_DIR>/:
  - train.parquet, val.parquet, test.parquet if they exist (all three)
  - otherwise train.csv.gz, val.csv.gz, test.csv.gz
- Split datasets are PA-level and include:
  - feature columns listed in metadata.json["features"]["feature_list_ordered"]
  - a target column named "y" (integer class id; int8 OK)
  - optional per-example weights (e.g., "pa_example_weight") may or may not exist
- Categorical feature columns are exactly:
    batter_id, pitcher_id, stadium_id
  These are already integer IDs with reserved semantics:
    PAD/NA = 0
    UNK    = 1
- v2 tendency mixture/uncertainty columns (mix_* and *_logN) are numeric inputs and must be used.


Key compatibility differences vs train_model.py:
- Target column is "y" (already an integer class index; no remapping).
- Labels come from metadata.json["labels"]["label_set_ordered"] and ["label_to_id"].
- Categorical vocab sizes come from metadata.json["features"]["categorical_vocab_sizes"]
  keyed by the exact column names (no suffix stripping). (A small compatibility
  fallback is included for older metadata that stores sizes under meta["categorical_vocabs"]["vocab_sizes"].)
- Example weights are optional; if missing, defaults to 1.0 (and training must not crash).
- Feature order is strictly metadata.json["features"]["feature_list_ordered"]; missing columns
  produce a clear error.
- Saves inference-supporting artifacts: model.pt, train_config.json, label maps, metadata.json copy, metrics.json.

No external services; only local reading/writing.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

# -----------------------------
# Top-of-file configuration constants
# -----------------------------
PREPROCESSED_DIR = "preprocessed"
ARTIFACT_DIR = "model_artifacts"

BATCH_SIZE = 4096
NUM_EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
EMBED_DIM_DEFAULT = 16  # used for categorical embeddings unless overridden by _choose_emb_dim
HIDDEN_DIMS = [256, 128]
DROPOUT = 0.2
EARLY_STOP_PATIENCE = 8  # stop if val loss doesn’t improve
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0  # safe default with IterableDataset
MAX_TRAIN_ROWS = None  # optional cap for quick testing
SEED = 1337
SHUFFLE_TRAIN_ROWS_WITHIN_BATCH = True
GRAD_CLIP_NORM = 1.0

# =========================
# Architecture Options
# =========================
USE_SIMPLE_VECTOR_WEIGHTS = False  # Disable for hybrid experiment
USE_EMBEDDING_INTERACTIONS = False  # Standard concat for clean comparison
USE_HYBRID_MODEL = True  # Hybrid: learnable a*emb_probs + b*vec_probs

# Embedding dimensions (must match for interaction)
BATTER_EMBED_DIM = 32
PITCHER_EMBED_DIM = 32   # Must equal BATTER_EMBED_DIM for element-wise ops
STADIUM_EMBED_DIM = 16
HANDEDNESS_EMBED_DIM = 4  # For stand and p_throws

# IMPORTANT:
# - For probability forecasting, do NOT force class balancing by default.
USE_CLASS_WEIGHTS = False  # optional; if enabled, computed from train split y distribution

# Target and optional per-example weights
TARGET_COL = "y"
WEIGHT_COL = "pa_example_weight"  # optional column; may be missing

# Fixed categorical columns for preprocessing2
CATEGORICAL_COLS = ["batter_id", "pitcher_id", "stadium_id", "stand", "p_throws"]
PAD_NA_ID = 0
UNK_ID = 1

# Sanity checks
SANITY_CHECK_MAX_ROWS = 200_000  # scan up to this many rows from train split for basic checks/logging
RAISE_ON_OOR_CATEGORICAL = False  # if True, raise on out-of-range categorical IDs instead of clamping to UNK

# -----------------------------
# Utilities
# -----------------------------
def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _resolve_splits(pre_dir: str) -> Tuple[str, Dict[str, str]]:
    """
    Returns (format, paths) where format is "parquet" or "csv"
    and paths has keys: train, val, test.
    """
    parquet_paths = {
        "train": os.path.join(pre_dir, "train.parquet"),
        "val": os.path.join(pre_dir, "val.parquet"),
        "test": os.path.join(pre_dir, "test.parquet"),
    }
    csv_paths = {
        "train": os.path.join(pre_dir, "train.csv.gz"),
        "val": os.path.join(pre_dir, "val.csv.gz"),
        "test": os.path.join(pre_dir, "test.csv.gz"),
    }

    parquet_exists = all(os.path.exists(p) for p in parquet_paths.values())
    csv_exists = all(os.path.exists(p) for p in csv_paths.values())

    if parquet_exists:
        return "parquet", parquet_paths
    if csv_exists:
        return "csv", csv_paths

    missing = []
    for p in parquet_paths.values():
        if not os.path.exists(p):
            missing.append(p)
    for p in csv_paths.values():
        if not os.path.exists(p):
            missing.append(p)

    raise FileNotFoundError(
        "Could not find a complete set of split files. Expected either all parquet:\n"
        f"  {list(parquet_paths.values())}\n"
        "or all csv.gz:\n"
        f"  {list(csv_paths.values())}\n"
        f"Missing: {missing}"
    )


def _require_meta(meta: dict, path: List[str], helpful: str) -> object:
    cur: object = meta
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            dotted = ".".join(path)
            raise KeyError(f"{helpful} (missing metadata key: {dotted})")
        cur = cur[key]
    return cur


def _get_feature_order(meta: dict) -> List[str]:
    raw = _require_meta(meta, ["features", "feature_list_ordered"], "metadata.json missing required feature list")
    if not isinstance(raw, list) or not raw:
        raise ValueError("metadata.json meta['features']['feature_list_ordered'] must be a non-empty list")
    cols = [str(c) for c in raw]
    if TARGET_COL in cols:
        # In preprocessing2, y is typically not in the feature list; but be defensive.
        print(f"[WARN] feature_list_ordered contains target column '{TARGET_COL}'. It will be excluded from inputs.")
        cols = [c for c in cols if c != TARGET_COL]
    return cols


def _get_label_info(meta: dict) -> Tuple[List[str], Dict[str, int], Dict[int, str], int]:
    """
    Returns:
      - label_set_ordered: List[str]
      - label_to_index: Dict[str, int]  (same as metadata labels.label_to_id)
      - index_to_label: Dict[int, str]
      - num_classes: int
    """
    labels = _require_meta(meta, ["labels"], "metadata.json missing labels info")
    if not isinstance(labels, dict):
        raise ValueError("metadata.json meta['labels'] must be a dict")

    label_set_ordered = labels.get("label_set_ordered")
    label_to_id = labels.get("label_to_id")

    if not isinstance(label_set_ordered, list) or not label_set_ordered:
        raise KeyError("metadata.json missing meta['labels']['label_set_ordered'] (non-empty list required)")
    if not isinstance(label_to_id, dict) or not label_to_id:
        raise KeyError("metadata.json missing meta['labels']['label_to_id'] (dict required)")

    label_set_ordered = [str(x) for x in label_set_ordered]
    label_to_index = {str(k): int(v) for k, v in label_to_id.items()}

    # Determine num_classes using max id + 1 (robust if IDs are sparse, though they shouldn't be).
    try:
        max_id = max(int(v) for v in label_to_index.values())
    except Exception as e:
        raise ValueError("metadata.json labels.label_to_id must map labels to integer ids") from e

    num_classes = max_id + 1
    if num_classes != len(label_set_ordered):
        print(
            f"[WARN] labels.label_set_ordered length ({len(label_set_ordered)}) "
            f"!= max(labels.label_to_id)+1 ({num_classes}). Using num_classes={num_classes}."
        )

    # Build index_to_label consistent with label_set_ordered when possible.
    if all(label_to_index.get(lab) == i for i, lab in enumerate(label_set_ordered)):
        index_to_label = {i: lab for i, lab in enumerate(label_set_ordered)}
    else:
        print("[WARN] labels.label_to_id does not match enumerate(label_set_ordered). Building index_to_label by id sort.")
        inv = {}
        for lab, idx in label_to_index.items():
            inv[int(idx)] = str(lab)
        index_to_label = {i: inv.get(i, "") for i in range(num_classes)}

    return label_set_ordered, label_to_index, index_to_label, num_classes


def _get_vocab_sizes_from_meta2(meta: dict, cat_cols: List[str]) -> Dict[str, int]:
    """
    Reads categorical vocab sizes for preprocessing2.

    Preferred source (new contract):
      meta["features"]["categorical_vocab_sizes"] keyed by exact column names,
      and values are embedding-ready sizes (include PAD=0 and UNK=1).

    Backward-compatible fallback:
      meta["categorical_vocabs"]["vocab_sizes"] may exist but older variants stored
      len(vocab) (EXCLUDING reserved IDs). If meta["categorical_vocabs"]["vocabs"]
      is present, we derive the correct embedding size as max_id+1.
    """
    if not isinstance(meta, dict):
        raise ValueError("metadata.json root must be a dict")

    source = None
    raw_sizes = None
    vocabs_dict = None

    features = meta.get("features")
    if isinstance(features, dict) and isinstance(features.get("categorical_vocab_sizes"), dict):
        raw_sizes = features["categorical_vocab_sizes"]
        source = "features.categorical_vocab_sizes"
    else:
        cv = meta.get("categorical_vocabs")
        if isinstance(cv, dict):
            raw_sizes = cv.get("vocab_sizes") or cv.get("vocab_size") or cv.get("vocab_sizes_by_col")
            vocabs_dict = cv.get("vocabs") if isinstance(cv.get("vocabs"), dict) else None
            source = "categorical_vocabs.vocab_sizes"

    if not isinstance(raw_sizes, dict):
        raise KeyError(
            "metadata.json missing categorical vocab sizes. Expected either:\n"
            "  meta['features']['categorical_vocab_sizes'] (dict)\n"
            "or fallback:\n"
            "  meta['categorical_vocabs']['vocab_sizes'] (dict)"
        )

    out: Dict[str, int] = {}
    available = sorted(list(raw_sizes.keys()))

    def _derived_size_from_vocab(v: dict) -> int:
        # v maps token -> id, with reserved ids 0/1 and real tokens starting at 2
        if not isinstance(v, dict) or len(v) == 0:
            return 2
        try:
            return int(max(int(x) for x in v.values()) + 1)  # max_id + 1
        except Exception:
            return 2

    for col in cat_cols:
        # If size is missing but we have the vocab itself, derive it.
        if col not in raw_sizes:
            if vocabs_dict and isinstance(vocabs_dict.get(col), dict):
                out[col] = _derived_size_from_vocab(vocabs_dict[col])
                continue
            raise ValueError(
                f"Missing vocab size in metadata for categorical column '{col}'. "
                f"Available vocab keys include: {available[:50]}{'...' if len(available) > 50 else ''}"
            )

        try:
            vs = int(raw_sizes[col])
        except Exception as e:
            raise ValueError(f"Non-integer vocab size for '{col}': {raw_sizes[col]!r}") from e

        # If we're using fallback metadata AND we can derive a safer size from vocabs, do it.
        if source == "categorical_vocabs.vocab_sizes" and vocabs_dict and isinstance(vocabs_dict.get(col), dict):
            derived = _derived_size_from_vocab(vocabs_dict[col])
            if derived > vs:
                print(
                    f"[WARN] metadata {source} for '{col}' is {vs}, "
                    f"but derived embedding size from vocabs is {derived}. Using {derived}."
                )
                vs = derived

        if vs < 2:
            print(f"[WARN] Vocab size for '{col}' is {vs}; clamping to 2 to support PAD/UNK.")
            vs = 2

        out[col] = vs

    return out



def _safe_series_to_int64(s: pd.Series) -> np.ndarray:
    # Fill NaNs and coerce to integer safely
    if s.dtype.kind in ("i", "u"):
        arr = s.to_numpy(copy=False)
        return arr.astype(np.int64, copy=False)
    s2 = s.fillna(PAD_NA_ID)
    if s2.dtype == object:
        s2 = pd.to_numeric(s2, errors="coerce").fillna(PAD_NA_ID)
    return s2.astype(np.int64, copy=False).to_numpy(copy=False)


def _safe_series_to_float32(s: pd.Series) -> np.ndarray:
    if s.dtype.kind == "f":
        return s.fillna(0.0).astype(np.float32, copy=False).to_numpy(copy=False)
    s2 = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return s2.astype(np.float32, copy=False).to_numpy(copy=False)


def _get_file_columns(path: str, file_format: str) -> List[str]:
    """
    Returns column names from a split file without reading the full dataset.
    """
    if file_format == "parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as e:
            raise ImportError(
                "Parquet splits were detected but pyarrow is not available. "
                "Install pyarrow or use csv.gz splits."
            ) from e
        pf = pq.ParquetFile(path)
        return list(pf.schema.names)

    if file_format == "csv":
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            header = f.readline().strip()
        if not header:
            raise ValueError(f"Empty CSV header in file: {path}")
        # Simple split (columns produced by our pipeline have safe names).
        return [c.strip() for c in header.split(",")]

    raise ValueError(f"Unsupported file_format={file_format}")


def _check_required_columns(
    split_name: str,
    split_path: str,
    file_format: str,
    required_cols: List[str],
) -> List[str]:
    cols = _get_file_columns(split_path, file_format)
    col_set = set(cols)
    missing = [c for c in required_cols if c not in col_set]
    if missing:
        preview = cols[:50]
        raise ValueError(
            f"[{split_name}] Missing required columns in split file: {missing}\n"
            f"File: {split_path}\n"
            f"Available columns (first 50): {preview}{'...' if len(cols) > 50 else ''}"
        )
    return cols


def _choose_emb_dim(col_name: str, vocab_size: int) -> int:
    """
    Choose embedding dimension based on column name and vocab size.
    For batter/pitcher, use fixed dimensions to enable interaction.
    """
    if col_name == "batter_id":
        return BATTER_EMBED_DIM
    elif col_name == "pitcher_id":
        return PITCHER_EMBED_DIM
    elif col_name == "stadium_id":
        return STADIUM_EMBED_DIM
    elif col_name in ("stand", "p_throws"):
        return HANDEDNESS_EMBED_DIM
    else:
        # Default heuristic for unknown columns
        vs = int(vocab_size)
        if vs <= 10:
            return 8
        if vs <= 50:
            return 16
        return int(min(64, max(16, round(math.sqrt(vs)))))


def _quick_sanity_scan_train(
    train_path: str,
    file_format: str,
    num_classes: int,
    cat_cols: List[str],
    vocab_sizes: Dict[str, int],
    max_rows: int,
) -> None:
    """
    Lightweight scan of up to `max_rows` from the TRAIN split:
      - checks y min/max and mismatch with num_classes
      - counts out-of-range categorical ids (will be clamped to UNK at runtime unless RAISE_ON_OOR_CATEGORICAL=True)

    This is best-effort and will not crash unless y is invalid/out-of-range.
    """
    scan_cols = [TARGET_COL] + list(cat_cols)
    rows = 0
    y_min = None
    y_max = None
    oor_counts = {c: 0 for c in cat_cols}

    def _update(df: pd.DataFrame) -> None:
        nonlocal rows, y_min, y_max, oor_counts
        if df is None or df.empty:
            return

        # y must exist and be integer-ish
        y_raw = pd.to_numeric(df[TARGET_COL], errors="coerce")
        if y_raw.isna().any():
            bad = int(y_raw.isna().sum())
            raise ValueError(
                f"[SANITY] Found {bad} NaN/non-numeric values in target column '{TARGET_COL}' while scanning train split. "
                "Expected integer class ids."
            )

        y_np = y_raw.astype(np.int64, copy=False).to_numpy(copy=False)
        if y_np.size == 0:
            return

        cur_min = int(y_np.min())
        cur_max = int(y_np.max())
        y_min = cur_min if y_min is None else min(y_min, cur_min)
        y_max = cur_max if y_max is None else max(y_max, cur_max)

        if cur_min < 0:
            raise ValueError(f"[SANITY] Found negative class id in '{TARGET_COL}': min={cur_min}")
        if cur_max >= num_classes:
            raise ValueError(
                f"[SANITY] Found class id >= num_classes in '{TARGET_COL}': max={cur_max}, num_classes={num_classes}. "
                "This indicates a mismatch between metadata labels and the dataset."
            )

        for c in cat_cols:
            vs = int(vocab_sizes[c])
            x = pd.to_numeric(df[c], errors="coerce").fillna(PAD_NA_ID).astype(np.int64, copy=False).to_numpy(copy=False)
            oor = int(((x < 0) | (x >= vs)).sum())
            oor_counts[c] += oor

        rows += len(df)

    if file_format == "parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception:
            print("[SANITY][WARN] pyarrow unavailable; skipping train split sanity scan.")
            return
        pf = pq.ParquetFile(train_path)
        for batch in pf.iter_batches(batch_size=min(50_000, max_rows), columns=scan_cols):
            df = batch.to_pandas()
            _update(df)
            if rows >= max_rows:
                break
    else:
        reader = pd.read_csv(
            train_path,
            compression="gzip",
            usecols=scan_cols,
            chunksize=min(50_000, max_rows),
            low_memory=True,
        )
        for df in reader:
            _update(df)
            if rows >= max_rows:
                break

    # Informational warnings
    if y_min is None or y_max is None:
        print("[SANITY][WARN] Train split sanity scan found no rows.")
        return

    expected = num_classes
    observed = y_max + 1
    if observed != expected:
        # This can happen if some classes do not appear in the sampled rows (or even in full train split).
        print(
            f"[SANITY][WARN] metadata num_classes={expected} but observed max(y)+1={observed} in scanned train rows "
            f"(y_min={y_min}, y_max={y_max}, rows_scanned={rows}). "
            "This is usually OK if some classes are rare/absent in the sample."
        )

    total_oor = sum(oor_counts.values())
    if total_oor > 0:
        msg = ", ".join([f"{c}={oor_counts[c]}" for c in cat_cols])
        action = "raise" if RAISE_ON_OOR_CATEGORICAL else "clamp-to-UNK"
        print(
            f"[SANITY][WARN] Detected out-of-range categorical ids in scanned train rows "
            f"(action during training: {action}). Counts: {msg}."
        )


# -----------------------------
# Dataset
# -----------------------------
class SplitBatchIterableDataset(IterableDataset):
    """
    IterableDataset that streams data in batches from parquet (pyarrow) or csv.gz (pandas chunks),
    yielding (x_cat, x_num, y, w) where each is a tensor batch.

    Requirements met:
    - No full-file reads into RAM
    - Parquet: ParquetFile.iter_batches
    - CSV: pd.read_csv(..., chunksize=...)
    - Strict feature order driven by metadata feature_list_ordered
    - Optional per-example weights (defaults to 1.0 if absent)
    """

    def __init__(
        self,
        path: str,
        file_format: str,
        feature_order: List[str],
        cat_cols: List[str],
        num_cols: List[str],
        vocab_sizes: Dict[str, int],
        num_classes: int,
        batch_size: int,
        available_columns: List[str],
        max_rows: Optional[int] = None,
        shuffle_rows: bool = False,
        seed: int = 0,
        weight_col: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.file_format = file_format
        self.feature_order = list(feature_order)
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.vocab_sizes = {k: int(v) for k, v in vocab_sizes.items()}
        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)
        self.max_rows = None if max_rows is None else int(max_rows)
        self.shuffle_rows = bool(shuffle_rows)
        self._rng = np.random.default_rng(int(seed))
        self.weight_col = weight_col

        self._available = set(str(c) for c in (available_columns or []))

        # Read only needed columns. Include weight column only if it exists in this split.
        self.columns = list(self.feature_order) + [TARGET_COL]
        if self.weight_col is not None and self.weight_col in self._available:
            self.columns.append(self.weight_col)

        # Defensive: ensure required features exist (main should have validated, but keep explicit errors).
        missing = [c for c in self.feature_order if c not in self._available]
        if missing:
            raise ValueError(
                f"Missing required feature columns in {self.path}: {missing}. "
                "This indicates a mismatch between metadata feature_list_ordered and the split file schema."
            )
        if TARGET_COL not in self._available:
            raise ValueError(f"Missing target column '{TARGET_COL}' in {self.path}.")

        # Log control for out-of-range categorical ids (avoid spam).
        self._oor_logged = {c: 0 for c in self.cat_cols}

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.file_format == "parquet":
            yield from self._iter_parquet()
        elif self.file_format == "csv":
            yield from self._iter_csv()
        else:
            raise ValueError(f"Unsupported file_format={self.file_format}")

    def _iter_parquet(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as e:
            raise ImportError(
                "Parquet splits were detected but pyarrow is not available. "
                "Install pyarrow or use csv.gz splits."
            ) from e

        pf = pq.ParquetFile(self.path)
        rows_read = 0

        for batch in pf.iter_batches(batch_size=self.batch_size, columns=self.columns):
            df = batch.to_pandas()
            if df is None or len(df) == 0:
                continue

            if self.max_rows is not None:
                remaining = self.max_rows - rows_read
                if remaining <= 0:
                    break
                if len(df) > remaining:
                    df = df.iloc[:remaining].copy()

            rows_read += len(df)

            if self.shuffle_rows and len(df) > 1:
                rs = int(self._rng.integers(0, 2**31 - 1))
                df = df.sample(frac=1.0, random_state=rs).reset_index(drop=True)

            yield self._df_to_tensors(df)

    def _iter_csv(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        rows_read = 0
        reader = pd.read_csv(
            self.path,
            compression="gzip",
            usecols=self.columns,
            chunksize=self.batch_size,
            low_memory=True,
        )

        for df in reader:
            if df is None or len(df) == 0:
                continue

            if self.max_rows is not None:
                remaining = self.max_rows - rows_read
                if remaining <= 0:
                    break
                if len(df) > remaining:
                    df = df.iloc[:remaining].copy()

            rows_read += len(df)

            if self.shuffle_rows and len(df) > 1:
                rs = int(self._rng.integers(0, 2**31 - 1))
                df = df.sample(frac=1.0, random_state=rs).reset_index(drop=True)

            yield self._df_to_tensors(df)

    def _df_to_tensors(
        self, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Target
        y_raw = pd.to_numeric(df[TARGET_COL], errors="coerce")
        if y_raw.isna().any():
            bad = int(y_raw.isna().sum())
            raise ValueError(
                f"Found {bad} NaN/non-numeric values in target column '{TARGET_COL}' in file {self.path}. "
                "Expected integer class ids from preprocessing2."
            )
        y_np = y_raw.astype(np.int64, copy=False).to_numpy(copy=False)
        if y_np.size == 0:
            # Empty batch is legal; yield empty tensors
            x_cat = torch.empty((0, len(self.cat_cols)), dtype=torch.int64)
            x_num = torch.empty((0, len(self.num_cols)), dtype=torch.float32)
            y = torch.empty((0,), dtype=torch.int64)
            w = torch.empty((0,), dtype=torch.float32)
            return x_cat, x_num, y, w

        y_min = int(y_np.min())
        y_max = int(y_np.max())
        if y_min < 0 or y_max >= self.num_classes:
            raise ValueError(
                f"Target '{TARGET_COL}' out of range in batch from {self.path}: "
                f"min={y_min}, max={y_max}, num_classes={self.num_classes}. "
                "This indicates a mismatch between metadata labels and the encoded dataset."
            )

        # Features: categorical
        if self.cat_cols:
            cat_arrays = []
            for c in self.cat_cols:
                arr = _safe_series_to_int64(df[c])
                vs = int(self.vocab_sizes[c])

                oor_mask = (arr < 0) | (arr >= vs)
                oor = int(oor_mask.sum())
                if oor > 0:
                    if RAISE_ON_OOR_CATEGORICAL:
                        min_bad = int(arr[oor_mask].min()) if oor > 0 else 0
                        max_bad = int(arr[oor_mask].max()) if oor > 0 else 0
                        raise ValueError(
                            f"Out-of-range categorical IDs for column '{c}' in {self.path}: "
                            f"bad_count={oor}, bad_min={min_bad}, bad_max={max_bad}, vocab_size={vs}. "
                            "This indicates a mismatch between metadata vocab sizes and encoded IDs."
                        )

                    # Clamp to UNK and log a few times.
                    arr = arr.copy()
                    arr[oor_mask] = UNK_ID
                    if self._oor_logged.get(c, 0) < 3:
                        print(
                            f"[WARN] Out-of-range IDs in '{c}' ({oor} rows) while reading {os.path.basename(self.path)}. "
                            f"Clamping to UNK={UNK_ID}. (vocab_size={vs})"
                        )
                        self._oor_logged[c] = self._oor_logged.get(c, 0) + 1

                cat_arrays.append(arr)

            x_cat_np = np.stack(cat_arrays, axis=1) if len(cat_arrays) > 1 else cat_arrays[0].reshape(-1, 1)
            x_cat = torch.from_numpy(x_cat_np.astype(np.int64, copy=False))
        else:
            x_cat = torch.empty((len(df), 0), dtype=torch.int64)

        # Features: numeric
        if self.num_cols:
            num_arrays = [_safe_series_to_float32(df[c]) for c in self.num_cols]
            x_num_np = np.stack(num_arrays, axis=1) if len(num_arrays) > 1 else num_arrays[0].reshape(-1, 1)
            x_num = torch.from_numpy(x_num_np.astype(np.float32, copy=False))
        else:
            x_num = torch.empty((len(df), 0), dtype=torch.float32)

        y = torch.from_numpy(y_np.astype(np.int64, copy=False))

        # Weights (default = 1.0)
        if self.weight_col is not None and self.weight_col in df.columns:
            w_np = pd.to_numeric(df[self.weight_col], errors="coerce").fillna(1.0).to_numpy(dtype=np.float32, copy=False)
            # Guard against negative weights (shouldn't happen).
            w_np = np.maximum(w_np, 0.0)
            w = torch.from_numpy(w_np)
        else:
            w = torch.ones((len(df),), dtype=torch.float32)

        return x_cat, x_num, y, w


# -----------------------------
# Model
# -----------------------------
class PitchOutcomeModel(nn.Module):
    def __init__(
        self,
        cat_cols: List[str],
        num_numeric: int,
        vocab_sizes: Dict[str, int],
        num_classes: int,
        hidden_dims: List[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.num_numeric = int(num_numeric)
        self.num_norm = nn.LayerNorm(self.num_numeric) if self.num_numeric > 0 else None
        self.vocab_sizes = {k: int(v) for k, v in vocab_sizes.items()}
        self.num_classes = int(num_classes)

        self.embeddings = nn.ModuleDict()
        emb_out_dim = 0
        self.emb_dims: Dict[str, int] = {}

        for col in self.cat_cols:
            vs = int(self.vocab_sizes[col])
            ed = _choose_emb_dim(col, vs)
            self.emb_dims[col] = ed
            # padding_idx=0 corresponds to PAD/NA.
            self.embeddings[col] = nn.Embedding(num_embeddings=vs, embedding_dim=ed, padding_idx=PAD_NA_ID)
            emb_out_dim += ed

        in_dim = emb_out_dim + self.num_numeric
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, self.num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        parts: List[torch.Tensor] = []

        if self.cat_cols and x_cat.numel() > 0:
            # x_cat shape: [B, n_cat]
            for i, col in enumerate(self.cat_cols):
                ids = x_cat[:, i]
                vs = self.vocab_sizes[col]

                if RAISE_ON_OOR_CATEGORICAL and ids.numel() > 0:
                    min_id = int(ids.min().item())
                    max_id = int(ids.max().item())
                    if min_id < 0 or max_id >= vs:
                        raise ValueError(
                            f"Out-of-range categorical IDs for column '{col}': "
                            f"min={min_id}, max={max_id}, vocab_size={vs}. "
                            "This indicates a mismatch between metadata vocab sizes and encoded IDs."
                        )

                emb = self.embeddings[col](ids)
                parts.append(emb)

        if x_num is not None and x_num.numel() > 0:
            if self.num_norm is not None:
                x_num = self.num_norm(x_num)
            parts.append(x_num)

        if parts:
            x = torch.cat(parts, dim=1)
        else:
            x = torch.empty((x_cat.shape[0], 0), device=x_cat.device)

        logits = self.mlp(x)
        return logits


class PitchOutcomeModelWithInteractions(nn.Module):
    """
    Model with explicit batter x pitcher embedding interactions.

    Instead of just concatenating embeddings, we compute:
    - Element-wise product: batter_emb * pitcher_emb
    - Element-wise difference: batter_emb - pitcher_emb
    - Absolute difference: |batter_emb - pitcher_emb|

    These interaction features help the model learn matchup-specific patterns
    that are hard to capture with simple concatenation.
    """

    def __init__(
        self,
        cat_cols: List[str],
        num_numeric: int,
        vocab_sizes: Dict[str, int],
        num_classes: int,
        hidden_dims: List[int],
        dropout: float,
    ) -> None:
        super().__init__()

        self.cat_cols = list(cat_cols)
        self.num_numeric = int(num_numeric)
        self.vocab_sizes = {k: int(v) for k, v in vocab_sizes.items()}
        self.num_classes = int(num_classes)

        # Build embeddings
        self.embeddings = nn.ModuleDict()
        self.emb_dims: Dict[str, int] = {}

        for col in self.cat_cols:
            vs = self.vocab_sizes[col]
            ed = _choose_emb_dim(col, vs)
            self.emb_dims[col] = ed
            self.embeddings[col] = nn.Embedding(
                num_embeddings=vs,
                embedding_dim=ed,
                padding_idx=PAD_NA_ID,
            )

        # Verify batter and pitcher have same dimension for interaction
        if "batter_id" in self.emb_dims and "pitcher_id" in self.emb_dims:
            assert self.emb_dims["batter_id"] == self.emb_dims["pitcher_id"], \
                f"Batter and pitcher embedding dims must match for interaction: " \
                f"{self.emb_dims['batter_id']} vs {self.emb_dims['pitcher_id']}"
            self.interaction_dim = self.emb_dims["batter_id"]
            self.use_interactions = True
        else:
            self.interaction_dim = 0
            self.use_interactions = False

        # Calculate input dimension to MLP
        base_emb_dim = sum(self.emb_dims.values())

        if self.use_interactions:
            # Add 3 interaction vectors: product, diff, abs_diff
            interaction_features = self.interaction_dim * 3
        else:
            interaction_features = 0

        mlp_input_dim = base_emb_dim + interaction_features + self.num_numeric

        # Layer norm for numeric features
        self.num_norm = nn.LayerNorm(self.num_numeric) if self.num_numeric > 0 else None

        # Build MLP
        layers: List[nn.Module] = []
        prev = mlp_input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, self.num_classes))
        self.mlp = nn.Sequential(*layers)

        # Log architecture
        print(f"[MODEL] PitchOutcomeModelWithInteractions:")
        print(f"  Embedding dims: {self.emb_dims}")
        print(f"  Interaction dim: {self.interaction_dim} x 3 = {interaction_features}")
        print(f"  Numeric features: {self.num_numeric}")
        print(f"  MLP input dim: {mlp_input_dim}")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Output classes: {self.num_classes}")

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        parts: List[torch.Tensor] = []

        batter_emb = None
        pitcher_emb = None

        # Compute embeddings
        for i, col in enumerate(self.cat_cols):
            ids = x_cat[:, i]
            vs = self.vocab_sizes[col]

            # Clamp out-of-range IDs to UNK
            ids = torch.clamp(ids, min=0, max=vs - 1)

            emb = self.embeddings[col](ids)
            parts.append(emb)

            # Store batter/pitcher for interaction
            if col == "batter_id":
                batter_emb = emb
            elif col == "pitcher_id":
                pitcher_emb = emb

        # Compute interactions if both embeddings exist
        if self.use_interactions and batter_emb is not None and pitcher_emb is not None:
            # Element-wise product: captures "how do these tendencies combine"
            bp_product = batter_emb * pitcher_emb

            # Difference: captures "who dominates"
            bp_diff = batter_emb - pitcher_emb

            # Absolute difference: captures "magnitude of style mismatch"
            bp_abs_diff = torch.abs(bp_diff)

            parts.append(bp_product)
            parts.append(bp_diff)
            parts.append(bp_abs_diff)

        # Add numeric features
        if x_num is not None and x_num.numel() > 0:
            if self.num_norm is not None:
                x_num = self.num_norm(x_num)
            parts.append(x_num)

        # Concatenate all features
        if parts:
            x = torch.cat(parts, dim=1)
        else:
            x = torch.empty((x_cat.shape[0], 0), device=x_cat.device)

        # MLP
        logits = self.mlp(x)
        return logits


class SimpleVectorWeightModel(nn.Module):
    """
    Simple weighted combination of six probability vectors.

    P(outcome) = a*v1 + b*v2 + c*v3 + d*v4 + e*v5 + f*v6

    Where weights sum to 1 (enforced via softmax).
    With log_n weighting, weights are per-example based on sample sizes.
    """

    def __init__(
        self,
        num_vectors: int = 6,
        num_outcomes: int = 6,
        use_log_n_weighting: bool = False,
    ) -> None:
        super().__init__()
        self.num_vectors = num_vectors
        self.num_outcomes = num_outcomes
        self.num_classes = num_outcomes
        self.use_log_n_weighting = use_log_n_weighting
        self.cat_cols: List[str] = []
        self.emb_dims: Dict[str, int] = {}

        # Initialize to equal weights (zeros -> softmax -> 1/6 each)
        self.raw_weights = nn.Parameter(torch.zeros(num_vectors))

        if use_log_n_weighting:
            self.log_n_scale = nn.Parameter(torch.zeros(num_vectors))

        n_params = num_vectors + (num_vectors if use_log_n_weighting else 0)
        print(f"[MODEL] SimpleVectorWeightModel:")
        print(f"  Num vectors: {num_vectors}")
        print(f"  Num outcomes: {num_outcomes}")
        print(f"  Use log_n weighting: {use_log_n_weighting}")
        print(f"  Total parameters: {n_params}")

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        batch_size = x_num.shape[0]

        # Feature order from _build_feature_list():
        # 0: pitcher_fatigue
        # 1-6: v1 rates, 7: v1_log_n
        # 8-13: v2 rates, 14: v2_log_n
        # 15-20: v3 rates, 21: v3_log_n
        # 22-27: v4 rates, 28: v4_log_n
        # 29-34: v5 rates, 35: v5_log_n
        # 36-41: v6 rates, 42: v6_log_n
        v1 = x_num[:, 1:7]
        v2 = x_num[:, 8:14]
        v3 = x_num[:, 15:21]
        v4 = x_num[:, 22:28]
        v5 = x_num[:, 29:35]
        v6 = x_num[:, 36:42]

        # Stack: (batch_size, 6_vectors, 6_outcomes)
        vectors = torch.stack([v1, v2, v3, v4, v5, v6], dim=1)

        if self.use_log_n_weighting:
            log_n = torch.stack([
                x_num[:, 7], x_num[:, 14], x_num[:, 21],
                x_num[:, 28], x_num[:, 35], x_num[:, 42],
            ], dim=1)  # (batch_size, 6)
            adjusted = self.raw_weights.unsqueeze(0) + self.log_n_scale.unsqueeze(0) * log_n
            weights = torch.softmax(adjusted, dim=1)  # (batch_size, 6)
        else:
            weights = torch.softmax(self.raw_weights, dim=0)  # (6,)
            weights = weights.unsqueeze(0).expand(batch_size, -1)  # (batch_size, 6)

        # Weighted sum: (batch_size, 6_vectors, 6_outcomes) * (batch_size, 6_vectors, 1)
        predicted_probs = (vectors * weights.unsqueeze(2)).sum(dim=1)

        # Convert to logits for cross-entropy
        logits = torch.log(predicted_probs + 1e-8)
        return logits

    def get_learned_weights(self) -> Dict[str, float]:
        with torch.no_grad():
            weights = torch.softmax(self.raw_weights, dim=0).cpu().numpy()
        return {
            "v1_batter_overall": float(weights[0]),
            "v2_pitcher_overall": float(weights[1]),
            "v3_stadium": float(weights[2]),
            "v4_batter_platoon": float(weights[3]),
            "v5_pitcher_platoon": float(weights[4]),
            "v6_pitch_mix": float(weights[5]),
        }


class HybridModel(nn.Module):
    """
    Hybrid model: learns a = softmax(raw_ab)[0], b = softmax(raw_ab)[1]
    such that combined_probs = a * emb_probs + b * vec_probs.

    Embedding head: categorical embeddings -> MLP -> softmax -> 6 probs
    Vector head: weighted sum of 6 probability vectors (like SimpleVectorWeightModel)
    """

    def __init__(
        self,
        cat_cols: List[str],
        num_numeric: int,
        vocab_sizes: Dict[str, int],
        num_classes: int,
        hidden_dims: List[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.num_classes = int(num_classes)

        # --- Embedding head ---
        self.embeddings = nn.ModuleDict()
        self.emb_dims: Dict[str, int] = {}
        emb_out_dim = 0
        for col in self.cat_cols:
            vs = int(vocab_sizes[col])
            ed = _choose_emb_dim(col, vs)
            self.emb_dims[col] = ed
            self.embeddings[col] = nn.Embedding(
                num_embeddings=vs, embedding_dim=ed, padding_idx=PAD_NA_ID
            )
            emb_out_dim += ed
        self.vocab_sizes = {k: int(v) for k, v in vocab_sizes.items()}

        # Embedding MLP: embeddings -> hidden -> num_classes (logits)
        emb_layers: List[nn.Module] = []
        prev = emb_out_dim
        for h in hidden_dims:
            emb_layers.append(nn.Linear(prev, int(h)))
            emb_layers.append(nn.ReLU())
            emb_layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        emb_layers.append(nn.Linear(prev, self.num_classes))
        self.emb_mlp = nn.Sequential(*emb_layers)

        # --- Vector head (same as SimpleVectorWeightModel with log_n) ---
        self.num_vectors = 6
        self.raw_vec_weights = nn.Parameter(torch.zeros(self.num_vectors))
        self.log_n_scale = nn.Parameter(torch.zeros(self.num_vectors))

        # --- Combination weights: a (embedding), b (vector) ---
        self.raw_ab = nn.Parameter(torch.zeros(2))

        # Log architecture
        print(f"[MODEL] HybridModel:")
        print(f"  Embedding head: {self.emb_dims} -> MLP {hidden_dims} -> {self.num_classes}")
        print(f"  Vector head: {self.num_vectors} vectors with log_n weighting")
        print(f"  Combination: softmax(raw_ab) -> a*emb_probs + b*vec_probs")

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        # --- Embedding head ---
        emb_parts: List[torch.Tensor] = []
        for i, col in enumerate(self.cat_cols):
            ids = x_cat[:, i]
            vs = self.vocab_sizes[col]
            ids = torch.clamp(ids, min=0, max=vs - 1)
            emb_parts.append(self.embeddings[col](ids))
        emb_concat = torch.cat(emb_parts, dim=1)
        emb_logits = self.emb_mlp(emb_concat)
        emb_probs = torch.softmax(emb_logits, dim=1)  # (B, 6)

        # --- Vector head ---
        # Feature layout: [pitcher_fatigue, v1(6+1), v2(6+1), ..., v6(6+1)]
        v1 = x_num[:, 1:7]
        v2 = x_num[:, 8:14]
        v3 = x_num[:, 15:21]
        v4 = x_num[:, 22:28]
        v5 = x_num[:, 29:35]
        v6 = x_num[:, 36:42]

        vectors = torch.stack([v1, v2, v3, v4, v5, v6], dim=1)  # (B, 6, 6)

        log_n = torch.stack([
            x_num[:, 7], x_num[:, 14], x_num[:, 21],
            x_num[:, 28], x_num[:, 35], x_num[:, 42],
        ], dim=1)  # (B, 6)

        adjusted = self.raw_vec_weights.unsqueeze(0) + self.log_n_scale.unsqueeze(0) * log_n
        vec_weights = torch.softmax(adjusted, dim=1)  # (B, 6)
        vec_probs = (vectors * vec_weights.unsqueeze(2)).sum(dim=1)  # (B, 6)

        # --- Combine ---
        ab = torch.softmax(self.raw_ab, dim=0)  # (2,)
        combined_probs = ab[0] * emb_probs + ab[1] * vec_probs  # (B, 6)

        logits = torch.log(combined_probs + 1e-8)
        return logits

    def get_learned_weights(self) -> Dict[str, float]:
        """Return the learned vector head weights (v1-v6)."""
        with torch.no_grad():
            weights = torch.softmax(self.raw_vec_weights, dim=0).cpu().numpy()
        return {
            "v1_batter_overall": float(weights[0]),
            "v2_pitcher_overall": float(weights[1]),
            "v3_stadium": float(weights[2]),
            "v4_batter_platoon": float(weights[3]),
            "v5_pitcher_platoon": float(weights[4]),
            "v6_pitch_mix": float(weights[5]),
        }

    def get_ab_weights(self) -> Dict[str, float]:
        """Return the learned a (embedding) and b (vector) combination weights."""
        with torch.no_grad():
            ab = torch.softmax(self.raw_ab, dim=0).cpu().numpy()
        return {
            "a_embedding": float(ab[0]),
            "b_vector": float(ab[1]),
        }


class StatcastLogitHybridModel(nn.Module):
    """
    Logit-space hybrid with pitch-type statcast block fed to embedding head.

    Embedding head input: concat(categorical_embeddings, x_pt)  where x_pt is the 90-col block
    Vector head: same as before (weighted sum of v1-v6 from the first 43 numeric cols)
    Combination: a * emb_logits + b * log(vec_probs)
    """

    def __init__(
        self,
        cat_cols: List[str],
        num_numeric: int,
        vocab_sizes: Dict[str, int],
        num_classes: int,
        hidden_dims: List[int],
        dropout: float,
        num_pt_statcast_cols: int = 90,
    ) -> None:
        super().__init__()
        self.cat_cols = list(cat_cols)
        self.num_classes = int(num_classes)
        self.vocab_sizes = {k: int(v) for k, v in vocab_sizes.items()}
        self.num_pt_statcast_cols = int(num_pt_statcast_cols)

        # --- Embedding head ---
        self.embeddings = nn.ModuleDict()
        self.emb_dims: Dict[str, int] = {}
        emb_out_dim = 0
        for col in self.cat_cols:
            vs = int(vocab_sizes[col])
            ed = _choose_emb_dim(col, vs)
            self.emb_dims[col] = ed
            self.embeddings[col] = nn.Embedding(
                num_embeddings=vs, embedding_dim=ed, padding_idx=PAD_NA_ID
            )
            emb_out_dim += ed

        # LayerNorm for the statcast block before feeding to MLP
        self.pt_norm = nn.LayerNorm(self.num_pt_statcast_cols)

        # Embedding MLP input: embeddings + statcast block
        mlp_in = emb_out_dim + self.num_pt_statcast_cols
        emb_layers: List[nn.Module] = []
        prev = mlp_in
        for h in hidden_dims:
            emb_layers.append(nn.Linear(prev, int(h)))
            emb_layers.append(nn.ReLU())
            emb_layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        emb_layers.append(nn.Linear(prev, self.num_classes))
        self.emb_mlp = nn.Sequential(*emb_layers)

        # --- Vector head (unchanged) ---
        self.num_vectors = 6
        self.raw_vec_weights = nn.Parameter(torch.zeros(self.num_vectors))
        self.log_n_scale = nn.Parameter(torch.zeros(self.num_vectors))

        # --- Combination weights ---
        self.raw_ab = nn.Parameter(torch.zeros(2))

        print(f"[MODEL] StatcastLogitHybridModel:")
        print(f"  Embedding head: {self.emb_dims} + {self.num_pt_statcast_cols} statcast cols -> MLP {hidden_dims} -> {self.num_classes}")
        print(f"  MLP input dim: {mlp_in}")
        print(f"  Vector head: {self.num_vectors} vectors with log_n weighting (unchanged)")
        print(f"  Combination: a * emb_logits + b * log(vec_probs)")

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        # --- Embedding head with statcast block ---
        emb_parts: List[torch.Tensor] = []
        for i, col in enumerate(self.cat_cols):
            ids = torch.clamp(x_cat[:, i], min=0, max=self.vocab_sizes[col] - 1)
            emb_parts.append(self.embeddings[col](ids))

        # Statcast block: last num_pt_statcast_cols columns of x_num
        x_pt = x_num[:, -self.num_pt_statcast_cols:]
        x_pt = self.pt_norm(x_pt)

        emb_input = torch.cat(emb_parts + [x_pt], dim=1)
        emb_logits = self.emb_mlp(emb_input)

        # --- Vector head (uses first 43 cols, indices unchanged) ---
        v1 = x_num[:, 1:7]
        v2 = x_num[:, 8:14]
        v3 = x_num[:, 15:21]
        v4 = x_num[:, 22:28]
        v5 = x_num[:, 29:35]
        v6 = x_num[:, 36:42]

        vectors = torch.stack([v1, v2, v3, v4, v5, v6], dim=1)

        log_n = torch.stack([
            x_num[:, 7], x_num[:, 14], x_num[:, 21],
            x_num[:, 28], x_num[:, 35], x_num[:, 42],
        ], dim=1)

        adjusted = self.raw_vec_weights.unsqueeze(0) + self.log_n_scale.unsqueeze(0) * log_n
        vec_weights = torch.softmax(adjusted, dim=1)
        vec_probs = (vectors * vec_weights.unsqueeze(2)).sum(dim=1)
        vec_logits = torch.log(vec_probs + 1e-8)

        # --- Combine in logit space ---
        ab = torch.softmax(self.raw_ab, dim=0)
        combined_logits = ab[0] * emb_logits + ab[1] * vec_logits

        return combined_logits

    def get_learned_weights(self) -> Dict[str, float]:
        with torch.no_grad():
            w = torch.softmax(self.raw_vec_weights, dim=0).cpu().numpy()
        return {
            "v1_batter_overall": float(w[0]),
            "v2_pitcher_overall": float(w[1]),
            "v3_stadium": float(w[2]),
            "v4_batter_platoon": float(w[3]),
            "v5_pitcher_platoon": float(w[4]),
            "v6_pitch_mix": float(w[5]),
        }

    def get_ab_weights(self) -> Dict[str, float]:
        with torch.no_grad():
            ab = torch.softmax(self.raw_ab, dim=0).cpu().numpy()
        return {"a_embedding": float(ab[0]), "b_vector": float(ab[1])}


# -----------------------------
# Training / Evaluation
# -----------------------------
@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    n: int


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: str,
) -> EpochMetrics:
    model.eval()
    loss_num = 0.0
    loss_den = 0.0
    total_correct = 0
    total_n = 0

    for x_cat, x_num, y, w in dataset:
        if y.numel() == 0:
            continue

        x_cat = x_cat.to(device, non_blocking=True)
        x_num = x_num.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True).float()

        logits = model(x_cat, x_num)
        per_ex_loss = criterion(logits, y)  # (N,)

        batch_num = (per_ex_loss * w).sum()
        batch_den = w.sum().clamp_min(1e-12)

        n = int(y.numel())
        loss_num += float(batch_num.item())
        loss_den += float(batch_den.item())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += n

    if total_n == 0:
        return EpochMetrics(loss=float("nan"), accuracy=float("nan"), n=0)

    return EpochMetrics(
        loss=(loss_num / max(loss_den, 1e-12)),
        accuracy=total_correct / total_n,
        n=total_n,
    )


def train_one_epoch(
    model: nn.Module,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> EpochMetrics:
    model.train()
    loss_num = 0.0
    loss_den = 0.0
    total_correct = 0
    total_n = 0

    for x_cat, x_num, y, w in dataset:
        if y.numel() == 0:
            continue

        x_cat = x_cat.to(device, non_blocking=True)
        x_num = x_num.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_cat, x_num)

        per_ex_loss = criterion(logits, y)  # (N,)
        batch_num = (per_ex_loss * w).sum()
        batch_den = w.sum().clamp_min(1e-12)
        loss = batch_num / batch_den

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        n = int(y.numel())
        loss_num += float(batch_num.item())
        loss_den += float(batch_den.item())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += n

    if total_n == 0:
        return EpochMetrics(loss=float("nan"), accuracy=float("nan"), n=0)

    return EpochMetrics(
        loss=(loss_num / max(loss_den, 1e-12)),
        accuracy=total_correct / total_n,
        n=total_n,
    )


def _compute_class_weights_from_y(
    train_path: str, file_format: str, num_classes: int, max_rows: Optional[int] = None
) -> torch.Tensor:
    """
    Optional class weights computed from the training split's y distribution.
    Uses an inverse-frequency heuristic: weight_c = total / (num_classes * count_c),
    with small eps to avoid division-by-zero.
    """
    counts = np.zeros((num_classes,), dtype=np.int64)
    rows = 0

    def _update(df: pd.DataFrame) -> None:
        nonlocal counts, rows
        y_raw = pd.to_numeric(df[TARGET_COL], errors="coerce")
        y_raw = y_raw.dropna()
        if y_raw.empty:
            return
        y_np = y_raw.astype(np.int64, copy=False).to_numpy(copy=False)
        if y_np.size == 0:
            return
        if y_np.min() < 0 or y_np.max() >= num_classes:
            raise ValueError(
                f"Target '{TARGET_COL}' out of range while computing class weights: "
                f"min={int(y_np.min())}, max={int(y_np.max())}, num_classes={num_classes}"
            )
        counts += np.bincount(y_np, minlength=num_classes)
        rows += len(df)

    if file_format == "parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as e:
            raise ImportError("USE_CLASS_WEIGHTS=True requires pyarrow for parquet splits.") from e
        pf = pq.ParquetFile(train_path)
        for batch in pf.iter_batches(batch_size=50_000, columns=[TARGET_COL]):
            df = batch.to_pandas()
            _update(df)
            if max_rows is not None and rows >= max_rows:
                break
    else:
        reader = pd.read_csv(
            train_path,
            compression="gzip",
            usecols=[TARGET_COL],
            chunksize=50_000,
            low_memory=True,
        )
        for df in reader:
            _update(df)
            if max_rows is not None and rows >= max_rows:
                break

    total = float(counts.sum())
    eps = 1e-9
    denom = np.maximum(counts.astype(np.float64), eps)
    weights = total / (num_classes * denom)
    # Avoid extreme blow-ups for completely absent classes
    weights = np.minimum(weights, np.percentile(weights, 95) * 10.0) if np.isfinite(weights).all() else weights
    return torch.tensor(weights.astype(np.float32), dtype=torch.float32)

@torch.no_grad()
def evaluate_logloss_only(
    model: nn.Module,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: str,
) -> float:
    """
    Unweighted multiclass logloss (cross-entropy) on a split.
    This scores the *probability distribution* produced by the model:
      logloss = mean_i[-log p_model(y_i)]
    If example weights are present in the dataset iterator, we apply them.
    """
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="none")  # no class weights; proper logloss

    loss_num = 0.0
    loss_den = 0.0

    for x_cat, x_num, y, w in dataset:
        if y.numel() == 0:
            continue

        x_cat = x_cat.to(device, non_blocking=True)
        x_num = x_num.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True).float()

        logits = model(x_cat, x_num)
        per_ex = crit(logits, y)  # (N,)
        loss_num += float((per_ex * w).sum().item())
        loss_den += float(w.sum().clamp_min(1e-12).item())

    return float(loss_num / max(loss_den, 1e-12))


def _stream_y_w(
    path: str,
    file_format: str,
    weight_col: Optional[str],
    batch_rows: int = 200_000,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Stream (y, w) from a split file without loading full data.
    y: int64 array
    w: float64 array (all 1.0 if weight col missing)
    """
    cols = [TARGET_COL]
    if weight_col is not None:
        cols.append(weight_col)

    if file_format == "parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as e:
            raise ImportError("Parquet splits detected but pyarrow is not available.") from e

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
            df = batch.to_pandas()
            if df is None or len(df) == 0:
                continue

            y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(np.int64, copy=False).to_numpy()
            if weight_col is not None and weight_col in df.columns:
                w = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0).astype(np.float64, copy=False).to_numpy()
                w = np.maximum(w, 0.0)
            else:
                w = np.ones((len(df),), dtype=np.float64)

            yield y, w
        return

    if file_format == "csv":
        reader = pd.read_csv(
            path,
            compression="gzip",
            usecols=cols,
            chunksize=batch_rows,
            low_memory=True,
        )
        for df in reader:
            if df is None or len(df) == 0:
                continue

            y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(np.int64, copy=False).to_numpy()
            if weight_col is not None and weight_col in df.columns:
                w = pd.to_numeric(df[weight_col], errors="coerce").fillna(1.0).astype(np.float64, copy=False).to_numpy()
                w = np.maximum(w, 0.0)
            else:
                w = np.ones((len(df),), dtype=np.float64)

            yield y, w
        return

    raise ValueError(f"Unsupported file_format={file_format}")


def compute_distribution_scorecard(
    model: nn.Module,
    train_path: str,
    test_path: str,
    file_format: str,
    num_classes: int,
    test_loader: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: str,
    weight_col_for_baselines: Optional[str],
) -> Dict[str, float]:
    """
    Computes:
      - uniform_logloss
      - base_rate_logloss (train freq baseline evaluated on test)
      - model_test_logloss (proper logloss on test)
      - pct_better_than_base_rates (positive means model beats base rates)
      - delta_logloss_vs_base_rates (base - model; positive means better)
      - majority_class_accuracy (always predicting train-majority class, measured on test)
    """
    K = int(num_classes)
    eps = 1e-12

    # --- uniform baseline ---
    uniform_logloss = float(math.log(K))

    # --- train base-rate distribution p_train ---
    counts = np.zeros((K,), dtype=np.float64)
    total_w = 0.0
    for y, w in _stream_y_w(train_path, file_format, weight_col_for_baselines):
        # weighted frequency if weights exist; otherwise plain counts
        counts += np.bincount(y, minlength=K).astype(np.float64) if w is None else np.bincount(y, minlength=K, weights=w)
        total_w += float(w.sum())
    if total_w <= 0:
        raise ValueError("Train split appears empty (cannot compute base rates).")

    p = counts / max(counts.sum(), eps)
    p = np.clip(p, eps, 1.0)  # avoid log(0)
    p = p / p.sum()

    maj = int(np.argmax(p))

    # --- base-rate logloss on test + majority acc on test ---
    loss_sum = 0.0
    w_sum = 0.0
    maj_correct = 0.0

    for y, w in _stream_y_w(test_path, file_format, weight_col_for_baselines):
        w = w.astype(np.float64, copy=False)
        w_sum += float(w.sum())
        loss_sum += float((-np.log(p[y]) * w).sum())
        maj_correct += float(((y == maj).astype(np.float64) * w).sum())

    if w_sum <= 0:
        raise ValueError("Test split appears empty (cannot compute scorecard).")

    base_rate_logloss = float(loss_sum / w_sum)
    majority_class_accuracy = float(maj_correct / w_sum)

    # --- model logloss on test (proper scoring rule) ---
    model_test_logloss = float(evaluate_logloss_only(model, test_loader, device))

    delta = float(base_rate_logloss - model_test_logloss)  # positive => model better
    pct_better = float(100.0 * delta / max(base_rate_logloss, eps))

    return {
        "uniform_logloss": uniform_logloss,
        "base_rate_logloss": base_rate_logloss,
        "model_test_logloss": model_test_logloss,
        "delta_logloss_vs_base_rates": delta,
        "pct_better_than_base_rates": pct_better,
        "majority_class_accuracy": majority_class_accuracy,
    }

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument("preprocessed_dir", nargs="?", default=PREPROCESSED_DIR)
    _parser.add_argument("--use_pitchtype_statcast_block", action="store_true", default=False,
                         help="Feed 90 pitch-type statcast cols into the embedding head")
    _parser.add_argument("--use_pitchtype_statcast_v2", action="store_true", default=False,
                         help="Feed 210 pitch-type statcast v2 cols into the embedding head")
    _parser.add_argument("--artifact_dir", type=str, default=None,
                         help="Override artifact output directory")
    _args = _parser.parse_args()
    pre_dir = _args.preprocessed_dir
    use_pt_statcast = _args.use_pitchtype_statcast_block or _args.use_pitchtype_statcast_v2
    artifact_dir = _args.artifact_dir if _args.artifact_dir else ARTIFACT_DIR

    # Reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    meta_path = os.path.join(pre_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    meta = _read_json(meta_path)

    # Feature order from metadata (strict)
    feature_order = _get_feature_order(meta)

    # Categorical / numeric columns - read from metadata (supports no-embeddings mode)
    meta_cat_features = meta.get("features", {}).get("categorical_features", [])
    if meta_cat_features:
        cat_cols = list(meta_cat_features)
        missing_cats = [c for c in cat_cols if c not in feature_order]
        if missing_cats:
            raise ValueError(
                f"metadata feature_list_ordered missing required categorical columns: {missing_cats}. "
                f"Expected categorical columns: {cat_cols}"
            )
    else:
        cat_cols = []
        print("[CONFIG] No categorical features in metadata - running without embeddings")
    num_cols = [c for c in feature_order if c not in set(cat_cols)]

    # Labels from metadata2
    label_set_ordered, label_to_index, index_to_label, num_classes = _get_label_info(meta)

    # Split files
    file_format, split_paths = _resolve_splits(pre_dir)

    # Vocab sizes for categorical features
    vocab_sizes = _get_vocab_sizes_from_meta2(meta, cat_cols) if cat_cols else {}

    # Validate split schemas against required columns (features + y)
    required_cols = list(feature_order) + [TARGET_COL]
    split_columns: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        cols = _check_required_columns(split, split_paths[split], file_format, required_cols)
        split_columns[split] = cols

    # Optional weights (must not crash if missing)
    weights_used = WEIGHT_COL in set(split_columns["train"])
    if weights_used:
        print(f"[INFO] Found weight column '{WEIGHT_COL}' in train split; using per-example weights.")
    else:
        print(f"[INFO] Weight column '{WEIGHT_COL}' not found in train split; defaulting all weights to 1.0.")

    # Basic sanity checks on train split y and categorical ranges
    _quick_sanity_scan_train(
        train_path=split_paths["train"],
        file_format=file_format,
        num_classes=num_classes,
        cat_cols=cat_cols,
        vocab_sizes=vocab_sizes,
        max_rows=SANITY_CHECK_MAX_ROWS,
    )

    # Optional class weights from y distribution (no dependence on old label-mapping structures)
    class_weights = None
    if USE_CLASS_WEIGHTS:
        print("[INFO] Computing class weights from training split y distribution...")
        class_weights = _compute_class_weights_from_y(split_paths["train"], file_format, num_classes).to(DEVICE)

    # Build datasets (IterableDataset yielding batches)
    train_ds = SplitBatchIterableDataset(
        path=split_paths["train"],
        file_format=file_format,
        feature_order=feature_order,
        cat_cols=cat_cols,
        num_cols=num_cols,
        vocab_sizes=vocab_sizes,
        num_classes=num_classes,
        batch_size=BATCH_SIZE,
        available_columns=split_columns["train"],
        max_rows=MAX_TRAIN_ROWS,
        shuffle_rows=SHUFFLE_TRAIN_ROWS_WITHIN_BATCH,
        seed=SEED,
        weight_col=WEIGHT_COL if weights_used else None,
    )
    val_ds = SplitBatchIterableDataset(
        path=split_paths["val"],
        file_format=file_format,
        feature_order=feature_order,
        cat_cols=cat_cols,
        num_cols=num_cols,
        vocab_sizes=vocab_sizes,
        num_classes=num_classes,
        batch_size=BATCH_SIZE,
        available_columns=split_columns["val"],
        max_rows=None,
        shuffle_rows=False,
        seed=SEED + 1,
        weight_col=WEIGHT_COL if WEIGHT_COL in set(split_columns["val"]) else None,
    )
    test_ds = SplitBatchIterableDataset(
        path=split_paths["test"],
        file_format=file_format,
        feature_order=feature_order,
        cat_cols=cat_cols,
        num_cols=num_cols,
        vocab_sizes=vocab_sizes,
        num_classes=num_classes,
        batch_size=BATCH_SIZE,
        available_columns=split_columns["test"],
        max_rows=None,
        shuffle_rows=False,
        seed=SEED + 2,
        weight_col=WEIGHT_COL if WEIGHT_COL in set(split_columns["test"]) else None,
    )

    # Wrap in DataLoaders (batch_size=None because dataset already yields batches)
    train_loader = DataLoader(train_ds, batch_size=None, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=None, num_workers=NUM_WORKERS)

    # Build model
    # Detect pitch-type statcast columns
    pt_statcast_cols = [c for c in num_cols if c.startswith("pitcher_pitch_rate_") or
                        c.startswith("pitcher_FF_") or c.startswith("pitcher_SI_") or
                        c.startswith("pitcher_FC_") or c.startswith("pitcher_SL_") or
                        c.startswith("pitcher_CU_") or c.startswith("pitcher_CH_") or
                        c.startswith("pitcher_FS_") or c.startswith("pitcher_ST_") or
                        c.startswith("pitcher_SV_") or c.startswith("pitcher_OTHER_") or
                        c.startswith("batter_FF_") or c.startswith("batter_SI_") or
                        c.startswith("batter_FC_") or c.startswith("batter_SL_") or
                        c.startswith("batter_CU_") or c.startswith("batter_CH_") or
                        c.startswith("batter_FS_") or c.startswith("batter_ST_") or
                        c.startswith("batter_SV_") or c.startswith("batter_OTHER_")]
    num_pt_statcast = len(pt_statcast_cols)

    if use_pt_statcast and num_pt_statcast > 0 and len(cat_cols) > 0:
        print(f"[CONFIG] Using StatcastLogitHybridModel ({num_pt_statcast} statcast cols in embedding head)")
        model = StatcastLogitHybridModel(
            cat_cols=cat_cols,
            num_numeric=len(num_cols),
            vocab_sizes=vocab_sizes,
            num_classes=num_classes,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
            num_pt_statcast_cols=num_pt_statcast,
        ).to(DEVICE)
    elif USE_HYBRID_MODEL and len(cat_cols) > 0:
        print("[CONFIG] Using hybrid model (embeddings + vectors with learnable a,b)")
        model = HybridModel(
            cat_cols=cat_cols,
            num_numeric=len(num_cols),
            vocab_sizes=vocab_sizes,
            num_classes=num_classes,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
        ).to(DEVICE)
    elif USE_SIMPLE_VECTOR_WEIGHTS and len(cat_cols) == 0:
        print("[CONFIG] Using simple weighted vector combination model")
        model = SimpleVectorWeightModel(
            num_vectors=6,
            num_outcomes=num_classes,
            use_log_n_weighting=True,
        ).to(DEVICE)
    elif USE_EMBEDDING_INTERACTIONS and len(cat_cols) > 0:
        print("[CONFIG] Using embedding interactions (batter x pitcher)")
        model = PitchOutcomeModelWithInteractions(
            cat_cols=cat_cols,
            num_numeric=len(num_cols),
            vocab_sizes=vocab_sizes,
            num_classes=num_classes,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
        ).to(DEVICE)
    elif len(cat_cols) > 0:
        print("[CONFIG] Using standard concatenation (no interactions)")
        model = PitchOutcomeModel(
            cat_cols=cat_cols,
            num_numeric=len(num_cols),
            vocab_sizes=vocab_sizes,
            num_classes=num_classes,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
        ).to(DEVICE)
    else:
        print("[CONFIG] No embeddings - numeric features only (MLP)")
        model = PitchOutcomeModel(
            cat_cols=cat_cols,
            num_numeric=len(num_cols),
            vocab_sizes=vocab_sizes,
            num_classes=num_classes,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
        ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Summary logging
    print("=" * 72)
    print("Training summary")
    print(f"Start time:          {_now_str()}")
    print(f"Preprocessed dir:    {pre_dir}")
    print(f"Split format:        {file_format}")
    print(f"Features:            {len(feature_order)}")
    print(f"  Categorical cols:  {len(cat_cols)}  ({cat_cols})")
    print(f"  Numeric:           {len(num_cols)}")
    print(f"Classes:             {num_classes}")
    print(f"Device:              {DEVICE}")
    print(f"Batch size:          {BATCH_SIZE}")
    print(f"Epochs:              {NUM_EPOCHS}")
    print(f"Early stop patience: {EARLY_STOP_PATIENCE}")
    print(f"Max train rows:      {MAX_TRAIN_ROWS}")
    print(f"Use class weights:   {USE_CLASS_WEIGHTS}")
    print(f"Use example weights: {weights_used}")
    print("=" * 72)

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_acc = 0.0
    best_state = None
    best_train_metrics: Optional[EpochMetrics] = None
    best_val_metrics: Optional[EpochMetrics] = None
    no_improve = 0

    history: List[dict] = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train loss: {train_metrics.loss:.6f}, train acc: {train_metrics.accuracy:.4f} | "
            f"val loss: {val_metrics.loss:.6f}, val acc: {val_metrics.accuracy:.4f} | "
            f"time: {dt:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics.loss),
                "train_accuracy": float(train_metrics.accuracy),
                "val_loss": float(val_metrics.loss),
                "val_accuracy": float(val_metrics.accuracy),
                "seconds": float(dt),
                "timestamp": _now_str(),
            }
        )

        improved = (val_metrics.loss < best_val_loss) if not math.isnan(val_metrics.loss) else False
        if improved:
            best_val_loss = float(val_metrics.loss)
            best_val_acc = float(val_metrics.accuracy)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best val loss {best_val_loss:.6f} at epoch {best_epoch})."
                )
                break

    # Restore best model for final evaluation + saving
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, DEVICE)
    # --- Distribution scorecard (proper probability evaluation) ---
    # Evaluate scorecard on VAL split (the primary evaluation window)
    baseline_weight_col = WEIGHT_COL if (WEIGHT_COL in set(split_columns["train"]) and WEIGHT_COL in set(split_columns["val"])) else None

    scorecard = compute_distribution_scorecard(
        model=model,
        train_path=split_paths["train"],
        test_path=split_paths["val"],
        file_format=file_format,
        num_classes=num_classes,
        test_loader=val_loader,
        device=DEVICE,
        weight_col_for_baselines=baseline_weight_col,
    )

    print("[SCORECARD] Evaluated on VAL split (primary eval window)")
    print("[SCORECARD] Uniform logloss:        {:.6f}".format(scorecard["uniform_logloss"]))
    print("[SCORECARD] Base-rate logloss:      {:.6f}".format(scorecard["base_rate_logloss"]))
    print("[SCORECARD] Model val logloss:      {:.6f}".format(scorecard["model_test_logloss"]))
    print("[SCORECARD] Delta vs base-rates:     {:+.6f}".format(scorecard["delta_logloss_vs_base_rates"]))
    print("[SCORECARD] % better than base:     {:+.2f}%".format(scorecard["pct_better_than_base_rates"]))
    print("[SCORECARD] Majority-class acc:     {:.4f}".format(scorecard["majority_class_accuracy"]))

    print("-" * 72)
    print(f"Best val loss: {best_val_loss:.6f} (epoch {best_epoch}), best val acc: {best_val_acc:.4f}")
    print(f"Test loss:     {test_metrics.loss:.6f}, test acc:     {test_metrics.accuracy:.4f}")
    print("-" * 72)

    # Save artifacts
    _ensure_dir(artifact_dir)

    model_path = os.path.join(artifact_dir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Preserve tendency artifact paths for downstream inference (if present)
    tendency_artifact_paths = {}
    if isinstance(meta.get("tendencies"), dict):
        ap = meta["tendencies"].get("artifact_paths")
        if isinstance(ap, dict):
            tendency_artifact_paths = ap

    # Train config + resolved lists
    train_config = {
        "PREPROCESSED_DIR": pre_dir,
        "ARTIFACT_DIR": artifact_dir,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "EMBED_DIM_DEFAULT": EMBED_DIM_DEFAULT,
        "HIDDEN_DIMS": HIDDEN_DIMS,
        "DROPOUT": DROPOUT,
        "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
        "DEVICE": DEVICE,
        "NUM_WORKERS": NUM_WORKERS,
        "MAX_TRAIN_ROWS": MAX_TRAIN_ROWS,
        "TARGET_COL": TARGET_COL,
        "WEIGHT_COL": WEIGHT_COL,
        "weights_used": bool(weights_used),
        "use_class_weights": bool(USE_CLASS_WEIGHTS),
        "USE_EMBEDDING_INTERACTIONS": USE_EMBEDDING_INTERACTIONS,
        "BATTER_EMBED_DIM": BATTER_EMBED_DIM,
        "PITCHER_EMBED_DIM": PITCHER_EMBED_DIM,
        "STADIUM_EMBED_DIM": STADIUM_EMBED_DIM,
        "HANDEDNESS_EMBED_DIM": HANDEDNESS_EMBED_DIM,
        "resolved": {
            "file_format": file_format,
            "split_paths": split_paths,
            "feature_list_ordered": feature_order,
            "categorical_cols": cat_cols,
            "numeric_cols": num_cols,
            "categorical_vocab_sizes": vocab_sizes,
            "embedding_dims": getattr(model, "emb_dims", {}),
            "num_classes": num_classes,
            "label_set_ordered": label_set_ordered,
            "tendency_artifact_paths": tendency_artifact_paths,
            "model_class": type(model).__name__,
            "interaction_dim": BATTER_EMBED_DIM if USE_EMBEDDING_INTERACTIONS else 0,
        },
        "timestamp": _now_str(),
    }
    _write_json(os.path.join(artifact_dir, "train_config.json"), train_config)

    # Label maps (consistent with preprocessing2)
    _write_json(os.path.join(artifact_dir, "label_to_index.json"), {k: int(v) for k, v in label_to_index.items()})
    _write_json(os.path.join(artifact_dir, "index_to_label.json"), {str(k): v for k, v in index_to_label.items()})

    # Copy metadata.json used
    shutil.copy(meta_path, os.path.join(artifact_dir, "metadata.json"))

    # Metrics
    metrics = {
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_acc),
        "best_train_loss": float(best_train_metrics.loss) if best_train_metrics else None,
        "best_train_accuracy": float(best_train_metrics.accuracy) if best_train_metrics else None,
        "best_val_loss_epoch": float(best_val_metrics.loss) if best_val_metrics else None,
        "best_val_accuracy_epoch": float(best_val_metrics.accuracy) if best_val_metrics else None,
        "final_test_loss": float(test_metrics.loss),
        "final_test_accuracy": float(test_metrics.accuracy),
        "history": history,  # per-epoch train/val metrics
        "timestamp": _now_str(),
    }

    # Log learned vector weights (if using simple or hybrid model)
    if hasattr(model, 'get_learned_weights'):
        learned_weights = model.get_learned_weights()
        print("\n" + "=" * 50)
        print("LEARNED VECTOR WEIGHTS:")
        print("=" * 50)
        for name, weight in learned_weights.items():
            print(f"  {name}: {weight:.4f} ({weight*100:.1f}%)")
        print("=" * 50 + "\n")
        metrics["learned_vector_weights"] = learned_weights

    # Log hybrid a,b combination weights
    if hasattr(model, 'get_ab_weights'):
        ab_weights = model.get_ab_weights()
        print("=" * 50)
        print("LEARNED COMBINATION WEIGHTS (a*emb + b*vec):")
        print("=" * 50)
        for name, weight in ab_weights.items():
            print(f"  {name}: {weight:.4f} ({weight*100:.1f}%)")
        print("=" * 50 + "\n")
        metrics["hybrid_ab_weights"] = ab_weights

    _write_json(os.path.join(artifact_dir, "metrics.json"), metrics)

    print(f"Artifacts saved to: {artifact_dir}")
    print(f"  - {model_path}")
    print("  - train_config.json, label_to_index.json, index_to_label.json, metadata.json, metrics.json")

    # Comparison table (when using pitchtype statcast experiment)
    if use_pt_statcast:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        val_ll = scorecard["model_test_logloss"]  # val logloss from scorecard
        test_ll = float(evaluate_logloss_only(model, test_loader, DEVICE))

        base_val = 1.4438
        base_test = 1.4724
        prior = [
            ("A: Embeddings only",         1.4320, 1.4688, 174_920, 32),
            ("B: Vector weights",           1.4312, 1.4689,      12,  1),
            ("C: Hybrid (prob mix)",        1.4311, 1.4719, 174_676, 13),
            ("D: Hybrid (logit mix)",       1.4307, 1.4695, 174_676, 13),
            ("E: Hybrid + PT Statcast v1",  1.4302, 1.4674, 188_330, 32),
        ]

        lines = []
        sep = "=" * 90
        hdr = f"  {'Model':<35s} {'Params':>8s} {'Epoch':>6s} {'Val LL':>10s} {'%vsBase':>8s} {'Test LL':>10s} {'%vsBase':>8s}"
        dash = "-" * 90

        lines.append(sep)
        lines.append("  COMPARISON TABLE")
        lines.append(sep)
        lines.append(hdr)
        lines.append(dash)
        for name, vl, tl, params, ep in prior:
            vp = 100.0 * (base_val - vl) / base_val
            tp = 100.0 * (base_test - tl) / base_test
            lines.append(f"  {name:<35s} {params:>8,d} {ep:>6d} {vl:>10.4f} {vp:>+7.2f}% {tl:>10.4f} {tp:>+7.2f}%")
        # This run
        vp = 100.0 * (base_val - val_ll) / base_val
        tp = 100.0 * (base_test - test_ll) / base_test
        lines.append(f"  {'F: Hybrid + PT Statcast v2':<35s} {n_params:>8,d} {best_epoch:>6d} {val_ll:>10.4f} {vp:>+7.2f}% {test_ll:>10.4f} {tp:>+7.02f}%")
        lines.append(dash)
        lines.append(f"  {'Base rate':<35s} {'--':>8s} {'--':>6s} {base_val:>10.4f} {'--':>8s} {base_test:>10.4f} {'--':>8s}")
        lines.append(sep)

        table_str = "\n".join(lines)
        print("\n" + table_str)

        # Save comparison
        with open(os.path.join(artifact_dir, "comparison.txt"), "w") as f:
            f.write(table_str + "\n")
        _write_json(os.path.join(artifact_dir, "comparison.json"), {
            "models": {
                "A_embeddings_only": {"val_ll": 1.4320, "test_ll": 1.4688},
                "B_vector_weights": {"val_ll": 1.4312, "test_ll": 1.4689},
                "C_hybrid_prob": {"val_ll": 1.4311, "test_ll": 1.4719},
                "D_hybrid_logit": {"val_ll": 1.4307, "test_ll": 1.4695},
                "E_hybrid_pt_statcast_v1": {"val_ll": 1.4302, "test_ll": 1.4674},
                "F_hybrid_pt_statcast_v2": {"val_ll": val_ll, "test_ll": test_ll,
                                            "params": n_params, "best_epoch": best_epoch},
            },
            "base_rate": {"val_ll": base_val, "test_ll": base_test},
        })
        print(f"\nComparison saved to {artifact_dir}/comparison.txt and comparison.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
