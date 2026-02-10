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
- Tendency feature columns (e.g., batter_*, pitcher_*) are numeric inputs and must be used.

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
PREPROCESSED_DIR = "preprocessed_test8"
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

# IMPORTANT:
# - For probability forecasting, do NOT force class balancing by default.
USE_CLASS_WEIGHTS = True  # optional; if enabled, computed from train split y distribution

# Target and optional per-example weights
TARGET_COL = "y"
WEIGHT_COL = "pa_example_weight"  # optional column; may be missing

# Fixed categorical columns for preprocessing2
CATEGORICAL_COLS = ["batter_id", "pitcher_id", "stadium_id"]
PAD_NA_ID = 0
UNK_ID = 1

# Sanity checks
SANITY_CHECK_MAX_ROWS = 200_000  # scan up to this many rows from train split for basic checks/logging
# Fail-fast categorical integrity checks.
# Default: enabled. Set env RAISE_ON_OOR_CATEGORICAL=0 to fall back to clamping-to-UNK (still logged and counted).
RAISE_ON_OOR_CATEGORICAL = os.environ.get("RAISE_ON_OOR_CATEGORICAL", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "t",
)
ECE_NUM_BINS = 15
ROW_ID_COL_CANDIDATES = ["pa_id"]  # used only for diagnostics in integrity errors (if present in the split files)

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


def _safe_series_to_float32(s: pd.Series, col_name: str = "", nan_ratio_threshold: float = 0.01) -> np.ndarray:
    raw = s
    raw_nonnull = raw.notna()
    out = pd.to_numeric(raw, errors="coerce")
    introduced = raw_nonnull & out.isna()
    denom = int(raw_nonnull.sum())
    if denom > 0:
        ratio = float(int(introduced.sum())) / float(denom)
        if ratio > float(nan_ratio_threshold):
            name = col_name or getattr(s, "name", None) or "<unnamed>"
            raise ValueError(
                f"Numeric coercion introduced NaNs in {name!r}: "
                f"{int(introduced.sum())}/{denom} ({ratio:.2%}) > {float(nan_ratio_threshold):.2%}"
            )
    out = out.fillna(0.0)
    return out.astype(np.float32, copy=False).to_numpy(copy=False)



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


def _choose_emb_dim(vocab_size: int) -> int:
    """
    A small heuristic for embedding dimension based on vocab size.
    Matches train_model.py style.
    """
    vs = int(vocab_size)
    if vs <= 10:
        return min(EMBED_DIM_DEFAULT, 8)
    if vs <= 50:
        return EMBED_DIM_DEFAULT
    # Common rule-of-thumb: ~sqrt(vocab) capped
    return int(min(64, max(EMBED_DIM_DEFAULT, round(math.sqrt(vs)))))


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



def _scan_split_categorical_stats(
    split_name: str,
    split_path: str,
    file_format: str,
    cat_cols: List[str],
    vocab_sizes: Dict[str, int],
    available_columns: List[str],
    max_rows: Optional[int] = None,
) -> dict:
    """Scan a split for UNK rates and out-of-range (OOR) categorical IDs.

    - Always counts UNK and OOR occurrences per categorical field.
    - If RAISE_ON_OOR_CATEGORICAL is enabled and any OOR is found, raises ValueError with split/field/value details.
    """
    unk_counts = {c: 0 for c in cat_cols}
    oor_counts = {c: 0 for c in cat_cols}
    rows = 0

    available = set(str(c) for c in (available_columns or []))
    row_id_col = next((c for c in ROW_ID_COL_CANDIDATES if c in available), None)

    columns = list(cat_cols)
    if row_id_col is not None and row_id_col not in columns:
        columns.append(row_id_col)

    def _process_chunk(df: pd.DataFrame, row_offset: int) -> None:
        nonlocal rows, unk_counts, oor_counts

        if df is None or len(df) == 0:
            return

        n = len(df)
        if max_rows is not None:
            remaining = max_rows - rows
            if remaining <= 0:
                return
            if n > remaining:
                df = df.iloc[:remaining].copy()
                n = len(df)

        for c in cat_cols:
            arr = _safe_series_to_int64(df[c])
            vs = int(vocab_sizes[c])

            unk_counts[c] += int((arr == UNK_ID).sum())

            oor_mask = (arr < 0) | (arr >= vs)
            oor = int(oor_mask.sum())
            if oor > 0:
                oor_counts[c] += oor

                if RAISE_ON_OOR_CATEGORICAL:
                    bad_vals = arr[oor_mask]
                    k = int(min(8, bad_vals.size))
                    sample_vals = bad_vals[:k].tolist()
                    idxs = np.nonzero(oor_mask)[0][:k]
                    sample_rows = (row_offset + idxs).tolist()

                    msg = (
                        f"Out-of-range categorical IDs detected in split='{split_name}'. "
                        f"Field='{c}', vocab_size={vs}, bad_count={oor}. "
                        f"Examples bad_values={sample_vals}, row_indices={sample_rows}"
                    )
                    if row_id_col is not None and row_id_col in df.columns:
                        try:
                            sample_ids = df[row_id_col].iloc[idxs].astype(str).tolist()
                            msg += f", row_ids({row_id_col})={sample_ids}"
                        except Exception:
                            pass

                    msg += (
                        ". This indicates a mismatch between metadata vocab sizes and encoded IDs. "
                        "(Set env RAISE_ON_OOR_CATEGORICAL=0 to clamp-to-UNK for debugging.)"
                    )
                    raise ValueError(msg)

        rows += n

    if file_format == "parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as e:
            raise ImportError("Parquet splits were detected but pyarrow is not available.") from e

        pf = pq.ParquetFile(split_path)
        row_offset = 0
        for batch in pf.iter_batches(batch_size=50_000, columns=columns):
            df = batch.to_pandas()
            _process_chunk(df, row_offset=row_offset)
            row_offset += 0 if df is None else len(df)
            if max_rows is not None and rows >= max_rows:
                break
    else:
        reader = pd.read_csv(
            split_path,
            compression="gzip",
            usecols=columns,
            chunksize=50_000,
            low_memory=True,
        )
        row_offset = 0
        for df in reader:
            _process_chunk(df, row_offset=row_offset)
            row_offset += 0 if df is None else len(df)
            if max_rows is not None and rows >= max_rows:
                break

    return {
        "split": split_name,
        "rows_scanned": int(rows),
        "capped": bool(max_rows is not None and rows >= max_rows),
        "row_id_col": row_id_col,
        "unk_counts": {k: int(v) for k, v in unk_counts.items()},
        "oor_counts": {k: int(v) for k, v in oor_counts.items()},
    }


def _compute_train_label_priors(
    train_path: str, file_format: str, num_classes: int, max_rows: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute train-label class counts and prior probabilities p_k = count_k / N.

    Returns (counts, prior_probs_safe) where prior_probs_safe are clipped/renormalized to avoid log(0) in baselines.
    """
    counts = np.zeros((num_classes,), dtype=np.int64)
    rows = 0

    def _update(df: pd.DataFrame) -> None:
        nonlocal counts, rows
        y_raw = pd.to_numeric(df[TARGET_COL], errors="coerce")
        y_raw = y_raw.dropna()
        if y_raw.empty:
            rows += len(df)
            return
        y_np = y_raw.astype(np.int64, copy=False).to_numpy(copy=False)
        if y_np.size == 0:
            rows += len(df)
            return
        if int(y_np.min()) < 0 or int(y_np.max()) >= num_classes:
            raise ValueError(
                f"Target '{TARGET_COL}' out of range while computing train priors: "
                f"min={int(y_np.min())}, max={int(y_np.max())}, num_classes={num_classes}"
            )
        counts += np.bincount(y_np, minlength=num_classes)
        rows += len(df)

    if file_format == "parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as e:
            raise ImportError("Parquet splits were detected but pyarrow is not available.") from e
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
    if total <= 0.0:
        raise ValueError("Train priors could not be computed (no labels found in train split).")

    priors = counts.astype(np.float64) / total
    priors_safe = np.clip(priors, 1e-12, 1.0)
    priors_safe = priors_safe / float(priors_safe.sum())

    return counts, priors_safe


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
        split_name: str,
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
        self.split_name = str(split_name)

        self._available = set(str(c) for c in (available_columns or []))

        # Optional row id column for diagnostics (never used as a model feature).
        self.row_id_col = next((c for c in ROW_ID_COL_CANDIDATES if c in self._available), None)

        # Read only needed columns. Include weight column only if it exists in this split.
        self.columns = list(self.feature_order) + [TARGET_COL]
        if self.weight_col is not None and self.weight_col in self._available:
            self.columns.append(self.weight_col)
        if self.row_id_col is not None and self.row_id_col in self._available and self.row_id_col not in self.columns:
            self.columns.append(self.row_id_col)

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
        self._oor_total = {c: 0 for c in self.cat_cols}

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

            row_offset = rows_read
            row_offset = rows_read
            rows_read += len(df)

            if self.shuffle_rows and len(df) > 1:
                rs = int(self._rng.integers(0, 2**31 - 1))
                df = df.sample(frac=1.0, random_state=rs).reset_index(drop=True)

            yield self._df_to_tensors(df, row_offset=row_offset)

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

            yield self._df_to_tensors(df, row_offset=row_offset)

    def _df_to_tensors(
        self, df: pd.DataFrame, row_offset: int = 0
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
                    self._oor_total[c] = self._oor_total.get(c, 0) + oor
                    if RAISE_ON_OOR_CATEGORICAL:
                        bad_vals = arr[oor_mask]
                        k = int(min(8, bad_vals.size))
                        sample_vals = bad_vals[:k].tolist()
                        idxs = np.nonzero(oor_mask)[0][:k]
                        sample_rows = (row_offset + idxs).tolist()

                        msg = (
                            f"Out-of-range categorical IDs detected (split={self.split_name}, file={os.path.basename(self.path)}). "
                            f"Field='{c}', vocab_size={vs}, bad_count={oor}. "
                            f"Examples bad_values={sample_vals}, row_indices={sample_rows}"
                        )
                        if self.row_id_col is not None and self.row_id_col in df.columns:
                            try:
                                sample_ids = df[self.row_id_col].iloc[idxs].astype(str).tolist()
                                msg += f", row_ids({self.row_id_col})={sample_ids}"
                            except Exception:
                                pass
                        raise ValueError(msg)

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
            num_arrays = [_safe_series_to_float32(df[c], col_name=c) for c in self.num_cols]
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
            ed = _choose_emb_dim(vs)
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


# -----------------------------
# Training / Evaluation
# -----------------------------
@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    n: int

    # Probabilistic evaluation (VAL/TEST): populated by evaluate()
    nll: Optional[float] = None
    brier: Optional[float] = None
    ece: Optional[float] = None
    prior_nll: Optional[float] = None
    skill: Optional[float] = None
    majority_accuracy: Optional[float] = None

    # Per-class diagnostics
    class_support: Optional[List[int]] = None
    per_class_nll: Optional[List[float]] = None
    macro_nll: Optional[float] = None

    # Prediction distribution diagnostics (primarily VAL)
    pred_hist: Optional[List[int]] = None
    mean_prob: Optional[List[float]] = None

    # Probabilistic evaluation (VAL/TEST): populated by evaluate()
    nll: Optional[float] = None
    brier: Optional[float] = None
    ece: Optional[float] = None
    prior_nll: Optional[float] = None
    skill: Optional[float] = None
    majority_accuracy: Optional[float] = None

    # Per-class diagnostics
    class_support: Optional[List[int]] = None
    per_class_nll: Optional[List[float]] = None
    macro_nll: Optional[float] = None

    # Prediction-collapse diagnostics
    pred_hist: Optional[List[int]] = None
    mean_prob: Optional[List[float]] = None


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: str,
    train_prior_probs: Optional[torch.Tensor] = None,
    majority_class: Optional[int] = None,
    ece_bins: int = ECE_NUM_BINS,
) -> EpochMetrics:
    """Evaluate model on a split with probability-first metrics.

    Notes:
    - NLL is computed as (weighted) CrossEntropyLoss.
    - Brier and ECE use probs = softmax(logits).
    - Train-prior baselines use priors computed from TRAIN labels only.
    """
    model.eval()

    loss_num = 0.0
    loss_den = 0.0
    total_correct = 0
    total_n = 0

    # Probability metrics
    brier_num = 0.0
    brier_den = 0.0

    # Baselines
    prior_nll_num = 0.0
    prior_nll_den = 0.0
    majority_correct = 0

    # Tensors initialized lazily once we know num_classes
    num_classes: Optional[int] = None
    class_loss_sum = None
    class_weight_sum = None
    class_count = None
    pred_hist = None
    prob_sum = None
    ece_bin_weight = None
    ece_bin_correct = None
    ece_bin_conf = None
    prior_probs_dev = None

    for x_cat, x_num, y, w in dataset:
        if y.numel() == 0:
            continue

        x_cat = x_cat.to(device, non_blocking=True)
        x_num = x_num.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True).float()

        logits = model(x_cat, x_num)
        if num_classes is None:
            num_classes = int(logits.shape[1])
            class_loss_sum = torch.zeros((num_classes,), dtype=torch.float32, device=device)
            class_weight_sum = torch.zeros((num_classes,), dtype=torch.float32, device=device)
            class_count = torch.zeros((num_classes,), dtype=torch.long, device=device)
            pred_hist = torch.zeros((num_classes,), dtype=torch.long, device=device)
            prob_sum = torch.zeros((num_classes,), dtype=torch.float32, device=device)
            ece_bin_weight = torch.zeros((ece_bins,), dtype=torch.float32, device=device)
            ece_bin_correct = torch.zeros((ece_bins,), dtype=torch.float32, device=device)
            ece_bin_conf = torch.zeros((ece_bins,), dtype=torch.float32, device=device)

            if train_prior_probs is not None:
                if int(train_prior_probs.numel()) != num_classes:
                    raise ValueError(
                        f"train_prior_probs length={int(train_prior_probs.numel())} != num_classes={num_classes}"
                    )
                prior_probs_dev = train_prior_probs.to(device=device, dtype=torch.float32)
                prior_probs_dev = prior_probs_dev.clamp_min(1e-12)

        per_ex_loss = criterion(logits, y)  # (N,)

        batch_num = (per_ex_loss * w).sum()
        batch_den = w.sum().clamp_min(1e-12)

        n = int(y.numel())
        loss_num += float(batch_num.item())
        loss_den += float(batch_den.item())

        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total_n += n

        # Probabilities for Brier/ECE + prediction distribution diagnostics
        probs = torch.softmax(logits, dim=1)

        # Brier score: mean(sum_k (p_hat_k - onehot_k)^2)
        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, y.view(-1, 1), 1.0)
        brier_per_ex = ((probs - onehot) ** 2).sum(dim=1)
        brier_num += float((brier_per_ex * w).sum().item())
        brier_den += float(batch_den.item())

        # ECE(15 bins): confidence=max prob; bin on confidence
        conf, _ = probs.max(dim=1)
        correct = (preds == y).float()
        # bin index in [0, ece_bins-1]
        bin_idx = torch.clamp((conf * float(ece_bins)).long(), min=0, max=ece_bins - 1)

        # Accumulate bin-wise weighted sums
        ece_bin_weight.scatter_add_(0, bin_idx, w)
        ece_bin_correct.scatter_add_(0, bin_idx, correct * w)
        ece_bin_conf.scatter_add_(0, bin_idx, conf * w)

        # Per-class NLL (weighted) + support counts
        class_loss_sum.scatter_add_(0, y, per_ex_loss * w)
        class_weight_sum.scatter_add_(0, y, w)
        class_count.scatter_add_(0, y, torch.ones_like(y, dtype=torch.long))

        # Prediction distribution diagnostics
        pred_hist.scatter_add_(0, preds, torch.ones_like(preds, dtype=torch.long))
        prob_sum += probs.sum(dim=0)

        # Baselines
        if prior_probs_dev is not None:
            prior_y = prior_probs_dev.gather(0, y)
            prior_nll_num += float((-torch.log(prior_y) * w).sum().item())
            prior_nll_den += float(batch_den.item())

        if majority_class is not None:
            majority_correct += int((y == int(majority_class)).sum().item())

    if total_n == 0 or num_classes is None:
        return EpochMetrics(loss=float("nan"), accuracy=float("nan"), n=0)

    nll = loss_num / max(loss_den, 1e-12)
    accuracy = total_correct / total_n

    brier = brier_num / max(brier_den, 1e-12)

    # ECE
    total_w = float(ece_bin_weight.sum().clamp_min(1e-12).item())
    bin_acc = (ece_bin_correct / ece_bin_weight.clamp_min(1e-12)).detach()
    bin_conf = (ece_bin_conf / ece_bin_weight.clamp_min(1e-12)).detach()
    ece = float((torch.abs(bin_acc - bin_conf) * (ece_bin_weight / total_w)).sum().item())

    # Per-class NLL and macro NLL (over classes present)
    per_class = (class_loss_sum / class_weight_sum.clamp_min(1e-12)).detach().cpu().numpy()
    support = class_count.detach().cpu().numpy()
    per_class_list: List[float] = []
    for i in range(int(num_classes)):
        if int(support[i]) == 0:
            per_class_list.append(float("nan"))
        else:
            per_class_list.append(float(per_class[i]))
    macro_nll_vals = [v for v, s in zip(per_class_list, support.tolist()) if s > 0 and math.isfinite(v)]
    macro_nll = float(np.mean(macro_nll_vals)) if macro_nll_vals else float("nan")

    prior_nll = None
    skill = None
    if prior_probs_dev is not None and prior_nll_den > 0.0:
        prior_nll = prior_nll_num / max(prior_nll_den, 1e-12)
        if prior_nll > 0.0 and math.isfinite(prior_nll):
            skill = (prior_nll - nll) / prior_nll

    maj_acc = None
    if majority_class is not None:
        maj_acc = majority_correct / total_n

    mean_prob = (prob_sum / float(total_n)).detach().cpu().numpy().astype(np.float64).tolist()
    pred_hist_list = pred_hist.detach().cpu().numpy().astype(np.int64).tolist()
    support_list = support.astype(np.int64).tolist()

    return EpochMetrics(
        loss=float(nll),  # kept for backward-compat (early stopping); equals NLL
        accuracy=float(accuracy),
        n=int(total_n),
        nll=float(nll),
        brier=float(brier),
        ece=float(ece),
        prior_nll=None if prior_nll is None else float(prior_nll),
        skill=None if skill is None else float(skill),
        majority_accuracy=None if maj_acc is None else float(maj_acc),
        class_support=[int(x) for x in support_list],
        per_class_nll=per_class_list,
        macro_nll=float(macro_nll),
        pred_hist=[int(x) for x in pred_hist_list],
        mean_prob=[float(x) for x in mean_prob],
    )


def _print_probabilistic_report(split_name: str, m: EpochMetrics, index_to_label: Dict[int, str]) -> None:
    if m is None or m.n <= 0 or m.nll is None:
        print(f"[{split_name}] No samples; skipping probabilistic report.")
        return

    print("-" * 72)
    print(f"[{split_name}] Probabilistic evaluation (n={m.n})")

    print("Baselines (from TRAIN priors):")
    if m.majority_accuracy is not None:
        print(f"  Majority accuracy: {m.majority_accuracy:.4f}")
    if m.prior_nll is not None:
        print(f"  Train-prior NLL:   {m.prior_nll:.6f}")
    else:
        print("  Train-prior NLL:   n/a")

    print("Model:")
    print(f"  Accuracy:          {m.accuracy:.4f}")
    print(f"  NLL:               {m.nll:.6f}")
    if m.brier is not None:
        print(f"  Brier:             {m.brier:.6f}")
    if m.ece is not None:
        print(f"  ECE({ECE_NUM_BINS} bins):        {m.ece:.6f}")
    if m.skill is not None:
        print(f"  Skill:             {m.skill:.6f}")

    if m.class_support is not None:
        print("Per-class diagnostics:")
        for i, sup in enumerate(m.class_support):
            label = index_to_label.get(i, str(i))
            nll_i = None if m.per_class_nll is None else m.per_class_nll[i]
            nll_str = f"{nll_i:.6f}" if nll_i is not None and math.isfinite(nll_i) else "nan"
            print(f"  class {i} ({label}): support={sup}, nll={nll_str}")
        if m.macro_nll is not None:
            print(f"  Macro NLL (unweighted over classes present): {m.macro_nll:.6f}")

    # Prediction distribution diagnostics (VAL primarily)
    if m.pred_hist is not None:
        total = float(sum(m.pred_hist)) if sum(m.pred_hist) > 0 else 1.0
        parts = []
        for i, cnt in enumerate(m.pred_hist):
            label = index_to_label.get(i, str(i))
            frac = cnt / total
            parts.append(f"{i}({label})={cnt} ({frac:.1%})")
        print("Predicted argmax histogram:")
        print("  " + ", ".join(parts))

    if m.mean_prob is not None:
        parts = []
        for i, mp in enumerate(m.mean_prob):
            label = index_to_label.get(i, str(i))
            parts.append(f"{i}({label})={mp:.4f}")
        print("Mean predicted probability per class:")
        print("  " + ", ".join(parts))


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


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    pre_dir = sys.argv[1] if len(sys.argv) > 1 else PREPROCESSED_DIR

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

    # Categorical / numeric columns (strict for preprocessing2)
    cat_cols = list(CATEGORICAL_COLS)
    missing_cats = [c for c in cat_cols if c not in feature_order]
    if missing_cats:
        raise ValueError(
            f"metadata feature_list_ordered missing required categorical columns: {missing_cats}. "
            f"Expected categorical columns: {cat_cols}"
        )
    num_cols = [c for c in feature_order if c not in cat_cols]

    # Labels from metadata2
    label_set_ordered, label_to_index, index_to_label, num_classes = _get_label_info(meta)

    # Split files
    file_format, split_paths = _resolve_splits(pre_dir)

    # Vocab sizes for categorical features
    vocab_sizes = _get_vocab_sizes_from_meta2(meta, cat_cols)

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

    # ------------------------------------------------------------------
    # Task #2 diagnostics: categorical UNK/OOR rates (per split) + fail-fast mode
    # ------------------------------------------------------------------
    print("=" * 72)
    print("[DATA INTEGRITY] Categorical UNK/OOR diagnostics")
    print(f"  UNK_ID={UNK_ID}, PAD_NA_ID={PAD_NA_ID}")
    print(f"  RAISE_ON_OOR_CATEGORICAL={RAISE_ON_OOR_CATEGORICAL}")
    for split in ("train", "val", "test"):
        max_scan = SANITY_CHECK_MAX_ROWS if split == "train" else None
        stats = _scan_split_categorical_stats(
            split_name=split,
            split_path=split_paths[split],
            file_format=file_format,
            cat_cols=cat_cols,
            vocab_sizes=vocab_sizes,
            available_columns=split_columns[split],
            max_rows=max_scan,
        )
        rows = int(stats["rows_scanned"])
        cap_str = " (capped)" if stats.get("capped") else ""
        print(f"  [{split}] rows_scanned={rows:,}{cap_str}")
        for c in cat_cols:
            unk_raw = int(stats["unk_counts"].get(c, 0))
            oor = int(stats["oor_counts"].get(c, 0))
            # If clamping is enabled, OOR values become UNK at training time.
            unk_effective = unk_raw if RAISE_ON_OOR_CATEGORICAL else (unk_raw + oor)
            rate = (unk_effective / max(rows, 1)) if rows > 0 else float("nan")
            print(f"    {c}: UNK_rate={rate:.4%} (UNK_raw={unk_raw}, OOR_count={oor})")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Task #1 baselines: train-prior probabilities for NLL baseline + skill score
    # ------------------------------------------------------------------
    train_counts, train_prior_probs_np = _compute_train_label_priors(
        train_path=split_paths["train"], file_format=file_format, num_classes=num_classes
    )
    train_prior_probs = torch.tensor(train_prior_probs_np, dtype=torch.float32)
    majority_class = int(np.argmax(train_counts))

    print("[BASELINES] From TRAIN labels only")
    print(f"  Train label counts: {train_counts.tolist()}")
    print(f"  Train prior probs:  {[float(x) for x in train_prior_probs_np.tolist()]}")
    print(f"  Majority class id:  {majority_class} ({index_to_label.get(majority_class, str(majority_class))})")
    print("=" * 72)

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
        split_name="train",
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
        split_name="val",
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
        split_name="test",
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
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE,
            train_prior_probs=train_prior_probs,
            majority_class=majority_class,
            ece_bins=ECE_NUM_BINS,
        )

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train loss: {train_metrics.loss:.6f}, train acc: {train_metrics.accuracy:.4f} | "
            f"val loss: {val_metrics.loss:.6f}, val acc: {val_metrics.accuracy:.4f} | "
            f"time: {dt:.1f}s"
        )

        _print_probabilistic_report("VAL", val_metrics, index_to_label)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics.loss),
                "train_accuracy": float(train_metrics.accuracy),
                "val_loss": float(val_metrics.loss),
                "val_accuracy": float(val_metrics.accuracy),
                "val_nll": float(val_metrics.nll) if val_metrics.nll is not None else None,
                "val_brier": float(val_metrics.brier) if val_metrics.brier is not None else None,
                "val_ece": float(val_metrics.ece) if val_metrics.ece is not None else None,
                "val_prior_nll": float(val_metrics.prior_nll) if val_metrics.prior_nll is not None else None,
                "val_skill": float(val_metrics.skill) if val_metrics.skill is not None else None,
                "val_macro_nll": float(val_metrics.macro_nll) if val_metrics.macro_nll is not None else None,
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

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE,
        train_prior_probs=train_prior_probs,
        majority_class=majority_class,
        ece_bins=ECE_NUM_BINS,
    )
    print("-" * 72)
    print(f"Best val loss: {best_val_loss:.6f} (epoch {best_epoch}), best val acc: {best_val_acc:.4f}")
    print(f"Test loss:     {test_metrics.loss:.6f}, test acc:     {test_metrics.accuracy:.4f}")
    if best_val_metrics is not None:
        _print_probabilistic_report("VAL(best)", best_val_metrics, index_to_label)
    _print_probabilistic_report("TEST", test_metrics, index_to_label)
    print("-" * 72)

    # Save artifacts
    _ensure_dir(ARTIFACT_DIR)

    model_path = os.path.join(ARTIFACT_DIR, "model.pt")
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
        "ARTIFACT_DIR": ARTIFACT_DIR,
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
        },
        "timestamp": _now_str(),
    }
    _write_json(os.path.join(ARTIFACT_DIR, "train_config.json"), train_config)

    # Label maps (consistent with preprocessing2)
    _write_json(os.path.join(ARTIFACT_DIR, "label_to_index.json"), {k: int(v) for k, v in label_to_index.items()})
    _write_json(os.path.join(ARTIFACT_DIR, "index_to_label.json"), {str(k): v for k, v in index_to_label.items()})

    # Copy metadata.json used
    shutil.copy(meta_path, os.path.join(ARTIFACT_DIR, "metadata.json"))

    # Metrics
    metrics = {
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "best_val_accuracy": float(best_val_acc),
        "best_val_nll": float(best_val_metrics.nll) if best_val_metrics and best_val_metrics.nll is not None else None,
        "best_val_brier": float(best_val_metrics.brier) if best_val_metrics and best_val_metrics.brier is not None else None,
        "best_val_ece": float(best_val_metrics.ece) if best_val_metrics and best_val_metrics.ece is not None else None,
        "best_val_prior_nll": float(best_val_metrics.prior_nll) if best_val_metrics and best_val_metrics.prior_nll is not None else None,
        "best_val_skill": float(best_val_metrics.skill) if best_val_metrics and best_val_metrics.skill is not None else None,
        "best_val_macro_nll": float(best_val_metrics.macro_nll) if best_val_metrics and best_val_metrics.macro_nll is not None else None,
        "best_train_loss": float(best_train_metrics.loss) if best_train_metrics else None,
        "best_train_accuracy": float(best_train_metrics.accuracy) if best_train_metrics else None,
        "best_val_loss_epoch": float(best_val_metrics.loss) if best_val_metrics else None,
        "best_val_accuracy_epoch": float(best_val_metrics.accuracy) if best_val_metrics else None,
        "final_test_loss": float(test_metrics.loss),
        "final_test_accuracy": float(test_metrics.accuracy),
        "final_test_nll": float(test_metrics.nll) if test_metrics.nll is not None else None,
        "final_test_brier": float(test_metrics.brier) if test_metrics.brier is not None else None,
        "final_test_ece": float(test_metrics.ece) if test_metrics.ece is not None else None,
        "final_test_prior_nll": float(test_metrics.prior_nll) if test_metrics.prior_nll is not None else None,
        "final_test_skill": float(test_metrics.skill) if test_metrics.skill is not None else None,
        "final_test_macro_nll": float(test_metrics.macro_nll) if test_metrics.macro_nll is not None else None,
        "history": history,  # per-epoch train/val metrics
        "timestamp": _now_str(),
    }
    _write_json(os.path.join(ARTIFACT_DIR, "metrics.json"), metrics)

    print(f"Artifacts saved to: {ARTIFACT_DIR}")
    print(f"  - {model_path}")
    print("  - train_config.json, label_to_index.json, index_to_label.json, metadata.json, metrics.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
