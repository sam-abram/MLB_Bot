
#!/usr/bin/env python3
"""
train_model.py

Train a PyTorch multiclass model using outputs from preprocess_data.py.

Runnable:
    python train_model.py

Assumptions:
- Reads preprocessed/metadata.json
- Reads train/val/test from preprocessed/:
  - train.parquet, val.parquet, test.parquet if they exist (all three)
  - otherwise train.csv.gz, val.csv.gz, test.csv.gz
- Uses meta["features"]["feature_list_ordered"] for exact feature order
- Uses "label" column as target
- Uses meta["label_mapping"]["final_label_set"] as the class set
- Treats feature columns ending in "_id" as categorical ID features
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
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
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
EMBED_DIM_DEFAULT = 16  # used for _id embeddings unless overridden
HIDDEN_DIMS = [256, 128]
DROPOUT = 0.2
EARLY_STOP_PATIENCE = 2  # stop if val loss doesn’t improve
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0  # safe default
MAX_TRAIN_ROWS = None  # optional cap for quick testing


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
    for k, p in parquet_paths.items():
        if not os.path.exists(p):
            missing.append(p)
    for k, p in csv_paths.items():
        if not os.path.exists(p):
            missing.append(p)
    raise FileNotFoundError(
        "Could not find a complete set of split files. Expected either all parquet:\n"
        f"  {list(parquet_paths.values())}\n"
        "or all csv.gz:\n"
        f"  {list(csv_paths.values())}\n"
        f"Missing: {missing}"
    )


def _get_label_list(meta: dict) -> List[str]:
    try:
        labels = meta["label_mapping"]["final_label_set"]
    except KeyError as e:
        raise KeyError("metadata.json missing meta['label_mapping']['final_label_set']") from e
    # Use string keys deterministically
    return [str(x) for x in labels]


def _build_label_maps(label_list: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label_to_index = {lab: i for i, lab in enumerate(label_list)}
    index_to_label = {i: lab for i, lab in enumerate(label_list)}
    return label_to_index, index_to_label


def _safe_series_to_int64(s: pd.Series) -> np.ndarray:
    # Fill NaNs and coerce to integer safely
    if s.dtype.kind in ("i", "u"):
        arr = s.to_numpy(copy=False)
        return arr.astype(np.int64, copy=False)
    # For floats/objects, fill NA then convert
    s2 = s.fillna(0)
    # If objects, try numeric coercion; otherwise, fallback to 0
    if s2.dtype == object:
        s2 = pd.to_numeric(s2, errors="coerce").fillna(0)
    return s2.astype(np.int64, copy=False).to_numpy(copy=False)


def _safe_series_to_float32(s: pd.Series) -> np.ndarray:
    if s.dtype.kind == "f":
        return s.fillna(0.0).astype(np.float32, copy=False).to_numpy(copy=False)
    if s.dtype.kind in ("i", "u"):
        return s.astype(np.float32, copy=False).to_numpy(copy=False)
    # objects
    s2 = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return s2.astype(np.float32, copy=False).to_numpy(copy=False)


def _choose_emb_dim(vocab_size: int) -> int:
    """
    Simple rule; if EMBED_DIM_DEFAULT is set, use it.
    Otherwise, fallback to a heuristic.
    """
    if EMBED_DIM_DEFAULT is not None:
        return int(EMBED_DIM_DEFAULT)
    dim = int(round(math.sqrt(max(2, vocab_size)) * 2))
    dim = max(4, min(64, dim))
    return dim


def _get_vocab_sizes_from_meta(meta: dict, cat_cols: List[str]) -> Dict[str, int]:
    """
    For each categorical encoded column like 'pitch_type_id', look up vocab size
    in metadata using the raw key 'pitch_type' (strip trailing '_id').

    Returned dict is keyed by the encoded feature column name (the _id name),
    because the model iterates cat_cols which are _id feature names.

    If metadata lacks a vocab size for a given raw key, raise ValueError.
    """
    if not isinstance(meta, dict):
        raise ValueError("metadata.json root must be a dict")

    vocabs = meta.get("categorical_vocabs")
    if not isinstance(vocabs, dict):
        raise ValueError("metadata.json missing meta['categorical_vocabs']")

    vocab_sizes = vocabs.get("vocab_sizes") or vocabs.get("vocab_size") or vocabs.get("vocab_sizes_by_col")
    if not isinstance(vocab_sizes, dict):
        raise ValueError("metadata.json missing meta['categorical_vocabs']['vocab_sizes'] (dict)")

    out: Dict[str, int] = {}
    available_keys = set(str(k) for k in vocab_sizes.keys())

    for enc_col in cat_cols:
        if not enc_col.endswith("_id"):
            raise ValueError(f"Expected categorical encoded column to end with '_id', got: {enc_col}")
        raw_key = enc_col[: -len("_id")]

        if raw_key not in vocab_sizes:
            raise ValueError(
                f"Missing vocab size in metadata for categorical feature raw key '{raw_key}' "
                f"(from encoded column '{enc_col}'). Available vocab keys include: "
                f"{sorted(list(available_keys))[:50]}{'...' if len(available_keys) > 50 else ''}"
            )

        try:
            vs = int(vocab_sizes[raw_key])
        except Exception as e:
            raise ValueError(
                f"Non-integer vocab size for raw key '{raw_key}' (encoded '{enc_col}'): {vocab_sizes[raw_key]!r}"
            ) from e

        # Add 2 reserved IDs (safe default); keep at least 2
        vs = max(2, vs + 2)
        out[enc_col] = vs

    return out


def _maybe_class_weights_from_meta(meta: dict, label_list: List[str]) -> torch.Tensor:
    """
    Compute class weights aligned to meta["label_mapping"]["final_label_set"].

    Preferred source (already in final-label space, post-rare-remap):
      meta["transform_summary"]["label_counts_by_split"]["train"]

    Fallback:
      Build final-label counts from canonical train counts + rare-label list:
        - non-OTHER labels: canonical_count[label]
        - OTHER: sum(canonical_count[rare_label] for rare_label in rare_labels_remapped_to_other)

    Weighting:
      For labels with count>0: weight_i = total / (C * count_i) using only nonzero-count labels
      For labels with count==0: weight_i = 0.0 (no training examples anyway)
    """
    if not isinstance(meta, dict):
        raise ValueError("metadata.json root must be a dict")

    # --- 1) Preferred: transform_summary train counts (final-label space) ---
    counts_final = None
    ts = meta.get("transform_summary")
    if isinstance(ts, dict):
        lcs = ts.get("label_counts_by_split")
        if isinstance(lcs, dict):
            train_counts = lcs.get("train")
            if isinstance(train_counts, dict) and len(train_counts) > 0:
                counts_final = {str(k): float(v) for k, v in train_counts.items() if v is not None}

    # --- 2) Fallback: build final-label counts from canonical + rare list ---
    if counts_final is None:
        lm = meta.get("label_mapping", {})
        if not isinstance(lm, dict):
            raise ValueError("metadata.json missing meta['label_mapping'] dict")

        canonical = lm.get("train_label_counts_canonical")
        rare_labels = lm.get("rare_labels_remapped_to_other")
        if not isinstance(canonical, dict) or not isinstance(rare_labels, list):
            raise ValueError(
                "Could not compute class weights. Expected either:\n"
                "  meta['transform_summary']['label_counts_by_split']['train'] (preferred)\n"
                "or fallback keys:\n"
                "  meta['label_mapping']['train_label_counts_canonical'] (dict)\n"
                "  meta['label_mapping']['rare_labels_remapped_to_other'] (list)"
            )

        canonical_norm = {str(k): float(v) for k, v in canonical.items() if v is not None}
        rare_norm = [str(x) for x in rare_labels]

        counts_final = {}
        for lab in label_list:
            lab_str = str(lab)
            if lab_str == "OTHER":
                counts_final[lab_str] = float(sum(canonical_norm.get(r, 0.0) for r in rare_norm))
            else:
                counts_final[lab_str] = float(canonical_norm.get(lab_str, 0.0))

    # --- 3) Convert to weights aligned with label_list ---
    class_counts = np.array([counts_final.get(str(lab), 0.0) for lab in label_list], dtype=np.float64)
    nonzero = class_counts > 0.0
    if not np.any(nonzero):
        raise ValueError("Cannot compute class weights: all class counts are 0.")

    total = float(class_counts[nonzero].sum())
    C = int(nonzero.sum())

    weights = np.zeros_like(class_counts, dtype=np.float64)
    weights[nonzero] = total / (C * class_counts[nonzero])

    return torch.tensor(weights, dtype=torch.float32)


# -----------------------------
# Streaming IterableDataset that yields BATCHED tensors
# -----------------------------
class SplitBatchIterableDataset(IterableDataset):
    """
    IterableDataset that streams data in batches from parquet (pyarrow) or csv.gz (pandas chunks),
    yielding (x_cat, x_num, y) where each is a tensor batch.

    Requirements met:
    - No full-file reads into RAM
    - Parquet: ParquetFile.iter_batches
    - CSV: pd.read_csv(..., chunksize=...)
    """

    def __init__(
        self,
        path: str,
        file_format: str,
        feature_order: List[str],
        cat_cols: List[str],
        num_cols: List[str],
        label_to_index: Dict[str, int],
        batch_size: int,
        max_rows: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.path = path
        self.file_format = file_format
        self.feature_order = feature_order
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.label_to_index = label_to_index
        self.batch_size = int(batch_size)
        self.max_rows = None if max_rows is None else int(max_rows)

        # Read only needed columns
        self.columns = list(feature_order) + ["label"]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.file_format == "parquet":
            yield from self._iter_parquet()
        elif self.file_format == "csv":
            yield from self._iter_csv()
        else:
            raise ValueError(f"Unsupported file_format={self.file_format}")

    def _iter_parquet(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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

            out = self._df_to_tensors(df)
            if out is None:
                continue
            yield out

    def _iter_csv(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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

            out = self._df_to_tensors(df)
            if out is None:
                continue
            yield out

    def _df_to_tensors(
        self, df: pd.DataFrame
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Map labels via string form to keep deterministic behavior
        labels_raw = df["label"].astype(str)
        y_idx = labels_raw.map(self.label_to_index)

        # Drop unknown/missing labels defensively
        mask = y_idx.notna()
        if not mask.any():
            return None

        df2 = df.loc[mask].copy()
        y_idx2 = y_idx.loc[mask].astype(np.int64, copy=False).to_numpy(copy=False)

        # Build x_cat, x_num in the exact feature order partition
        if self.cat_cols:
            cat_arrays = [_safe_series_to_int64(df2[c]) for c in self.cat_cols]
            x_cat_np = np.stack(cat_arrays, axis=1) if len(cat_arrays) > 1 else cat_arrays[0].reshape(-1, 1)
            x_cat = torch.from_numpy(x_cat_np.astype(np.int64, copy=False))
        else:
            x_cat = torch.empty((len(df2), 0), dtype=torch.int64)

        if self.num_cols:
            num_arrays = [_safe_series_to_float32(df2[c]) for c in self.num_cols]
            x_num_np = np.stack(num_arrays, axis=1) if len(num_arrays) > 1 else num_arrays[0].reshape(-1, 1)
            x_num = torch.from_numpy(x_num_np.astype(np.float32, copy=False))
        else:
            x_num = torch.empty((len(df2), 0), dtype=torch.float32)

        y = torch.from_numpy(y_idx2.astype(np.int64, copy=False))
        return x_cat, x_num, y


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
        self.vocab_sizes = {k: int(v) for k, v in vocab_sizes.items()}
        self.num_classes = int(num_classes)

        self.embeddings = nn.ModuleDict()
        emb_out_dim = 0
        self.emb_dims: Dict[str, int] = {}

        for col in self.cat_cols:
            vs = int(self.vocab_sizes[col])
            ed = _choose_emb_dim(vs)
            self.emb_dims[col] = ed
            # padding_idx=0 is a safe default for "missing" IDs if preprocessing uses 0.
            self.embeddings[col] = nn.Embedding(num_embeddings=vs, embedding_dim=ed, padding_idx=0)
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

                # Fail loudly if preprocessing produced out-of-range IDs
                if ids.numel() > 0:
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
            parts.append(x_num)

        if parts:
            x = torch.cat(parts, dim=1)
        else:
            # Edge case: no features at all
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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: str,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x_cat, x_num, y in dataset:
        if y.numel() == 0:
            continue
        x_cat = x_cat.to(device, non_blocking=True)
        x_num = x_num.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x_cat, x_num)
        loss = criterion(logits, y)

        n = int(y.numel())
        total_loss += float(loss.item()) * n
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += n

    if total_n == 0:
        return EpochMetrics(loss=float("nan"), accuracy=float("nan"), n=0)

    return EpochMetrics(
        loss=total_loss / total_n,
        accuracy=total_correct / total_n,
        n=total_n,
    )


def train_one_epoch(
    model: nn.Module,
    dataset: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x_cat, x_num, y in dataset:
        if y.numel() == 0:
            continue
        x_cat = x_cat.to(device, non_blocking=True)
        x_num = x_num.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_cat, x_num)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        n = int(y.numel())
        total_loss += float(loss.item()) * n
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_n += n

    if total_n == 0:
        return EpochMetrics(loss=float("nan"), accuracy=float("nan"), n=0)

    return EpochMetrics(
        loss=total_loss / total_n,
        accuracy=total_correct / total_n,
        n=total_n,
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    meta_path = os.path.join(PREPROCESSED_DIR, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    meta = _read_json(meta_path)

    try:
        feature_order = meta["features"]["feature_list_ordered"]
    except KeyError as e:
        raise KeyError("metadata.json missing meta['features']['feature_list_ordered']") from e
    if not isinstance(feature_order, list) or not feature_order:
        raise ValueError("meta['features']['feature_list_ordered'] must be a non-empty list")

    # Partition features by suffix rule, preserving order
    feature_order = [str(c) for c in feature_order]
    cat_cols = [c for c in feature_order if c.endswith("_id")]
    num_cols = [c for c in feature_order if not c.endswith("_id")]

    label_list = _get_label_list(meta)
    label_to_index, index_to_label = _build_label_maps(label_list)
    num_classes = len(label_list)

    file_format, split_paths = _resolve_splits(PREPROCESSED_DIR)

    # Vocab sizes for categorical features
    vocab_sizes = _get_vocab_sizes_from_meta(meta, cat_cols)

    # Class weights (required; will raise if metadata missing/incomplete)
    class_weights = _maybe_class_weights_from_meta(meta, label_list).to(DEVICE)

    # Build datasets (IterableDataset yielding batches)
    train_ds = SplitBatchIterableDataset(
        path=split_paths["train"],
        file_format=file_format,
        feature_order=feature_order,
        cat_cols=cat_cols,
        num_cols=num_cols,
        label_to_index=label_to_index,
        batch_size=BATCH_SIZE,
        max_rows=MAX_TRAIN_ROWS,
    )
    val_ds = SplitBatchIterableDataset(
        path=split_paths["val"],
        file_format=file_format,
        feature_order=feature_order,
        cat_cols=cat_cols,
        num_cols=num_cols,
        label_to_index=label_to_index,
        batch_size=BATCH_SIZE,
        max_rows=None,
    )
    test_ds = SplitBatchIterableDataset(
        path=split_paths["test"],
        file_format=file_format,
        feature_order=feature_order,
        cat_cols=cat_cols,
        num_cols=num_cols,
        label_to_index=label_to_index,
        batch_size=BATCH_SIZE,
        max_rows=None,
    )

    # Wrap in DataLoaders (batch_size=None because dataset already yields batches)
    # NUM_WORKERS is fixed to 0 by default for safety with IterableDataset.
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

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Summary logging
    print("=" * 72)
    print("Training summary")
    print(f"Start time:         {_now_str()}")
    print(f"Preprocessed dir:   {PREPROCESSED_DIR}")
    print(f"Split format:       {file_format}")
    print(f"Features:           {len(feature_order)}")
    print(f"  Categorical _id:  {len(cat_cols)}")
    print(f"  Numeric:          {len(num_cols)}")
    print(f"Classes:            {num_classes}")
    print(f"Device:             {DEVICE}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Epochs:             {NUM_EPOCHS}")
    print(f"Early stop patience:{EARLY_STOP_PATIENCE}")
    print(f"Max train rows:     {MAX_TRAIN_ROWS}")
    print("=" * 72)

    best_val_loss = float("inf")
    best_epoch = -1
    best_val_acc = 0.0
    best_state = None
    no_improve = 0

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

        improved = (val_metrics.loss < best_val_loss) if not math.isnan(val_metrics.loss) else False
        if improved:
            best_val_loss = float(val_metrics.loss)
            best_val_acc = float(val_metrics.accuracy)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
    print("-" * 72)
    print(f"Best val loss: {best_val_loss:.6f} (epoch {best_epoch}), best val acc: {best_val_acc:.4f}")
    print(f"Test loss:     {test_metrics.loss:.6f}, test acc:     {test_metrics.accuracy:.4f}")
    print("-" * 72)

    # Save artifacts
    _ensure_dir(ARTIFACT_DIR)

    model_path = os.path.join(ARTIFACT_DIR, "model.pt")
    torch.save(model.state_dict(), model_path)

    # Train config + resolved lists
    train_config = {
        "PREPROCESSED_DIR": PREPROCESSED_DIR,
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
        "resolved": {
            "file_format": file_format,
            "split_paths": split_paths,
            "feature_list_ordered": feature_order,
            "categorical_id_cols": cat_cols,
            "numeric_cols": num_cols,
            "num_classes": num_classes,
            "vocab_sizes": vocab_sizes,
            "embedding_dims": getattr(model, "emb_dims", {}),
        },
        "timestamp": _now_str(),
    }
    _write_json(os.path.join(ARTIFACT_DIR, "train_config.json"), train_config)

    # Label maps
    _write_json(os.path.join(ARTIFACT_DIR, "label_to_index.json"), {k: int(v) for k, v in label_to_index.items()})
    _write_json(os.path.join(ARTIFACT_DIR, "index_to_label.json"), {str(k): v for k, v in index_to_label.items()})

    # Copy metadata.json used
    shutil.copy(meta_path, os.path.join(ARTIFACT_DIR, "metadata.json"))

    # Metrics
    metrics = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "final_test_loss": float(test_metrics.loss),
        "final_test_accuracy": float(test_metrics.accuracy),
        "timestamp": _now_str(),
    }
    _write_json(os.path.join(ARTIFACT_DIR, "metrics.json"), metrics)

    print(f"Artifacts saved to: {ARTIFACT_DIR}")
    print(f"  - {model_path}")
    print(f"  - train_config.json, label_to_index.json, index_to_label.json, metadata.json, metrics.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
