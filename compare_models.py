#!/usr/bin/env python3
"""
compare_models.py

Trains and evaluates all three model architectures on identical data splits
with the same random seed for a controlled head-to-head comparison.

  Model A: Embeddings only (PitchOutcomeModel with cat features, no numeric six-vectors)
  Model B: Simple vector weights (SimpleVectorWeightModel, 12 params)
  Model C: Hybrid (a * emb_probs + b * vec_probs)

Usage:
    python compare_models.py
    python compare_models.py /path/to/preprocessed_dir
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_model2 import (
    # Constants
    BATCH_SIZE,
    DEVICE,
    DROPOUT,
    EARLY_STOP_PATIENCE,
    GRAD_CLIP_NORM,
    HIDDEN_DIMS,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    PREPROCESSED_DIR,
    SEED,
    SHUFFLE_TRAIN_ROWS_WITHIN_BATCH,
    TARGET_COL,
    WEIGHT_COL,
    # Utilities
    _get_feature_order,
    _get_label_info,
    _get_vocab_sizes_from_meta2,
    _now_str,
    _read_json,
    _resolve_splits,
    _check_required_columns,
    _stream_y_w,
    # Dataset
    SplitBatchIterableDataset,
    # Models
    PitchOutcomeModel,
    SimpleVectorWeightModel,
    HybridModel,
    # Training / eval
    train_one_epoch,
    evaluate,
    evaluate_logloss_only,
    WEIGHT_DECAY,
)


def _reset_seed():
    """Reset all random seeds to SEED for reproducibility."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _make_loader(
    path: str,
    file_format: str,
    feature_order: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    vocab_sizes: Dict[str, int],
    num_classes: int,
    available_columns: List[str],
    shuffle: bool = False,
    weight_col: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> DataLoader:
    ds = SplitBatchIterableDataset(
        path=path,
        file_format=file_format,
        feature_order=feature_order,
        cat_cols=cat_cols,
        num_cols=num_cols,
        vocab_sizes=vocab_sizes,
        num_classes=num_classes,
        batch_size=BATCH_SIZE,
        available_columns=available_columns,
        max_rows=max_rows,
        shuffle_rows=shuffle,
        seed=SEED,
        weight_col=weight_col,
    )
    return DataLoader(ds, batch_size=None, num_workers=NUM_WORKERS)


def _compute_base_rate_logloss(
    train_path: str,
    test_path: str,
    file_format: str,
    num_classes: int,
    weight_col: Optional[str],
) -> float:
    """Compute base-rate logloss: train frequencies evaluated on test."""
    K = num_classes
    eps = 1e-12
    counts = np.zeros((K,), dtype=np.float64)
    for y, w in _stream_y_w(train_path, file_format, weight_col):
        counts += np.bincount(y, minlength=K, weights=w)
    p = counts / max(counts.sum(), eps)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()

    loss_sum = 0.0
    w_sum = 0.0
    for y, w in _stream_y_w(test_path, file_format, weight_col):
        w = w.astype(np.float64, copy=False)
        w_sum += float(w.sum())
        loss_sum += float((-np.log(p[y]) * w).sum())

    return float(loss_sum / max(w_sum, eps))


def train_and_evaluate(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> dict:
    """Train a model and return results dict."""
    print(f"\n{'='*72}")
    print(f"  Training: {model_name}")
    print(f"  Parameters: {_count_parameters(model):,}")
    print(f"{'='*72}")

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0

    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_m = evaluate(model, val_loader, criterion, DEVICE)
        dt = time.time() - t0

        print(
            f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train: {train_m.loss:.6f} | val: {val_m.loss:.6f} | {dt:.1f}s"
        )

        improved = val_m.loss < best_val_loss if not math.isnan(val_m.loss) else False
        if improved:
            best_val_loss = float(val_m.loss)
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"  Early stop at epoch {epoch} (best: {best_val_loss:.6f} @ epoch {best_epoch})")
                break

    total_time = time.time() - t_start

    if best_state is not None:
        model.load_state_dict(best_state)

    val_logloss = evaluate_logloss_only(model, val_loader, DEVICE)
    test_logloss = evaluate_logloss_only(model, test_loader, DEVICE)

    result = {
        "model_name": model_name,
        "num_params": _count_parameters(model),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_logloss": val_logloss,
        "test_logloss": test_logloss,
        "train_time_s": total_time,
    }

    # Extra info for hybrid model
    if hasattr(model, "get_ab_weights"):
        result["ab_weights"] = model.get_ab_weights()
    if hasattr(model, "get_learned_weights"):
        result["vector_weights"] = model.get_learned_weights()

    return result


def main() -> None:
    pre_dir = sys.argv[1] if len(sys.argv) > 1 else PREPROCESSED_DIR

    meta = _read_json(os.path.join(pre_dir, "metadata.json"))
    feature_order = _get_feature_order(meta)
    _, _, _, num_classes = _get_label_info(meta)
    file_format, split_paths = _resolve_splits(pre_dir)

    # Full feature set (both embeddings + six vectors)
    cat_cols = list(meta.get("features", {}).get("categorical_features", []))
    num_cols = [c for c in feature_order if c not in set(cat_cols)]
    vocab_sizes = _get_vocab_sizes_from_meta2(meta, cat_cols) if cat_cols else {}

    # Validate columns
    required_cols = list(feature_order) + [TARGET_COL]
    split_columns: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        cols = _check_required_columns(split, split_paths[split], file_format, required_cols)
        split_columns[split] = cols

    weight_col_avail = WEIGHT_COL if WEIGHT_COL in set(split_columns["train"]) else None

    # Compute base-rate logloss once (evaluated on val split, matching scorecard)
    print("Computing base-rate logloss...")
    base_rate_val = _compute_base_rate_logloss(
        split_paths["train"], split_paths["val"], file_format, num_classes, weight_col_avail
    )
    base_rate_test = _compute_base_rate_logloss(
        split_paths["train"], split_paths["test"], file_format, num_classes, weight_col_avail
    )
    print(f"  Base-rate val logloss:  {base_rate_val:.6f}")
    print(f"  Base-rate test logloss: {base_rate_test:.6f}")

    # Helper to build loaders with full feature set
    def make_loaders():
        train_ldr = _make_loader(
            split_paths["train"], file_format, feature_order, cat_cols, num_cols,
            vocab_sizes, num_classes, split_columns["train"],
            shuffle=SHUFFLE_TRAIN_ROWS_WITHIN_BATCH, weight_col=weight_col_avail,
        )
        val_ldr = _make_loader(
            split_paths["val"], file_format, feature_order, cat_cols, num_cols,
            vocab_sizes, num_classes, split_columns["val"],
        )
        test_ldr = _make_loader(
            split_paths["test"], file_format, feature_order, cat_cols, num_cols,
            vocab_sizes, num_classes, split_columns["test"],
        )
        return train_ldr, val_ldr, test_ldr

    results = []

    # -------------------------------------------------------
    # Model A: Embeddings only
    #   Uses cat_cols embeddings + pitcher_fatigue (1 numeric)
    #   Ignores the 42 six-vector features
    # -------------------------------------------------------
    _reset_seed()
    emb_num_cols = ["pitcher_fatigue"]  # only non-six-vector numeric
    emb_feature_order = list(cat_cols) + emb_num_cols

    train_a = _make_loader(
        split_paths["train"], file_format, emb_feature_order, cat_cols, emb_num_cols,
        vocab_sizes, num_classes, split_columns["train"],
        shuffle=SHUFFLE_TRAIN_ROWS_WITHIN_BATCH, weight_col=weight_col_avail,
    )
    val_a = _make_loader(
        split_paths["val"], file_format, emb_feature_order, cat_cols, emb_num_cols,
        vocab_sizes, num_classes, split_columns["val"],
    )
    test_a = _make_loader(
        split_paths["test"], file_format, emb_feature_order, cat_cols, emb_num_cols,
        vocab_sizes, num_classes, split_columns["test"],
    )

    model_a = PitchOutcomeModel(
        cat_cols=cat_cols,
        num_numeric=len(emb_num_cols),
        vocab_sizes=vocab_sizes,
        num_classes=num_classes,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    ).to(DEVICE)
    results.append(train_and_evaluate("A: Embeddings only", model_a, train_a, val_a, test_a))
    del model_a

    # -------------------------------------------------------
    # Model B: Simple vector weights (no embeddings)
    #   Uses only the 43 numeric features (pitcher_fatigue + 42 six-vector)
    # -------------------------------------------------------
    _reset_seed()
    vec_feature_order = list(num_cols)  # all numeric, no cat

    train_b = _make_loader(
        split_paths["train"], file_format, vec_feature_order, [], num_cols,
        {}, num_classes, split_columns["train"],
        shuffle=SHUFFLE_TRAIN_ROWS_WITHIN_BATCH, weight_col=weight_col_avail,
    )
    val_b = _make_loader(
        split_paths["val"], file_format, vec_feature_order, [], num_cols,
        {}, num_classes, split_columns["val"],
    )
    test_b = _make_loader(
        split_paths["test"], file_format, vec_feature_order, [], num_cols,
        {}, num_classes, split_columns["test"],
    )

    model_b = SimpleVectorWeightModel(
        num_vectors=6,
        num_outcomes=num_classes,
        use_log_n_weighting=True,
    ).to(DEVICE)
    results.append(train_and_evaluate("B: Vector weights", model_b, train_b, val_b, test_b))
    del model_b

    # -------------------------------------------------------
    # Model C: Hybrid (a * emb_probs + b * vec_probs)
    #   Uses all 48 features (5 cat + 43 numeric)
    # -------------------------------------------------------
    _reset_seed()
    train_c, val_c, test_c = make_loaders()

    model_c = HybridModel(
        cat_cols=cat_cols,
        num_numeric=len(num_cols),
        vocab_sizes=vocab_sizes,
        num_classes=num_classes,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    ).to(DEVICE)
    results.append(train_and_evaluate("C: Hybrid (a*emb + b*vec)", model_c, train_c, val_c, test_c))
    del model_c

    # -------------------------------------------------------
    # Print comparison table
    # -------------------------------------------------------
    print("\n")
    print("=" * 80)
    print("  COMPARISON TABLE")
    print("=" * 80)
    print(
        f"  {'Model':<30s} {'Params':>8s} {'Epoch':>6s} "
        f"{'Val LL':>10s} {'Test LL':>10s} {'% vs Base':>10s}"
    )
    print("-" * 80)
    for r in results:
        pct = 100.0 * (base_rate_val - r["val_logloss"]) / base_rate_val
        print(
            f"  {r['model_name']:<30s} {r['num_params']:>8,d} {r['best_epoch']:>6d} "
            f"{r['val_logloss']:>10.6f} {r['test_logloss']:>10.6f} {pct:>+9.2f}%"
        )
    print("-" * 80)
    print(f"  {'Base rate':<30s} {'--':>8s} {'--':>6s} {base_rate_val:>10.6f} {base_rate_test:>10.6f} {'0.00%':>10s}")
    print("=" * 80)

    # Print hybrid details if present
    for r in results:
        if "ab_weights" in r:
            ab = r["ab_weights"]
            print(f"\n  Hybrid combination weights: a_embedding={ab['a_embedding']:.4f} ({ab['a_embedding']*100:.1f}%), b_vector={ab['b_vector']:.4f} ({ab['b_vector']*100:.1f}%)")
        if "vector_weights" in r:
            vw = r["vector_weights"]
            print(f"  Vector weights: " + ", ".join(f"{k}={v:.3f}" for k, v in vw.items()))

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
