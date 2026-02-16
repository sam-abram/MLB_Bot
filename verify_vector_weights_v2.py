#!/usr/bin/env python3
"""
verify_vector_weights_v2.py

Train SimpleVectorWeightModel with higher LR and longer patience
to let weights properly differentiate.

Usage:
    python verify_vector_weights_v2.py
"""

from __future__ import annotations

import math
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_model2 import (
    BATCH_SIZE,
    DEVICE,
    GRAD_CLIP_NORM,
    NUM_EPOCHS,
    NUM_WORKERS,
    PREPROCESSED_DIR,
    SHUFFLE_TRAIN_ROWS_WITHIN_BATCH,
    TARGET_COL,
    WEIGHT_COL,
    WEIGHT_DECAY,
    _get_feature_order,
    _get_label_info,
    _read_json,
    _resolve_splits,
    _check_required_columns,
    _stream_y_w,
    SplitBatchIterableDataset,
    SimpleVectorWeightModel,
    train_one_epoch,
    evaluate,
    evaluate_logloss_only,
)

# Overrides for this experiment
LR = 3e-3          # 10x higher
PATIENCE = 20       # longer patience
SEED = 42

VECTOR_NAMES = ["v1_batter", "v2_pitcher", "v3_stadium", "v4_bat_plat", "v5_pit_plat", "v6_mix"]


def make_loader(path, file_format, feature_order, num_cols, num_classes, available_columns, seed, shuffle=False, weight_col=None):
    ds = SplitBatchIterableDataset(
        path=path, file_format=file_format, feature_order=feature_order,
        cat_cols=[], num_cols=num_cols, vocab_sizes={}, num_classes=num_classes,
        batch_size=BATCH_SIZE, available_columns=available_columns,
        shuffle_rows=shuffle, seed=seed, weight_col=weight_col,
    )
    return DataLoader(ds, batch_size=None, num_workers=NUM_WORKERS)


def get_weights(model):
    with torch.no_grad():
        w = torch.softmax(model.raw_weights, dim=0).cpu().numpy()
    return [float(x) for x in w]


def get_log_n_scales(model):
    with torch.no_grad():
        s = model.log_n_scale.cpu().numpy()
    return [float(x) for x in s]


def main():
    pre_dir = sys.argv[1] if len(sys.argv) > 1 else PREPROCESSED_DIR
    meta = _read_json(f"{pre_dir}/metadata.json")
    feature_order = _get_feature_order(meta)
    _, _, _, num_classes = _get_label_info(meta)
    file_format, split_paths = _resolve_splits(pre_dir)

    cat_cols = list(meta.get("features", {}).get("categorical_features", []))
    num_cols = [c for c in feature_order if c not in set(cat_cols)]
    vec_feature_order = list(num_cols)

    required = vec_feature_order + [TARGET_COL]
    split_cols = {}
    for s in ("train", "val", "test"):
        split_cols[s] = _check_required_columns(s, split_paths[s], file_format, required)

    weight_col = WEIGHT_COL if WEIGHT_COL in set(split_cols["train"]) else None

    # Base rate
    K = num_classes
    eps = 1e-12
    counts = np.zeros((K,), dtype=np.float64)
    for y, w in _stream_y_w(split_paths["train"], file_format, weight_col):
        counts += np.bincount(y, minlength=K, weights=w)
    p = counts / max(counts.sum(), eps)
    p = np.clip(p, eps, 1.0)
    p /= p.sum()
    loss_sum = 0.0
    w_sum = 0.0
    for y, w in _stream_y_w(split_paths["val"], file_format, weight_col):
        w = w.astype(np.float64)
        w_sum += float(w.sum())
        loss_sum += float((-np.log(p[y]) * w).sum())
    base_rate_val = loss_sum / max(w_sum, eps)
    print(f"Base-rate val logloss: {base_rate_val:.6f}")

    # Setup
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_ldr = make_loader(split_paths["train"], file_format, vec_feature_order, num_cols, num_classes, split_cols["train"], SEED, shuffle=SHUFFLE_TRAIN_ROWS_WITHIN_BATCH, weight_col=weight_col)
    val_ldr = make_loader(split_paths["val"], file_format, vec_feature_order, num_cols, num_classes, split_cols["val"], SEED + 1)
    test_ldr = make_loader(split_paths["test"], file_format, vec_feature_order, num_cols, num_classes, split_cols["test"], SEED + 2)

    model = SimpleVectorWeightModel(num_vectors=6, num_outcomes=num_classes, use_log_n_weighting=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"\nLR={LR}, patience={PATIENCE}, seed={SEED}")
    print(f"Initial weights: {['%.4f' % x for x in get_weights(model)]}")

    # Header
    print(f"\n{'Ep':>3s} {'Train':>10s} {'Val':>10s} {'%vsBase':>8s}", end="")
    for vn in VECTOR_NAMES:
        print(f" {vn:>10s}", end="")
    print(f" {'log_n_scales':>60s}")
    print("-" * 160)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0
    epoch_data = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_ldr, criterion, optimizer, DEVICE)
        val_m = evaluate(model, val_ldr, criterion, DEVICE)
        dt = time.time() - t0

        w = get_weights(model)
        s = get_log_n_scales(model)
        pct = 100.0 * (base_rate_val - val_m.loss) / base_rate_val

        print(f"{epoch:>3d} {train_m.loss:>10.6f} {val_m.loss:>10.6f} {pct:>+7.2f}%", end="")
        for wi in w:
            print(f" {wi:>10.4f}", end="")
        print(f"  [{', '.join('%.4f' % si for si in s)}]")

        epoch_data.append({"epoch": epoch, "train": train_m.loss, "val": val_m.loss, "weights": w, "scales": s})

        if not math.isnan(val_m.loss) and val_m.loss < best_val_loss:
            best_val_loss = float(val_m.loss)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stop at epoch {epoch} (best: {best_val_loss:.6f} @ epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_ll = evaluate_logloss_only(model, val_ldr, DEVICE)
    test_ll = evaluate_logloss_only(model, test_ldr, DEVICE)

    final_w = get_weights(model)
    final_s = get_log_n_scales(model)

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS (best epoch {best_epoch})")
    print(f"{'='*80}")
    print(f"  Val logloss:  {val_ll:.6f}  ({100*(base_rate_val - val_ll)/base_rate_val:+.2f}% vs base)")
    print(f"  Test logloss: {test_ll:.6f}")
    print(f"\n  Final vector weights:")
    for name, wi in zip(VECTOR_NAMES, final_w):
        print(f"    {name:<15s}: {wi:.4f} ({wi*100:.1f}%)")
    print(f"\n  Log-n scale parameters (how much sample size matters):")
    for name, si in zip(VECTOR_NAMES, final_s):
        print(f"    {name:<15s}: {si:+.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
