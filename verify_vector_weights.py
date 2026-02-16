#!/usr/bin/env python3
"""
verify_vector_weights.py

Trains SimpleVectorWeightModel 10 times with different random seeds
to verify that gradient descent converges to a consistent optimum.

Usage:
    python verify_vector_weights.py
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
    EARLY_STOP_PATIENCE,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
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
    SplitBatchIterableDataset,
    SimpleVectorWeightModel,
    train_one_epoch,
    evaluate,
    evaluate_logloss_only,
)

NUM_RUNS = 10
VECTOR_NAMES = [
    "v1_batter", "v2_pitcher", "v3_stadium",
    "v4_bat_plat", "v5_pit_plat", "v6_mix",
]


def make_loader(path, file_format, feature_order, num_cols, num_classes, available_columns, seed, shuffle=False, weight_col=None):
    ds = SplitBatchIterableDataset(
        path=path, file_format=file_format, feature_order=feature_order,
        cat_cols=[], num_cols=num_cols, vocab_sizes={}, num_classes=num_classes,
        batch_size=BATCH_SIZE, available_columns=available_columns,
        shuffle_rows=shuffle, seed=seed, weight_col=weight_col,
    )
    return DataLoader(ds, batch_size=None, num_workers=NUM_WORKERS)


def train_one_run(seed, train_path, val_path, test_path, file_format, feature_order,
                  num_cols, num_classes, train_cols, val_cols, test_cols, weight_col):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_ldr = make_loader(train_path, file_format, feature_order, num_cols, num_classes, train_cols, seed, shuffle=SHUFFLE_TRAIN_ROWS_WITHIN_BATCH, weight_col=weight_col)
    val_ldr = make_loader(val_path, file_format, feature_order, num_cols, num_classes, val_cols, seed + 1000)
    test_ldr = make_loader(test_path, file_format, feature_order, num_cols, num_classes, test_cols, seed + 2000)

    model = SimpleVectorWeightModel(num_vectors=6, num_outcomes=num_classes, use_log_n_weighting=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, train_ldr, criterion, optimizer, DEVICE)
        val_m = evaluate(model, val_ldr, criterion, DEVICE)

        if not math.isnan(val_m.loss) and val_m.loss < best_val_loss:
            best_val_loss = float(val_m.loss)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_ll = evaluate_logloss_only(model, val_ldr, DEVICE)
    test_ll = evaluate_logloss_only(model, test_ldr, DEVICE)
    weights = model.get_learned_weights()

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_logloss": val_ll,
        "test_logloss": test_ll,
        "weights": weights,
    }


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

    print(f"Running {NUM_RUNS} training runs with seeds 1-{NUM_RUNS}...\n")

    results = []
    for i in range(1, NUM_RUNS + 1):
        t0 = time.time()
        print(f"--- Run {i}/{NUM_RUNS} (seed={i}) ---")
        r = train_one_run(
            seed=i,
            train_path=split_paths["train"], val_path=split_paths["val"],
            test_path=split_paths["test"], file_format=file_format,
            feature_order=vec_feature_order, num_cols=num_cols,
            num_classes=num_classes, train_cols=split_cols["train"],
            val_cols=split_cols["val"], test_cols=split_cols["test"],
            weight_col=weight_col,
        )
        results.append(r)
        dt = time.time() - t0
        print(f"  epoch={r['best_epoch']}, val={r['val_logloss']:.6f}, test={r['test_logloss']:.6f} ({dt:.1f}s)\n")

    # Print results table
    weight_keys = list(results[0]["weights"].keys())

    print("=" * 100)
    print("  RESULTS: SimpleVectorWeightModel across 10 seeds")
    print("=" * 100)
    header = f"  {'Seed':>4s} {'Epoch':>5s} {'Val LL':>10s} {'Test LL':>10s}"
    for vn in VECTOR_NAMES:
        header += f" {vn:>10s}"
    print(header)
    print("-" * 100)

    all_weights = {k: [] for k in weight_keys}
    val_lls = []
    test_lls = []

    for r in results:
        row = f"  {r['seed']:>4d} {r['best_epoch']:>5d} {r['val_logloss']:>10.6f} {r['test_logloss']:>10.6f}"
        for k in weight_keys:
            w = r["weights"][k]
            row += f" {w:>10.4f}"
            all_weights[k].append(w)
        val_lls.append(r["val_logloss"])
        test_lls.append(r["test_logloss"])
        print(row)

    print("-" * 100)

    # Mean
    mean_row = f"  {'Mean':>4s} {'':>5s} {np.mean(val_lls):>10.6f} {np.mean(test_lls):>10.6f}"
    for k in weight_keys:
        mean_row += f" {np.mean(all_weights[k]):>10.4f}"
    print(mean_row)

    # Std
    std_row = f"  {'Std':>4s} {'':>5s} {np.std(val_lls):>10.6f} {np.std(test_lls):>10.6f}"
    for k in weight_keys:
        std_row += f" {np.std(all_weights[k]):>10.4f}"
    print(std_row)

    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
