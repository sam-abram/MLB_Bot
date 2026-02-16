#!/usr/bin/env python3
"""
train_logit_hybrid_test.py

Logit-space hybrid: combined_logits = a * emb_logits + b * vec_logits
where a,b are softmax-normalized learnable weights.

Uses identical data, splits, seed, and training settings as train_model2.py.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_model2 import (
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
    WEIGHT_DECAY,
    PAD_NA_ID,
    _choose_emb_dim,
    _get_feature_order,
    _get_label_info,
    _get_vocab_sizes_from_meta2,
    _now_str,
    _read_json,
    _write_json,
    _ensure_dir,
    _resolve_splits,
    _check_required_columns,
    _stream_y_w,
    SplitBatchIterableDataset,
    train_one_epoch,
    evaluate,
    evaluate_logloss_only,
)

ARTIFACT_DIR = "model_artifacts_logit_hybrid"


class LogitHybridModel(nn.Module):
    """
    Hybrid that combines in logit space:
      combined_logits = a * emb_logits + b * vec_logits
    where vec_logits = log(vec_probs + eps).
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
        self.vocab_sizes = {k: int(v) for k, v in vocab_sizes.items()}

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

        emb_layers: List[nn.Module] = []
        prev = emb_out_dim
        for h in hidden_dims:
            emb_layers.append(nn.Linear(prev, int(h)))
            emb_layers.append(nn.ReLU())
            emb_layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        emb_layers.append(nn.Linear(prev, self.num_classes))
        self.emb_mlp = nn.Sequential(*emb_layers)

        # --- Vector head ---
        self.num_vectors = 6
        self.raw_vec_weights = nn.Parameter(torch.zeros(self.num_vectors))
        self.log_n_scale = nn.Parameter(torch.zeros(self.num_vectors))

        # --- Combination weights ---
        self.raw_ab = nn.Parameter(torch.zeros(2))

        print(f"[MODEL] LogitHybridModel:")
        print(f"  Embedding head: {self.emb_dims} -> MLP {hidden_dims} -> {self.num_classes}")
        print(f"  Vector head: {self.num_vectors} vectors with log_n weighting")
        print(f"  Combination: a * emb_logits + b * log(vec_probs)")

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        # --- Embedding head (raw logits) ---
        emb_parts: List[torch.Tensor] = []
        for i, col in enumerate(self.cat_cols):
            ids = torch.clamp(x_cat[:, i], min=0, max=self.vocab_sizes[col] - 1)
            emb_parts.append(self.embeddings[col](ids))
        emb_logits = self.emb_mlp(torch.cat(emb_parts, dim=1))  # (B, 6)

        # --- Vector head (probabilities -> logits) ---
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
        vec_logits = torch.log(vec_probs + 1e-8)  # (B, 6)

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


def main() -> None:
    pre_dir = sys.argv[1] if len(sys.argv) > 1 else PREPROCESSED_DIR

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    meta = _read_json(os.path.join(pre_dir, "metadata.json"))
    feature_order = _get_feature_order(meta)
    label_set_ordered, label_to_index, index_to_label, num_classes = _get_label_info(meta)
    file_format, split_paths = _resolve_splits(pre_dir)

    cat_cols = list(meta.get("features", {}).get("categorical_features", []))
    num_cols = [c for c in feature_order if c not in set(cat_cols)]
    vocab_sizes = _get_vocab_sizes_from_meta2(meta, cat_cols) if cat_cols else {}

    required_cols = list(feature_order) + [TARGET_COL]
    split_columns: Dict[str, List[str]] = {}
    for split in ("train", "val", "test"):
        split_columns[split] = _check_required_columns(split, split_paths[split], file_format, required_cols)

    weights_used = WEIGHT_COL in set(split_columns["train"])

    def make_loader(split, shuffle=False, max_rows=None):
        wc = WEIGHT_COL if weights_used and WEIGHT_COL in set(split_columns[split]) else None
        ds = SplitBatchIterableDataset(
            path=split_paths[split], file_format=file_format,
            feature_order=feature_order, cat_cols=cat_cols, num_cols=num_cols,
            vocab_sizes=vocab_sizes, num_classes=num_classes, batch_size=BATCH_SIZE,
            available_columns=split_columns[split], max_rows=max_rows,
            shuffle_rows=shuffle, seed=SEED, weight_col=wc,
        )
        return DataLoader(ds, batch_size=None, num_workers=NUM_WORKERS)

    train_loader = make_loader("train", shuffle=SHUFFLE_TRAIN_ROWS_WITHIN_BATCH)
    val_loader = make_loader("val")
    test_loader = make_loader("test")

    # Base rate
    eps = 1e-12
    baseline_wc = WEIGHT_COL if weights_used and WEIGHT_COL in set(split_columns["val"]) else None
    counts = np.zeros((num_classes,), dtype=np.float64)
    for y, w in _stream_y_w(split_paths["train"], file_format, baseline_wc):
        counts += np.bincount(y, minlength=num_classes, weights=w)
    p = np.clip(counts / max(counts.sum(), eps), eps, 1.0)
    p /= p.sum()

    base_val_sum, base_val_w = 0.0, 0.0
    for y, w in _stream_y_w(split_paths["val"], file_format, baseline_wc):
        w = w.astype(np.float64)
        base_val_w += float(w.sum())
        base_val_sum += float((-np.log(p[y]) * w).sum())
    base_rate_val = base_val_sum / max(base_val_w, eps)

    base_test_sum, base_test_w = 0.0, 0.0
    for y, w in _stream_y_w(split_paths["test"], file_format, baseline_wc):
        w = w.astype(np.float64)
        base_test_w += float(w.sum())
        base_test_sum += float((-np.log(p[y]) * w).sum())
    base_rate_test = base_test_sum / max(base_test_w, eps)

    # Build model
    model = LogitHybridModel(
        cat_cols=cat_cols, num_numeric=len(num_cols), vocab_sizes=vocab_sizes,
        num_classes=num_classes, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print("=" * 72)
    print(f"LogitHybridModel | params={n_params:,} | device={DEVICE}")
    print(f"Base-rate val={base_rate_val:.6f}, test={base_rate_test:.6f}")
    print("=" * 72)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_m = evaluate(model, val_loader, criterion, DEVICE)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train: {train_m.loss:.6f} | val: {val_m.loss:.6f} | {dt:.1f}s"
        )
        history.append({
            "epoch": epoch, "train_loss": float(train_m.loss),
            "val_loss": float(val_m.loss), "seconds": float(dt),
        })

        if not math.isnan(val_m.loss) and val_m.loss < best_val_loss:
            best_val_loss = float(val_m.loss)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stop at epoch {epoch} (best: {best_val_loss:.6f} @ epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_ll = evaluate_logloss_only(model, val_loader, DEVICE)
    test_ll = evaluate_logloss_only(model, test_loader, DEVICE)

    ab = model.get_ab_weights()
    vw = model.get_learned_weights()

    # Save artifacts
    _ensure_dir(ARTIFACT_DIR)
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "model.pt"))
    shutil.copy(os.path.join(pre_dir, "metadata.json"), os.path.join(ARTIFACT_DIR, "metadata.json"))
    _write_json(os.path.join(ARTIFACT_DIR, "metrics.json"), {
        "best_epoch": best_epoch, "best_val_loss": best_val_loss,
        "val_logloss": val_ll, "test_logloss": test_ll,
        "num_params": n_params, "ab_weights": ab, "vector_weights": vw,
        "history": history, "timestamp": _now_str(),
    })

    # Print results
    print(f"\n{'='*72}")
    print(f"  Logit Hybrid: val={val_ll:.6f}, test={test_ll:.6f}, epoch={best_epoch}")
    print(f"  a_embedding={ab['a_embedding']:.4f} ({ab['a_embedding']*100:.1f}%)")
    print(f"  b_vector={ab['b_vector']:.4f} ({ab['b_vector']*100:.1f}%)")
    print(f"  Vector weights: {', '.join(f'{k}={v:.4f}' for k,v in vw.items())}")
    print(f"{'='*72}")

    # Comparison table with prior results
    prior = [
        ("A: Embeddings only",        174_920, 32, 1.432024, 1.468802),
        ("B: Vector weights (12p)",         12,  1, 1.431225, 1.468902),
        ("C: Hybrid (prob mix)",       174_676, 13, 1.431102, 1.471873),
    ]

    print(f"\n{'='*72}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*72}")
    print(f"  {'Model':<30s} {'Params':>8s} {'Epoch':>6s} {'Val LL':>10s} {'Test LL':>10s}")
    print(f"  {'-'*66}")
    for name, params, ep, vl, tl in prior:
        print(f"  {name:<30s} {params:>8,d} {ep:>6d} {vl:>10.6f} {tl:>10.6f}")
    print(f"  {'D: Hybrid (logit mix)':<30s} {n_params:>8,d} {best_epoch:>6d} {val_ll:>10.6f} {test_ll:>10.6f}")
    print(f"  {'-'*66}")
    print(f"  {'Base rate':<30s} {'--':>8s} {'--':>6s} {base_rate_val:>10.6f} {base_rate_test:>10.6f}")
    print(f"{'='*72}")
    print(f"\nArtifacts saved to: {ARTIFACT_DIR}/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
