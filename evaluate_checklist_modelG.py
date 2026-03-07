#!/usr/bin/env python3
"""
evaluate_checklist_modelG.py

Deployment-readiness evaluation checklist for Model G (or any compatible model artifact).

Checks:
  1. Overall val + test logloss vs base rate
  2. Paired bootstrap significance test (test set)
  3. Calibration: ECE (top-prob binning) + Brier score
  4. Segment breakdowns:
       - Time slices by YYYY-MM (if game_date present)
       - Logloss by batter/pitcher log_n decile (if columns present)
       - Logloss by pitch-mix entropy decile (if pitcher_pitch_rate_* cols present)

Usage:
  python evaluate_checklist_modelG.py \
      --preprocessed_dir preprocessed_modelG \
      --artifact_dir model_artifacts_modelG \
      [--num_bootstrap 1000] [--seed 1337] \
      [--out_json PATH] [--out_txt PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers to reconstruct model from artifacts without modifying train_model2
# ---------------------------------------------------------------------------

def _import_model_classes():
    """Import model classes from train_model2.py (no side-effects expected at import)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_model2 as tm
    return tm


def _choose_emb_dim_local(col: str, vocab_size: int,
                           batter_dim: int = 32, pitcher_dim: int = 32,
                           stadium_dim: int = 16, hand_dim: int = 4) -> int:
    if col == "batter_id":
        return batter_dim
    elif col == "pitcher_id":
        return pitcher_dim
    elif col == "stadium_id":
        return stadium_dim
    elif col in ("stand", "p_throws"):
        return hand_dim
    return max(4, min(50, vocab_size // 2))


def _detect_pt_statcast_cols(num_cols: List[str]) -> List[str]:
    """Mirror the detection logic in train_model2.py."""
    prefixes = (
        "pitcher_pitch_rate_",
        "pitcher_FF_", "pitcher_SI_", "pitcher_FC_", "pitcher_SL_",
        "pitcher_CU_", "pitcher_CH_", "pitcher_FS_", "pitcher_ST_",
        "pitcher_SV_", "pitcher_OTHER_",
        "batter_FF_", "batter_SI_", "batter_FC_", "batter_SL_",
        "batter_CU_", "batter_CH_", "batter_FS_", "batter_ST_",
        "batter_SV_", "batter_OTHER_",
    )
    return [c for c in num_cols if any(c.startswith(p) for p in prefixes)]


def _build_model(tm, train_cfg: dict, meta: dict,
                 cat_cols: List[str], num_cols: List[str],
                 vocab_sizes: dict, n_classes: int) -> torch.nn.Module:
    model_class = train_cfg.get("resolved", {}).get("model_class", "HybridModel")
    hidden_dims = train_cfg.get("HIDDEN_DIMS", [256, 128])
    dropout = train_cfg.get("DROPOUT", 0.2)

    pt_cols = _detect_pt_statcast_cols(num_cols)
    num_pt = len(pt_cols)

    if model_class == "ContextualGateLogitHybridModel" and num_pt > 0:
        model = tm.ContextualGateLogitHybridModel(
            cat_cols=cat_cols,
            num_numeric=len(num_cols),
            vocab_sizes=vocab_sizes,
            num_classes=n_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_pt_statcast_cols=num_pt,
        )
    elif model_class == "PerClassGateLogitHybridModel" and num_pt > 0:
        model = tm.PerClassGateLogitHybridModel(
            cat_cols=cat_cols,
            num_numeric=len(num_cols),
            vocab_sizes=vocab_sizes,
            num_classes=n_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_pt_statcast_cols=num_pt,
        )
    elif model_class == "StatcastLogitHybridModel" and num_pt > 0:
        model = tm.StatcastLogitHybridModel(
            cat_cols=cat_cols,
            num_numeric=len(num_cols),
            vocab_sizes=vocab_sizes,
            num_classes=n_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_pt_statcast_cols=num_pt,
        )
    else:
        model = tm.HybridModel(
            cat_cols=cat_cols,
            num_numeric=len(num_cols),
            vocab_sizes=vocab_sizes,
            num_classes=n_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_parquet_chunked(path: str, chunk_rows: int = 50_000) -> pd.DataFrame:
    """Load parquet in chunks (memory-safe), return full df."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    chunks = []
    for batch in pf.iter_batches(batch_size=chunk_rows):
        chunks.append(batch.to_pandas())
    return pd.concat(chunks, ignore_index=True)


def _load_split(split_dir: str, split_name: str) -> pd.DataFrame:
    parquet_path = os.path.join(split_dir, f"{split_name}.parquet")
    csv_path = os.path.join(split_dir, f"{split_name}.csv.gz")
    if os.path.exists(parquet_path):
        return _load_parquet_chunked(parquet_path)
    elif os.path.exists(csv_path):
        chunks = []
        for chunk in pd.read_csv(csv_path, compression="gzip", chunksize=50_000):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    else:
        raise FileNotFoundError(f"No split file found for {split_name} in {split_dir}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(model: torch.nn.Module, df: pd.DataFrame,
                   cat_cols: List[str], num_cols: List[str],
                   batch_size: int = 4096) -> np.ndarray:
    """Run model inference; returns (N, C) log-probability array."""
    model.eval()
    x_cat = torch.tensor(df[cat_cols].values.astype(np.int64), dtype=torch.long)
    x_num = torch.tensor(df[num_cols].values.astype(np.float32), dtype=torch.float32)

    all_logits = []
    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            logits = model(x_cat[i:i+batch_size], x_num[i:i+batch_size])
            all_logits.append(logits.cpu())
    logits = torch.cat(all_logits, dim=0)
    log_probs = F.log_softmax(logits, dim=1).numpy()
    return log_probs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _nll_per_example(log_probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-example negative log-likelihood."""
    return -log_probs[np.arange(len(y)), y]


def _logloss(nll: np.ndarray) -> float:
    return float(np.mean(nll))


def _accuracy(log_probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(log_probs, axis=1) == y))


def _base_rate_nll(train_y: np.ndarray, n_classes: int) -> Tuple[np.ndarray, float]:
    """Compute per-class base rate from train split; return (probs, log_probs implied)."""
    counts = np.bincount(train_y, minlength=n_classes).astype(float)
    probs = counts / counts.sum()
    return probs


def _base_nll_per_example(base_probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(base_probs[y] + 1e-12)


def _bootstrap_significance(model_nll: np.ndarray, base_nll: np.ndarray,
                              n_bootstrap: int = 1000, seed: int = 1337
                              ) -> dict:
    """
    Paired bootstrap on test set.
    delta_i = model_nll_i - base_nll_i  (negative = model better)
    Returns: mean_delta, 95% CI, mean % improvement vs base
    """
    rng = np.random.default_rng(seed)
    n = len(model_nll)
    delta = model_nll - base_nll
    observed_mean = float(np.mean(delta))

    boot_means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = np.mean(delta[idx])

    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    p_positive = float(np.mean(boot_means > 0))  # fraction where model is worse

    mean_base = float(np.mean(base_nll))
    pct_improvement = float(-observed_mean / mean_base * 100)  # positive = model better

    return {
        "mean_delta_nll": observed_mean,
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
        "p_model_worse": p_positive,
        "mean_pct_improvement_vs_base": pct_improvement,
        "significant_at_95": bool(ci_hi < 0),  # true if whole CI is negative (model better)
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error using top-prob binning."""
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(y)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (max_probs >= lo) & (max_probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = max_probs[mask].mean()
        ece_val += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece_val)


def _brier_score(probs: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """Multi-class Brier score."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Segment breakdowns
# ---------------------------------------------------------------------------

def _logloss_by_group(nll: np.ndarray, groups: pd.Series) -> List[dict]:
    result = []
    for g, idx in groups.groupby(groups).groups.items():
        idx = list(idx)
        result.append({
            "group": str(g),
            "n": len(idx),
            "logloss": float(np.mean(nll[idx])),
        })
    return sorted(result, key=lambda x: str(x["group"]))


def _decile_groups(series: pd.Series) -> pd.Series:
    try:
        return pd.qcut(series, q=10, labels=False, duplicates="drop")
    except Exception:
        return None


def _pitch_mix_entropy(df: pd.DataFrame) -> Optional[pd.Series]:
    rate_cols = [c for c in df.columns if c.startswith("pitcher_pitch_rate_")]
    if not rate_cols:
        return None
    rates = df[rate_cols].values.astype(float)
    eps = 1e-8
    entropy = -np.sum(rates * np.log(rates + eps), axis=1)
    return pd.Series(entropy, index=df.index)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Model G deployment-readiness checklist")
    parser.add_argument("--preprocessed_dir", default="preprocessed_modelG")
    parser.add_argument("--artifact_dir", default="model_artifacts_modelG")
    parser.add_argument("--num_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--out_txt", default=None)
    args = parser.parse_args()

    pre_dir = args.preprocessed_dir
    art_dir = args.artifact_dir

    if args.out_json is None:
        args.out_json = os.path.join(art_dir, "eval_checklist.json")
    if args.out_txt is None:
        args.out_txt = os.path.join(art_dir, "eval_checklist.txt")

    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load artifacts
    # ------------------------------------------------------------------
    meta_path = os.path.join(pre_dir, "metadata.json")
    train_cfg_path = os.path.join(art_dir, "train_config.json")
    model_path = os.path.join(art_dir, "model.pt")

    for p in [meta_path, train_cfg_path, model_path]:
        if not os.path.exists(p):
            print(f"ERROR: Required file not found: {p}")
            sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    with open(train_cfg_path) as f:
        train_cfg = json.load(f)

    feature_order = meta["features"]["feature_list_ordered"]
    cat_cols = meta["features"].get("categorical_features", [])
    num_cols = [c for c in feature_order if c not in set(cat_cols)]
    vocab_sizes = meta["features"].get("categorical_vocab_sizes", {})
    label_to_id = meta["labels"]["label_to_id"]
    id_to_label = {v: k for k, v in label_to_id.items()}
    n_classes = len(label_to_id)
    label_set = meta["labels"]["label_set_ordered"]

    print(f"[1/5] Loaded metadata: {len(feature_order)} features, {n_classes} classes")
    print(f"      cat_cols={cat_cols}, num_cols={len(num_cols)}")

    # Import model classes and build model
    tm = _import_model_classes()
    model = _build_model(tm, train_cfg, meta, cat_cols, num_cols, vocab_sizes, n_classes)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model_class = train_cfg.get("resolved", {}).get("model_class", "unknown")
    print(f"      Model class: {model_class}")

    # ------------------------------------------------------------------
    # 2. Load splits
    # ------------------------------------------------------------------
    print("[2/5] Loading splits...")
    train_df = _load_split(pre_dir, "train")
    val_df   = _load_split(pre_dir, "val")
    test_df  = _load_split(pre_dir, "test")
    print(f"      train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    train_y = train_df["y"].values.astype(int)
    val_y   = val_df["y"].values.astype(int)
    test_y  = test_df["y"].values.astype(int)

    # Base rate from TRAIN split
    base_probs = _base_rate_nll(train_y, n_classes)
    print(f"      Base-rate probs (from train): { {label_set[i]: round(float(base_probs[i]),4) for i in range(n_classes)} }")

    # ------------------------------------------------------------------
    # 3. Run inference
    # ------------------------------------------------------------------
    print("[3/5] Running inference...")
    val_logp  = _run_inference(model, val_df,  cat_cols, num_cols)
    test_logp = _run_inference(model, test_df, cat_cols, num_cols)

    val_probs  = np.exp(val_logp)
    test_probs = np.exp(test_logp)

    val_nll  = _nll_per_example(val_logp,  val_y)
    test_nll = _nll_per_example(test_logp, test_y)
    base_nll_val  = _base_nll_per_example(base_probs, val_y)
    base_nll_test = _base_nll_per_example(base_probs, test_y)

    base_ll_val  = float(np.mean(base_nll_val))
    base_ll_test = float(np.mean(base_nll_test))
    model_ll_val  = _logloss(val_nll)
    model_ll_test = _logloss(test_nll)

    val_acc  = _accuracy(val_logp,  val_y)
    test_acc = _accuracy(test_logp, test_y)

    # Per-class avg log-prob of true class
    def per_class_avg_logp(logp, y, n):
        out = {}
        for c in range(n):
            mask = y == c
            if mask.sum() > 0:
                out[label_set[c]] = float(np.mean(logp[mask, c]))
        return out

    val_per_class  = per_class_avg_logp(val_logp,  val_y,  n_classes)
    test_per_class = per_class_avg_logp(test_logp, test_y, n_classes)

    overall = {
        "val": {
            "logloss": model_ll_val,
            "base_rate_logloss": base_ll_val,
            "pct_improvement_vs_base": float(-(model_ll_val - base_ll_val) / base_ll_val * 100),
            "accuracy": val_acc,
            "per_class_avg_logp": val_per_class,
            "n": int(len(val_y)),
        },
        "test": {
            "logloss": model_ll_test,
            "base_rate_logloss": base_ll_test,
            "pct_improvement_vs_base": float(-(model_ll_test - base_ll_test) / base_ll_test * 100),
            "accuracy": test_acc,
            "per_class_avg_logp": test_per_class,
            "n": int(len(test_y)),
        },
    }

    # ------------------------------------------------------------------
    # 4. Bootstrap significance (test)
    # ------------------------------------------------------------------
    print(f"[4/5] Bootstrap significance (n={args.num_bootstrap}, seed={args.seed})...")
    bootstrap = _bootstrap_significance(test_nll, base_nll_test,
                                        n_bootstrap=args.num_bootstrap,
                                        seed=args.seed)

    # ------------------------------------------------------------------
    # 5. Calibration
    # ------------------------------------------------------------------
    print("[5/5] Calibration + segment breakdowns...")
    calibration = {
        "val": {
            "ece": _ece(val_probs, val_y),
            "brier": _brier_score(val_probs, val_y, n_classes),
        },
        "test": {
            "ece": _ece(test_probs, test_y),
            "brier": _brier_score(test_probs, test_y, n_classes),
        },
    }

    # ------------------------------------------------------------------
    # 6. Segment breakdowns (test; val where easy)
    # ------------------------------------------------------------------
    segments = {}

    # Time slices (YYYY-MM) from game_date
    for split_name, df, nll in [("val", val_df, val_nll), ("test", test_df, test_nll)]:
        if "game_date" in df.columns:
            try:
                months = pd.to_datetime(df["game_date"]).dt.to_period("M").astype(str)
                segments.setdefault("month_logloss", {})[split_name] = _logloss_by_group(nll, months)
            except Exception as e:
                print(f"      [WARN] month breakdown failed for {split_name}: {e}")

    # batter/pitcher log_n deciles (test only)
    for log_n_col in ["v1_batter_log_n", "v2_pitcher_log_n", "batter_log_n", "pitcher_log_n"]:
        if log_n_col in test_df.columns:
            deciles = _decile_groups(test_df[log_n_col])
            if deciles is not None:
                segments[f"{log_n_col}_decile_logloss"] = _logloss_by_group(test_nll, deciles)

    # Pitch-mix entropy deciles (test)
    entropy = _pitch_mix_entropy(test_df)
    if entropy is not None:
        deciles = _decile_groups(entropy)
        if deciles is not None:
            segments["pitch_mix_entropy_decile_logloss"] = _logloss_by_group(test_nll, deciles)

    # ------------------------------------------------------------------
    # Compile results
    # ------------------------------------------------------------------
    results = {
        "model_class": model_class,
        "preprocessed_dir": pre_dir,
        "artifact_dir": art_dir,
        "base_rate_probs_from_train": {label_set[i]: float(base_probs[i]) for i in range(n_classes)},
        "overall": overall,
        "bootstrap_significance_test": bootstrap,
        "calibration": calibration,
        "segments": segments,
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    os.makedirs(art_dir, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Build text report
    lines = []
    lines.append("=" * 66)
    lines.append("  MODEL G DEPLOYMENT CHECKLIST")
    lines.append(f"  Model:  {model_class}")
    lines.append(f"  Prepr:  {pre_dir}")
    lines.append(f"  Artif:  {art_dir}")
    lines.append("=" * 66)

    lines.append("\n--- OVERALL PERFORMANCE ---")
    for split in ("val", "test"):
        o = overall[split]
        sign = "+" if o["pct_improvement_vs_base"] >= 0 else ""
        lines.append(f"  {split.upper():5s}  logloss={o['logloss']:.5f}  "
                     f"base={o['base_rate_logloss']:.5f}  "
                     f"vs_base={sign}{o['pct_improvement_vs_base']:.2f}%  "
                     f"acc={o['accuracy']*100:.1f}%  n={o['n']:,}")

    lines.append("\n--- BOOTSTRAP SIGNIFICANCE (TEST, 95% CI) ---")
    b = bootstrap
    sig = "YES" if b["significant_at_95"] else "NO"
    lines.append(f"  Mean delta NLL  : {b['mean_delta_nll']:+.5f}  (negative = model better)")
    lines.append(f"  95% CI          : [{b['ci_95_lo']:+.5f}, {b['ci_95_hi']:+.5f}]")
    lines.append(f"  Significant@95% : {sig}  (P(model worse)={b['p_model_worse']:.3f})")
    lines.append(f"  Mean improvement: {b['mean_pct_improvement_vs_base']:+.3f}% vs base rate")

    lines.append("\n--- CALIBRATION ---")
    for split in ("val", "test"):
        c = calibration[split]
        lines.append(f"  {split.upper():5s}  ECE={c['ece']:.5f}  Brier={c['brier']:.5f}")

    if "month_logloss" in segments:
        lines.append("\n--- LOGLOSS BY MONTH ---")
        for split_name, rows in segments["month_logloss"].items():
            lines.append(f"  [{split_name.upper()}]")
            for r in rows:
                lines.append(f"    {r['group']}  n={r['n']:5d}  ll={r['logloss']:.5f}")

    for seg_key in segments:
        if seg_key == "month_logloss":
            continue
        friendly = seg_key.replace("_", " ")
        lines.append(f"\n--- {friendly.upper()} (TEST) ---")
        rows = segments[seg_key]
        if isinstance(rows, list):
            for r in rows:
                lines.append(f"  decile={r['group']}  n={r['n']:5d}  ll={r['logloss']:.5f}")

    lines.append(f"\n  [Elapsed: {results['elapsed_seconds']}s]")
    lines.append("=" * 66)

    report = "\n".join(lines)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    print(f"\n[DONE] JSON: {args.out_json}")
    print(f"       TXT:  {args.out_txt}")


if __name__ == "__main__":
    main()
