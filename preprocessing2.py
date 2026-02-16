#!/usr/bin/env python3
"""
preprocessing2.py

Three-pass preprocessing pipeline:

PASS 0: Fit train-only artifacts
  - Categoricals vocabularies (train only)
  - Batter/pitcher tendency tables (train only)
  - League fallback JSON for tendency columns

PASS 1: Materialize pitch-level rows with labels + split assignments (optional debug)

PASS 2: Fit/train-only tendencies (v1) + (v2) and write frozen tables/artifacts

PASS 3: Build PA-level split CSVs by aggregating first pitch of PA (or provided PA rows),
        join tendencies, encode categoricals, write train/val/test split files.

Strictly no leakage: anything fit is trained on TRAIN split only, with split logic based on game_date.
"""

import argparse
import datetime as dt
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =========================
# Config
# =========================

READ_CHUNK_ROWS = 1_000_000

OUTPUT_DIR = "preprocessed"
INPUT_CSV = "statcast_pitches.csv"  # override via CLI
OUTPUT_FORMAT = "parquet"  # or "csv"

# Split strategy: train <= TRAIN_END_DATE, val <= VAL_END_DATE, else test
TRAIN_END_DATE = "2025-06-30"
VAL_END_DATE = "2025-09-27"

# Tendency smoothing / shrink parameters
# =========================
# PA-Ending Tendency Configuration
# =========================

# Pitch type remapping (before bucketing)
PITCH_TYPE_REMAP = {
    # Fastballs (keep as-is)
    "FF": "FF",
    "SI": "SI",
    "FC": "FC",
    # Breaking balls
    "SL": "SL",
    "CU": "CU",
    "KC": "CU",   # Knuckle-curve -> Curveball
    "SV": "SV",
    "ST": "ST",
    # Offspeed
    "CH": "CH",
    "FS": "FS",
    # Rare -> Other
    "KN": "OTHER",  # Knuckleball -> Other
    "EP": "OTHER",  # Eephus -> Other
    "SC": "OTHER",  # Screwball -> Other
    "FA": "FF",     # Generic fastball -> 4-seam
    "PO": "OTHER",  # Pitchout -> Other
    "IN": "OTHER",  # Intentional ball -> Other
}

# Pitch types (10 total after remapping)
PITCH_TYPES = ["FF", "SI", "FC", "SL", "CU", "CH", "FS", "ST", "SV", "OTHER"]
PITCH_TYPE_BINS = list(PITCH_TYPES)  # alias for backward compat
PITCH_TYPE_OTHER_BIN = "OTHER"  # fallback for unknown pitch types

# Zone bins (raw Statcast zones)
ZONE_BINS = list(range(0, 15))

# Zone tiers (4)
ZONE_TIERS = ["HEART", "SHADOW", "CHASE", "WASTE"]

# All buckets: pitch_type x zone (40 total)
ALL_BUCKETS = [
    f"{pt}|{zone}"
    for pt in PITCH_TYPES
    for zone in ZONE_TIERS
]

# Categorical features (5)
CATEGORICAL_COLS = ["batter_id", "pitcher_id", "stadium_id", "stand", "p_throws"]

# 6-class outcomes
OUTCOMES = ["K", "BIPO", "BB", "1B", "XBH", "HR"]

# Smoothing parameters
USE_SMOOTHING = False           # NO smoothing - raw rates only
TENDENCY_ALPHA = 0.0            # Alpha = 0 means no shrinkage toward prior
TENDENCY_DIRICHLET_ALPHA = 2.0  # alias for backward compat (old bucket system)
MEAN_SHRINK_K = 50.0

# Feature toggles for ablation experiments
ENABLE_SIX_VECTORS = True       # Six-vector interpretable features (42)
ENABLE_EMBEDDINGS = True        # Enabled for hybrid model experiment
ENABLE_BUCKET_FEATURES = False  # Set to False to disable bucket-weighted tendency features
ENABLE_PARK_FACTORS = False     # Set to False to disable park factor features

# PA-level feature columns
PA_ID_COL = "pa_id"
BATTER_ID_COL = "batter_id"
PITCHER_ID_COL = "pitcher_id"
STADIUM_ID_COL = "stadium_id"
PITCH_TYPE_COL = "pitch_type"
ZONE_COL = "zone"
GAME_DATE_COL = "game_date"
OUTCOME_COL = "pa_outcome"
DESCRIPTION_COL = "description"

# Required columns for processing pitch rows
REQUIRED_PITCH_COLUMNS = [GAME_DATE_COL, OUTCOME_COL, BATTER_ID_COL, PITCHER_ID_COL, STADIUM_ID_COL, PITCH_TYPE_COL, ZONE_COL, "stand", "p_throws"]


# Output columns (pitch-level)
PITCH_LEVEL_BASE_COLS = [
    GAME_DATE_COL,
    BATTER_ID_COL,
    PITCHER_ID_COL,
    "pitcher_fatigue",
    "pitch_type_bin",
    "zone_bin",
]

# Output columns (PA-level)
BATTER_HAS_STATS_COL = "batter_has_stats"
PITCHER_HAS_STATS_COL = "pitcher_has_stats"


# =========================
# Dataclasses
# =========================

@dataclass
class SplitInfo:
    train_end_date: str
    val_end_date: str


@dataclass
class FitStats:
    split_info: SplitInfo
    vocabs: Dict[str, Dict[str, int]]
    tendency_spec: "TendencySpec"
    tendency_paths: Dict[str, str]
    output_columns: List[str]


@dataclass
class TendencySpec:
    pitch_type_bins: List[str]
    pitch_type_other_bin: str
    zone_bins: List[int]
    zone_tiers: List[str]
    outcomes: List[str]
    dirichlet_alpha: float
    mean_shrink_k: float
    batter_feature_cols: List[str]
    pitcher_feature_cols: List[str]


# =========================
# Utilities
# =========================

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _coerce_int_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).astype("int64")


def _assign_split(game_dates: pd.Series, train_end: pd.Timestamp, val_end: pd.Timestamp) -> pd.Series:
    # game_dates should be normalized to dates (no time)
    out = pd.Series(["test"] * len(game_dates), index=game_dates.index, dtype="string")
    out.loc[game_dates <= val_end] = "val"
    out.loc[game_dates <= train_end] = "train"
    return out


def _write_table(df: pd.DataFrame, out_path_no_ext: str, output_format: str) -> str:
    if output_format == "csv":
        out_path = out_path_no_ext + ".csv.gz"
        df.to_csv(out_path, index=False, compression="gzip")
        return out_path
    if output_format == "parquet":
        out_path = out_path_no_ext + ".parquet"
        df.to_parquet(out_path, index=False)
        return out_path
    raise ValueError(f"Unknown output_format={output_format}")


def _read_table(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=True)
    if path.endswith(".csv"):
        return pd.read_csv(path, low_memory=True)
    raise ValueError(f"Unsupported table extension: {path}")


def _encode_categorical(s: pd.Series, vocab: Dict[str, int], unk_id: int = 1) -> pd.Series:
    # 0 reserved for PAD; 1 reserved for UNK
    # vocab contains entries for observed tokens starting at 2
    return s.astype("string").fillna("").map(vocab).fillna(unk_id).astype("int64")


# =========================
# Label mapping (example)
# =========================

SIX_LABELS = ["K", "BIPO", "BB", "1B", "XBH", "HR"]

PA_OUTCOME_TO_SIX = {
    # Strikeouts
    "strikeout": "K",
    "strikeout_double_play": "K",

    # Ball in play outs
    "field_out": "BIPO",
    "force_out": "BIPO",
    "grounded_into_double_play": "BIPO",
    "double_play": "BIPO",
    "fielders_choice": "BIPO",
    "fielders_choice_out": "BIPO",
    "sac_fly": "BIPO",
    "sac_bunt": "BIPO",
    "sac_fly_double_play": "BIPO",
    "sac_bunt_double_play": "BIPO",
    "triple_play": "BIPO",
    "field_error": "BIPO",
    "catcher_interf": "BIPO",

    # Walks/HBP
    "walk": "BB",
    "hit_by_pitch": "BB",
    "intent_walk": "BB",

    # Singles
    "single": "1B",

    # Extra base hits (2B + 3B combined)
    "double": "XBH",
    "triple": "XBH",

    # Home runs
    "home_run": "HR",
}


def _map_to_six_labels(pa_outcome: pd.Series) -> pd.Series:
    return pa_outcome.astype("string").map(PA_OUTCOME_TO_SIX)


# =========================
# Pitch binning helpers
# =========================

def _pitch_type_bin(pitch_type: pd.Series) -> pd.Series:
    """
    Bin pitch types, remapping rare/similar types.
    KC -> CU, KN -> OTHER, etc.
    """
    pt = pitch_type.astype("string").fillna("OTHER").str.upper()
    mapped = pt.map(PITCH_TYPE_REMAP)
    mapped = mapped.fillna("OTHER")
    valid_types = set(PITCH_TYPE_BINS)
    mapped = mapped.where(mapped.isin(valid_types), "OTHER")
    return mapped


def _zone_bin(zone: pd.Series) -> pd.Series:
    z = pd.to_numeric(zone, errors="coerce").fillna(0).astype("int64")
    z = z.where(z.isin(ZONE_BINS), 0)
    return z


def _inplay_indicator_from_description(desc: pd.Series) -> pd.Series:
    d = desc.astype("string").fillna("")
    inplay = d.isin(["hit_into_play", "hit_into_play_no_out", "hit_into_play_score"]) | d.str.contains("hit_into_play", regex=False)
    return inplay.astype("int64")


# =========================
# Pitch characteristic helpers: zone tier, velocity tier, pitch-bin key
# =========================

def _zone_tier_from_zone(zone_int: pd.Series) -> pd.Series:
    """
    Map Statcast zone (1-14) to tier.

    Statcast zones:
    - 1-9: Strike zone grid (1=upper-left to 9=lower-right, 5=center)
    - 11-14: Chase zones (just outside)
    - 10+: Waste (way outside)

    Mapping:
    - HEART: Zone 5 (dead center)
    - SHADOW: Zones 1,2,3,4,6,7,8,9 (rest of strike zone)
    - CHASE: Zones 11,12,13,14 (just outside)
    - WASTE: Zone 10 or missing (way outside / unknown)
    """
    z = pd.to_numeric(zone_int, errors="coerce").fillna(0).astype("int64")
    out = pd.Series(["WASTE"] * len(z), index=z.index, dtype="string")

    out.loc[z == 5] = "HEART"
    out.loc[z.isin([1, 2, 3, 4, 6, 7, 8, 9])] = "SHADOW"
    out.loc[z.isin([11, 12, 13, 14])] = "CHASE"
    # Everything else stays WASTE

    return out


def _sanitize_pitch_bin_key(k: str) -> str:
    # Keep deterministic but filesystem/column-name safe-ish
    return (k or "").replace("|", "_").replace(" ", "")


def _assign_pitch_bucket(
    pitch_type: pd.Series,
    zone: pd.Series,
) -> pd.Series:
    """
    Assign each pitch to one of 40 buckets (pitch_type x zone).

    No velocity tiers.
    """
    # Remap pitch types
    pt = pitch_type.astype("string").str.upper().fillna("OTHER")
    pt_mapped = pt.map(PITCH_TYPE_REMAP).fillna("OTHER")

    # Get zone tier
    zone_tier = _zone_tier_from_zone(zone)

    # Combine
    bucket = pt_mapped + "|" + zone_tier

    return bucket


# =========================
# PASS 0: Fit vocabs + v1 tendencies
# =========================

def pass0_fit_vocabs_and_tendencies(input_csv: str, split_info: SplitInfo) -> Tuple[Dict[str, Dict[str, int]], TendencySpec]:
    print("[PASS 0] Fitting train-only vocabs and v1 tendency tables...")

    train_end = pd.to_datetime(split_info.train_end_date)
    val_end = pd.to_datetime(split_info.val_end_date)

    # Vocabs
    vocabs = {
        "batter_id": {},
        "pitcher_id": {},
        "stadium_id": {},
        "stand": {"__PAD__": 0, "__UNK__": 1, "L": 2, "R": 3},
        "p_throws": {"__PAD__": 0, "__UNK__": 1, "L": 2, "R": 3},
        "pitch_type_bin": {},
        "zone_bin": {},
    }


    # Track tokens on train only
    token_counts = {k: Counter() for k in vocabs.keys()}

    # v1 accumulators: mean fatigue and simple distributions
    batter_zone_counts = defaultdict(lambda: np.zeros(len(ZONE_BINS), dtype=np.float64))
    pitcher_pt_counts = defaultdict(lambda: np.zeros(len(PITCH_TYPE_BINS), dtype=np.float64))
    batter_N = defaultdict(float)
    pitcher_N = defaultdict(float)

    rows_read = 0

    for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=READ_CHUNK_ROWS, low_memory=True)):
        rows_read += len(chunk)
        missing = [c for c in REQUIRED_PITCH_COLUMNS if c not in chunk.columns]
        _require(not missing, f"Missing required columns in input: {missing}")

        chunk["game_date"] = pd.to_datetime(chunk["game_date"], errors="coerce").dt.normalize()
        y_str = _map_to_six_labels(chunk["pa_outcome"])
        good = chunk["game_date"].notna() & y_str.notna()
        df = chunk.loc[good].copy()
        if df.empty:
            continue

        split = _assign_split(df["game_date"], train_end, val_end)
        df_train = df.loc[split == "train"].copy()
        if df_train.empty:
            continue

        # Derived bins
        df_train["pitch_type_bin"] = _pitch_type_bin(df_train["pitch_type"])
        df_train["zone_bin"] = _zone_bin(df_train["zone"])
        # Stadium id (stable categorical)
        if STADIUM_ID_COL not in df_train.columns:
            for c in ["park", "park_id", "stadium", "venue_name", "park_proxy_home_team"]:
                if c in df_train.columns:
                    df_train[STADIUM_ID_COL] = df_train[c]
                    break
        if STADIUM_ID_COL not in df_train.columns:
            df_train[STADIUM_ID_COL] = "__UNK__"
        df_train[STADIUM_ID_COL] = df_train[STADIUM_ID_COL].astype("string").fillna("__UNK__").replace("", "__UNK__")


        # Update vocab counts
        token_counts["batter_id"].update(df_train["batter_id"].astype("string").tolist())
        token_counts["pitcher_id"].update(df_train["pitcher_id"].astype("string").tolist())
        token_counts["pitch_type_bin"].update(df_train["pitch_type_bin"].astype("string").tolist())
        token_counts["zone_bin"].update(df_train["zone_bin"].astype("string").tolist())
        token_counts["stadium_id"].update(df_train[STADIUM_ID_COL].astype("string").tolist())


        # Update v1 tendencies
        # batter_zone_counts: P(zone_bin | batter)
        g_bz = df_train.groupby("batter_id")["zone_bin"].value_counts()
        for (bid, z), cnt in g_bz.items():
            z_idx = int(z)
            if z_idx < 0 or z_idx >= len(ZONE_BINS):
                continue
            batter_zone_counts[int(bid)][z_idx] += float(cnt)
            batter_N[int(bid)] += float(cnt)

        # pitcher_pt_counts: P(pitch_type_bin | pitcher)
        pt_to_idx = {pt: j for j, pt in enumerate(PITCH_TYPE_BINS)}
        other_idx = pt_to_idx.get("OTHER", len(PITCH_TYPE_BINS) - 1)
        g_pp = df_train.groupby("pitcher_id")["pitch_type_bin"].value_counts()
        for (pid, pt), cnt in g_pp.items():
            j = pt_to_idx.get(str(pt), other_idx)
            pitcher_pt_counts[int(pid)][j] += float(cnt)
            pitcher_N[int(pid)] += float(cnt)

        if (i + 1) % 10 == 0:
            print(f"  - [PASS 0] processed {rows_read:,} pitch rows...")

    # Build vocabs with PAD=0, UNK=1, then tokens by frequency
    def _build_vocab(cntr: Counter) -> Dict[str, int]:
        items = [t for t, _ in cntr.most_common()]
        vocab = {"__PAD__": 0, "__UNK__": 1}
        for j, t in enumerate(items, start=2):
            vocab[str(t)] = j
        return vocab

    for k, cntr in token_counts.items():
        vocabs[k] = _build_vocab(cntr)

    # Build v1 tables with Dirichlet smoothing
    alpha = TENDENCY_DIRICHLET_ALPHA

    # Batter zone distribution
    league_zone_counts = np.zeros(len(ZONE_BINS), dtype=np.float64)
    for bid, v in batter_zone_counts.items():
        league_zone_counts += v
    league_zone_probs = (league_zone_counts + alpha) / (league_zone_counts.sum() + alpha * len(ZONE_BINS))

    batter_rows = []
    for bid, counts in batter_zone_counts.items():
        denom = counts.sum() + alpha
        probs = (counts + alpha * league_zone_probs) / denom
        probs = probs / probs.sum()
        row = {"batter_id": int(bid), "batter_N": float(batter_N.get(bid, 0.0)), "batter_logN": float(np.log1p(batter_N.get(bid, 0.0)))}
        for z in ZONE_BINS:
            row[f"batter_p_zone_{z}"] = float(probs[int(z)])
        batter_rows.append(row)
    batter_df = pd.DataFrame(batter_rows)

    # Pitcher pitch type distribution
    league_pt_counts = np.zeros(len(PITCH_TYPE_BINS), dtype=np.float64)
    for pid, v in pitcher_pt_counts.items():
        league_pt_counts += v
    league_pt_probs = (league_pt_counts + alpha) / (league_pt_counts.sum() + alpha * len(PITCH_TYPE_BINS))
    league_pt_probs = league_pt_probs / league_pt_probs.sum()

    pitcher_rows = []
    for pid, counts in pitcher_pt_counts.items():
        denom = counts.sum() + alpha
        probs = (counts + alpha * league_pt_probs) / denom
        probs = probs / probs.sum()
        row = {"pitcher_id": int(pid), "pitcher_N": float(pitcher_N.get(pid, 0.0)), "pitcher_logN": float(np.log1p(pitcher_N.get(pid, 0.0)))}
        for j, pt in enumerate(PITCH_TYPE_BINS):
            row[f"pitcher_p_pt_{pt}"] = float(probs[j])
        pitcher_rows.append(row)
    pitcher_df = pd.DataFrame(pitcher_rows)

    # Feature columns for new PA-ending tendency system
    batter_feature_cols = [
        "batter_K_expected", "batter_BIPO_expected", "batter_BB_expected",
        "batter_1B_expected", "batter_XBH_expected", "batter_HR_expected",
        "batter_log_n",
    ]
    pitcher_feature_cols = [
        "pitcher_K_expected", "pitcher_BIPO_expected", "pitcher_BB_expected",
        "pitcher_1B_expected", "pitcher_XBH_expected", "pitcher_HR_expected",
        "pitcher_log_n",
    ]

    tendency_spec = TendencySpec(
        pitch_type_bins=PITCH_TYPE_BINS,
        pitch_type_other_bin=PITCH_TYPE_OTHER_BIN,
        zone_bins=ZONE_BINS,
        zone_tiers=ZONE_TIERS,
        outcomes=OUTCOMES,
        dirichlet_alpha=alpha,
        mean_shrink_k=MEAN_SHRINK_K,
        batter_feature_cols=batter_feature_cols,
        pitcher_feature_cols=pitcher_feature_cols,
    )

    return vocabs, tendency_spec


# =========================
# PASS 2: Fit PA-ending tendencies (NEW SYSTEM)
# =========================

def _hierarchical_smooth(
    bucket_counts: np.ndarray,
    bucket_n: int,
    overall_counts: np.ndarray,
    overall_n: int,
    league_rates: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Smooth bucket rates: bucket -> player overall -> league.

    NO platoon in hierarchy (platoon is handled separately).
    """
    # League prior
    league_prior = league_rates / (league_rates.sum() + 1e-12)

    # Player overall (smoothed toward league)
    if overall_n > 0:
        overall_rate = (overall_counts + alpha * league_prior) / (overall_n + alpha)
    else:
        overall_rate = league_prior.copy()
    overall_rate = overall_rate / (overall_rate.sum() + 1e-12)

    # Bucket (smoothed toward player overall)
    if bucket_n > 0:
        bucket_rate = (bucket_counts + alpha * overall_rate) / (bucket_n + alpha)
    else:
        bucket_rate = overall_rate.copy()
    bucket_rate = bucket_rate / (bucket_rate.sum() + 1e-12)

    return bucket_rate


def pass2_fit_pa_ending_tendencies(
    input_csv: str,
    split_info: SplitInfo,
    alpha: float = TENDENCY_ALPHA,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Compute PA-outcome rates by pitch bucket from TRAIN split only.

    40 buckets (pitch_type x zone). No platoon dimension.
    Hierarchical smoothing: bucket -> player overall -> league.

    Returns:
        batter_df: batter_id -> outcome rates by bucket + overall stats
        pitcher_df: pitcher_id -> outcome rates by bucket + usage rates
        league_fallback: league-level rates and metadata for inference
    """
    print("[PASS 2] Fitting PA-ending tendencies from TRAIN split only...")

    train_end = pd.to_datetime(split_info.train_end_date)
    val_end = pd.to_datetime(split_info.val_end_date)

    n_outcomes = len(OUTCOMES)
    outcome_to_idx = {o: i for i, o in enumerate(OUTCOMES)}

    # 40 buckets (pitch_type x zone)
    all_buckets = list(ALL_BUCKETS)
    n_buckets = len(all_buckets)

    # --- Accumulators ---
    batter_counts = defaultdict(lambda: {
        "by_bucket": defaultdict(lambda: np.zeros(n_outcomes, dtype=np.float64)),
        "n_by_bucket": defaultdict(int),
        "overall": np.zeros(n_outcomes, dtype=np.float64),
        "n_overall": 0,
    })
    pitcher_counts = defaultdict(lambda: {
        "by_bucket": defaultdict(lambda: np.zeros(n_outcomes, dtype=np.float64)),
        "n_by_bucket": defaultdict(int),
        "overall": np.zeros(n_outcomes, dtype=np.float64),
        "n_overall": 0,
        "usage_by_bucket": defaultdict(int),
        "usage_total": 0,
    })
    league_counts = {
        "by_bucket": defaultdict(lambda: np.zeros(n_outcomes, dtype=np.float64)),
        "n_by_bucket": defaultdict(int),
        "overall": np.zeros(n_outcomes, dtype=np.float64),
        "n_overall": 0,
        "usage_by_bucket": defaultdict(int),
        "usage_total": 0,
    }

    rows_read = 0
    train_pitch_rows = 0
    pa_ending_rows = 0

    for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=READ_CHUNK_ROWS, low_memory=True)):
        rows_read += len(chunk)
        missing = [c for c in REQUIRED_PITCH_COLUMNS if c not in chunk.columns]
        _require(not missing, f"Missing required columns in input: {missing}")

        chunk["game_date"] = pd.to_datetime(chunk["game_date"], errors="coerce").dt.normalize()
        y_str = _map_to_six_labels(chunk["pa_outcome"])
        good = chunk["game_date"].notna() & y_str.notna()
        df = chunk.loc[good].copy()
        if df.empty:
            continue

        split = _assign_split(df["game_date"], train_end, val_end)
        df = df.loc[split == "train"].copy()
        if df.empty:
            continue

        train_pitch_rows += len(df)

        # IDs
        batter_id = _coerce_int_series(df.get("batter_id", pd.Series([0] * len(df), index=df.index))).astype("int64")
        pitcher_id = _coerce_int_series(df.get("pitcher_id", pd.Series([0] * len(df), index=df.index))).astype("int64")

        ok_ids = (batter_id > 0) & (pitcher_id > 0)
        if not bool(ok_ids.any()):
            continue
        df = df.loc[ok_ids].copy()
        batter_id = batter_id.loc[ok_ids]
        pitcher_id = pitcher_id.loc[ok_ids]

        n = len(df)

        # Compute bucket (pitch_type x zone, no velocity)
        bucket = _assign_pitch_bucket(df["pitch_type"], df["zone"])

        # Map outcomes
        pa_outcome = df["pa_outcome"].astype("string")
        six_outcome = _map_to_six_labels(pa_outcome)

        # Detect PA-ending pitches
        if "is_last_pitch_of_pa" in df.columns:
            is_last = pd.to_numeric(df["is_last_pitch_of_pa"], errors="coerce").fillna(0).astype("int64")
        elif "pitch_number_in_pa" in df.columns and "pa_id" in df.columns:
            max_pitch = df.groupby("pa_id")["pitch_number_in_pa"].transform("max")
            is_last = (df["pitch_number_in_pa"] == max_pitch).astype("int64")
        else:
            is_last = pd.Series([0] * n, index=df.index, dtype="int64")

        # 1) Accumulate pitcher usage (ALL pitches, 40 buckets)
        usage_df = pd.DataFrame({"pid": pitcher_id.values, "bucket": bucket.values})
        for (pid_val, b_val), cnt in usage_df.groupby(["pid", "bucket"]).size().items():
            pitcher_counts[int(pid_val)]["usage_by_bucket"][b_val] += int(cnt)
            pitcher_counts[int(pid_val)]["usage_total"] += int(cnt)
        for b_val, cnt in usage_df["bucket"].value_counts().items():
            league_counts["usage_by_bucket"][b_val] += int(cnt)
            league_counts["usage_total"] += int(cnt)

        # 2) Accumulate PA-ending outcomes
        pa_ending_mask = is_last == 1
        outcome_idx = six_outcome.map(outcome_to_idx)
        valid_pa = pa_ending_mask & outcome_idx.notna()
        if valid_pa.any():
            df_pa = pd.DataFrame({
                "bid": batter_id.loc[valid_pa].values,
                "pid": pitcher_id.loc[valid_pa].values,
                "oi": outcome_idx.loc[valid_pa].astype("int64").values,
                "bucket": bucket.loc[valid_pa].values,
            })
            pa_ending_rows += len(df_pa)

            # --- League ---
            for oi_val, cnt in df_pa["oi"].value_counts().items():
                league_counts["overall"][int(oi_val)] += float(cnt)
                league_counts["n_overall"] += int(cnt)
            for (b_val, oi_val), cnt in df_pa.groupby(["bucket", "oi"]).size().items():
                league_counts["by_bucket"][b_val][int(oi_val)] += float(cnt)
                league_counts["n_by_bucket"][b_val] += int(cnt)

            # --- Batter ---
            for (bid_val, oi_val), cnt in df_pa.groupby(["bid", "oi"]).size().items():
                batter_counts[int(bid_val)]["overall"][int(oi_val)] += float(cnt)
                batter_counts[int(bid_val)]["n_overall"] += int(cnt)
            for (bid_val, b_val, oi_val), cnt in df_pa.groupby(["bid", "bucket", "oi"]).size().items():
                batter_counts[int(bid_val)]["by_bucket"][b_val][int(oi_val)] += float(cnt)
                batter_counts[int(bid_val)]["n_by_bucket"][b_val] += int(cnt)

            # --- Pitcher ---
            for (pid_val, oi_val), cnt in df_pa.groupby(["pid", "oi"]).size().items():
                pitcher_counts[int(pid_val)]["overall"][int(oi_val)] += float(cnt)
                pitcher_counts[int(pid_val)]["n_overall"] += int(cnt)
            for (pid_val, b_val, oi_val), cnt in df_pa.groupby(["pid", "bucket", "oi"]).size().items():
                pitcher_counts[int(pid_val)]["by_bucket"][b_val][int(oi_val)] += float(cnt)
                pitcher_counts[int(pid_val)]["n_by_bucket"][b_val] += int(cnt)

        if (i + 1) % 10 == 0:
            print(f"  - [PASS 2] processed {rows_read:,} pitch rows... (train: {train_pitch_rows:,}, PA-ending: {pa_ending_rows:,})")

    _require(train_pitch_rows > 0, "No TRAIN pitch rows found (after filtering).")
    _require(pa_ending_rows > 0, "No PA-ending pitch rows found in TRAIN split.")

    # Compute league rates
    league_overall_n = float(league_counts["overall"].sum())
    league_rates = league_counts["overall"] / max(league_overall_n, 1.0)
    league_rates = league_rates / league_rates.sum()

    # League usage rates per bucket (40 buckets)
    league_usage_total = float(league_counts["usage_total"])
    league_usage_rates = {}
    for b in all_buckets:
        league_usage_rates[b] = float(league_counts["usage_by_bucket"].get(b, 0)) / max(league_usage_total, 1.0)

    print(f"[PASS 2] League outcome rates: {dict(zip(OUTCOMES, league_rates.tolist()))}")
    print(f"[PASS 2] Total PA-ending pitches in train: {pa_ending_rows:,}")

    # Build batter DataFrame
    print(f"[PASS 2] Building batter tendency table for {len(batter_counts):,} batters...")
    batter_rows = []
    for bid, data in batter_counts.items():
        row = {"batter_id": int(bid)}
        overall_n = data["n_overall"]
        row["batter_n_pa"] = overall_n

        # Overall rates (smoothed toward league)
        overall_rate = _hierarchical_smooth(
            data["overall"], overall_n,
            data["overall"], overall_n,
            league_rates, alpha
        )
        for oi, outcome in enumerate(OUTCOMES):
            row[f"batter_{outcome}_rate"] = float(overall_rate[oi])

        # Per-bucket rates (40 buckets)
        for bucket in all_buckets:
            bucket_safe = _sanitize_pitch_bin_key(bucket)
            b_counts = data["by_bucket"].get(bucket, np.zeros(n_outcomes, dtype=np.float64))
            b_n = int(b_counts.sum())

            rates = _hierarchical_smooth(
                b_counts, b_n,
                data["overall"], overall_n,
                league_rates, alpha
            )
            for oi, outcome in enumerate(OUTCOMES):
                row[f"batter_{outcome}_{bucket_safe}"] = float(rates[oi])
            row[f"batter_n_{bucket_safe}"] = b_n

        batter_rows.append(row)

    batter_df = pd.DataFrame(batter_rows)

    # Build pitcher DataFrame
    print(f"[PASS 2] Building pitcher tendency table for {len(pitcher_counts):,} pitchers...")
    pitcher_rows = []
    for pid, data in pitcher_counts.items():
        row = {"pitcher_id": int(pid)}
        overall_n = data["n_overall"]
        row["pitcher_n_pa"] = overall_n

        # Overall rates
        overall_rate = _hierarchical_smooth(
            data["overall"], overall_n,
            data["overall"], overall_n,
            league_rates, alpha
        )
        for oi, outcome in enumerate(OUTCOMES):
            row[f"pitcher_{outcome}_rate"] = float(overall_rate[oi])

        # Per-bucket rates + usage (40 buckets)
        usage_total = float(data["usage_total"])
        for bucket in all_buckets:
            bucket_safe = _sanitize_pitch_bin_key(bucket)
            b_counts = data["by_bucket"].get(bucket, np.zeros(n_outcomes, dtype=np.float64))
            b_n = int(b_counts.sum())

            rates = _hierarchical_smooth(
                b_counts, b_n,
                data["overall"], overall_n,
                league_rates, alpha
            )
            for oi, outcome in enumerate(OUTCOMES):
                row[f"pitcher_{outcome}_{bucket_safe}"] = float(rates[oi])

            usage_rate = float(data["usage_by_bucket"].get(bucket, 0)) / max(usage_total, 1.0)
            row[f"pitcher_usage_{bucket_safe}"] = float(usage_rate)
            row[f"pitcher_n_{bucket_safe}"] = b_n

        pitcher_rows.append(row)

    pitcher_df = pd.DataFrame(pitcher_rows)

    # Build league fallback
    league_fallback = {
        "outcomes": OUTCOMES,
        "buckets": all_buckets,
        "pitch_types": PITCH_TYPES,
        "zone_tiers": ZONE_TIERS,
        "pitch_type_remap": PITCH_TYPE_REMAP,
        "league_rates": league_rates.tolist(),
        "league_usage": {bucket: float(league_usage_rates.get(bucket, 0)) for bucket in all_buckets},
        "smoothing_alpha": alpha,
        "n_pa_ending_train": pa_ending_rows,
    }

    print("[PASS 2] PA-ending tendency fitting complete.")
    return batter_df, pitcher_df, league_fallback


# =========================
# PASS 2b: Six-vector statistics (NEW SYSTEM)
# =========================

def pass2_compute_six_vector_stats(
    input_csv: str,
    split_info: SplitInfo,
    alpha: float = TENDENCY_ALPHA,
) -> Dict:
    """
    Compute statistics needed for the six-vector model from TRAIN split only.

    Returns dict with:
        - batter_overall: batter_id -> overall outcome rates
        - pitcher_overall: pitcher_id -> overall outcome rates
        - stadium_rates: stadium_id -> outcome rates
        - batter_platoon: batter_id -> outcome rates vs L and R pitchers
        - pitcher_platoon: pitcher_id -> outcome rates vs L and R batters
        - batter_pitch_type: batter_id -> outcome rates by pitch type
        - pitcher_mix: pitcher_id -> pitch type usage rates
        - league_rates: league-wide outcome rates (numpy array)
    """
    print("[PASS 2b] Computing six-vector statistics from TRAIN split only...")

    train_end = pd.to_datetime(split_info.train_end_date)
    val_end = pd.to_datetime(split_info.val_end_date)

    n_outcomes = len(OUTCOMES)
    outcome_to_idx = {o: i for i, o in enumerate(OUTCOMES)}

    # --- Accumulators ---
    batter_overall_counts = defaultdict(lambda: np.zeros(n_outcomes, dtype=np.float64))
    batter_overall_n = defaultdict(int)

    pitcher_overall_counts = defaultdict(lambda: np.zeros(n_outcomes, dtype=np.float64))
    pitcher_overall_n = defaultdict(int)

    stadium_counts = defaultdict(lambda: np.zeros(n_outcomes, dtype=np.float64))
    stadium_n = defaultdict(int)

    # Batter platoon: (batter_id, pitcher_hand) -> outcome counts
    batter_platoon_counts = defaultdict(lambda: np.zeros(n_outcomes, dtype=np.float64))
    batter_platoon_n = defaultdict(int)

    # Pitcher platoon: (pitcher_id, batter_hand) -> outcome counts
    pitcher_platoon_counts = defaultdict(lambda: np.zeros(n_outcomes, dtype=np.float64))
    pitcher_platoon_n = defaultdict(int)

    # Batter by pitch type: (batter_id, pitch_type) -> outcome counts (PA-ending only)
    batter_pitch_type_counts = defaultdict(lambda: np.zeros(n_outcomes, dtype=np.float64))
    batter_pitch_type_n = defaultdict(int)

    # Pitcher pitch mix: pitcher_id -> {pitch_type: count} (ALL pitches)
    pitcher_pitch_mix_counts = defaultdict(lambda: defaultdict(int))
    pitcher_pitch_mix_total = defaultdict(int)

    league_counts = np.zeros(n_outcomes, dtype=np.float64)
    league_n = 0

    rows_read = 0
    train_pitch_rows = 0
    pa_ending_rows = 0

    for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=READ_CHUNK_ROWS, low_memory=True)):
        rows_read += len(chunk)
        missing = [c for c in REQUIRED_PITCH_COLUMNS if c not in chunk.columns]
        _require(not missing, f"Missing required columns in input: {missing}")

        chunk["game_date"] = pd.to_datetime(chunk["game_date"], errors="coerce").dt.normalize()
        y_str = _map_to_six_labels(chunk["pa_outcome"])
        good = chunk["game_date"].notna() & y_str.notna()
        df = chunk.loc[good].copy()
        if df.empty:
            continue

        split = _assign_split(df["game_date"], train_end, val_end)
        df = df.loc[split == "train"].copy()
        if df.empty:
            continue

        train_pitch_rows += len(df)

        # IDs
        batter_id = _coerce_int_series(df["batter_id"]).astype("int64")
        pitcher_id = _coerce_int_series(df["pitcher_id"]).astype("int64")
        ok_ids = (batter_id > 0) & (pitcher_id > 0)
        if not bool(ok_ids.any()):
            continue
        df = df.loc[ok_ids].copy()
        batter_id = batter_id.loc[ok_ids]
        pitcher_id = pitcher_id.loc[ok_ids]

        # Remap pitch types
        df["pitch_type_mapped"] = _pitch_type_bin(df["pitch_type"])

        # Clean handedness
        df["stand_clean"] = df["stand"].astype("string").str.upper().str.strip()
        df["p_throws_clean"] = df["p_throws"].astype("string").str.upper().str.strip()

        # Ensure stadium_id
        if STADIUM_ID_COL not in df.columns:
            for c in ["park", "park_id", "stadium", "venue_name", "park_proxy_home_team"]:
                if c in df.columns:
                    df[STADIUM_ID_COL] = df[c]
                    break
        if STADIUM_ID_COL not in df.columns:
            df[STADIUM_ID_COL] = "__UNK__"
        df[STADIUM_ID_COL] = df[STADIUM_ID_COL].astype("string").fillna("__UNK__").replace("", "__UNK__")

        # =====================================================
        # ALL PITCHES: Accumulate pitcher pitch mix
        # =====================================================
        mix_df = pd.DataFrame({
            "pid": pitcher_id.values,
            "pt": df["pitch_type_mapped"].values,
        })
        for (pid_val, pt_val), cnt in mix_df.groupby(["pid", "pt"]).size().items():
            pitcher_pitch_mix_counts[int(pid_val)][pt_val] += int(cnt)
            pitcher_pitch_mix_total[int(pid_val)] += int(cnt)

        # =====================================================
        # PA-ENDING PITCHES: Accumulate outcome counts
        # =====================================================
        if "is_last_pitch_of_pa" in df.columns:
            is_last = pd.to_numeric(df["is_last_pitch_of_pa"], errors="coerce").fillna(0).astype("int64")
        elif "pitch_number_in_pa" in df.columns and "pa_id" in df.columns:
            max_pitch = df.groupby("pa_id")["pitch_number_in_pa"].transform("max")
            is_last = (df["pitch_number_in_pa"] == max_pitch).astype("int64")
        else:
            is_last = pd.Series([0] * len(df), index=df.index, dtype="int64")

        pa_ending_mask = is_last == 1
        six_outcome = _map_to_six_labels(df["pa_outcome"])
        outcome_idx = six_outcome.map(outcome_to_idx)
        valid_pa = pa_ending_mask & outcome_idx.notna()

        if not valid_pa.any():
            if (i + 1) % 10 == 0:
                print(f"  - [PASS 2b] processed {rows_read:,} pitch rows...")
            continue

        pa_df = pd.DataFrame({
            "bid": batter_id.loc[valid_pa].values,
            "pid": pitcher_id.loc[valid_pa].values,
            "sid": df[STADIUM_ID_COL].loc[valid_pa].values,
            "p_hand": df["p_throws_clean"].loc[valid_pa].values,
            "b_hand": df["stand_clean"].loc[valid_pa].values,
            "pt": df["pitch_type_mapped"].loc[valid_pa].values,
            "oi": outcome_idx.loc[valid_pa].astype("int64").values,
        })
        pa_ending_rows += len(pa_df)

        # --- League ---
        for oi_val, cnt in pa_df["oi"].value_counts().items():
            league_counts[int(oi_val)] += float(cnt)
            league_n += int(cnt)

        # --- Batter overall ---
        for (bid_val, oi_val), cnt in pa_df.groupby(["bid", "oi"]).size().items():
            batter_overall_counts[int(bid_val)][int(oi_val)] += float(cnt)
            batter_overall_n[int(bid_val)] += int(cnt)

        # --- Pitcher overall ---
        for (pid_val, oi_val), cnt in pa_df.groupby(["pid", "oi"]).size().items():
            pitcher_overall_counts[int(pid_val)][int(oi_val)] += float(cnt)
            pitcher_overall_n[int(pid_val)] += int(cnt)

        # --- Stadium ---
        for (sid_val, oi_val), cnt in pa_df.groupby(["sid", "oi"]).size().items():
            stadium_counts[str(sid_val)][int(oi_val)] += float(cnt)
            stadium_n[str(sid_val)] += int(cnt)

        # --- Batter platoon (batter vs pitcher's hand) ---
        plat_b = pa_df[pa_df["p_hand"].isin(["L", "R"])]
        for (bid_val, p_hand, oi_val), cnt in plat_b.groupby(["bid", "p_hand", "oi"]).size().items():
            batter_platoon_counts[(int(bid_val), p_hand)][int(oi_val)] += float(cnt)
            batter_platoon_n[(int(bid_val), p_hand)] += int(cnt)

        # --- Pitcher platoon (pitcher vs batter's hand) ---
        plat_p = pa_df[pa_df["b_hand"].isin(["L", "R"])]
        for (pid_val, b_hand, oi_val), cnt in plat_p.groupby(["pid", "b_hand", "oi"]).size().items():
            pitcher_platoon_counts[(int(pid_val), b_hand)][int(oi_val)] += float(cnt)
            pitcher_platoon_n[(int(pid_val), b_hand)] += int(cnt)

        # --- Batter by pitch type ---
        for (bid_val, pt_val, oi_val), cnt in pa_df.groupby(["bid", "pt", "oi"]).size().items():
            batter_pitch_type_counts[(int(bid_val), pt_val)][int(oi_val)] += float(cnt)
            batter_pitch_type_n[(int(bid_val), pt_val)] += int(cnt)

        if (i + 1) % 10 == 0:
            print(f"  - [PASS 2b] processed {rows_read:,} pitch rows... (train: {train_pitch_rows:,}, PA-ending: {pa_ending_rows:,})")

    _require(train_pitch_rows > 0, "No TRAIN pitch rows found.")
    _require(pa_ending_rows > 0, "No PA-ending pitch rows found in TRAIN split.")

    # Compute league rates
    league_rates = league_counts / max(league_n, 1)
    league_rates = league_rates / (league_rates.sum() + 1e-12)

    print(f"[PASS 2b] League rates: {dict(zip(OUTCOMES, league_rates.round(4)))}")
    print(f"[PASS 2b] Total PA-ending pitches: {pa_ending_rows:,}")

    # =====================================================
    # Build output DataFrames
    # =====================================================

    def _compute_rates(counts, n, league_rates):
        """
        Compute outcome rates WITHOUT smoothing.

        - If n > 0: return raw rates (counts / n)
        - If n = 0: return league average (fallback for unseen players)

        No Dirichlet smoothing. The model uses log_n to learn weighting.
        """
        if n > 0:
            rates = counts / n
            total = rates.sum()
            if total > 0:
                rates = rates / total
            else:
                rates = league_rates.copy()
        else:
            rates = league_rates.copy()
        return rates

    # 1. Batter overall
    batter_overall_rows = []
    for bid, counts in batter_overall_counts.items():
        n = batter_overall_n[bid]
        rates = _compute_rates(counts, n, league_rates)
        row = {"batter_id": bid, "batter_n_pa": n}
        for oi, outcome in enumerate(OUTCOMES):
            row[f"batter_{outcome}_rate"] = float(rates[oi])
        batter_overall_rows.append(row)
    batter_overall_df = pd.DataFrame(batter_overall_rows)

    # 2. Pitcher overall
    pitcher_overall_rows = []
    for pid, counts in pitcher_overall_counts.items():
        n = pitcher_overall_n[pid]
        rates = _compute_rates(counts, n, league_rates)
        row = {"pitcher_id": pid, "pitcher_n_pa": n}
        for oi, outcome in enumerate(OUTCOMES):
            row[f"pitcher_{outcome}_rate"] = float(rates[oi])
        pitcher_overall_rows.append(row)
    pitcher_overall_df = pd.DataFrame(pitcher_overall_rows)

    # 3. Stadium rates
    stadium_rows = []
    for sid, counts in stadium_counts.items():
        n = stadium_n[sid]
        rates = _compute_rates(counts, n, league_rates)
        row = {"stadium_id": sid, "stadium_n_pa": n}
        for oi, outcome in enumerate(OUTCOMES):
            row[f"stadium_{outcome}_rate"] = float(rates[oi])
        stadium_rows.append(row)
    stadium_df = pd.DataFrame(stadium_rows)

    # 4. Batter platoon (vs L and vs R pitchers)
    batter_platoon_rows = []
    for bid in batter_overall_counts.keys():
        row = {"batter_id": bid}
        for hand in ("L", "R"):
            key = (bid, hand)
            counts = batter_platoon_counts.get(key, np.zeros(n_outcomes))
            n = batter_platoon_n.get(key, 0)
            rates = _compute_rates(counts, n, league_rates)
            for oi, outcome in enumerate(OUTCOMES):
                row[f"batter_{outcome}_vs_{hand}"] = float(rates[oi])
            row[f"batter_n_vs_{hand}"] = n
        batter_platoon_rows.append(row)
    batter_platoon_df = pd.DataFrame(batter_platoon_rows)

    # 5. Pitcher platoon (vs L and vs R batters)
    pitcher_platoon_rows = []
    for pid in pitcher_overall_counts.keys():
        row = {"pitcher_id": pid}
        for hand in ("L", "R"):
            key = (pid, hand)
            counts = pitcher_platoon_counts.get(key, np.zeros(n_outcomes))
            n = pitcher_platoon_n.get(key, 0)
            rates = _compute_rates(counts, n, league_rates)
            for oi, outcome in enumerate(OUTCOMES):
                row[f"pitcher_{outcome}_vs_{hand}"] = float(rates[oi])
            row[f"pitcher_n_vs_{hand}"] = n
        pitcher_platoon_rows.append(row)
    pitcher_platoon_df = pd.DataFrame(pitcher_platoon_rows)

    # 6. Batter by pitch type (for pitch mix interaction)
    batter_pitch_type_rows = []
    for bid in batter_overall_counts.keys():
        row = {"batter_id": bid}
        for pt in PITCH_TYPES:
            key = (bid, pt)
            counts = batter_pitch_type_counts.get(key, np.zeros(n_outcomes))
            n = batter_pitch_type_n.get(key, 0)
            rates = _compute_rates(counts, n, league_rates)
            for oi, outcome in enumerate(OUTCOMES):
                row[f"batter_{outcome}_vs_{pt}"] = float(rates[oi])
            row[f"batter_n_vs_{pt}"] = n
        batter_pitch_type_rows.append(row)
    batter_pitch_type_df = pd.DataFrame(batter_pitch_type_rows)

    # 7. Pitcher pitch mix (usage rates)
    pitcher_mix_rows = []
    for pid in pitcher_overall_counts.keys():
        row = {"pitcher_id": pid}
        total = max(pitcher_pitch_mix_total.get(pid, 0), 1)
        for pt in PITCH_TYPES:
            count = pitcher_pitch_mix_counts.get(pid, {}).get(pt, 0)
            row[f"pitcher_mix_{pt}"] = float(count) / total
        row["pitcher_n_pitches"] = total
        pitcher_mix_rows.append(row)
    pitcher_mix_df = pd.DataFrame(pitcher_mix_rows)

    print(f"[PASS 2b] Built tables: {len(batter_overall_df)} batters, {len(pitcher_overall_df)} pitchers, {len(stadium_df)} stadiums")

    # ===========================================
    # DIAGNOSTIC: Verify player-specific variation
    # ===========================================
    print("\n" + "=" * 70)
    print("SIX-VECTOR DIAGNOSTICS (No Smoothing)")
    print("=" * 70)

    print("\n[SAMPLE SIZES]")
    print(f"  Total PA-ending pitches in training: {pa_ending_rows:,}")
    print(f"  Unique batters: {len(batter_overall_df):,}")
    print(f"  Unique pitchers: {len(pitcher_overall_df):,}")
    print(f"  Unique stadiums: {len(stadium_df):,}")

    print(f"\n  Batter PA: min={batter_overall_df['batter_n_pa'].min()}, "
          f"median={batter_overall_df['batter_n_pa'].median():.0f}, "
          f"mean={batter_overall_df['batter_n_pa'].mean():.0f}, "
          f"max={batter_overall_df['batter_n_pa'].max()}")
    print(f"  Pitcher PA: min={pitcher_overall_df['pitcher_n_pa'].min()}, "
          f"median={pitcher_overall_df['pitcher_n_pa'].median():.0f}, "
          f"mean={pitcher_overall_df['pitcher_n_pa'].mean():.0f}, "
          f"max={pitcher_overall_df['pitcher_n_pa'].max()}")
    print(f"  Stadium PA: min={stadium_df['stadium_n_pa'].min()}, "
          f"median={stadium_df['stadium_n_pa'].median():.0f}, "
          f"mean={stadium_df['stadium_n_pa'].mean():.0f}, "
          f"max={stadium_df['stadium_n_pa'].max()}")

    print("\n[RATE VARIATION - Should be HIGH with no smoothing]")
    for outcome in OUTCOMES:
        b_col = f"batter_{outcome}_rate"
        p_col = f"pitcher_{outcome}_rate"
        s_col = f"stadium_{outcome}_rate"
        b_std = batter_overall_df[b_col].std()
        b_min = batter_overall_df[b_col].min()
        b_max = batter_overall_df[b_col].max()
        p_std = pitcher_overall_df[p_col].std()
        p_min = pitcher_overall_df[p_col].min()
        p_max = pitcher_overall_df[p_col].max()
        s_std = stadium_df[s_col].std()
        s_min = stadium_df[s_col].min()
        s_max = stadium_df[s_col].max()
        print(f"\n  {outcome}:")
        print(f"    Batter:  std={b_std:.4f}, range=[{b_min:.3f}, {b_max:.3f}]")
        print(f"    Pitcher: std={p_std:.4f}, range=[{p_min:.3f}, {p_max:.3f}]")
        print(f"    Stadium: std={s_std:.4f}, range=[{s_min:.3f}, {s_max:.3f}]")

    print("\n[PLATOON VARIATION]")
    for outcome in OUTCOMES:
        vs_L = batter_platoon_df[f"batter_{outcome}_vs_L"]
        vs_R = batter_platoon_df[f"batter_{outcome}_vs_R"]
        diff = (vs_L - vs_R).abs()
        print(f"  Batter {outcome} vs L vs R: mean_diff={diff.mean():.4f}, max_diff={diff.max():.4f}")

    print("\n[PITCH-TYPE VARIATION]")
    for pt in PITCH_TYPES[:5]:
        col = f"batter_K_vs_{pt}"
        if col in batter_pitch_type_df.columns:
            std = batter_pitch_type_df[col].std()
            print(f"  Batter K vs {pt}: std={std:.4f}")

    print("=" * 70 + "\n")
    print("[PASS 2b] Six-vector statistics computation complete.")

    return {
        "batter_overall": batter_overall_df,
        "pitcher_overall": pitcher_overall_df,
        "stadium_rates": stadium_df,
        "batter_platoon": batter_platoon_df,
        "pitcher_platoon": pitcher_platoon_df,
        "batter_pitch_type": batter_pitch_type_df,
        "pitcher_mix": pitcher_mix_df,
        "league_rates": league_rates,
    }


# =========================
# Six-vector PA-level feature computation
# =========================

def compute_six_vectors(
    pa_chunk: pd.DataFrame,
    stats: Dict,
) -> pd.DataFrame:
    """
    Compute the 6 vectors for each PA.

    Each vector has 6 outcome rates + 1 log(n) = 7 values.
    Total: 6 vectors x 7 = 42 features.
    """
    league_rates = stats["league_rates"]

    # Build indexed lookup tables
    batter_overall = stats["batter_overall"].set_index("batter_id")
    pitcher_overall = stats["pitcher_overall"].set_index("pitcher_id")
    stadium_rates = stats["stadium_rates"].set_index("stadium_id")
    batter_platoon = stats["batter_platoon"].set_index("batter_id")
    pitcher_platoon = stats["pitcher_platoon"].set_index("pitcher_id")
    batter_pitch_type = stats["batter_pitch_type"].set_index("batter_id")
    pitcher_mix = stats["pitcher_mix"].set_index("pitcher_id")

    n_pa = len(pa_chunk)
    features = {}

    # Extract raw IDs for lookups
    b_ids = _coerce_int_series(pa_chunk["batter_id"]).astype("int64")
    p_ids = _coerce_int_series(pa_chunk["pitcher_id"]).astype("int64")
    s_ids = pa_chunk["stadium_id"].astype("string").values
    p_throws = pa_chunk["p_throws"].astype("string").str.upper().str.strip().values
    stand = pa_chunk["stand"].astype("string").str.upper().str.strip().values

    # Reindex all tables to PA order
    batter_overall_rows = batter_overall.reindex(b_ids)
    pitcher_overall_rows = pitcher_overall.reindex(p_ids)
    stadium_rows = stadium_rates.reindex(s_ids)
    batter_platoon_rows = batter_platoon.reindex(b_ids)
    pitcher_platoon_rows = pitcher_platoon.reindex(p_ids)
    batter_pt_rows = batter_pitch_type.reindex(b_ids)
    pitcher_mix_rows = pitcher_mix.reindex(p_ids)

    # =====================================================
    # VECTOR 1: Batter overall
    # =====================================================
    for oi, outcome in enumerate(OUTCOMES):
        col = f"batter_{outcome}_rate"
        features[f"v1_batter_{outcome}"] = batter_overall_rows[col].fillna(league_rates[oi]).values.astype(np.float32)
    features["v1_batter_log_n"] = np.log1p(batter_overall_rows["batter_n_pa"].fillna(0).values).astype(np.float32)

    # =====================================================
    # VECTOR 2: Pitcher overall
    # =====================================================
    for oi, outcome in enumerate(OUTCOMES):
        col = f"pitcher_{outcome}_rate"
        features[f"v2_pitcher_{outcome}"] = pitcher_overall_rows[col].fillna(league_rates[oi]).values.astype(np.float32)
    features["v2_pitcher_log_n"] = np.log1p(pitcher_overall_rows["pitcher_n_pa"].fillna(0).values).astype(np.float32)

    # =====================================================
    # VECTOR 3: Stadium
    # =====================================================
    for oi, outcome in enumerate(OUTCOMES):
        col = f"stadium_{outcome}_rate"
        features[f"v3_stadium_{outcome}"] = stadium_rows[col].fillna(league_rates[oi]).values.astype(np.float32)
    features["v3_stadium_log_n"] = np.log1p(stadium_rows["stadium_n_pa"].fillna(0).values).astype(np.float32)

    # =====================================================
    # VECTOR 4: Batter platoon (vs this pitcher's hand)
    # =====================================================
    for oi, outcome in enumerate(OUTCOMES):
        vs_L = batter_platoon_rows[f"batter_{outcome}_vs_L"].fillna(league_rates[oi]).values
        vs_R = batter_platoon_rows[f"batter_{outcome}_vs_R"].fillna(league_rates[oi]).values
        features[f"v4_batter_platoon_{outcome}"] = np.where(
            p_throws == "L", vs_L, vs_R
        ).astype(np.float32)
    n_vs_L = batter_platoon_rows["batter_n_vs_L"].fillna(0).values
    n_vs_R = batter_platoon_rows["batter_n_vs_R"].fillna(0).values
    features["v4_batter_platoon_log_n"] = np.log1p(
        np.where(p_throws == "L", n_vs_L, n_vs_R)
    ).astype(np.float32)

    # =====================================================
    # VECTOR 5: Pitcher platoon (vs this batter's hand)
    # =====================================================
    for oi, outcome in enumerate(OUTCOMES):
        vs_L = pitcher_platoon_rows[f"pitcher_{outcome}_vs_L"].fillna(league_rates[oi]).values
        vs_R = pitcher_platoon_rows[f"pitcher_{outcome}_vs_R"].fillna(league_rates[oi]).values
        features[f"v5_pitcher_platoon_{outcome}"] = np.where(
            stand == "L", vs_L, vs_R
        ).astype(np.float32)
    n_vs_L = pitcher_platoon_rows["pitcher_n_vs_L"].fillna(0).values
    n_vs_R = pitcher_platoon_rows["pitcher_n_vs_R"].fillna(0).values
    features["v5_pitcher_platoon_log_n"] = np.log1p(
        np.where(stand == "L", n_vs_L, n_vs_R)
    ).astype(np.float32)

    # =====================================================
    # VECTOR 6: Pitch mix interaction
    # E[outcome] = sum_pt P(pitch_type=pt | pitcher) * P(outcome | batter, pt)
    # =====================================================
    for oi, outcome in enumerate(OUTCOMES):
        interaction = np.zeros(n_pa, dtype=np.float32)
        for pt in PITCH_TYPES:
            # P(pitch_type | pitcher)
            mix_col = f"pitcher_mix_{pt}"
            if mix_col in pitcher_mix_rows.columns:
                p_pitch = pitcher_mix_rows[mix_col].fillna(0).values
            else:
                p_pitch = np.zeros(n_pa)
            # P(outcome | batter, pitch_type)
            batter_col = f"batter_{outcome}_vs_{pt}"
            if batter_col in batter_pt_rows.columns:
                p_outcome = batter_pt_rows[batter_col].fillna(league_rates[oi]).values
            else:
                p_outcome = np.full(n_pa, league_rates[oi])
            interaction += p_pitch * p_outcome
        features[f"v6_mix_{outcome}"] = interaction.astype(np.float32)

    # Sample size: min of pitcher total pitches and batter PAs
    pitcher_n_pitches = pitcher_mix_rows["pitcher_n_pitches"].fillna(0).values if "pitcher_n_pitches" in pitcher_mix_rows.columns else np.zeros(n_pa)
    batter_n_pa = batter_overall_rows["batter_n_pa"].fillna(0).values
    features["v6_mix_log_n"] = np.log1p(np.minimum(pitcher_n_pitches, batter_n_pa)).astype(np.float32)

    # ===========================================
    # DIAGNOSTIC: PA-level feature statistics
    # ===========================================
    print("\n[PA-LEVEL FEATURE DIAGNOSTICS]")
    print("  Checking that features vary across PAs...")
    for oi, outcome in enumerate(OUTCOMES):
        v1 = features[f"v1_batter_{outcome}"]
        v2 = features[f"v2_pitcher_{outcome}"]
        v3 = features[f"v3_stadium_{outcome}"]
        v4 = features[f"v4_batter_platoon_{outcome}"]
        v5 = features[f"v5_pitcher_platoon_{outcome}"]
        v6 = features[f"v6_mix_{outcome}"]
        print(f"\n  {outcome}:")
        print(f"    v1_batter:  mean={v1.mean():.4f}, std={v1.std():.4f}, range=[{v1.min():.3f}, {v1.max():.3f}]")
        print(f"    v2_pitcher: mean={v2.mean():.4f}, std={v2.std():.4f}, range=[{v2.min():.3f}, {v2.max():.3f}]")
        print(f"    v3_stadium: mean={v3.mean():.4f}, std={v3.std():.4f}, range=[{v3.min():.3f}, {v3.max():.3f}]")
        print(f"    v4_b_plat:  mean={v4.mean():.4f}, std={v4.std():.4f}, range=[{v4.min():.3f}, {v4.max():.3f}]")
        print(f"    v5_p_plat:  mean={v5.mean():.4f}, std={v5.std():.4f}, range=[{v5.min():.3f}, {v5.max():.3f}]")
        print(f"    v6_mix:     mean={v6.mean():.4f}, std={v6.std():.4f}, range=[{v6.min():.3f}, {v6.max():.3f}]")
    print("\n  Sample size features (log_n):")
    print(f"    v1_batter_log_n:  mean={features['v1_batter_log_n'].mean():.2f}, std={features['v1_batter_log_n'].std():.2f}")
    print(f"    v2_pitcher_log_n: mean={features['v2_pitcher_log_n'].mean():.2f}, std={features['v2_pitcher_log_n'].std():.2f}")
    print(f"    v3_stadium_log_n: mean={features['v3_stadium_log_n'].mean():.2f}, std={features['v3_stadium_log_n'].std():.2f}")
    print(f"    v4_platoon_log_n: mean={features['v4_batter_platoon_log_n'].mean():.2f}, std={features['v4_batter_platoon_log_n'].std():.2f}")
    print(f"    v5_platoon_log_n: mean={features['v5_pitcher_platoon_log_n'].mean():.2f}, std={features['v5_pitcher_platoon_log_n'].std():.2f}")
    print(f"    v6_mix_log_n:     mean={features['v6_mix_log_n'].mean():.2f}, std={features['v6_mix_log_n'].std():.2f}")

    return pd.DataFrame(features, index=pa_chunk.index)


# =========================
# PASS 3: Build PA-level rows, join tendencies, encode categoricals, write splits
# =========================

def _compute_park_factors(
    input_csv: str,
    split_info: SplitInfo,
    alpha: float = 5.0,
) -> pd.DataFrame:
    """
    Compute park factors from TRAIN split only.

    For each stadium, compute the rate of each outcome relative to league average.
    Apply Dirichlet smoothing toward league rates.

    Returns DataFrame with columns:
        stadium_id, park_K_factor, park_BIPO_factor, park_BB_factor,
        park_1B_factor, park_XBH_factor, park_HR_factor, park_n_pa
    """
    train_end = pd.to_datetime(split_info.train_end_date)

    # Accumulators
    stadium_counts = defaultdict(lambda: np.zeros(len(OUTCOMES), dtype=np.float64))
    stadium_n = defaultdict(int)
    league_counts_pf = np.zeros(len(OUTCOMES), dtype=np.float64)
    league_n = 0

    outcome_to_idx = {o: i for i, o in enumerate(OUTCOMES)}

    for chunk in pd.read_csv(input_csv, chunksize=READ_CHUNK_ROWS, low_memory=True):
        chunk["game_date"] = pd.to_datetime(chunk["game_date"], errors="coerce").dt.normalize()
        y_str = _map_to_six_labels(chunk["pa_outcome"])
        good = chunk["game_date"].notna() & y_str.notna()
        chunk = chunk.loc[good].copy()
        if chunk.empty:
            continue

        split = _assign_split(chunk["game_date"], train_end, pd.to_datetime(split_info.val_end_date))
        chunk = chunk.loc[split == "train"].copy()
        if chunk.empty:
            continue

        # Detect PA-ending pitches
        if "is_last_pitch_of_pa" in chunk.columns:
            is_last = pd.to_numeric(chunk["is_last_pitch_of_pa"], errors="coerce").fillna(0).astype("int64")
        elif "pitch_number_in_pa" in chunk.columns and "pa_id" in chunk.columns:
            max_pitch = chunk.groupby("pa_id")["pitch_number_in_pa"].transform("max")
            is_last = (chunk["pitch_number_in_pa"] == max_pitch).astype("int64")
        else:
            is_last = pd.Series([0] * len(chunk), index=chunk.index, dtype="int64")

        pa_end = chunk[is_last == 1].copy()
        if pa_end.empty:
            continue

        pa_end["outcome"] = _map_to_six_labels(pa_end["pa_outcome"])
        pa_end = pa_end[pa_end["outcome"].notna()]
        if pa_end.empty:
            continue

        # Ensure stadium_id exists
        if "stadium_id" not in pa_end.columns:
            for c in ["park", "park_id", "stadium", "venue_name", "park_proxy_home_team"]:
                if c in pa_end.columns:
                    pa_end["stadium_id"] = pa_end[c]
                    break
        if "stadium_id" not in pa_end.columns:
            pa_end["stadium_id"] = "__UNK__"
        pa_end["stadium_id"] = pa_end["stadium_id"].astype("string").fillna("__UNK__").replace("", "__UNK__")

        # Vectorized accumulation
        pa_end["oi"] = pa_end["outcome"].map(outcome_to_idx)
        pa_end = pa_end[pa_end["oi"].notna()]
        pa_end["oi"] = pa_end["oi"].astype("int64")

        for (stadium, oi_val), cnt in pa_end.groupby(["stadium_id", "oi"]).size().items():
            stadium_counts[str(stadium)][int(oi_val)] += float(cnt)
            stadium_n[str(stadium)] += int(cnt)
            league_counts_pf[int(oi_val)] += float(cnt)
            league_n += int(cnt)

    # Compute league rates
    league_rates_pf = league_counts_pf / max(league_n, 1)

    # Build park factors DataFrame
    rows = []
    for stadium, counts in stadium_counts.items():
        n = stadium_n[stadium]

        # Smoothed rates: (counts + alpha * league_rates) / (n + alpha)
        smoothed = (counts + alpha * league_rates_pf) / (n + alpha)

        # Park factor = smoothed_rate / league_rate
        factors = smoothed / (league_rates_pf + 1e-8)

        row = {"stadium_id": stadium, "park_n_pa": n}
        for i, outcome in enumerate(OUTCOMES):
            row[f"park_{outcome}_factor"] = factors[i]

        rows.append(row)

    park_df = pd.DataFrame(rows)

    # Log summary
    print(f"[PARK FACTORS] Computed for {len(park_df)} stadiums")
    for outcome in OUTCOMES:
        col = f"park_{outcome}_factor"
        if col in park_df.columns and len(park_df) > 0:
            print(f"  {outcome}: min={park_df[col].min():.3f}, max={park_df[col].max():.3f}, mean={park_df[col].mean():.3f}")

    return park_df


def _build_feature_list():
    """Build feature lists dynamically based on feature toggles."""
    categorical_cols = list(CATEGORICAL_COLS) if ENABLE_EMBEDDINGS else []
    numeric_cols = ["pitcher_fatigue"]

    if ENABLE_SIX_VECTORS:
        # Vector 1: Batter overall (6 rates + 1 log_n)
        for outcome in OUTCOMES:
            numeric_cols.append(f"v1_batter_{outcome}")
        numeric_cols.append("v1_batter_log_n")
        # Vector 2: Pitcher overall (6 rates + 1 log_n)
        for outcome in OUTCOMES:
            numeric_cols.append(f"v2_pitcher_{outcome}")
        numeric_cols.append("v2_pitcher_log_n")
        # Vector 3: Stadium (6 rates + 1 log_n)
        for outcome in OUTCOMES:
            numeric_cols.append(f"v3_stadium_{outcome}")
        numeric_cols.append("v3_stadium_log_n")
        # Vector 4: Batter platoon (6 rates + 1 log_n)
        for outcome in OUTCOMES:
            numeric_cols.append(f"v4_batter_platoon_{outcome}")
        numeric_cols.append("v4_batter_platoon_log_n")
        # Vector 5: Pitcher platoon (6 rates + 1 log_n)
        for outcome in OUTCOMES:
            numeric_cols.append(f"v5_pitcher_platoon_{outcome}")
        numeric_cols.append("v5_pitcher_platoon_log_n")
        # Vector 6: Pitch mix interaction (6 rates + 1 log_n)
        for outcome in OUTCOMES:
            numeric_cols.append(f"v6_mix_{outcome}")
        numeric_cols.append("v6_mix_log_n")

    if ENABLE_BUCKET_FEATURES:
        for prefix in ["batter", "pitcher", "matchup"]:
            for outcome in OUTCOMES:
                numeric_cols.append(f"{prefix}_{outcome}_expected")

    if ENABLE_PARK_FACTORS:
        for outcome in OUTCOMES:
            numeric_cols.append(f"park_{outcome}_factor")

    if ENABLE_BUCKET_FEATURES:
        numeric_cols.extend(["batter_log_n", "pitcher_log_n"])

    return categorical_cols, numeric_cols


def _compute_matchup_features(
    pa_chunk: pd.DataFrame,
    batter_tend: pd.DataFrame,
    pitcher_tend: pd.DataFrame,
    park_factors: pd.DataFrame,
    league_fallback: Dict,
) -> pd.DataFrame:
    """
    Compute PA-level features:
    1. Bucket-weighted expected rates (18 features) - from 40 buckets (if enabled)
    2. Park factors (6 features) (if enabled)
    3. Uncertainty (2 features) (if bucket features enabled)

    Note: stand and p_throws are handled as categorical columns, not here.
    Respects ENABLE_BUCKET_FEATURES and ENABLE_PARK_FACTORS toggles.
    """
    outcomes = league_fallback["outcomes"]
    league_rates = np.array(league_fallback["league_rates"], dtype=np.float32)

    n_pa = len(pa_chunk)
    features = {}

    # =====================================================
    # PART 1: Bucket-weighted expected rates (40 buckets)
    # =====================================================
    if ENABLE_BUCKET_FEATURES:
        batter_map = batter_tend.set_index("batter_id") if not batter_tend.empty else pd.DataFrame()
        pitcher_map = pitcher_tend.set_index("pitcher_id") if not pitcher_tend.empty else pd.DataFrame()

        b_ids = pa_chunk["batter_id"].values
        p_ids = pa_chunk["pitcher_id"].values

        batter_rows = batter_map.reindex(b_ids) if not batter_map.empty else pd.DataFrame(index=range(n_pa))
        pitcher_rows = pitcher_map.reindex(p_ids) if not pitcher_map.empty else pd.DataFrame(index=range(n_pa))

        for oi, outcome in enumerate(outcomes):
            batter_weighted = np.zeros(n_pa, dtype=np.float32)
            pitcher_weighted = np.zeros(n_pa, dtype=np.float32)
            matchup_weighted = np.zeros(n_pa, dtype=np.float32)
            total_usage = np.zeros(n_pa, dtype=np.float32)

            for bucket in ALL_BUCKETS:
                bucket_safe = bucket.replace("|", "_")

                # Pitcher usage
                usage_col = f"pitcher_usage_{bucket_safe}"
                if usage_col in pitcher_rows.columns:
                    usage = pitcher_rows[usage_col].fillna(1.0 / len(ALL_BUCKETS)).values
                else:
                    usage = np.full(n_pa, 1.0 / len(ALL_BUCKETS), dtype=np.float32)

                # Batter bucket rate
                b_col = f"batter_{outcome}_{bucket_safe}"
                if b_col in batter_rows.columns:
                    b_rate = batter_rows[b_col].fillna(league_rates[oi]).values
                else:
                    b_rate = np.full(n_pa, league_rates[oi], dtype=np.float32)

                # Pitcher bucket rate
                p_col = f"pitcher_{outcome}_{bucket_safe}"
                if p_col in pitcher_rows.columns:
                    p_rate = pitcher_rows[p_col].fillna(league_rates[oi]).values
                else:
                    p_rate = np.full(n_pa, league_rates[oi], dtype=np.float32)

                # Accumulate
                batter_weighted += b_rate * usage
                pitcher_weighted += p_rate * usage
                matchup_weighted += (b_rate * p_rate / (league_rates[oi] + 1e-8)) * usage
                total_usage += usage

            total_usage = np.maximum(total_usage, 1e-8)

            features[f"batter_{outcome}_expected"] = (batter_weighted / total_usage).astype(np.float32)
            features[f"pitcher_{outcome}_expected"] = (pitcher_weighted / total_usage).astype(np.float32)
            features[f"matchup_{outcome}_expected"] = (matchup_weighted / total_usage).astype(np.float32)

        # Uncertainty features
        if "batter_n_pa" in batter_rows.columns:
            features["batter_log_n"] = np.log1p(batter_rows["batter_n_pa"].fillna(0).values).astype(np.float32)
        else:
            features["batter_log_n"] = np.zeros(n_pa, dtype=np.float32)

        if "pitcher_n_pa" in pitcher_rows.columns:
            features["pitcher_log_n"] = np.log1p(pitcher_rows["pitcher_n_pa"].fillna(0).values).astype(np.float32)
        else:
            features["pitcher_log_n"] = np.zeros(n_pa, dtype=np.float32)

    # =====================================================
    # PART 2: Park factors (6 features)
    # =====================================================
    if ENABLE_PARK_FACTORS:
        park_map = park_factors.set_index("stadium_id") if (not park_factors.empty and "stadium_id" in park_factors.columns) else pd.DataFrame()
        s_ids = pa_chunk["stadium_id"].astype("string").values
        park_rows = park_map.reindex(s_ids) if not park_map.empty else pd.DataFrame(index=range(n_pa))

        for outcome in outcomes:
            col = f"park_{outcome}_factor"
            if col in park_rows.columns:
                features[col] = park_rows[col].fillna(1.0).values.astype(np.float32)
            else:
                features[col] = np.ones(n_pa, dtype=np.float32)

    return pd.DataFrame(features, index=pa_chunk.index)


def pass3_write_pa_splits(
    input_csv: str,
    split_info: SplitInfo,
    vocabs: Dict[str, Dict[str, int]],
    batter_tend: pd.DataFrame,
    pitcher_tend: pd.DataFrame,
    park_factors: pd.DataFrame,
    league_fallback: Dict,
    six_vector_stats: Optional[Dict],
    output_format: str,
) -> Dict[str, object]:
    """
    Build PA-level rows with matchup features, encode categoricals, write splits.

    Uses the new PA-ending tendency system with expected outcome rates.
    Now includes park factors and handedness as categorical embeddings.
    """
    print("[PASS 3] Building PA-level rows with matchup features, encoding categoricals, writing splits...")

    train_end = pd.to_datetime(split_info.train_end_date)
    val_end = pd.to_datetime(split_info.val_end_date)

    # Dynamic feature lists based on toggles
    categorical_cols, numeric_feature_cols = _build_feature_list()
    print(f"  [PASS 3] Feature toggles: SIX_VECTORS={ENABLE_SIX_VECTORS}, EMBEDDINGS={ENABLE_EMBEDDINGS}, BUCKET={ENABLE_BUCKET_FEATURES}, PARK={ENABLE_PARK_FACTORS}")
    print(f"  [PASS 3] Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"  [PASS 3] Numeric features ({len(numeric_feature_cols)}): {numeric_feature_cols}")

    feature_cols = categorical_cols + numeric_feature_cols
    out_cols = [PA_ID_COL] + feature_cols + ["y"]

    writers = {"train": None, "val": None, "test": None}
    rows_read = 0
    pa_rows_written = {"train": 0, "val": 0, "test": 0}
    label_counts_by_split = {"train": Counter(), "val": Counter(), "test": Counter()}

    for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=READ_CHUNK_ROWS, low_memory=True)):
        rows_read += len(chunk)
        missing = [c for c in REQUIRED_PITCH_COLUMNS if c not in chunk.columns]
        _require(not missing, f"Missing required columns in input: {missing}")

        chunk[GAME_DATE_COL] = pd.to_datetime(chunk[GAME_DATE_COL], errors="coerce").dt.normalize()
        y_str = _map_to_six_labels(chunk[OUTCOME_COL])
        y_id = y_str.map({lab: idx for idx, lab in enumerate(SIX_LABELS)})
        good = chunk[GAME_DATE_COL].notna() & y_id.notna()
        df = chunk.loc[good].copy()
        if df.empty:
            continue

        df["y"] = y_id.loc[good].astype("int64")

        # Split assignment
        split = _assign_split(df[GAME_DATE_COL], train_end, val_end)
        df["split"] = split

        # Ensure pitcher_fatigue exists
        if "pitcher_fatigue" not in df.columns:
            if "pitcher_game_pitch_count" in df.columns:
                ppc = pd.to_numeric(df["pitcher_game_pitch_count"], errors="coerce").fillna(0).astype("int64")
                df["pitcher_fatigue"] = (ppc - 1).clip(lower=0).astype("int64")
            else:
                df["pitcher_fatigue"] = 0

        # Ensure stadium_id exists
        if STADIUM_ID_COL not in df.columns:
            for c in ["park", "park_id", "stadium", "venue_name", "park_proxy_home_team"]:
                if c in df.columns:
                    df[STADIUM_ID_COL] = df[c]
                    break
        if STADIUM_ID_COL not in df.columns:
            df[STADIUM_ID_COL] = "__UNK__"
        df[STADIUM_ID_COL] = df[STADIUM_ID_COL].astype("string").fillna("__UNK__").replace("", "__UNK__")

        # PA id
        if PA_ID_COL not in df.columns:
            df[PA_ID_COL] = (
                df[GAME_DATE_COL].astype("string")
                + "|" + df[BATTER_ID_COL].astype("string")
                + "|" + df[PITCHER_ID_COL].astype("string")
                + "|" + df.index.astype("string")
            )

        # Aggregate to PA level by taking first pitch row per PA
        pa_first = df.sort_index().groupby(PA_ID_COL, sort=False).head(1).copy()

        # Partition by split
        pa_rows = {
            "train": pa_first.loc[pa_first["split"] == "train"].copy(),
            "val": pa_first.loc[pa_first["split"] == "val"].copy(),
            "test": pa_first.loc[pa_first["split"] == "test"].copy(),
        }

        for split_name, pa_chunk in pa_rows.items():
            if pa_chunk.empty:
                continue

            # Base columns (include stand and p_throws for both matchup features and encoding)
            pa_chunk = pa_chunk[[PA_ID_COL, BATTER_ID_COL, PITCHER_ID_COL, STADIUM_ID_COL, "pitcher_fatigue", "stand", "p_throws", "y"]].copy()
            pa_chunk["pitcher_fatigue"] = pd.to_numeric(pa_chunk["pitcher_fatigue"], errors="coerce").fillna(0.0).astype("float32")

            # Compute six-vector features
            if ENABLE_SIX_VECTORS and six_vector_stats is not None:
                six_vector_features = compute_six_vectors(pa_chunk, six_vector_stats)
                for col in six_vector_features.columns:
                    pa_chunk[col] = six_vector_features[col].values

            # Compute matchup features (bucket-weighted + park factors + uncertainty)
            if ENABLE_BUCKET_FEATURES or ENABLE_PARK_FACTORS:
                matchup_features = _compute_matchup_features(
                    pa_chunk,
                    batter_tend,
                    pitcher_tend,
                    park_factors,
                    league_fallback,
                )
                for col in matchup_features.columns:
                    pa_chunk[col] = matchup_features[col].values

            # Encode categoricals with TRAIN vocabs (UNK=1) - only if embeddings enabled
            if ENABLE_EMBEDDINGS:
                pa_chunk[BATTER_ID_COL] = _encode_categorical(pa_chunk[BATTER_ID_COL].astype("string"), vocabs["batter_id"])
                pa_chunk[PITCHER_ID_COL] = _encode_categorical(pa_chunk[PITCHER_ID_COL].astype("string"), vocabs["pitcher_id"])
                pa_chunk[STADIUM_ID_COL] = _encode_categorical(pa_chunk[STADIUM_ID_COL].astype("string"), vocabs["stadium_id"])

                pa_chunk["stand"] = pa_chunk["stand"].astype("string").str.upper().str.strip()
                pa_chunk["stand"] = pa_chunk["stand"].map(vocabs["stand"]).fillna(1).astype("int64")

                pa_chunk["p_throws"] = pa_chunk["p_throws"].astype("string").str.upper().str.strip()
                pa_chunk["p_throws"] = pa_chunk["p_throws"].map(vocabs["p_throws"]).fillna(1).astype("int64")

            # Ensure all feature columns exist with defaults
            for col in numeric_feature_cols:
                if col not in pa_chunk.columns:
                    pa_chunk[col] = 0.0

            # Reorder columns
            pa_chunk = pa_chunk[[PA_ID_COL] + feature_cols + ["y"]]

            # Write
            out_path_no_ext = os.path.join(OUTPUT_DIR, f"{split_name}")
            if writers[split_name] is None:
                writers[split_name] = _write_table(pa_chunk, out_path_no_ext, output_format)
            else:
                existing = _read_table(writers[split_name])
                combined = pd.concat([existing, pa_chunk], axis=0, ignore_index=True)
                writers[split_name] = _write_table(combined, out_path_no_ext, output_format)

            pa_rows_written[split_name] += len(pa_chunk)
            label_counts_by_split[split_name].update(pa_chunk["y"].astype("string").tolist())

        if (i + 1) % 10 == 0:
            print(f"  - [PASS 3] processed {rows_read:,} pitch rows... (PA rows written: {pa_rows_written})")

    print("[PASS 3] Done.")
    print("[PASS 3] Rows written:", pa_rows_written)
    print("[PASS 3] Label counts:", {k: dict(v) for k, v in label_counts_by_split.items()})

    return {
        "output_columns": out_cols,
        "feature_cols": feature_cols,
        "split_paths": {k: writers[k] for k in ["train", "val", "test"]},
        "rows_written": pa_rows_written,
        "label_counts_by_split": {k: dict(v) for k, v in label_counts_by_split.items()},
    }



# =========================
# Metadata writer
# =========================

def write_metadata_json(fit: FitStats, transform_summary: Dict[str, object], league_fallback: Dict) -> str:
    # Vocab sizes (include PAD=0 and UNK=1)
    vocab_sizes = {k: len(v) for k, v in fit.vocabs.items()}

    # Dynamic feature lists based on toggles
    categorical_features, numeric_features = _build_feature_list()
    feature_list_ordered = list(transform_summary.get("feature_cols", []))
    y_col = "y"

    if ENABLE_EMBEDDINGS:
        categorical_vocab_sizes = {
            "batter_id": int(vocab_sizes.get("batter_id", 0)),
            "pitcher_id": int(vocab_sizes.get("pitcher_id", 0)),
            "stadium_id": int(vocab_sizes.get("stadium_id", 0)),
            "stand": 4,       # PAD, UNK, L, R
            "p_throws": 4,    # PAD, UNK, L, R
        }
    else:
        categorical_vocab_sizes = {}

    metadata = {
        "created_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "split_info": asdict(fit.split_info),
        "vocabs": {"sizes": vocab_sizes},
        "tendency_spec": asdict(fit.tendency_spec),
        "tendency_paths": fit.tendency_paths,
        "labels": {
            "label_set_ordered": SIX_LABELS,
            "label_to_id": {lab: i for i, lab in enumerate(SIX_LABELS)},
        },
        "features": {
            "label_col": y_col,
            "categorical_features": categorical_features,
            "categorical_vocab_sizes": categorical_vocab_sizes,
            "numeric_features": numeric_features,
            "feature_list_ordered": feature_list_ordered,
        },
        "tendency_info": {
            "all_buckets": list(ALL_BUCKETS),
            "pitch_types": list(PITCH_TYPES),
            "zone_tiers": list(ZONE_TIERS),
            "smoothing_alpha": league_fallback.get("smoothing_alpha", TENDENCY_ALPHA),
        },
        "feature_toggles": {
            "ENABLE_SIX_VECTORS": ENABLE_SIX_VECTORS,
            "ENABLE_EMBEDDINGS": ENABLE_EMBEDDINGS,
            "ENABLE_BUCKET_FEATURES": ENABLE_BUCKET_FEATURES,
            "ENABLE_PARK_FACTORS": ENABLE_PARK_FACTORS,
        },
    }

    out_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return out_path



# =========================
# CLI / main
# =========================

def main() -> None:
    global INPUT_CSV, OUTPUT_DIR, OUTPUT_FORMAT
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--output_format", type=str, default=OUTPUT_FORMAT, choices=["csv", "parquet"])
    parser.add_argument("--train_end_date", type=str, default=TRAIN_END_DATE)
    parser.add_argument("--val_end_date", type=str, default=VAL_END_DATE)
    args = parser.parse_args()

    INPUT_CSV = args.input_csv
    OUTPUT_DIR = args.output_dir
    OUTPUT_FORMAT = args.output_format

    split_info = SplitInfo(train_end_date=args.train_end_date, val_end_date=args.val_end_date)

    _safe_mkdir(OUTPUT_DIR)

    # PASS 0: Fit vocabularies (always run for vocab tracking; embeddings optional)
    vocabs, tendency_spec = pass0_fit_vocabs_and_tendencies(INPUT_CSV, split_info)

    # Write vocabs
    vocabs_path = os.path.join(OUTPUT_DIR, "vocabs.json")
    with open(vocabs_path, "w", encoding="utf-8") as f:
        json.dump(vocabs, f, indent=2, sort_keys=True)

    # PASS 2: Fit PA-ending tendencies (old bucket system) - only if needed
    batter_tend_df = pd.DataFrame()
    pitcher_tend_df = pd.DataFrame()
    league_fallback = {"outcomes": OUTCOMES, "league_rates": [1.0 / len(OUTCOMES)] * len(OUTCOMES)}
    batter_path = ""
    pitcher_path = ""
    fallback_path = ""

    if ENABLE_BUCKET_FEATURES:
        batter_tend_df, pitcher_tend_df, league_fallback = pass2_fit_pa_ending_tendencies(
            INPUT_CSV, split_info, alpha=TENDENCY_ALPHA
        )
        batter_path = _write_table(batter_tend_df, os.path.join(OUTPUT_DIR, "batter_tendencies"), OUTPUT_FORMAT)
        pitcher_path = _write_table(pitcher_tend_df, os.path.join(OUTPUT_DIR, "pitcher_tendencies"), OUTPUT_FORMAT)
        fallback_path = os.path.join(OUTPUT_DIR, "league_fallback.json")
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(league_fallback, f, indent=2, sort_keys=False)

    # Compute park factors (old system) - only if needed
    park_factors_df = pd.DataFrame()
    park_factors_path = ""
    if ENABLE_PARK_FACTORS:
        park_factors_df = _compute_park_factors(INPUT_CSV, split_info, alpha=5.0)
        park_factors_path = _write_table(park_factors_df, os.path.join(OUTPUT_DIR, "park_factors"), OUTPUT_FORMAT)

    # PASS 2b: Compute six-vector statistics (new system)
    six_vector_stats = None
    if ENABLE_SIX_VECTORS:
        six_vector_stats = pass2_compute_six_vector_stats(
            INPUT_CSV, split_info, alpha=TENDENCY_ALPHA
        )
        # Save stats tables
        for name, df_or_arr in six_vector_stats.items():
            if isinstance(df_or_arr, pd.DataFrame):
                path = os.path.join(OUTPUT_DIR, f"six_vector_{name}")
                _write_table(df_or_arr, path, OUTPUT_FORMAT)
                print(f"  Saved {name}: {len(df_or_arr)} rows")
        # Save league rates
        league_rates_path = os.path.join(OUTPUT_DIR, "league_rates.json")
        with open(league_rates_path, "w", encoding="utf-8") as f:
            json.dump({
                "outcomes": OUTCOMES,
                "rates": six_vector_stats["league_rates"].tolist(),
            }, f, indent=2)

    # PASS 3: Build PA-level splits with features
    transform_summary = pass3_write_pa_splits(
        INPUT_CSV,
        split_info,
        vocabs,
        batter_tend_df,
        pitcher_tend_df,
        park_factors_df,
        league_fallback,
        six_vector_stats,
        OUTPUT_FORMAT,
    )

    fit = FitStats(
        split_info=split_info,
        vocabs=vocabs,
        tendency_spec=tendency_spec,
        tendency_paths={
            "batter_tendencies": batter_path,
            "pitcher_tendencies": pitcher_path,
            "park_factors": park_factors_path,
            "league_fallback": fallback_path,
            "vocabs": vocabs_path,
        },
        output_columns=transform_summary["output_columns"],
    )

    meta_path = write_metadata_json(fit, transform_summary, league_fallback)
    print("[DONE] Wrote metadata:", meta_path)
    print("[DONE] Outputs:", transform_summary["split_paths"])
    if ENABLE_BUCKET_FEATURES:
        print("[DONE] Tendency tables:", batter_path, pitcher_path)
    if ENABLE_PARK_FACTORS:
        print("[DONE] Park factors:", park_factors_path)
    if ENABLE_SIX_VECTORS:
        print("[DONE] Six-vector stats saved to preprocessed/six_vector_*.parquet")


if __name__ == "__main__":
    main()
