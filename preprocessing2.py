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
TRAIN_END_DATE = "2025-08-31"
VAL_END_DATE = "2025-09-14"

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
SHRINKAGE_K = 150               # PA-equivalents of prior weight for Bayesian shrinkage

# Feature toggles for ablation experiments
ENABLE_SIX_VECTORS = True       # Six-vector interpretable features (42)
ENABLE_EMBEDDINGS = True        # Enabled for hybrid model experiment
ENABLE_PITCHTYPE_STATCAST_V2 = False    # Pitch-type statcast v2 block (extended cols, CLI toggle)

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
        Bayesian shrinkage toward league average.

        With k=150, a player needs ~150 PAs before their rates are weighted
        equally with the prior. This means:
        - 8-PA player:   ~95% league average, ~5% observed
        - 50-PA player:  ~75% league, ~25% observed
        - 150-PA player: ~50/50
        - 600-PA player: ~20% league, ~80% observed

        The shrinkage naturally bounds log-odds deviations by sample size,
        eliminating the need for LayerNorm or log_n_scale on the vector head.
        """
        k = SHRINKAGE_K
        rates = (counts + k * league_rates) / (n + k)
        total = rates.sum()
        if total > 0:
            rates = rates / total
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
        "league_logodds": _safe_logit(league_rates),
    }


# =========================
# Six-vector PA-level feature computation
# =========================

def _safe_logit(p, eps=1e-6):
    """Clamp and compute log-odds: log(p / (1-p)). Works on numpy arrays and scalars."""
    p_clamped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clamped / (1.0 - p_clamped))


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
    league_logodds = _safe_logit(league_rates)  # shape (6,), log-odds of league baseline

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
        raw_rate = batter_overall_rows[col].fillna(league_rates[oi]).values
        features[f"v1_batter_{outcome}"] = (_safe_logit(raw_rate) - league_logodds[oi]).astype(np.float32)
    features["v1_batter_log_n"] = np.log1p(batter_overall_rows["batter_n_pa"].fillna(0).values).astype(np.float32)

    # =====================================================
    # VECTOR 2: Pitcher overall
    # =====================================================
    for oi, outcome in enumerate(OUTCOMES):
        col = f"pitcher_{outcome}_rate"
        raw_rate = pitcher_overall_rows[col].fillna(league_rates[oi]).values
        features[f"v2_pitcher_{outcome}"] = (_safe_logit(raw_rate) - league_logodds[oi]).astype(np.float32)
    features["v2_pitcher_log_n"] = np.log1p(pitcher_overall_rows["pitcher_n_pa"].fillna(0).values).astype(np.float32)

    # =====================================================
    # VECTOR 3: Stadium
    # =====================================================
    for oi, outcome in enumerate(OUTCOMES):
        col = f"stadium_{outcome}_rate"
        raw_rate = stadium_rows[col].fillna(league_rates[oi]).values
        features[f"v3_stadium_{outcome}"] = (_safe_logit(raw_rate) - league_logodds[oi]).astype(np.float32)
    features["v3_stadium_log_n"] = np.log1p(stadium_rows["stadium_n_pa"].fillna(0).values).astype(np.float32)

    # =====================================================
    # VECTOR 4: Batter platoon (vs this pitcher's hand)
    # =====================================================
    for oi, outcome in enumerate(OUTCOMES):
        vs_L = batter_platoon_rows[f"batter_{outcome}_vs_L"].fillna(league_rates[oi]).values
        vs_R = batter_platoon_rows[f"batter_{outcome}_vs_R"].fillna(league_rates[oi]).values
        raw_rate = np.where(p_throws == "L", vs_L, vs_R)
        features[f"v4_batter_platoon_{outcome}"] = (_safe_logit(raw_rate) - league_logodds[oi]).astype(np.float32)
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
        raw_rate = np.where(stand == "L", vs_L, vs_R)
        features[f"v5_pitcher_platoon_{outcome}"] = (_safe_logit(raw_rate) - league_logodds[oi]).astype(np.float32)
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
        features[f"v6_mix_{outcome}"] = (_safe_logit(interaction) - league_logodds[oi]).astype(np.float32)

    # Sample size: min of pitcher total pitches and batter PAs
    pitcher_n_pitches = pitcher_mix_rows["pitcher_n_pitches"].fillna(0).values if "pitcher_n_pitches" in pitcher_mix_rows.columns else np.zeros(n_pa)
    batter_n_pa = batter_overall_rows["batter_n_pa"].fillna(0).values
    features["v6_mix_log_n"] = np.log1p(np.minimum(pitcher_n_pitches, batter_n_pa)).astype(np.float32)

    # =====================================================
    # EXTREMENESS FEATURES (indices 43-45)
    # =====================================================
    _vec_prefixes = ["v1_batter", "v2_pitcher", "v3_stadium",
                     "v4_batter_platoon", "v5_pitcher_platoon", "v6_mix"]
    all_deviations = []
    for oi, outcome in enumerate(OUTCOMES):
        for prefix in _vec_prefixes:
            all_deviations.append(features[f"{prefix}_{outcome}"])
    dev_stack = np.stack(all_deviations, axis=1)  # (n_pa, 36)
    features["ext_max_abs_dev"] = np.abs(dev_stack).max(axis=1).astype(np.float32)
    features["ext_mean_abs_dev"] = np.abs(dev_stack).mean(axis=1).astype(np.float32)
    # Vector disagreement: mean over outcomes of std across 6 vectors
    disagreements = []
    for oi, outcome in enumerate(OUTCOMES):
        pred_stack = np.stack([features[f"{prefix}_{outcome}"] for prefix in _vec_prefixes], axis=1)
        disagreements.append(pred_stack.std(axis=1))
    features["ext_vector_disagreement"] = np.mean(np.stack(disagreements, axis=1), axis=1).astype(np.float32)

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
# =========================
# Pitch-Type Statcast Block
# =========================


# =========================
# Pitch-Type Statcast V2 Block
# =========================

# Pitcher mean cols (same as v1 base)
_V2_PITCHER_MEAN_COLS = [
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z",
]

# Pitcher shape cols (extension, release point, effective speed)
_V2_PITCHER_SHAPE_COLS = [
    "release_extension", "release_pos_x", "release_pos_z", "effective_speed",
]

# Spin axis is special: encoded as sin/cos
_V2_HAS_SPIN_AXIS = True  # we verified it exists


def _pitchtype_statcast_v2_feature_names():
    """Return the v2 feature column names in deterministic order."""
    cols = []
    for pt in PITCH_TYPES:
        # Pitcher block (21 per pt)
        cols.append(f"pitcher_pitch_rate_{pt}")
        for stat in _V2_PITCHER_MEAN_COLS:
            cols.append(f"pitcher_{pt}_{stat}")
        cols.append(f"pitcher_{pt}_zone_rate")
        cols.append(f"pitcher_{pt}_heart_rate")
        cols.append(f"pitcher_{pt}_shadow_rate")
        cols.append(f"pitcher_{pt}_chase_rate")
        cols.append(f"pitcher_{pt}_waste_rate")
        cols.append(f"pitcher_{pt}_plate_x_std")
        cols.append(f"pitcher_{pt}_plate_z_std")
        for stat in _V2_PITCHER_SHAPE_COLS:
            cols.append(f"pitcher_{pt}_{stat}")
        if _V2_HAS_SPIN_AXIS:
            cols.append(f"pitcher_{pt}_spin_axis_sin")
            cols.append(f"pitcher_{pt}_spin_axis_cos")
        # Batter block (3 per pt)
        cols.append(f"batter_{pt}_launch_speed")
        cols.append(f"batter_{pt}_launch_angle")
        cols.append(f"batter_{pt}_contact_rate")
    return cols


def compute_pitchtype_statcast_block_v2(input_csv: str, split_info: SplitInfo):
    """
    Compute TRAIN-only extended pitcher/batter statcast stats per pitch type.

    Returns:
        pitcher_wide: DataFrame keyed by pitcher_id (str)
        batter_wide:  DataFrame keyed by batter_id (str)
    """
    print("[PT STATCAST V2] Computing TRAIN-only pitch-type statcast v2 block...")
    train_end = pd.to_datetime(split_info.train_end_date)

    pitcher_records = []
    batter_records = []

    all_pitcher_mean_cols = _V2_PITCHER_MEAN_COLS + ["plate_x", "plate_z"] + _V2_PITCHER_SHAPE_COLS
    if _V2_HAS_SPIN_AXIS:
        all_pitcher_mean_cols.append("spin_axis")

    for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=READ_CHUNK_ROWS, low_memory=True)):
        chunk[GAME_DATE_COL] = pd.to_datetime(chunk[GAME_DATE_COL], errors="coerce").dt.normalize()
        train_mask = chunk[GAME_DATE_COL].notna() & (chunk[GAME_DATE_COL] <= train_end)
        df = chunk.loc[train_mask].copy()
        if df.empty:
            continue

        df["pt_bin"] = _pitch_type_bin(df[PITCH_TYPE_COL])

        # Coerce numeric columns
        for col in all_pitcher_mean_cols + ["launch_speed", "launch_angle", "sz_top", "sz_bot"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = np.nan

        # Zone classification
        if "zone" in df.columns:
            zone_val = pd.to_numeric(df["zone"], errors="coerce")
            df["in_zone"] = zone_val.between(1, 9).astype("float64")
        else:
            df["in_zone"] = (
                (df["plate_x"].abs() <= 0.83) &
                (df["plate_z"] >= df["sz_bot"]) &
                (df["plate_z"] <= df["sz_top"])
            ).astype("float64")

        # Attack zone classification
        eps = 0.01
        sz_range = (df["sz_top"] - df["sz_bot"]).clip(lower=eps)
        z_norm = (df["plate_z"] - df["sz_bot"]) / sz_range
        px = df["plate_x"]

        heart = (px.abs() <= 0.40) & (z_norm >= 0.33) & (z_norm <= 0.66)
        shadow_box = (px.abs() <= 1.08) & (z_norm >= -0.20) & (z_norm <= 1.20)
        shadow = shadow_box & ~heart
        chase_box = (px.abs() <= 1.40) & (z_norm >= -0.50) & (z_norm <= 1.50)
        chase = chase_box & ~shadow_box
        waste = ~chase_box

        df["is_heart"] = heart.astype("float64")
        df["is_shadow"] = shadow.astype("float64")
        df["is_chase"] = chase.astype("float64")
        df["is_waste"] = waste.astype("float64")

        # Spin axis sin/cos
        if _V2_HAS_SPIN_AXIS and "spin_axis" in df.columns:
            sa_rad = df["spin_axis"] * (np.pi / 180.0)
            df["spin_axis_sin"] = np.sin(sa_rad)
            df["spin_axis_cos"] = np.cos(sa_rad)

        # --- Pitcher aggregation (ALL pitches) ---
        agg_dict = {"n_pitches": ("pt_bin", "size")}
        for c in _V2_PITCHER_MEAN_COLS + _V2_PITCHER_SHAPE_COLS:
            agg_dict[f"sum_{c}"] = (c, "sum")
            agg_dict[f"cnt_{c}"] = (c, "count")
        # plate_x/z for std
        agg_dict["sum_plate_x"] = ("plate_x", "sum")
        agg_dict["cnt_plate_x"] = ("plate_x", "count")
        agg_dict["ssq_plate_x"] = ("plate_x", lambda x: (x**2).sum())
        agg_dict["sum_plate_z"] = ("plate_z", "sum")
        agg_dict["cnt_plate_z"] = ("plate_z", "count")
        agg_dict["ssq_plate_z"] = ("plate_z", lambda x: (x**2).sum())
        # Zone rates
        agg_dict["sum_in_zone"] = ("in_zone", "sum")
        agg_dict["sum_heart"] = ("is_heart", "sum")
        agg_dict["sum_shadow"] = ("is_shadow", "sum")
        agg_dict["sum_chase"] = ("is_chase", "sum")
        agg_dict["sum_waste"] = ("is_waste", "sum")
        # Spin axis sin/cos
        if _V2_HAS_SPIN_AXIS:
            agg_dict["sum_spin_axis_sin"] = ("spin_axis_sin", "sum")
            agg_dict["cnt_spin_axis_sin"] = ("spin_axis_sin", "count")
            agg_dict["sum_spin_axis_cos"] = ("spin_axis_cos", "sum")
            agg_dict["cnt_spin_axis_cos"] = ("spin_axis_cos", "count")

        pitcher_agg = df.groupby([PITCHER_ID_COL, "pt_bin"]).agg(**agg_dict).reset_index()
        pitcher_records.append(pitcher_agg)

        # --- Batter aggregation (PA-ending pitches only) ---
        if "is_last_pitch_of_pa" in df.columns:
            is_last = pd.to_numeric(df["is_last_pitch_of_pa"], errors="coerce").fillna(0).astype("int64")
        elif "pitch_number_in_pa" in df.columns and "pa_id" in df.columns:
            max_pitch = df.groupby("pa_id")["pitch_number_in_pa"].transform("max")
            is_last = (df["pitch_number_in_pa"] == max_pitch).astype("int64")
        else:
            is_last = pd.Series(1, index=df.index)

        pa_df = df.loc[is_last == 1].copy()
        if not pa_df.empty:
            pa_df["has_contact"] = pa_df["launch_speed"].notna().astype("float64")
            batter_agg = pa_df.groupby([BATTER_ID_COL, "pt_bin"]).agg(
                n_pa=("pt_bin", "size"),
                sum_launch_speed=("launch_speed", "sum"),
                cnt_launch_speed=("launch_speed", "count"),
                sum_launch_angle=("launch_angle", "sum"),
                cnt_launch_angle=("launch_angle", "count"),
                sum_has_contact=("has_contact", "sum"),
            ).reset_index()
            batter_records.append(batter_agg)

        if (i + 1) % 5 == 0:
            print(f"  [PT STATCAST V2] processed {(i+1)*READ_CHUNK_ROWS:,} rows...")

    # --- Aggregate across chunks ---
    pitcher_all = pd.concat(pitcher_records, ignore_index=True)
    # Sum numeric columns across chunks
    sum_cols = [c for c in pitcher_all.columns if c not in [PITCHER_ID_COL, "pt_bin"]]
    pitcher_all = pitcher_all.groupby([PITCHER_ID_COL, "pt_bin"])[sum_cols].sum().reset_index()

    # Compute pitcher means
    for c in _V2_PITCHER_MEAN_COLS + _V2_PITCHER_SHAPE_COLS:
        pitcher_all[f"mean_{c}"] = pitcher_all[f"sum_{c}"] / pitcher_all[f"cnt_{c}"].replace(0, np.nan)

    # Spin axis sin/cos means
    if _V2_HAS_SPIN_AXIS:
        pitcher_all["mean_spin_axis_sin"] = pitcher_all["sum_spin_axis_sin"] / pitcher_all["cnt_spin_axis_sin"].replace(0, np.nan)
        pitcher_all["mean_spin_axis_cos"] = pitcher_all["sum_spin_axis_cos"] / pitcher_all["cnt_spin_axis_cos"].replace(0, np.nan)

    # Plate x/z std: std = sqrt(E[x^2] - E[x]^2)
    for coord in ["plate_x", "plate_z"]:
        n = pitcher_all[f"cnt_{coord}"].replace(0, np.nan)
        mean = pitcher_all[f"sum_{coord}"] / n
        mean_sq = pitcher_all[f"ssq_{coord}"] / n
        pitcher_all[f"std_{coord}"] = np.sqrt((mean_sq - mean**2).clip(lower=0))

    # Pitch rates
    pitcher_total = pitcher_all.groupby(PITCHER_ID_COL)["n_pitches"].transform("sum")
    pitcher_all["pitch_rate"] = pitcher_all["n_pitches"] / pitcher_total.replace(0, np.nan)

    # Zone/attack rates
    pitcher_all["zone_rate"] = pitcher_all["sum_in_zone"] / pitcher_all["n_pitches"].replace(0, np.nan)
    pitcher_all["heart_rate"] = pitcher_all["sum_heart"] / pitcher_all["n_pitches"].replace(0, np.nan)
    pitcher_all["shadow_rate"] = pitcher_all["sum_shadow"] / pitcher_all["n_pitches"].replace(0, np.nan)
    pitcher_all["chase_rate"] = pitcher_all["sum_chase"] / pitcher_all["n_pitches"].replace(0, np.nan)
    pitcher_all["waste_rate"] = pitcher_all["sum_waste"] / pitcher_all["n_pitches"].replace(0, np.nan)

    # Pitcher overall fallbacks
    pitcher_overall = {}
    for c in _V2_PITCHER_MEAN_COLS + _V2_PITCHER_SHAPE_COLS:
        pitcher_overall[c] = pitcher_all.groupby(PITCHER_ID_COL).apply(
            lambda g: g[f"sum_{c}"].sum() / max(g[f"cnt_{c}"].sum(), 1), include_groups=False
        )
    for coord in ["plate_x", "plate_z"]:
        pitcher_overall[f"std_{coord}"] = pitcher_all.groupby(PITCHER_ID_COL).apply(
            lambda g: np.sqrt(max(g[f"ssq_{coord}"].sum() / max(g[f"cnt_{coord}"].sum(), 1)
                              - (g[f"sum_{coord}"].sum() / max(g[f"cnt_{coord}"].sum(), 1))**2, 0)),
            include_groups=False
        )
    if _V2_HAS_SPIN_AXIS:
        for sc in ["spin_axis_sin", "spin_axis_cos"]:
            pitcher_overall[sc] = pitcher_all.groupby(PITCHER_ID_COL).apply(
                lambda g, _sc=sc: g[f"sum_{_sc}"].sum() / max(g[f"cnt_{_sc}"].sum(), 1), include_groups=False
            )
    # Zone rate overalls
    for zr in ["zone_rate", "heart_rate", "shadow_rate", "chase_rate", "waste_rate"]:
        sum_col = f"sum_{zr.replace('_rate', '')}" if zr != "zone_rate" else "sum_in_zone"
        pitcher_overall[zr] = pitcher_all.groupby(PITCHER_ID_COL).apply(
            lambda g, _sc=sum_col: g[_sc].sum() / max(g["n_pitches"].sum(), 1), include_groups=False
        )

    # Pivot pitcher wide
    all_pitchers = pitcher_all[PITCHER_ID_COL].unique()
    pitcher_wide_parts = []
    for pt in PITCH_TYPES:
        pt_data = pitcher_all.loc[pitcher_all["pt_bin"] == pt].set_index(PITCHER_ID_COL)
        pt_df = pd.DataFrame(index=all_pitchers)

        pt_df[f"pitcher_pitch_rate_{pt}"] = pt_data["pitch_rate"].reindex(all_pitchers).fillna(0.0)
        for c in _V2_PITCHER_MEAN_COLS:
            col = f"pitcher_{pt}_{c}"
            pt_df[col] = pt_data[f"mean_{c}"].reindex(all_pitchers)
            mask = pt_df[col].isna()
            if mask.any():
                pt_df.loc[mask, col] = pt_df.index[mask].map(pitcher_overall[c])
            pt_df[col] = pt_df[col].fillna(0.0)

        pt_df[f"pitcher_{pt}_zone_rate"] = pt_data["zone_rate"].reindex(all_pitchers)
        pt_df[f"pitcher_{pt}_heart_rate"] = pt_data["heart_rate"].reindex(all_pitchers)
        pt_df[f"pitcher_{pt}_shadow_rate"] = pt_data["shadow_rate"].reindex(all_pitchers)
        pt_df[f"pitcher_{pt}_chase_rate"] = pt_data["chase_rate"].reindex(all_pitchers)
        pt_df[f"pitcher_{pt}_waste_rate"] = pt_data["waste_rate"].reindex(all_pitchers)
        for zr in ["zone_rate", "heart_rate", "shadow_rate", "chase_rate", "waste_rate"]:
            col = f"pitcher_{pt}_{zr}"
            mask = pt_df[col].isna()
            if mask.any():
                pt_df.loc[mask, col] = pt_df.index[mask].map(pitcher_overall[zr])
            pt_df[col] = pt_df[col].fillna(0.0)

        for coord in ["plate_x", "plate_z"]:
            col = f"pitcher_{pt}_{coord}_std"
            pt_df[col] = pt_data[f"std_{coord}"].reindex(all_pitchers)
            mask = pt_df[col].isna()
            if mask.any():
                pt_df.loc[mask, col] = pt_df.index[mask].map(pitcher_overall[f"std_{coord}"])
            pt_df[col] = pt_df[col].fillna(0.0)

        for c in _V2_PITCHER_SHAPE_COLS:
            col = f"pitcher_{pt}_{c}"
            pt_df[col] = pt_data[f"mean_{c}"].reindex(all_pitchers)
            mask = pt_df[col].isna()
            if mask.any():
                pt_df.loc[mask, col] = pt_df.index[mask].map(pitcher_overall[c])
            pt_df[col] = pt_df[col].fillna(0.0)

        if _V2_HAS_SPIN_AXIS:
            for sc in ["spin_axis_sin", "spin_axis_cos"]:
                col = f"pitcher_{pt}_{sc}"
                pt_df[col] = pt_data[f"mean_{sc}"].reindex(all_pitchers)
                mask = pt_df[col].isna()
                if mask.any():
                    pt_df.loc[mask, col] = pt_df.index[mask].map(pitcher_overall[sc])
                pt_df[col] = pt_df[col].fillna(0.0)

        pitcher_wide_parts.append(pt_df)

    pitcher_wide = pd.concat(pitcher_wide_parts, axis=1).astype("float32")
    pitcher_wide.index.name = PITCHER_ID_COL

    # --- Batter aggregation ---
    batter_all = pd.concat(batter_records, ignore_index=True)
    batter_sum_cols = [c for c in batter_all.columns if c not in [BATTER_ID_COL, "pt_bin"]]
    batter_all = batter_all.groupby([BATTER_ID_COL, "pt_bin"])[batter_sum_cols].sum().reset_index()

    batter_all["mean_launch_speed"] = batter_all["sum_launch_speed"] / batter_all["cnt_launch_speed"].replace(0, np.nan)
    batter_all["mean_launch_angle"] = batter_all["sum_launch_angle"] / batter_all["cnt_launch_angle"].replace(0, np.nan)
    batter_all["contact_rate"] = batter_all["sum_has_contact"] / batter_all["n_pa"].replace(0, np.nan)

    # Batter overall fallbacks
    batter_overall_ls = batter_all.groupby(BATTER_ID_COL).apply(
        lambda g: g["sum_launch_speed"].sum() / max(g["cnt_launch_speed"].sum(), 1), include_groups=False)
    batter_overall_la = batter_all.groupby(BATTER_ID_COL).apply(
        lambda g: g["sum_launch_angle"].sum() / max(g["cnt_launch_angle"].sum(), 1), include_groups=False)
    batter_overall_cr = batter_all.groupby(BATTER_ID_COL).apply(
        lambda g: g["sum_has_contact"].sum() / max(g["n_pa"].sum(), 1), include_groups=False)

    all_batters = batter_all[BATTER_ID_COL].unique()
    batter_wide_parts = []
    for pt in PITCH_TYPES:
        pt_data = batter_all.loc[batter_all["pt_bin"] == pt].set_index(BATTER_ID_COL)
        pt_df = pd.DataFrame(index=all_batters)

        for stat, overall, src in [
            (f"batter_{pt}_launch_speed", batter_overall_ls, "mean_launch_speed"),
            (f"batter_{pt}_launch_angle", batter_overall_la, "mean_launch_angle"),
            (f"batter_{pt}_contact_rate", batter_overall_cr, "contact_rate"),
        ]:
            pt_df[stat] = pt_data[src].reindex(all_batters)
            mask = pt_df[stat].isna()
            if mask.any():
                pt_df.loc[mask, stat] = pt_df.index[mask].map(overall)
            pt_df[stat] = pt_df[stat].fillna(0.0)

        batter_wide_parts.append(pt_df)

    batter_wide = pd.concat(batter_wide_parts, axis=1).astype("float32")
    batter_wide.index.name = BATTER_ID_COL

    total_cols = len(pitcher_wide.columns) + len(batter_wide.columns)
    print(f"[PT STATCAST V2] Done. Pitcher: {len(pitcher_wide)} x {len(pitcher_wide.columns)} cols, "
          f"Batter: {len(batter_wide)} x {len(batter_wide.columns)} cols, Total per-PA cols: {total_cols}")

    return pitcher_wide, batter_wide


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
        # Extremeness features (indices 43-45 when six-vectors enabled)
        numeric_cols.extend([
            "ext_max_abs_dev",
            "ext_mean_abs_dev",
            "ext_vector_disagreement",
        ])

    if ENABLE_PITCHTYPE_STATCAST_V2:
        numeric_cols.extend(_pitchtype_statcast_v2_feature_names())

    return categorical_cols, numeric_cols




def pass3_write_pa_splits(
    input_csv: str,
    split_info: SplitInfo,
    vocabs: Dict[str, Dict[str, int]],
    six_vector_stats: Optional[Dict],
    output_format: str,
    pitchtype_statcast: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
    pitchtype_statcast_v2: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
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
    print(f"  [PASS 3] Feature toggles: SIX_VECTORS={ENABLE_SIX_VECTORS}, EMBEDDINGS={ENABLE_EMBEDDINGS}")
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

            # Merge pitch-type statcast v2 block
            if ENABLE_PITCHTYPE_STATCAST_V2 and pitchtype_statcast_v2 is not None:
                pitcher_wide_v2, batter_wide_v2 = pitchtype_statcast_v2
                pid_str = pa_chunk[PITCHER_ID_COL].astype("string")
                bid_str = pa_chunk[BATTER_ID_COL].astype("string")
                pitcher_matched = pitcher_wide_v2.reindex(pid_str.values)
                pitcher_matched.index = pa_chunk.index
                for col in pitcher_wide_v2.columns:
                    pa_chunk[col] = pitcher_matched[col].fillna(0.0).astype("float32").values
                batter_matched = batter_wide_v2.reindex(bid_str.values)
                batter_matched.index = pa_chunk.index
                for col in batter_wide_v2.columns:
                    pa_chunk[col] = batter_matched[col].fillna(0.0).astype("float32").values

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

def write_metadata_json(fit: FitStats, transform_summary: Dict[str, object]) -> str:
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
            "smoothing_alpha": TENDENCY_ALPHA,
        },
        "feature_toggles": {
            "ENABLE_SIX_VECTORS": ENABLE_SIX_VECTORS,
            "ENABLE_EMBEDDINGS": ENABLE_EMBEDDINGS,
            "ENABLE_PITCHTYPE_STATCAST_V2": ENABLE_PITCHTYPE_STATCAST_V2,
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
    global INPUT_CSV, OUTPUT_DIR, OUTPUT_FORMAT, ENABLE_PITCHTYPE_STATCAST_V2
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--output_format", type=str, default=OUTPUT_FORMAT, choices=["csv", "parquet"])
    parser.add_argument("--train_end_date", type=str, default=TRAIN_END_DATE)
    parser.add_argument("--val_end_date", type=str, default=VAL_END_DATE)
    parser.add_argument("--enable_pitchtype_statcast_v2", action="store_true", default=False,
                        help="Enable pitch-type statcast v2 block (210 extra numeric cols)")
    args = parser.parse_args()

    INPUT_CSV = args.input_csv
    OUTPUT_DIR = args.output_dir
    OUTPUT_FORMAT = args.output_format
    if args.enable_pitchtype_statcast_v2:
        ENABLE_PITCHTYPE_STATCAST_V2 = True

    split_info = SplitInfo(train_end_date=args.train_end_date, val_end_date=args.val_end_date)

    _safe_mkdir(OUTPUT_DIR)

    # PASS 0: Fit vocabularies (always run for vocab tracking; embeddings optional)
    vocabs, tendency_spec = pass0_fit_vocabs_and_tendencies(INPUT_CSV, split_info)

    # Write vocabs
    vocabs_path = os.path.join(OUTPUT_DIR, "vocabs.json")
    with open(vocabs_path, "w", encoding="utf-8") as f:
        json.dump(vocabs, f, indent=2, sort_keys=True)

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
        # Save league log-odds
        league_logodds_path = os.path.join(OUTPUT_DIR, "league_logodds.json")
        with open(league_logodds_path, "w", encoding="utf-8") as f:
            json.dump({
                "outcomes": OUTCOMES,
                "logodds": six_vector_stats["league_logodds"].tolist(),
            }, f, indent=2)

    # Compute pitch-type statcast v2 block (if enabled)
    pitchtype_statcast_v2 = None
    if ENABLE_PITCHTYPE_STATCAST_V2:
        pitcher_wide_v2, batter_wide_v2 = compute_pitchtype_statcast_block_v2(INPUT_CSV, split_info)
        pitchtype_statcast_v2 = (pitcher_wide_v2, batter_wide_v2)
        _write_table(pitcher_wide_v2.reset_index(), os.path.join(OUTPUT_DIR, "pitchtype_statcast_v2_pitcher"), OUTPUT_FORMAT)
        _write_table(batter_wide_v2.reset_index(), os.path.join(OUTPUT_DIR, "pitchtype_statcast_v2_batter"), OUTPUT_FORMAT)

    # PASS 3: Build PA-level splits with features
    transform_summary = pass3_write_pa_splits(
        INPUT_CSV,
        split_info,
        vocabs,
        six_vector_stats,
        OUTPUT_FORMAT,
        pitchtype_statcast_v2=pitchtype_statcast_v2,
    )

    fit = FitStats(
        split_info=split_info,
        vocabs=vocabs,
        tendency_spec=tendency_spec,
        tendency_paths={
            "vocabs": vocabs_path,
        },
        output_columns=transform_summary["output_columns"],
    )

    meta_path = write_metadata_json(fit, transform_summary)
    print("[DONE] Wrote metadata:", meta_path)
    print("[DONE] Outputs:", transform_summary["split_paths"])
    if ENABLE_SIX_VECTORS:
        print("[DONE] Six-vector stats saved to preprocessed/six_vector_*.parquet")


if __name__ == "__main__":
    main()
