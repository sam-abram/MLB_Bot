#!/usr/bin/env python3
"""
preprocess_data.py

Stream-preprocess Statcast pitch-level CSV (from data_pipeline.py) into train/val/test files
with leakage-safe splitting, feature selection, categorical encoding, and numeric imputation.

Design goals:
- Never load the full dataset into memory (stream in chunks).
- Preserve one row per pitch (no aggregation).
- Vectorized operations only (no row-wise .apply(axis=1)).
- Leakage-safe: fit vocabularies / imputations / rare-label handling on TRAIN only, apply to all splits.
"""

from __future__ import annotations

import os
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# =========================
# Configuration constants
# =========================

INPUT_CSV = "statcast_pitch_level.csv"
READ_CHUNK_ROWS = 500_000
OUTPUT_DIR = "preprocessed"

# Prefer parquet, but fall back safely if not available
OUTPUT_FORMAT = "parquet"  # "parquet" or "csv.gz" (auto-fallback if parquet unavailable)

FEATURE_MODE = "pre_pitch"  # allowed: "pre_pitch", "post_pitch", "batted_ball"
SPLIT_MODE = "time"         # only "time" supported here (by game_date)

# Choose either explicit date cutoffs OR fractions.
# If TRAIN_END_DATE and VAL_END_DATE are set (non-empty), they are used (inclusive).
TRAIN_END_DATE = ""  # e.g. "2023-08-31"
VAL_END_DATE = ""    # e.g. "2023-09-30"

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10  # must sum to 1.0 if using fractions

MIN_LABEL_COUNT = 200
RANDOM_SEED = 1337

WRITE_METADATA_JSON = True

# Optional behavior knobs (kept simple; default keeps raw categoricals out of final features)
KEEP_RAW_CATEGORICALS = False

# =========================
# Label mapping (pa_outcome -> canonical label)
# =========================

RAW_TO_CANON_LABEL = {
    # K / BB / HBP
    "strikeout": "SO",
    "strikeout_double_play": "SO",  # treat as SO (rare)
    "walk": "BB",
    "intent_walk": "BB",
    "hit_by_pitch": "HBP",

    # Hits
    "home_run": "HR",
    "single": "1B",
    "double": "2B",
    "triple": "3B",

    # In-play outs (common)
    "field_out": "INPLAY_OUT",
    "force_out": "INPLAY_OUT",
    "grounded_into_double_play": "INPLAY_OUT",
    "double_play": "INPLAY_OUT",
    "triple_play": "INPLAY_OUT",
    "fielders_choice_out": "INPLAY_OUT",

    # Reached
    "fielders_choice": "FC",
    "reached_on_error": "ROE",

    # Sacrifice
    "sac_fly": "SAC",
    "sac_bunt": "SAC",
}

UNRECOGNIZED_LABEL_BUCKET = "OTHER_RAW"
RARE_LABEL_BUCKET = "OTHER"  # after rare-label handling on train

# =========================
# Feature selection rules
# =========================

# These columns leak outcome / are post-result fields and should be excluded in pre_pitch.
# (Only excluded if present.)
PRE_PITCH_EXCLUDE = {
    "description",
    "type",
    "bb_type",
    "hit_location",
    "outs_on_play",
    "des",
    # batted-ball measurements
    "launch_speed",
    "launch_angle",
    "launch_speed_angle",
    "hit_distance_sc",
    "hc_x",
    "hc_y",
    "spray_angle",
    # leakage/result-derived
    "delta_run_exp",
    "delta_home_win_exp",
    "events",
}

# Post-pitch can include pitch-result fields (description/type),
# but still excludes batted-ball & estimated_* unless in batted_ball mode.
POST_PITCH_EXCLUDE = {
    # batted-ball measurements
    "bb_type",
    "hit_location",
    "outs_on_play",
    "launch_speed",
    "launch_angle",
    "launch_speed_angle",
    "hit_distance_sc",
    "hc_x",
    "hc_y",
    "spray_angle",
    "des",
}

# Always exclude these from model features (identifiers or targets / splitting)
ALWAYS_EXCLUDE = {
    "pa_outcome",
    "label",
}

# Any columns starting with this prefix:
ESTIMATED_PREFIX = "estimated_"

# =========================
# Column groups for coercion and engineering
# =========================

REQUIRED_COLUMNS = ["pa_id", "pitch_number_in_pa", "pa_outcome"]

# ID-like columns to coerce to integer if present (invalid -> 0)
ID_LIKE_INT_COLS = [
    "game_pk",
    "at_bat_number",
    "batter",
    "pitcher",
]

# Count/state fields to coerce to integer if present (missing -> 0)
STATE_INT_COLS = [
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "pitch_number_in_pa",
    "pitcher_game_pitch_count",
    "zone",
]

# Base occupancy columns may be runner IDs (on_1b/on_2b/on_3b) or 0/1 flags (base_*_occupied).
# We'll derive boolean flags from base_*_occupied when present; otherwise fall back to on_*.
BASE_OCCUPANCY_COLS = ["on_1b", "on_2b", "on_3b"]
BASE_OCCUPIED_FLAG_COLS = ["base_1b_occupied", "base_2b_occupied", "base_3b_occupied"]

# Common continuous numeric fields to coerce to float if present (missing stays NaN)
# (Not exhaustive; includes many common Statcast pitch metrics.)
FLOAT_COLS_CANDIDATES = [
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "spin_axis",
    "spin_rate_deprecated",
    "effective_speed",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "sz_top",
    "sz_bot",
    "vx0",
    "vy0",
    "vz0",
    "ax",
    "ay",
    "az",
    "delta_home_win_exp",
    "delta_run_exp",
    "score_diff_bat_minus_fld",
]

# Categorical candidates to encode (if present)
CATEGORICAL_COLS_CANDIDATES = [
    "pitch_type",
    "stand",
    "p_throws",
    "inning_topbot",
    "home_team",
    "away_team",
    "venue",
    "park",
    "park_id",
    "stadium",
    "venue_name",
    "park_proxy_home_team",
    # derived:
    "count_str",
]

# Missing-indicator numerics (only created if the base numeric exists and is in final numeric features)
MISSING_INDICATOR_NUMERICS = [
    "release_speed",
    "release_spin_rate",
    "plate_x",
    "plate_z",
    "release_extension",
]

# =========================
# Utilities
# =========================

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _parse_date_str(s: str) -> Optional[pd.Timestamp]:
    s = (s or "").strip()
    if not s:
        return None
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.normalize()

def _is_parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
        return True
    except Exception:
        return False

def _resolve_output_format() -> str:
    fmt = (OUTPUT_FORMAT or "").strip().lower()
    if fmt == "parquet":
        if _is_parquet_available():
            return "parquet"
        print("[WARN] OUTPUT_FORMAT=parquet but pyarrow not available; falling back to csv.gz")
        return "csv.gz"
    if fmt in ("csv.gz", "csv"):
        return "csv.gz" if fmt == "csv.gz" else "csv.gz"
    print(f"[WARN] Unknown OUTPUT_FORMAT={OUTPUT_FORMAT!r}; falling back to csv.gz")
    return "csv.gz"

def _split_fractions_ok() -> bool:
    s = TRAIN_FRAC + VAL_FRAC + TEST_FRAC
    return abs(s - 1.0) < 1e-9 and TRAIN_FRAC > 0 and VAL_FRAC >= 0 and TEST_FRAC >= 0

def _normalize_str_series(s: pd.Series) -> pd.Series:
    # Normalize categoricals consistently (string, strip, lower)
    return s.astype("string").fillna("").str.strip().str.lower()

def _coerce_int(df: pd.DataFrame, col: str, fill: int = 0, dtype: str = "int64") -> None:
    if col not in df.columns:
        return
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill).astype(dtype)

def _coerce_float(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        return
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

def _coerce_many_int(df: pd.DataFrame, cols: Iterable[str], fill: int = 0, dtype: str = "int64") -> None:
    for c in cols:
        _coerce_int(df, c, fill=fill, dtype=dtype)

def _coerce_many_float(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        _coerce_float(df, c)

def _map_labels(pa_outcome: pd.Series) -> pd.Series:
    raw = _normalize_str_series(pa_outcome)
    # Treat empty as missing -> will be dropped later
    mapped = raw.map(RAW_TO_CANON_LABEL).fillna(UNRECOGNIZED_LABEL_BUCKET)
    return mapped

def _feature_mode_exclusions(columns: List[str]) -> set:
    cols = set(columns)
    if FEATURE_MODE == "pre_pitch":
        exc = set(PRE_PITCH_EXCLUDE)
        exc |= {c for c in cols if c.startswith(ESTIMATED_PREFIX)}
        return exc
    if FEATURE_MODE == "post_pitch":
        exc = set(POST_PITCH_EXCLUDE)
        exc |= {c for c in cols if c.startswith(ESTIMATED_PREFIX)}
        return exc
    if FEATURE_MODE == "batted_ball":
        return set()  # allow everything
    raise ValueError(f"FEATURE_MODE must be one of pre_pitch/post_pitch/batted_ball; got {FEATURE_MODE!r}")

def _validate_feature_mode() -> None:
    _require(FEATURE_MODE in {"pre_pitch", "post_pitch", "batted_ball"},
             f"Invalid FEATURE_MODE={FEATURE_MODE!r}")

def _validate_split_mode() -> None:
    _require(SPLIT_MODE == "time", f"Only SPLIT_MODE='time' is supported; got {SPLIT_MODE!r}")

def _compute_cutoffs_from_date_counts(
    date_counts: Dict[pd.Timestamp, int]
) -> Tuple[pd.Timestamp, pd.Timestamp, int]:
    """
    Given counts per normalized date, compute TRAIN_END and VAL_END cutoffs using row-weighted percentiles.
    Returns (train_end, val_end, total_rows).
    """
    _require(_split_fractions_ok(), "TRAIN_FRAC+VAL_FRAC+TEST_FRAC must sum to 1.0 and be non-negative.")
    total = int(sum(date_counts.values()))
    _require(total > 0, "No valid rows found to compute split cutoffs.")

    # Sort dates ascending
    items = sorted(date_counts.items(), key=lambda x: x[0])
    cum = 0
    train_target = total * TRAIN_FRAC
    val_target = total * (TRAIN_FRAC + VAL_FRAC)

    train_end = items[-1][0]
    val_end = items[-1][0]

    for d, cnt in items:
        cum += cnt
        if cum >= train_target and train_end == items[-1][0]:
            train_end = d
        if cum >= val_target:
            val_end = d
            break

    # Ensure ordering
    if val_end < train_end:
        val_end = train_end

    return train_end, val_end, total

def _assign_split(game_date: pd.Series, train_end: pd.Timestamp, val_end: pd.Timestamp) -> pd.Series:
    """
    Vectorized split assignment by date, inclusive cutoffs:
      train: date <= train_end
      val:   train_end < date <= val_end
      test:  date > val_end
    """
    d = game_date
    split = pd.Series(np.full(len(d), "test", dtype=object), index=d.index)
    split.loc[d <= train_end] = "train"
    split.loc[(d > train_end) & (d <= val_end)] = "val"
    return split

# =========================
# Writers
# =========================

class SplitWriter:
    def __init__(self, output_dir: str, fmt: str, split_name: str, columns: List[str]):
        self.output_dir = output_dir
        self.fmt = fmt
        self.split_name = split_name
        self.columns = columns
        self._initialized = False
        self._parquet_writer = None
        self.path = self._make_path()

    def _make_path(self) -> str:
        if self.fmt == "parquet":
            return os.path.join(self.output_dir, f"{self.split_name}.parquet")
        return os.path.join(self.output_dir, f"{self.split_name}.csv.gz")

    def write_df(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df.reindex(columns=self.columns)

        if self.fmt == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df, preserve_index=False)
            if not self._initialized:
                # Overwrite if exists
                if os.path.exists(self.path):
                    os.remove(self.path)
                self._parquet_writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
                self._initialized = True
            self._parquet_writer.write_table(table)
        else:
            # csv.gz append
            mode = "wt" if not self._initialized else "at"
            header = not self._initialized
            df.to_csv(self.path, index=False, mode=mode, header=header, compression="gzip")
            self._initialized = True

    def close(self) -> None:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None

# =========================
# Pass logic
# =========================

@dataclass
class SplitInfo:
    train_end_date: str
    val_end_date: str
    total_rows_with_valid_date_and_label: int

@dataclass
class FitStats:
    # label stats from train (canonical, before rare remap)
    train_label_counts: Dict[str, int]

    # rare labels -> OTHER
    rare_labels: List[str]
    final_labels: List[str]

    # categorical vocabularies (fit on train)
    # vocab maps normalized string -> int id (>=2), with 0=PAD/NA and 1=UNK
    categorical_vocabs: Dict[str, Dict[str, int]]

    # numeric imputation values (fit on train)
    numeric_impute: Dict[str, float]

    # final feature list (ordered) used for model input
    feature_list: List[str]

    # metadata about which raw categorical columns were encoded
    categorical_raw_cols: List[str]
    categorical_id_cols: List[str]
    numeric_cols: List[str]
    int_feature_cols: List[str]
    derived_cols: List[str]

def pass1_date_histogram(input_csv: str) -> Tuple[Dict[pd.Timestamp, int], List[str]]:
    """
    First pass: read chunks, parse/clean game_date and label presence,
    build histogram of counts per day to compute time split cutoffs.
    Also return header columns.
    """
    print("[PASS 1] Building game_date histogram for time-based split...")
    date_counts: Dict[pd.Timestamp, int] = defaultdict(int)
    header_cols: List[str] = []

    rows_read = 0
    rows_kept = 0
    rows_dropped_bad_date = 0
    rows_dropped_bad_label = 0

    for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=READ_CHUNK_ROWS, low_memory=True)):
        if i == 0:
            header_cols = list(chunk.columns)
            missing = [c for c in REQUIRED_COLUMNS if c not in header_cols]
            _require(not missing, f"Missing required columns in input: {missing}")

        rows_read += len(chunk)

        # Parse date
        if "game_date" not in chunk.columns:
            raise ValueError("Missing required column 'game_date' for time-based split.")

        game_date = pd.to_datetime(chunk["game_date"], errors="coerce").dt.normalize()
        good_date_mask = game_date.notna()

        # Label presence (pa_outcome non-empty)
        raw_outcome = chunk["pa_outcome"]
        raw_norm = _normalize_str_series(raw_outcome)
        good_label_mask = raw_norm.ne("")  # non-empty

        mask = good_date_mask & good_label_mask

        rows_dropped_bad_date += int((~good_date_mask).sum())
        rows_dropped_bad_label += int((good_date_mask & ~good_label_mask).sum())

        if mask.any():
            gd = game_date.loc[mask]
            vc = gd.value_counts()
            for d, cnt in vc.items():
                date_counts[d] += int(cnt)
            rows_kept += int(mask.sum())

        if (i + 1) % 10 == 0:
            print(f"  - processed {rows_read:,} rows...")

    print(f"[PASS 1] Done. Rows read: {rows_read:,}")
    print(f"[PASS 1] Rows kept (valid date+label): {rows_kept:,}")
    print(f"[PASS 1] Dropped bad dates: {rows_dropped_bad_date:,}")
    print(f"[PASS 1] Dropped missing/empty pa_outcome: {rows_dropped_bad_label:,}")
    return date_counts, header_cols

def determine_split_info(date_counts: Dict[pd.Timestamp, int]) -> SplitInfo:
    """
    Determine split cutoffs from explicit dates or from percentile-based fractions.
    """
    train_end = _parse_date_str(TRAIN_END_DATE)
    val_end = _parse_date_str(VAL_END_DATE)

    if train_end is not None and val_end is not None:
        _require(val_end >= train_end, "VAL_END_DATE must be >= TRAIN_END_DATE.")
        total = int(sum(date_counts.values()))
        print("[SPLIT] Using explicit date cutoffs.")
        print(f"[SPLIT] TRAIN_END_DATE (inclusive): {train_end.date().isoformat()}")
        print(f"[SPLIT] VAL_END_DATE   (inclusive): {val_end.date().isoformat()}")
        return SplitInfo(
            train_end_date=train_end.date().isoformat(),
            val_end_date=val_end.date().isoformat(),
            total_rows_with_valid_date_and_label=total,
        )

    _require(train_end is None and val_end is None,
             "Set both TRAIN_END_DATE and VAL_END_DATE, or neither (use fractions).")

    train_end2, val_end2, total = _compute_cutoffs_from_date_counts(date_counts)
    print("[SPLIT] Using fraction-based date cutoffs from game_date distribution.")
    print(f"[SPLIT] Fractions: train={TRAIN_FRAC:.2f}, val={VAL_FRAC:.2f}, test={TEST_FRAC:.2f}")
    print(f"[SPLIT] TRAIN_END_DATE (inclusive): {train_end2.date().isoformat()}")
    print(f"[SPLIT] VAL_END_DATE   (inclusive): {val_end2.date().isoformat()}")

    return SplitInfo(
        train_end_date=train_end2.date().isoformat(),
        val_end_date=val_end2.date().isoformat(),
        total_rows_with_valid_date_and_label=total,
    )

def build_feature_plan(header_cols: List[str]) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Decide which raw columns to keep as features (subject to FEATURE_MODE), and which are categorical/numeric/int.
    Returns:
      raw_categorical_cols, float_cols, int_cols, passthrough_cols, derived_cols
    """
    cols = list(header_cols)
    exclusions = _feature_mode_exclusions(cols)

    # Identify raw categoricals present (we'll encode to *_id)
    raw_cats = [c for c in CATEGORICAL_COLS_CANDIDATES if c in cols]
    # batter/pitcher treated as categorical IDs for embeddings (encoded)
    for c in ["batter", "pitcher"]:
        if c in cols and c not in raw_cats:
            raw_cats.append(c)

    # Int features present
    int_cols = [c for c in (STATE_INT_COLS + ID_LIKE_INT_COLS) if c in cols]
    # We'll also derive base flags and derived ints later.

    # Float candidates present (exclude estimated_* unless mode allows; exclusions applied later)
    float_cols = [c for c in FLOAT_COLS_CANDIDATES if c in cols]

    # Additional float columns: include any existing numeric-looking columns not in int/cat/always-exclude,
    # but keep it conservative to avoid pulling in lots of junk strings.
    # If you want broader inclusion, extend FLOAT_COLS_CANDIDATES.
    # (We still coerce unknown continuous columns to float only if you add them.)

    # Passthrough (non-feature) columns we keep for trace/debug in outputs
    passthrough = [c for c in ["game_date", "pa_id", "pitch_number_in_pa", "game_pk"] if c in cols]

    # Derived columns we may add if inputs exist
    derived = [
        "count_str",
        "is_two_strike",
        "is_three_ball",
        "on_1b_flag",
        "on_2b_flag",
        "on_3b_flag",
        "base_state",
        "runners_on",
        "is_first_pitch_of_pa",
        "abs_score_diff",
    ]

    # Apply FEATURE_MODE exclusions to raw feature sources
    raw_cats = [c for c in raw_cats if c not in exclusions]
    int_cols = [c for c in int_cols if c not in exclusions]
    float_cols = [c for c in float_cols if c not in exclusions]

    # Also exclude ALWAYS_EXCLUDE even if they appear in candidates
    raw_cats = [c for c in raw_cats if c not in ALWAYS_EXCLUDE]
    int_cols = [c for c in int_cols if c not in ALWAYS_EXCLUDE]
    float_cols = [c for c in float_cols if c not in ALWAYS_EXCLUDE]

    return raw_cats, float_cols, int_cols, passthrough, derived

def preprocess_chunk_base(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Base preprocessing that is common across passes:
    - validate required columns exist (already validated globally, but safe)
    - parse game_date -> datetime (drop NaT)
    - build canonical label from pa_outcome (drop missing/empty outcomes)
    - coerce key columns (ints/floats)
    - derive occupancy flags and derived features (vectorized)
    Returns (processed_df, drop_counts dict)
    """
    drop_counts = {
        "dropped_bad_date": 0,
        "dropped_bad_label": 0,
    }

    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if "game_date" not in df.columns:
        raise ValueError("Missing required column 'game_date'.")

    # Parse and normalize date
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    bad_date_mask = df["game_date"].isna()

    # Drop empty/NA pa_outcome
    outcome_norm = _normalize_str_series(df["pa_outcome"])
    bad_label_mask = outcome_norm.eq("")

    bad_mask = bad_date_mask | bad_label_mask
    if bad_mask.any():
        drop_counts["dropped_bad_date"] = int(bad_date_mask.sum())
        drop_counts["dropped_bad_label"] = int((~bad_date_mask & bad_label_mask).sum())
        df = df.loc[~bad_mask].copy()
    else:
        df = df.copy()

    if df.empty:
        return df, drop_counts

    # Build canonical label
    df["label"] = _map_labels(df["pa_outcome"])

    # Coerce ID-like ints
    _coerce_many_int(df, [c for c in ID_LIKE_INT_COLS if c in df.columns], fill=0, dtype="int64")

    # Coerce state ints
    _coerce_many_int(df, [c for c in STATE_INT_COLS if c in df.columns], fill=0, dtype="int64")

    # Coerce base occupancy runner IDs to int
    for c in BASE_OCCUPANCY_COLS:
        if c in df.columns:
            _coerce_int(df, c, fill=0, dtype="int64")

    # Coerce base occupancy flags (0/1 ints) if present
    for c in BASE_OCCUPIED_FLAG_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    # Flags (prefer base_*_occupied if present; else fall back to runner id > 0)
    if "base_1b_occupied" in df.columns:
        df["on_1b_flag"] = (df["base_1b_occupied"] != 0).astype("int8")
    elif "on_1b" in df.columns:
        df["on_1b_flag"] = (df["on_1b"] > 0).astype("int8")
    else:
        df["on_1b_flag"] = 0

    if "base_2b_occupied" in df.columns:
        df["on_2b_flag"] = (df["base_2b_occupied"] != 0).astype("int8")
    elif "on_2b" in df.columns:
        df["on_2b_flag"] = (df["on_2b"] > 0).astype("int8")
    else:
        df["on_2b_flag"] = 0

    if "base_3b_occupied" in df.columns:
        df["on_3b_flag"] = (df["base_3b_occupied"] != 0).astype("int8")
    elif "on_3b" in df.columns:
        df["on_3b_flag"] = (df["on_3b"] > 0).astype("int8")
    else:
        df["on_3b_flag"] = 0

    # Float coercions (only for known candidates present)
    float_cols_present = [c for c in FLOAT_COLS_CANDIDATES if c in df.columns]
    _coerce_many_float(df, float_cols_present)

    # Derived: count_str, is_two_strike, is_three_ball
    if "balls" in df.columns and "strikes" in df.columns:
        df["count_str"] = df["balls"].astype("int64").astype("string") + "-" + df["strikes"].astype("int64").astype("string")
        df["is_two_strike"] = (df["strikes"] == 2).astype("int8")
        df["is_three_ball"] = (df["balls"] == 3).astype("int8")
    else:
        df["count_str"] = ""
        df["is_two_strike"] = 0
        df["is_three_ball"] = 0

    # Derived: base_state (0-7), runners_on
    df["base_state"] = (df["on_1b_flag"].astype("int8")
                        + 2 * df["on_2b_flag"].astype("int8")
                        + 4 * df["on_3b_flag"].astype("int8")).astype("int8")
    df["runners_on"] = (df["on_1b_flag"] + df["on_2b_flag"] + df["on_3b_flag"]).astype("int8")

    # Derived: is_first_pitch_of_pa
    if "pitch_number_in_pa" in df.columns:
        df["is_first_pitch_of_pa"] = (df["pitch_number_in_pa"] == 1).astype("int8")
    else:
        df["is_first_pitch_of_pa"] = 0

    # Derived: abs_score_diff
    if "score_diff_bat_minus_fld" in df.columns:
        df["abs_score_diff"] = df["score_diff_bat_minus_fld"].abs()
    else:
        df["abs_score_diff"] = np.nan

    return df, drop_counts

def pass2_fit_train_stats(
    input_csv: str,
    split_info: SplitInfo,
    raw_categorical_cols: List[str],
    float_cols: List[str],
    int_cols: List[str],
    derived_cols: List[str],
) -> FitStats:
    """
    Second pass: compute train-only label counts (canonical),
    train-only categorical vocabularies, and train-only numeric imputation stats (mean).
    """
    print("[PASS 2] Fitting train-only stats (vocabularies, numeric imputations, label counts)...")
    train_end = pd.to_datetime(split_info.train_end_date)
    val_end = pd.to_datetime(split_info.val_end_date)

    exclusions = _feature_mode_exclusions(list(pd.read_csv(input_csv, nrows=0).columns))

    # Prepare train-only stats containers
    train_label_counts = Counter()

    # vocab dict per categorical raw col
    categorical_vocabs: Dict[str, Dict[str, int]] = {}
    next_id: Dict[str, int] = {}

    for c in raw_categorical_cols:
        categorical_vocabs[c] = {}  # normalized string -> id (>=2)
        next_id[c] = 2

    # numeric mean stats (sum, count non-nan)
    num_sum = defaultdict(float)
    num_cnt = defaultdict(int)

    # To avoid including excluded columns in stats
    float_cols_fit = [c for c in float_cols if c not in exclusions and c not in ALWAYS_EXCLUDE]
    # include derived float cols like abs_score_diff if present (we'll treat it as numeric if it exists)
    # (abs_score_diff is derived and always created; keep as numeric)
    if "abs_score_diff" not in float_cols_fit:
        float_cols_fit = float_cols_fit + ["abs_score_diff"]

    # Derived + int features that matter are not imputed (ints already filled), but are part of final features.
    rows_read = 0
    rows_train = 0
    dropped_bad_date = 0
    dropped_bad_label = 0

    for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=READ_CHUNK_ROWS, low_memory=True)):
        rows_read += len(chunk)

        df, drops = preprocess_chunk_base(chunk)
        dropped_bad_date += drops["dropped_bad_date"]
        dropped_bad_label += drops["dropped_bad_label"]

        if df.empty:
            continue

        # Assign split
        split = _assign_split(df["game_date"], train_end, val_end)
        df_train = df.loc[split == "train"].copy()
        if df_train.empty:
            continue

        rows_train += len(df_train)

        # Train label counts (canonical, before rare remap)
        train_label_counts.update(df_train["label"].astype(str).tolist())

        # Fit vocabs on train
        for c in raw_categorical_cols:
            if c not in df_train.columns:
                continue
            # Special: batter/pitcher are ints -> treat as categorical IDs via string
            if c in ("batter", "pitcher"):
                s = df_train[c].astype("int64").astype("string")
                s = s.fillna("").str.strip()
                # do not lower numeric strings
                s_norm = s
            else:
                s_norm = _normalize_str_series(df_train[c])

            # Unique non-empty values
            u = pd.unique(s_norm)
            u = u[u != ""]
            if len(u) == 0:
                continue

            vocab = categorical_vocabs[c]
            nid = next_id[c]
            # Add unseen categories
            for val in u.tolist():
                if val not in vocab:
                    vocab[val] = nid
                    nid += 1
            next_id[c] = nid

        # Numeric mean stats on train only
        for c in float_cols_fit:
            if c not in df_train.columns:
                continue
            s = df_train[c]
            # s is float64 for coerced cols; abs_score_diff is float
            cnt = int(s.count())
            if cnt == 0:
                continue
            num_cnt[c] += cnt
            num_sum[c] += float(s.sum(skipna=True))

        if (i + 1) % 10 == 0:
            print(f"  - processed {rows_read:,} rows... (train seen: {rows_train:,})")

    # Compute numeric imputations (mean; default 0.0 if no train observations)
    numeric_impute: Dict[str, float] = {}
    for c in float_cols_fit:
        if num_cnt.get(c, 0) > 0:
            numeric_impute[c] = num_sum[c] / num_cnt[c]
        else:
            numeric_impute[c] = 0.0

    # Rare label handling based on train counts
    rare_labels = sorted([lbl for lbl, cnt in train_label_counts.items() if cnt < MIN_LABEL_COUNT])
    final_labels_set = set(train_label_counts.keys()) - set(rare_labels)
    final_labels_set.add(RARE_LABEL_BUCKET)
    final_labels = sorted(final_labels_set)

    # Build final feature list (ordered)
    # - int features (present or derived)
    # - numeric float features (present or derived)
    # - categorical encoded id columns
    # Keep raw categoricals out unless KEEP_RAW_CATEGORICALS=True
    derived_ints = [
        "is_two_strike",
        "is_three_ball",
        "on_1b_flag",
        "on_2b_flag",
        "on_3b_flag",
        "base_state",
        "runners_on",
        "is_first_pitch_of_pa",
    ]
    derived_float = ["abs_score_diff"]

    categorical_id_cols = sorted({f"{c}_id" for c in raw_categorical_cols})

    # Choose int features: provided + derived (exclude ids we encode separately? we keep batter/pitcher ints out and use *_id)
    int_feature_cols = []
    for c in int_cols:
        if c in ("batter", "pitcher"):
            continue
        if c in ALWAYS_EXCLUDE or c in exclusions:
            continue
        int_feature_cols.append(c)
    for c in derived_ints:
        int_feature_cols.append(c)
    # also include pitch_number_in_pa already in STATE_INT_COLS but is useful as feature; kept above.

    # Numeric features (floats) - exclude estimated_* unless allowed already; already handled via float_cols list
    numeric_cols = []
    for c in float_cols:
        if c in ALWAYS_EXCLUDE or c in exclusions:
            continue
        numeric_cols.append(c)
    for c in derived_float:
        if c not in numeric_cols:
            numeric_cols.append(c)

    # Missing indicators (only for those numeric cols we actually include)
    missing_indicator_cols = [f"{c}_missing" for c in MISSING_INDICATOR_NUMERICS if c in numeric_cols]

    # Final feature list order
    feature_list: List[str] = []
    feature_list += int_feature_cols
    feature_list += numeric_cols
    feature_list += missing_indicator_cols
    feature_list += categorical_id_cols
    if KEEP_RAW_CATEGORICALS:
        feature_list += [c for c in raw_categorical_cols if c in CATEGORICAL_COLS_CANDIDATES]

    # De-dup while preserving order
    seen = set()
    feature_list = [x for x in feature_list if not (x in seen or seen.add(x))]

    print("[PASS 2] Done.")
    print(f"[PASS 2] Train rows observed (valid date+label): {rows_train:,}")
    print(f"[PASS 2] Canonical labels in train: {len(train_label_counts)}")
    print(f"[PASS 2] Rare labels (<{MIN_LABEL_COUNT}) remapped to {RARE_LABEL_BUCKET}: {len(rare_labels)}")
    print(f"[PASS 2] Final label count: {len(final_labels)}")
    print(f"[PASS 2] Final feature count: {len(feature_list)}")
    print(f"[PASS 2] Total vocab sizes: " +
          ", ".join([f"{k}={len(v)}" for k, v in categorical_vocabs.items()]))

    return FitStats(
        train_label_counts=dict(train_label_counts),
        rare_labels=rare_labels,
        final_labels=final_labels,
        categorical_vocabs=categorical_vocabs,
        numeric_impute=numeric_impute,
        feature_list=feature_list,
        categorical_raw_cols=raw_categorical_cols,
        categorical_id_cols=categorical_id_cols,
        numeric_cols=numeric_cols,
        int_feature_cols=int_feature_cols,
        derived_cols=derived_cols,
    )

def _encode_categorical_series(s: pd.Series, vocab: Dict[str, int], treat_as_numeric_id: bool) -> pd.Series:
    """
    Encode a categorical series using vocab built on train only:
      0 = PAD/NA, 1 = UNK, >=2 = known categories
    Vectorized mapping.
    """
    if treat_as_numeric_id:
        # s is expected to be integer-like (already coerced)
        s_norm = s.astype("int64").astype("string").fillna("").str.strip()
    else:
        s_norm = _normalize_str_series(s)

    # map known -> id; unknown -> NaN; then fill unknown->1
    mapped = s_norm.map(vocab)

    # Identify NA/PAD where original normalized is empty
    is_pad = s_norm.eq("")
    # Fill unknowns with 1
    out = mapped.fillna(1).astype("int32")
    # Set PAD to 0
    out = out.mask(is_pad, 0).astype("int32")
    return out

def _enforce_output_dtypes(
    df: pd.DataFrame,
    numeric_cols: List[str],
    int_cols: List[str],
    derived_int8_cols: List[str],
) -> pd.DataFrame:
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    for c in derived_int8_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")

    # Any missing-indicator columns
    missing_cols = [c for c in df.columns if c.endswith("_missing")]
    for c in missing_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int8")

    # Any categorical encoded columns
    id_cols = [c for c in df.columns if c.endswith("_id")]
    for c in id_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int32")

    return df

def pass3_transform_and_write(
    input_csv: str,
    split_info: SplitInfo,
    fit: FitStats,
    passthrough_cols: List[str],
    output_format: str,
) -> Dict[str, object]:
    """
    Third pass: apply rare-label remap, categorical encoding (train vocab), numeric imputation (train mean),
    and write split outputs streamingly.
    """
    print("[PASS 3] Transforming chunks and writing outputs...")
    train_end = pd.to_datetime(split_info.train_end_date)
    val_end = pd.to_datetime(split_info.val_end_date)
    exclusions = _feature_mode_exclusions(list(pd.read_csv(input_csv, nrows=0).columns))

    # Prepare output directory (fresh)
    _safe_mkdir(OUTPUT_DIR)
    # Remove existing split outputs to avoid accidental append across runs
    for name in ("train", "val", "test"):
        for ext in (".parquet", ".csv.gz"):
            p = os.path.join(OUTPUT_DIR, f"{name}{ext}")
            if os.path.exists(p):
                os.remove(p)

    # Output columns: passthrough + label + features
    # Keep game_date for trace and to confirm split logic.
    base_cols = []
    for c in ["game_date", "pa_id", "pitch_number_in_pa", "game_pk"]:
        if c in passthrough_cols and c not in base_cols:
            base_cols.append(c)
    out_cols = base_cols + ["label"] + fit.feature_list

    writers = {
        "train": SplitWriter(OUTPUT_DIR, output_format, "train", out_cols),
        "val": SplitWriter(OUTPUT_DIR, output_format, "val", out_cols),
        "test": SplitWriter(OUTPUT_DIR, output_format, "test", out_cols),
    }

    # Tracking
    rows_read = 0
    dropped_bad_date = 0
    dropped_bad_label = 0
    rows_written = Counter()
    label_counts_by_split = {"train": Counter(), "val": Counter(), "test": Counter()}

    rare_set = set(fit.rare_labels)

    # For faster checks
    numeric_cols = [c for c in fit.numeric_cols if c not in exclusions and c not in ALWAYS_EXCLUDE]
    int_cols = [c for c in fit.int_feature_cols if c not in exclusions and c not in ALWAYS_EXCLUDE]

    derived_int8_cols = [
        "is_two_strike",
        "is_three_ball",
        "on_1b_flag",
        "on_2b_flag",
        "on_3b_flag",
        "base_state",
        "runners_on",
        "is_first_pitch_of_pa",
    ]
    int_cols_non_derived = [c for c in int_cols if c not in set(derived_int8_cols)]

    # Missing indicator base list (only if numeric col exists)
    missing_indicator_bases = [c for c in MISSING_INDICATOR_NUMERICS if c in numeric_cols]
    missing_indicator_cols = [f"{c}_missing" for c in missing_indicator_bases]

    for i, chunk in enumerate(pd.read_csv(input_csv, chunksize=READ_CHUNK_ROWS, low_memory=True)):
        rows_read += len(chunk)

        df, drops = preprocess_chunk_base(chunk)
        dropped_bad_date += drops["dropped_bad_date"]
        dropped_bad_label += drops["dropped_bad_label"]

        if df.empty:
            continue

        # Apply FEATURE_MODE exclusions by dropping excluded columns if present
        # (This is safe even if we don't use them as features; it prevents accidental inclusion.)
        drop_cols = [c for c in df.columns if c in exclusions or c.startswith(ESTIMATED_PREFIX)]
        # But in batted_ball mode, keep estimated_*
        if FEATURE_MODE == "batted_ball":
            drop_cols = [c for c in df.columns if c in exclusions]
        if drop_cols:
            df = df.drop(columns=drop_cols, errors="ignore")

        # Assign split
        split = _assign_split(df["game_date"], train_end, val_end)
        df["__split__"] = split

        # Rare-label remap (based on train counts only)
        df["label"] = df["label"].mask(df["label"].isin(rare_set), RARE_LABEL_BUCKET)

        # Ensure required int features exist; fill if missing
        for c in int_cols:
            if c not in df.columns:
                df[c] = 0
        # Ensure numeric features exist; set NaN if missing so we can impute
        for c in numeric_cols:
            if c not in df.columns:
                df[c] = np.nan

        # Missing indicators BEFORE imputation
        for base in missing_indicator_bases:
            df[f"{base}_missing"] = df[base].isna().astype("int8")

        # Numeric imputation (mean fit on train)
        # Use fit.numeric_impute values; default to 0.0
        for c in numeric_cols:
            imp = float(fit.numeric_impute.get(c, 0.0))
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
            df[c] = df[c].fillna(imp)

        # Int feature dtype enforcement (already mostly ints; re-coerce to be safe)
        for c in int_cols_non_derived:
            _coerce_int(df, c, fill=0, dtype="int64")

        # Encode categoricals
        for raw_c in fit.categorical_raw_cols:
            if raw_c not in df.columns:
                # create PAD-only
                df[f"{raw_c}_id"] = 0
                continue
            treat_as_id = raw_c in ("batter", "pitcher")
            # For batter/pitcher, ensure int coercion first
            if treat_as_id:
                _coerce_int(df, raw_c, fill=0, dtype="int64")

            vocab = fit.categorical_vocabs.get(raw_c, {})
            encoded = _encode_categorical_series(df[raw_c], vocab, treat_as_numeric_id=treat_as_id)
            df[f"{raw_c}_id"] = encoded

        # Write each split chunk
        for split_name in ("train", "val", "test"):
            part = df.loc[df["__split__"] == split_name, out_cols].copy()
            if part.empty:
                continue
            part = _enforce_output_dtypes(part, numeric_cols=numeric_cols, int_cols=int_cols_non_derived, derived_int8_cols=derived_int8_cols)
            writers[split_name].write_df(part)
            rows_written[split_name] += len(part)
            label_counts_by_split[split_name].update(part["label"].astype(str).tolist())

        if (i + 1) % 10 == 0:
            print(f"  - processed {rows_read:,} rows... written so far: "
                  f"train={rows_written['train']:,}, val={rows_written['val']:,}, test={rows_written['test']:,}")

    for w in writers.values():
        w.close()

    print("[PASS 3] Done writing.")
    print(f"[PASS 3] Rows read: {rows_read:,}")
    print(f"[PASS 3] Dropped bad dates: {dropped_bad_date:,}")
    print(f"[PASS 3] Dropped missing/empty pa_outcome: {dropped_bad_label:,}")
    print(f"[PASS 3] Rows written: train={rows_written['train']:,}, val={rows_written['val']:,}, test={rows_written['test']:,}")

    return {
        "rows_read": int(rows_read),
        "dropped_bad_date": int(dropped_bad_date),
        "dropped_bad_label": int(dropped_bad_label),
        "rows_written": {k: int(v) for k, v in rows_written.items()},
        "label_counts_by_split": {
            k: dict(v) for k, v in label_counts_by_split.items()
        },
        "output_columns": out_cols,
    }

def _print_label_top(counter: Counter, n: int = 15) -> str:
    items = counter.most_common(n)
    return ", ".join([f"{k}:{v}" for k, v in items])

def write_metadata_json(
    output_dir: str,
    output_format: str,
    split_info: SplitInfo,
    fit: FitStats,
    transform_summary: Dict[str, object],
) -> None:
    meta = {
        "created_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": {
            "INPUT_CSV": INPUT_CSV,
            "READ_CHUNK_ROWS": READ_CHUNK_ROWS,
            "OUTPUT_DIR": OUTPUT_DIR,
            "OUTPUT_FORMAT_RESOLVED": output_format,
            "FEATURE_MODE": FEATURE_MODE,
            "SPLIT_MODE": SPLIT_MODE,
            "TRAIN_END_DATE": TRAIN_END_DATE,
            "VAL_END_DATE": VAL_END_DATE,
            "TRAIN_FRAC": TRAIN_FRAC,
            "VAL_FRAC": VAL_FRAC,
            "TEST_FRAC": TEST_FRAC,
            "MIN_LABEL_COUNT": MIN_LABEL_COUNT,
            "RANDOM_SEED": RANDOM_SEED,
            "WRITE_METADATA_JSON": WRITE_METADATA_JSON,
            "KEEP_RAW_CATEGORICALS": KEEP_RAW_CATEGORICALS,
        },
        "split": asdict(split_info),
        "label_mapping": {
            "raw_to_canonical": RAW_TO_CANON_LABEL,
            "unrecognized_bucket": UNRECOGNIZED_LABEL_BUCKET,
            "rare_bucket": RARE_LABEL_BUCKET,
            "min_label_count": MIN_LABEL_COUNT,
            "train_label_counts_canonical": fit.train_label_counts,
            "rare_labels_remapped_to_other": fit.rare_labels,
            "final_label_set": fit.final_labels,
        },
        "features": {
            "feature_list_ordered": fit.feature_list,
            "categorical_raw_cols": fit.categorical_raw_cols,
            "categorical_id_cols": fit.categorical_id_cols,
            "numeric_cols": fit.numeric_cols,
            "int_feature_cols": fit.int_feature_cols,
            "derived_cols": fit.derived_cols,
            "missing_indicator_numerics": MISSING_INDICATOR_NUMERICS,
        },
        "categorical_vocabs": {
            "reserved_ids": {"PAD_NA": 0, "UNK": 1},
            "vocabs": fit.categorical_vocabs,
            "vocab_sizes": {k: len(v) for k, v in fit.categorical_vocabs.items()},
        },
        "numeric_imputation": {
            "strategy": "mean",
            "values": fit.numeric_impute,
        },
        "transform_summary": transform_summary,
    }

    path = os.path.join(output_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=False)
    print(f"[META] Wrote {path}")

# =========================
# Main
# =========================

def main() -> None:
    np.random.seed(RANDOM_SEED)
    _validate_feature_mode()
    _validate_split_mode()

    output_format = _resolve_output_format()

    _require(os.path.exists(INPUT_CSV), f"Input CSV not found: {INPUT_CSV!r}")

    # PASS 1: date histogram
    date_counts, header_cols = pass1_date_histogram(INPUT_CSV)

    # Determine split cutoffs
    split_info = determine_split_info(date_counts)
    train_end = pd.to_datetime(split_info.train_end_date)
    val_end = pd.to_datetime(split_info.val_end_date)

    # Quick row-count estimate per split using date histogram (row-weighted)
    train_rows_est = sum(cnt for d, cnt in date_counts.items() if d <= train_end)
    val_rows_est = sum(cnt for d, cnt in date_counts.items() if (d > train_end and d <= val_end))
    test_rows_est = sum(cnt for d, cnt in date_counts.items() if d > val_end)
    print("[SPLIT] Estimated rows by date histogram:")
    print(f"  train={train_rows_est:,}, val={val_rows_est:,}, test={test_rows_est:,}, total={train_rows_est+val_rows_est+test_rows_est:,}")

    # Feature plan
    raw_cats, float_cols, int_cols, passthrough_cols, derived_cols = build_feature_plan(header_cols)

    print("[FEATURES] Plan:")
    print(f"  FEATURE_MODE={FEATURE_MODE}")
    print(f"  raw categorical cols (to encode): {raw_cats}")
    print(f"  float numeric cols (candidates):  {float_cols}")
    print(f"  int cols (candidates):            {int_cols}")
    print(f"  passthrough cols:                 {passthrough_cols}")

    # PASS 2: fit train-only stats (label counts, vocabs, imputations)
    fit = pass2_fit_train_stats(
        INPUT_CSV,
        split_info,
        raw_cats,
        float_cols,
        int_cols,
        derived_cols,
    )

    # PASS 3: transform & write
    transform_summary = pass3_transform_and_write(
        INPUT_CSV,
        split_info,
        fit,
        passthrough_cols,
        output_format,
    )

    # Print final summary with label distributions
    label_counts_by_split = {
        k: Counter(v) for k, v in transform_summary["label_counts_by_split"].items()
    }
    print("\n[SUMMARY]")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Output format: {output_format}")
    print(f"  Train end date (inclusive): {split_info.train_end_date}")
    print(f"  Val end date (inclusive):   {split_info.val_end_date}")
    print(f"  Features: {len(fit.feature_list)}")
    print(f"  Labels (final): {len(fit.final_labels)} => {fit.final_labels}")

    for s in ("train", "val", "test"):
        c = label_counts_by_split.get(s, Counter())
        total = sum(c.values())
        print(f"  {s}: rows={transform_summary['rows_written'].get(s, 0):,}, labels_top={_print_label_top(c, n=15)}")
        if total == 0:
            print(f"    [WARN] {s} split has 0 rows written.")

    # Write metadata
    if WRITE_METADATA_JSON:
        write_metadata_json(OUTPUT_DIR, output_format, split_info, fit, transform_summary)

    print("\n[OK] Preprocessing complete.")

if __name__ == "__main__":
    main()
