
#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from pybaseball import statcast
except ImportError as e:
    raise SystemExit(
        "Missing dependency: pybaseball. Install with:\n  pip install pybaseball"
    ) from e


# =========================
# Config (edit these)
# =========================
START_DT = "2024-03-01"   # inclusive, YYYY-MM-DD
END_DT = "2024-10-31"     # inclusive, YYYY-MM-DD
CHUNK_DAYS = 31           # inclusive chunk length (default ~31)
OUTPUT_CSV = "statcast_pitch_level.csv"
# =========================


def _parse_date(dt_str: str) -> datetime.date:
    return datetime.strptime(dt_str, "%Y-%m-%d").date()


def _daterange_chunks(start_dt: str, end_dt: str, chunk_days: int) -> List[Tuple[str, str]]:
    start = _parse_date(start_dt)
    end = _parse_date(end_dt)
    if end < start:
        raise ValueError("END_DT must be >= START_DT")

    chunks: List[Tuple[str, str]] = []
    cur = start
    step = timedelta(days=chunk_days - 1)  # inclusive length
    one_day = timedelta(days=1)

    while cur <= end:
        chunk_end = min(cur + step, end)
        chunks.append((cur.isoformat(), chunk_end.isoformat()))
        cur = chunk_end + one_day

    return chunks


def _download_statcast_chunk(start_dt: str, end_dt: str, max_retries: int = 4, sleep_s: float = 3.0) -> pd.DataFrame:
    """
    Download a chunk with a small retry loop to be resilient to transient network issues.
    """
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            df = statcast(start_dt=start_dt, end_dt=end_dt)
            # Ensure pandas DataFrame
            if df is None:
                return pd.DataFrame()
            return df
        except BaseException as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(sleep_s * attempt)
            else:
                raise
    raise last_err  # pragma: no cover


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_int_id_series(s: pd.Series) -> pd.Series:
    # Coerce to numeric, fill missing with 0, cast to int
    return pd.to_numeric(s, errors="coerce").fillna(0).astype("int64")


def _build_output_columns(df: pd.DataFrame) -> List[str]:
    """
    Build columns in the required group order.
    Include only columns that exist in df, except derived columns that should exist once computed.
    """
    # (1) identifiers
    id_cols = [
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "pitch_number_in_pa",  # derived
        "pa_id",               # derived
    ]

    # (2) label
    label_cols = ["pa_outcome"]  # derived (after merge)

    # (3) player identity
    player_cols = ["batter", "pitcher", "stand", "p_throws"]

    # (4) stadium/park: first available OR park_proxy_home_team
    park_candidates = ["park", "park_id", "stadium", "venue_name"]
    park_col = _first_present(df, park_candidates)
    park_cols: List[str] = []
    if park_col is not None:
        park_cols = [park_col]
    else:
        # if none exist but home_team exists, write park_proxy_home_team and not home_team
        if "park_proxy_home_team" in df.columns:
            park_cols = ["park_proxy_home_team"]

    # (5) optional defense fielder_2..fielder_9
    defense_cols = [f"fielder_{i}" for i in range(2, 10) if f"fielder_{i}" in df.columns]

    # (6) pre-pitch state / game situation
    situation_cols = [
        "balls",
        "strikes",
        "outs_when_up",
        "inning",
        "inning_topbot",
        "bat_score",
        "fld_score",
        "score_diff_bat_minus_fld",  # derived when scores exist
        "base_1b_occupied",          # derived when on_1b exists
        "base_2b_occupied",          # derived when on_2b exists
        "base_3b_occupied",          # derived when on_3b exists
        "pitcher_game_pitch_count",  # derived when pitcher/game_pk exist
    ]

    # (7) observed pitch characteristics
    pitch_char_cols_base = [
        "pitch_type",
        "release_speed",
        "release_spin_rate",
        "spin_axis",
        "pfx_x",
        "pfx_z",
        "effective_speed",
        "release_extension",
        "release_pos_x",
        "release_pos_y",
        "release_pos_z",
        "plate_x",
        "plate_z",
        "zone",
        "sz_top",
        "sz_bot",
    ]

    # plus any extra strike-zone bound fields
    extra_sz_cols = [
        c for c in df.columns
        if (c.startswith("sz_") or ("strike_zone" in c))
        and c not in pitch_char_cols_base
    ]

    pitch_char_cols = pitch_char_cols_base + extra_sz_cols

    # (8) observed pitch result fields
    pitch_result_cols = [
        "description",
        "type",
        "bb_type",
        "hit_location",
        "outs_on_play",
        "des",
    ]

    # (9) observed batted-ball fields
    batted_ball_cols_base = [
        "launch_speed",
        "launch_angle",
        "launch_speed_angle",
        "hit_distance_sc",
        "hc_x",
        "hc_y",
        "spray_angle",
    ]
    estimated_cols = [c for c in df.columns if c.startswith("estimated_")]
    batted_ball_cols = batted_ball_cols_base + estimated_cols

    # Build ordered list, include only if present
    ordered_groups = (
        id_cols
        + label_cols
        + player_cols
        + park_cols
        + defense_cols
        + situation_cols
        + pitch_char_cols
        + pitch_result_cols
        + batted_ball_cols
    )

    # Derived columns should exist; otherwise, include only if present.
    derived_always = {"pa_id", "pa_outcome", "pitch_number_in_pa"}
    derived_conditional = {"score_diff_bat_minus_fld", "pitcher_game_pitch_count",
                           "base_1b_occupied", "base_2b_occupied", "base_3b_occupied"}

    out_cols: List[str] = []
    for c in ordered_groups:
        if c in df.columns:
            out_cols.append(c)
        elif c in derived_always:
            # should exist after derivation; include anyway (will KeyError if missing => signals bug)
            out_cols.append(c)
        elif c in derived_conditional:
            # include only if computed (i.e., exists in df)
            # (already handled by c in df.columns)
            continue
        else:
            continue

    # Ensure helper columns never leak
    forbidden = {"_row_order", "on_1b", "on_2b", "on_3b"}
    out_cols = [c for c in out_cols if c not in forbidden]

    # De-duplicate while preserving order
    seen = set()
    out_cols_unique: List[str] = []
    for c in out_cols:
        if c not in seen:
            seen.add(c)
            out_cols_unique.append(c)

    return out_cols_unique


def _process_chunk(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Process one chunk:
      - stable ordering + helper _row_order
      - pa_id, pitch_number_in_pa, pitcher_game_pitch_count
      - base occupancy flags (and ensure raw runner IDs not output)
      - score_diff_bat_minus_fld
      - pa_outcome via last terminal events row per PA, then drop unlabeled PAs
    Returns processed df and count of dropped rows due to missing pa_outcome.
    """
    if df is None or df.empty:
        return pd.DataFrame(), 0

    # Stable deterministic tie-breaker
    df = df.copy()
    df["_row_order"] = np.arange(len(df), dtype=np.int64)

    # Require identifiers for PA construction
    if "game_pk" not in df.columns or "at_bat_number" not in df.columns:
        raise KeyError("Downloaded dataframe missing required columns: game_pk and/or at_bat_number")

    # Sort (stable) as specified, using _row_order to break ties deterministically
    if "pitch_number" in df.columns:
        df = df.sort_values(
            by=["game_pk", "at_bat_number", "pitch_number", "_row_order"],
            kind="mergesort",
            na_position="last",
        )
    else:
        df = df.sort_values(
            by=["game_pk", "at_bat_number", "_row_order"],
            kind="mergesort",
            na_position="last",
        )

    # Vectorized pa_id = game_pk|at_bat_number
    df["pa_id"] = df["game_pk"].astype("int64").astype(str) + "|" + df["at_bat_number"].astype("int64").astype(str)

    # pitch_number_in_pa logic
    cc_in_pa = df.groupby("pa_id", sort=False).cumcount() + 1
    if "pitch_number" not in df.columns:
        df["pitch_number_in_pa"] = cc_in_pa.astype("int64")
    else:
        pn = pd.to_numeric(df["pitch_number"], errors="coerce")
        # If pitch_number exists but has some missing values, fill only missing with cumcount+1
        df["pitch_number_in_pa"] = pn.where(pn.notna(), cc_in_pa).astype("int64")

    # pitcher_game_pitch_count
    if "game_pk" in df.columns and "pitcher" in df.columns:
        df["pitcher_game_pitch_count"] = (
            df.groupby(["game_pk", "pitcher"], sort=False).cumcount() + 1
        ).astype("int64")

    # Base occupancy flags (0/1 ints) when source columns exist
    if "on_1b" in df.columns:
        df["base_1b_occupied"] = df["on_1b"].notna().astype("int64")
    if "on_2b" in df.columns:
        df["base_2b_occupied"] = df["on_2b"].notna().astype("int64")
    if "on_3b" in df.columns:
        df["base_3b_occupied"] = df["on_3b"].notna().astype("int64")

    # score_diff_bat_minus_fld
    if "bat_score" in df.columns and "fld_score" in df.columns:
        bat = pd.to_numeric(df["bat_score"], errors="coerce")
        fld = pd.to_numeric(df["fld_score"], errors="coerce")
        df["score_diff_bat_minus_fld"] = bat - fld

    # Park proxy if needed: if none of park/park_id/stadium/venue_name exist but home_team exists
    if _first_present(df, ["park", "park_id", "stadium", "venue_name"]) is None and "home_team" in df.columns:
        df["park_proxy_home_team"] = df["home_team"]
        # ensure original home_team is not written (we'll exclude it in selection)

    # Optional defense coercion: fielder_2..fielder_9 if present
    for i in range(2, 10):
        c = f"fielder_{i}"
        if c in df.columns:
            df[c] = _coerce_int_id_series(df[c])

    # pa_outcome: terminal events label per PA
    if "events" in df.columns:
        outcomes = (
            df.loc[df["events"].notna(), ["pa_id", "events"]]
            .drop_duplicates(subset=["pa_id"], keep="last")
            .rename(columns={"events": "pa_outcome"})
        )
        df = df.merge(outcomes, on="pa_id", how="left")
    else:
        # No events column => cannot label; will drop all rows
        df["pa_outcome"] = pd.NA

    before = len(df)
    df = df.loc[df["pa_outcome"].notna()].copy()
    dropped = before - len(df)

    # Never allow helper columns to remain in final output
    # (We'll also exclude them during column selection, but drop here for safety.)
    if "_row_order" in df.columns:
        df = df.drop(columns=["_row_order"])

    # Ensure runner ID columns are never output
    for c in ["on_1b", "on_2b", "on_3b"]:
        # keep them in df if you want for debugging? Requirement says never written; not necessarily drop.
        # We'll drop to reduce accidental leakage.
        if c in df.columns:
            df = df.drop(columns=[c])

    return df, dropped


def main() -> int:
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)) or ".", exist_ok=True)

    chunks = _daterange_chunks(START_DT, END_DT, CHUNK_DAYS)

    total_chunks = 0
    total_written = 0
    total_dropped_unlabeled = 0

    wrote_header = False
    header_cols: Optional[List[str]] = None

    print(f"Output: {OUTPUT_CSV}")
    print(f"Date range: {START_DT} to {END_DT} (inclusive)")
    print(f"Chunk size: {CHUNK_DAYS} days (inclusive)\n")

    for (chunk_start, chunk_end) in chunks:
        total_chunks += 1
        print(f"[Chunk {total_chunks}/{len(chunks)}] Downloading {chunk_start} to {chunk_end} ...", flush=True)

        df_raw = _download_statcast_chunk(chunk_start, chunk_end)
        if df_raw is None or df_raw.empty:
            print(f"  -> Downloaded 0 rows. Skipping write.\n")
            continue

        df_proc, dropped = _process_chunk(df_raw)
        total_dropped_unlabeled += dropped

        print(f"  -> Downloaded rows: {len(df_raw):,}")
        print(f"  -> Dropped rows (missing pa_outcome): {dropped:,}")
        print(f"  -> Rows remaining to write: {len(df_proc):,}")

        if df_proc.empty:
            print(f"  -> No rows to write for this chunk.\n")
            continue

        # Build columns to write for this chunk
        chunk_cols = _build_output_columns(df_proc)

        # Establish output schema from FIRST written chunk to keep a valid single CSV.
        # If later chunks are missing some of these columns, we create them as NA to match the header.
        if not wrote_header:
            header_cols = chunk_cols
        else:
            assert header_cols is not None
            # Add any missing header columns as NA (only those already in the established schema).
            for c in header_cols:
                if c not in df_proc.columns:
                    df_proc[c] = pd.NA
            # If the chunk has extra columns not in the header schema, ignore them.
            chunk_cols = header_cols

        # Ensure correct order and no accidental helper columns
        df_out = df_proc.loc[:, chunk_cols].copy()

        mode = "w" if not wrote_header else "a"
        df_out.to_csv(OUTPUT_CSV, mode=mode, header=(not wrote_header), index=False)

        wrote_header = True
        total_written += len(df_out)

        print(f"  -> Wrote {len(df_out):,} rows ({'with header' if mode=='w' else 'appended'})\n")

    if not wrote_header:
        print("No output written (no data returned for the given date range).")
        print(f"Summary: chunks processed={total_chunks}, rows written=0, rows dropped(unlabeled)={total_dropped_unlabeled}, columns written=0")
        return 0

    assert header_cols is not None
    print("======== Summary ========")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Total pitch rows written: {total_written:,}")
    print(f"Total pitch rows dropped due to unlabeled PAs: {total_dropped_unlabeled:,}")
    print(f"Total columns written: {len(header_cols)}")
    print("=========================")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted.")

