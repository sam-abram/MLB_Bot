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
START_DT = "2023-03-30"   # inclusive, YYYY-MM-DD
END_DT = "2025-09-28"     # inclusive, YYYY-MM-DD
CHUNK_DAYS = 7            # inclusive chunk length
# data_pipeline2.py (top config)
OUTPUT_CSV = "statcast_pitches.csv"

OUTPUT_PA_CSV = "test9_statcast_pa_level.csv"  # new: optional convenience output
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
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            df = statcast(start_dt=start_dt, end_dt=end_dt)
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
    return pd.to_numeric(s, errors="coerce").fillna(0).astype("int64")


def _build_output_columns(df: pd.DataFrame) -> List[str]:
    id_cols = [
        "game_date",
        "game_pk",
        "at_bat_number",
        "pitch_number",
        "pitch_number_in_pa",
        "pa_pitch_count",
        "is_last_pitch_of_pa",
        "pa_id",
    ]
    label_cols = ["pa_outcome"]

    player_cols = ["batter", "batter_id", "pitcher", "pitcher_id", "stand", "p_throws", "stadium_id"]



    park_candidates = ["park", "park_id", "stadium", "venue_name"]
    park_col = _first_present(df, park_candidates)
    park_cols: List[str] = []
    if park_col is not None:
        park_cols = [park_col]
    else:
        if "park_proxy_home_team" in df.columns:
            park_cols = ["park_proxy_home_team"]

    defense_cols = [f"fielder_{i}" for i in range(2, 10) if f"fielder_{i}" in df.columns]

    situation_cols = [
        "balls",
        "strikes",
        "outs_when_up",
        "inning",
        "inning_topbot",
        "bat_score",
        "fld_score",
        "score_diff_bat_minus_fld",
        "base_1b_occupied",
        "base_2b_occupied",
        "base_3b_occupied",
        "pitcher_game_pitch_count",
    ]

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

    extra_sz_cols = [
        c for c in df.columns
        if (c.startswith("sz_") or ("strike_zone" in c))
        and c not in pitch_char_cols_base
    ]
    pitch_char_cols = pitch_char_cols_base + extra_sz_cols

    pitch_result_cols = ["description", "type", "bb_type", "hit_location", "outs_on_play", "des"]

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

    derived_always = {
        "pa_id",
        "pa_outcome",
        "pitch_number_in_pa",
        "pa_pitch_count",
        "is_last_pitch_of_pa",
    }
    derived_conditional = {
        "score_diff_bat_minus_fld",
        "pitcher_game_pitch_count",
        "base_1b_occupied",
        "base_2b_occupied",
        "base_3b_occupied",
    }

    out_cols: List[str] = []
    for c in ordered_groups:
        if c in df.columns:
            out_cols.append(c)
        elif c in derived_always:
            out_cols.append(c)
        elif c in derived_conditional:
            continue

    forbidden = {"_row_order", "on_1b", "on_2b", "on_3b"}
    out_cols = [c for c in out_cols if c not in forbidden]

    seen = set()
    out_cols_unique: List[str] = []
    for c in out_cols:
        if c not in seen:
            seen.add(c)
            out_cols_unique.append(c)
    return out_cols_unique


def _process_chunk(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if df is None or df.empty:
        return pd.DataFrame(), 0

    df = df.copy()
    df["_row_order"] = np.arange(len(df), dtype=np.int64)

    if "game_pk" not in df.columns or "at_bat_number" not in df.columns:
        raise KeyError("Downloaded dataframe missing required columns: game_pk and/or at_bat_number")

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

    df["pa_id"] = df["game_pk"].astype("int64").astype(str) + "|" + df["at_bat_number"].astype("int64").astype(str)

    df["pitch_number_in_pa"] = (df.groupby("pa_id", sort=False).cumcount() + 1).astype("int64")
    df["pa_pitch_count"] = df.groupby("pa_id", sort=False)["pitch_number_in_pa"].transform("max").astype("int64")
    df["is_last_pitch_of_pa"] = (df["pitch_number_in_pa"] == df["pa_pitch_count"]).astype("int8")

    if "game_pk" in df.columns and "pitcher" in df.columns:
        df["pitcher_game_pitch_count"] = (df.groupby(["game_pk", "pitcher"], sort=False).cumcount() + 1).astype("int64")

    if "on_1b" in df.columns:
        df["base_1b_occupied"] = df["on_1b"].notna().astype("int64")
    if "on_2b" in df.columns:
        df["base_2b_occupied"] = df["on_2b"].notna().astype("int64")
    if "on_3b" in df.columns:
        df["base_3b_occupied"] = df["on_3b"].notna().astype("int64")

    if "bat_score" in df.columns and "fld_score" in df.columns:
        bat = pd.to_numeric(df["bat_score"], errors="coerce")
        fld = pd.to_numeric(df["fld_score"], errors="coerce")
        df["score_diff_bat_minus_fld"] = bat - fld

    if _first_present(df, ["park", "park_id", "stadium", "venue_name"]) is None and "home_team" in df.columns:
        df["park_proxy_home_team"] = df["home_team"]
    # Stable stadium identifier for downstream modeling
    park_candidates = ["park", "park_id", "stadium", "venue_name", "park_proxy_home_team"]
    park_col = _first_present(df, park_candidates)
    if park_col is None:
        df["stadium_id"] = "__UNK__"
    else:
        df["stadium_id"] = df[park_col].astype("string").fillna("__UNK__").replace("", "__UNK__")

    for i in range(2, 10):
        c = f"fielder_{i}"
        if c in df.columns:
            df[c] = _coerce_int_id_series(df[c])
            # Add standardized id column names expected by preprocessing2.py
    if "batter_id" not in df.columns and "batter" in df.columns:
        df["batter_id"] = _coerce_int_id_series(df["batter"])
    if "pitcher_id" not in df.columns and "pitcher" in df.columns:
        df["pitcher_id"] = _coerce_int_id_series(df["pitcher"])


    if "events" in df.columns:
        out_ev = df["events"].astype("string")
        ev_norm = out_ev.str.lower().fillna("")

        non_pa = (
            ev_norm.str.startswith("pickoff")
            | ev_norm.str.contains("caught_stealing")
            | ev_norm.str.contains("stolen_base")
            | ev_norm.str.contains("wild_pitch")
            | ev_norm.str.contains("passed_ball")
        )

        outcomes = (
            df.loc[df["events"].notna() & ~non_pa, ["pa_id", "pitch_number_in_pa", "_row_order", "events"]]
            .sort_values(["pa_id", "pitch_number_in_pa", "_row_order"], kind="mergesort")
            .drop_duplicates(subset=["pa_id"], keep="last")
            [["pa_id", "events"]]
            .rename(columns={"events": "pa_outcome"})
        )

        df = df.merge(outcomes, on="pa_id", how="left")
    else:
        df["pa_outcome"] = pd.NA

    before = len(df)
    df = df.loc[df["pa_outcome"].notna()].copy()
    dropped = before - len(df)

    if "_row_order" in df.columns:
        df = df.drop(columns=["_row_order"])

    for c in ["on_1b", "on_2b", "on_3b"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    return df, dropped


def _build_pa_rows_from_pitch_chunk(df_proc: pd.DataFrame) -> pd.DataFrame:
    """
    Build 1 row per PA (for convenience; preprocessing does full PA dataset creation anyway).

    Columns:
      - game_date, pa_id, batter_id, pitcher_id, stadium_id, pitcher_fatigue, pa_outcome
    """
    if df_proc is None or df_proc.empty:
        return pd.DataFrame()

    df = df_proc
    if "pitch_number_in_pa" not in df.columns:
        return pd.DataFrame()

    first = df.loc[df["pitch_number_in_pa"] == 1].copy()
    if first.empty:
        return pd.DataFrame()

    # Dedup within chunk (safety)
    if "pa_id" in first.columns:
        first = first.drop_duplicates(subset=["pa_id"], keep="first")

    # Stadium extraction
    # Stadium extraction
    park_candidates = ["stadium_id", "park", "park_id", "stadium", "venue_name", "park_proxy_home_team"]
    park_col = _first_present(first, park_candidates)
    if park_col is None:
        first["stadium_id"] = "__UNK__"
    else:
        first["stadium_id"] = first[park_col].astype("string").fillna("__UNK__").replace("", "__UNK__")


    # Fatigue at PA start = pitches thrown before first pitch of this PA
    if "pitcher_game_pitch_count" in first.columns:
        ppc = pd.to_numeric(first["pitcher_game_pitch_count"], errors="coerce").fillna(0).astype("int64")
        first["pitcher_fatigue"] = (ppc - 1).clip(lower=0).astype("int64")
    else:
        first["pitcher_fatigue"] = 0

    bat_col = "batter_id" if "batter_id" in first.columns else ("batter" if "batter" in first.columns else None)
    pit_col = "pitcher_id" if "pitcher_id" in first.columns else ("pitcher" if "pitcher" in first.columns else None)
    if bat_col is None or pit_col is None:
        raise ValueError(
            f"Missing batter/pitcher id columns for PA output. Need one of "
            f"batter/batter_id and pitcher/pitcher_id. Found columns: {sorted(first.columns.tolist())}"
        )
    first["batter_id"] = pd.to_numeric(first[bat_col], errors="coerce").fillna(0).astype("int64")
    first["pitcher_id"] = pd.to_numeric(first[pit_col], errors="coerce").fillna(0).astype("int64")


    out = first[["game_date", "pa_id", "batter_id", "pitcher_id", "stadium_id", "pitcher_fatigue", "pa_outcome"]].copy()
    return out


def main() -> int:
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PA_CSV)) or ".", exist_ok=True)

    chunks = _daterange_chunks(START_DT, END_DT, CHUNK_DAYS)

    total_chunks = 0
    total_written = 0
    total_pa_written = 0
    total_dropped_unlabeled = 0

    wrote_header = False
    wrote_pa_header = False
    header_cols: Optional[List[str]] = None
    pa_header_cols: Optional[List[str]] = None

    print(f"Pitch-level output: {OUTPUT_CSV}")
    print(f"PA-level output:    {OUTPUT_PA_CSV}")
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

        chunk_cols = _build_output_columns(df_proc)

        if not wrote_header:
            header_cols = chunk_cols
        else:
            assert header_cols is not None
            for c in header_cols:
                if c not in df_proc.columns:
                    df_proc[c] = pd.NA
            chunk_cols = header_cols

        df_out = df_proc.loc[:, chunk_cols].copy()
        mode = "w" if not wrote_header else "a"
        df_out.to_csv(OUTPUT_CSV, mode=mode, header=(not wrote_header), index=False)
        wrote_header = True
        total_written += len(df_out)

        # PA-level optional output
        df_pa = _build_pa_rows_from_pitch_chunk(df_proc)
        if not df_pa.empty:
            pa_cols = ["game_date", "pa_id", "batter_id", "pitcher_id", "stadium_id", "pitcher_fatigue", "pa_outcome"]
            if not wrote_pa_header:
                pa_header_cols = pa_cols
            df_pa.to_csv(OUTPUT_PA_CSV, mode=("w" if not wrote_pa_header else "a"),
                         header=(not wrote_pa_header), index=False)
            wrote_pa_header = True
            total_pa_written += len(df_pa)

        print(f"  -> Wrote pitch rows: {len(df_out):,} ({'with header' if mode=='w' else 'appended'})")
        print(f"  -> Wrote PA rows:    {len(df_pa):,}\n")

    if not wrote_header:
        print("No output written (no data returned for the given date range).")
        print(f"Summary: chunks processed={total_chunks}, rows written=0, rows dropped(unlabeled)={total_dropped_unlabeled}, columns written=0")
        return 0

    assert header_cols is not None
    print("======== Summary ========")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Total pitch rows written: {total_written:,}")
    print(f"Total PA rows written:    {total_pa_written:,}")
    print(f"Total pitch rows dropped due to unlabeled PAs: {total_dropped_unlabeled:,}")
    print(f"Total columns written: {len(header_cols)}")
    print("=========================")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("\nInterrupted.")
  