"""
build_caches.py — Rebuild the player name/ID and handedness caches from the
raw statcast CSV.

preprocessing2.py writes everything else in preprocessed/, but these two files
are built lazily by livematchup.py on a developer machine and then frozen
(name_id_cache.json carries a "built": true flag that short-circuits the
rebuild). In CI nothing regenerates them, so the site's player search stays
pinned to whatever roster existed when the cache was first created.

Handedness comes from the CSV's stand/p_throws columns. Names do not: the
statcast download has no name columns (batter/pitcher hold numeric IDs), so
they are resolved against the MLB StatsAPI people endpoint by ID.

Run after preprocessing2.py, before syncing preprocessed/ to S3.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

import pandas as pd

READ_CHUNK_ROWS = 500_000
USECOLS = ["batter_id", "pitcher_id", "stand", "p_throws"]

PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people?personIds={ids}"
PEOPLE_BATCH = 200


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _fetch_names(ids: Iterable[int], retries: int = 3) -> Dict[int, str]:
    """Resolve MLBAM ids -> full names via the public StatsAPI, batched."""
    ids = sorted(set(int(i) for i in ids))
    out: Dict[int, str] = {}
    for start in range(0, len(ids), PEOPLE_BATCH):
        batch: List[int] = ids[start:start + PEOPLE_BATCH]
        url = PEOPLE_URL.format(ids=",".join(str(i) for i in batch))
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read())
                for p in data.get("people", []):
                    full = p.get("fullName")
                    if full:
                        out[int(p["id"])] = full
                break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"[CACHE] WARNING: name lookup failed for batch "
                          f"{start}-{start + len(batch)}: {e}")
                else:
                    time.sleep(2 * (attempt + 1))
    return out


def build(csv_path: str, out_dir: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing source CSV: {csv_path}")

    hands: Dict[str, Dict[int, Counter]] = {
        "batters": defaultdict(Counter),
        "pitchers": defaultdict(Counter),
    }

    rows = 0
    for chunk in pd.read_csv(csv_path, chunksize=READ_CHUNK_ROWS,
                             low_memory=True, usecols=USECOLS):
        rows += len(chunk)
        for role, id_col, hand_col in (
            ("batters", "batter_id", "stand"),
            ("pitchers", "pitcher_id", "p_throws"),
        ):
            sub = chunk[[id_col, hand_col]].dropna(subset=[id_col])
            for pid, hand in zip(sub[id_col], sub[hand_col]):
                if isinstance(hand, str) and hand:
                    hands[role][int(pid)][hand] += 1

    id_to_name = _fetch_names(
        list(hands["batters"].keys()) + list(hands["pitchers"].keys())
    )

    names: Dict[str, Dict[str, dict]] = {"batters": {}, "pitchers": {}}
    missing = 0
    for role in ("batters", "pitchers"):
        for pid in hands[role]:
            full = id_to_name.get(pid)
            if not full:
                missing += 1
                continue
            names[role][_norm(full)] = {"id": pid, "name": full}
    if missing:
        print(f"[CACHE] {missing} player ids had no StatsAPI name; skipped.")

    name_cache = {"batters": names["batters"], "pitchers": names["pitchers"], "built": True}
    hand_cache = {
        role: {str(pid): c.most_common(1)[0][0] for pid, c in hands[role].items()}
        for role in ("batters", "pitchers")
    }

    os.makedirs(out_dir, exist_ok=True)
    name_path = os.path.join(out_dir, "name_id_cache.json")
    hand_path = os.path.join(out_dir, "handedness_cache.json")
    with open(name_path, "w", encoding="utf-8") as f:
        json.dump(name_cache, f)
    with open(hand_path, "w", encoding="utf-8") as f:
        json.dump(hand_cache, f)

    print(f"[CACHE] Scanned {rows:,} rows from {csv_path}")
    print(f"[CACHE] {len(name_cache['batters'])} batters, "
          f"{len(name_cache['pitchers'])} pitchers -> {name_path}")
    print(f"[CACHE] handedness for {len(hand_cache['batters'])} batters, "
          f"{len(hand_cache['pitchers'])} pitchers -> {hand_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="statcast_pitches.csv")
    parser.add_argument("--output_dir", type=str, default="preprocessed")
    args = parser.parse_args()
    build(args.input_csv, args.output_dir)


if __name__ == "__main__":
    main()
