#!/usr/bin/env python3
"""
livematchup.py  –  Live MLB plate-appearance probability predictor.

HOW TO RUN:
  # Named matchup
  python livematchup.py --preprocessed_dir preprocessed_2023_to_2025mid_test_rest2025 \
      --artifact_dir artifacts_modelG_2023_to_2025mid_test_rest2025 \
      --batter "Juan Soto" --pitcher "Gerrit Cole" --stadium "Yankee Stadium"

  # Discovery helpers
  python livematchup.py --list_batters "soto"
  python livematchup.py --list_pitchers "cole"
  python livematchup.py --list_stadiums "yank"

  # Interactive (no args)
  python livematchup.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PREPROCESSED_DIR = "preprocessed_2023_to_2025mid_test_rest2025"
DEFAULT_ARTIFACT_DIR = "artifacts_modelG_2023_to_2025mid_test_rest2025"
CACHE_FILENAME = "name_id_cache.json"
OUTCOMES = ["K", "BIPO", "BB", "1B", "XBH", "HR"]
PITCH_TYPES = ["FF", "SI", "FC", "SL", "CU", "CH", "FS", "ST", "SV", "OTHER"]

# Hardcoded stand/p_throws encoding (L=2, R=3, UNK=1; matches preprocessing2.py init)
HAND_VOCAB = {"L": 2, "R": 3}

# Stadium token -> list of human-readable names/aliases
STADIUM_ALIASES: Dict[str, List[str]] = {
    "ATH": ["athletics", "ath", "oakland", "sutter health park", "sutter"],
    "ATL": ["braves", "truist", "atl", "atlanta"],
    "AZ":  ["diamondbacks", "chase field", "chase", "arizona", "az"],
    "BAL": ["orioles", "camden yards", "camden", "baltimore", "bal"],
    "BOS": ["red sox", "fenway", "fenway park", "boston", "bos"],
    "CHC": ["cubs", "wrigley", "wrigley field", "chicago cubs", "chc"],
    "CIN": ["reds", "great american", "great american ball park", "cincinnati", "cin"],
    "CLE": ["guardians", "progressive field", "progressive", "cleveland", "cle"],
    "COL": ["rockies", "coors", "coors field", "colorado", "col"],
    "CWS": ["white sox", "guaranteed rate", "guaranteed rate field", "chicago white sox", "cws"],
    "DET": ["tigers", "comerica", "comerica park", "detroit", "det"],
    "HOU": ["astros", "minute maid", "minute maid park", "houston", "hou"],
    "KC":  ["royals", "kauffman", "kauffman stadium", "kansas city", "kc"],
    "LAA": ["angels", "angel stadium", "anaheim", "los angeles angels", "laa"],
    "LAD": ["dodgers", "dodger stadium", "los angeles dodgers", "lad"],
    "MIA": ["marlins", "loandepot park", "loandepot", "miami", "mia"],
    "MIL": ["brewers", "american family field", "american family", "milwaukee", "mil"],
    "MIN": ["twins", "target field", "target", "minnesota", "min"],
    "NYM": ["mets", "citi field", "citi", "new york mets", "nym"],
    "NYY": ["yankees", "yankee stadium", "new york yankees", "nyy"],
    "PHI": ["phillies", "citizens bank", "citizens bank park", "philadelphia", "phi"],
    "PIT": ["pirates", "pnc park", "pnc", "pittsburgh", "pit"],
    "SD":  ["padres", "petco", "petco park", "san diego", "sd"],
    "SEA": ["mariners", "t-mobile park", "t-mobile", "safeco", "seattle", "sea"],
    "SF":  ["giants", "oracle park", "oracle", "san francisco", "sf"],
    "STL": ["cardinals", "busch stadium", "busch", "st louis", "stl"],
    "TB":  ["rays", "tropicana", "tropicana field", "tampa bay", "tb"],
    "TEX": ["rangers", "globe life", "globe life field", "texas", "tex"],
    "TOR": ["blue jays", "rogers centre", "rogers", "toronto", "tor"],
    "WSH": ["nationals", "nationals park", "washington", "wsh"],
}

# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    """Lowercase, strip accents, remove punctuation."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Name -> ID cache (disk-backed)
# ---------------------------------------------------------------------------

def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {"batters": {}, "pitchers": {}, "built": False}


def _save_cache(cache: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------------------
# Build name->ID maps by scanning the source CSV
# ---------------------------------------------------------------------------

def _build_name_maps_from_csv(
    csv_path: str,
    cache: dict,
    cache_path: str,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Scans statcast_pitches.csv to extract batter/pitcher name->id maps.
    The CSV has 'batter' (name string) and 'batter_id' (int) columns.
    Results stored in cache['batters'] and cache['pitchers'].
    """
    print(f"[RESOLVER] Building name->ID maps from {csv_path} (one-time, will be cached)...")
    batter_map: Dict[str, int] = {}   # norm_name -> mlbam_id
    pitcher_map: Dict[str, int] = {}
    batter_raw: Dict[str, str] = {}   # norm_name -> display_name
    pitcher_raw: Dict[str, str] = {}

    try:
        for chunk in pd.read_csv(csv_path, chunksize=500_000, low_memory=True,
                                  usecols=["batter", "batter_id", "pitcher", "pitcher_id"]):
            for _, row in chunk.drop_duplicates(subset=["batter_id"]).iterrows():
                bid = int(row["batter_id"])
                # 'batter' col could be id or name - check dtype
                bname = str(row["batter"])
                if not bname.isdigit():
                    key = _norm(bname)
                    batter_map[key] = bid
                    batter_raw[key] = bname
            for _, row in chunk.drop_duplicates(subset=["pitcher_id"]).iterrows():
                pid = int(row["pitcher_id"])
                pname = str(row["pitcher"])
                if not pname.isdigit():
                    key = _norm(pname)
                    pitcher_map[key] = pid
                    pitcher_raw[key] = pname
    except Exception as e:
        print(f"[RESOLVER] Warning: could not fully scan CSV: {e}")

    # Store in cache
    cache["batters"] = {k: {"id": v, "name": batter_raw.get(k, k)} for k, v in batter_map.items()}
    cache["pitchers"] = {k: {"id": v, "name": pitcher_raw.get(k, k)} for k, v in pitcher_map.items()}
    cache["built"] = bool(batter_map or pitcher_map)
    _save_cache(cache, cache_path)
    print(f"[RESOLVER] Cached {len(batter_map)} batters, {len(pitcher_map)} pitchers -> {cache_path}")
    return batter_map, pitcher_map


def _fetch_mlbam_by_name(name: str) -> List[Tuple[str, int]]:
    """
    Query the public MLBAM People API to resolve a name.
    Returns list of (full_name, player_id) sorted by relevance.
    """
    try:
        import urllib.request, urllib.parse
        encoded = urllib.parse.quote(name)
        url = f"https://statsapi.mlb.com/api/v1/people/search?names={encoded}&sportId=1"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        people = data.get("people", [])
        return [(p["fullName"], int(p["id"])) for p in people]
    except Exception:
        return []


def _resolve_player(
    query: str,
    player_map: Dict[str, int],   # norm_name -> id
    player_raw: Dict[str, str],   # norm_name -> display name
    role: str,
    vocab: Dict[str, int],        # "123456" -> encoded_id
    cache: dict,
    cache_path: str,
    interactive: bool = True,
) -> Tuple[int, int, str, bool]:
    """
    Resolve a player name string to (raw_mlbam_id, encoded_id, display_name, in_vocab).
    Returns (raw_id, encoded_id, display_name, in_vocab).
    """
    q_norm = _norm(query)

    # 1) Exact key match in local map
    if q_norm in player_map:
        raw_id = player_map[q_norm]
        disp = player_raw.get(q_norm, query)
        enc = vocab.get(str(raw_id), 1)
        return raw_id, enc, disp, str(raw_id) in vocab

    # 2) Partial / substring matches
    matches = [(k, v) for k, v in player_map.items() if q_norm in k or k in q_norm]
    # Also token overlap
    q_tokens = set(q_norm.split())
    token_matches = [(k, v) for k, v in player_map.items()
                     if q_tokens & set(k.split()) and (k, v) not in matches]
    matches = matches + token_matches

    # Deduplicate by id
    seen_ids: set = set()
    unique_matches: List[Tuple[str, int]] = []
    for k, v in matches:
        if v not in seen_ids:
            seen_ids.add(v)
            unique_matches.append((k, v))

    if len(unique_matches) == 1:
        raw_id = unique_matches[0][1]
        disp = player_raw.get(unique_matches[0][0], query)
        return raw_id, vocab.get(str(raw_id), 1), disp, str(raw_id) in vocab

    if len(unique_matches) > 1:
        print(f"\n[RESOLVER] '{query}' matched {len(unique_matches)} {role}s. Top results:")
        for i, (k, v) in enumerate(unique_matches[:10]):
            in_v = "OK" if str(v) in vocab else "!! UNK"
            print(f"  [{i+1}] {player_raw.get(k, k):<30s} id={v}  {in_v}")
        if interactive:
            choice = input(f"Enter number (1-{min(len(unique_matches),10)}) or 0 to abort: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= min(len(unique_matches), 10):
                k, raw_id = unique_matches[int(choice) - 1]
                disp = player_raw.get(k, query)
                return raw_id, vocab.get(str(raw_id), 1), disp, str(raw_id) in vocab
            print("[RESOLVER] Aborted.")
            sys.exit(1)
        else:
            # non-interactive: pick first
            k, raw_id = unique_matches[0]
            disp = player_raw.get(k, query)
            return raw_id, vocab.get(str(raw_id), 1), disp, str(raw_id) in vocab

    # 3) Fall back to MLBAM API
    print(f"[RESOLVER] No local match for '{query}'. Querying MLBAM API...")
    api_results = _fetch_mlbam_by_name(query)
    if api_results:
        if len(api_results) == 1:
            disp, raw_id = api_results[0]
            # Add to cache
            cache[role + "s"][_norm(disp)] = {"id": raw_id, "name": disp}
            _save_cache(cache, cache_path)
            return raw_id, vocab.get(str(raw_id), 1), disp, str(raw_id) in vocab

        print(f"[RESOLVER] API returned {len(api_results)} results:")
        for i, (name, pid) in enumerate(api_results[:10]):
            in_v = "OK" if str(pid) in vocab else "!! UNK"
            print(f"  [{i+1}] {name:<30s} id={pid}  {in_v}")
        if interactive:
            choice = input(f"Enter number (1-{min(len(api_results),10)}) or 0 to abort: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= min(len(api_results), 10):
                disp, raw_id = api_results[int(choice) - 1]
                cache[role + "s"][_norm(disp)] = {"id": raw_id, "name": disp}
                _save_cache(cache, cache_path)
                return raw_id, vocab.get(str(raw_id), 1), disp, str(raw_id) in vocab
            print("[RESOLVER] Aborted.")
            sys.exit(1)

    print(f"[RESOLVER] ERROR: Could not resolve {role} '{query}'. Check spelling or use --list_{role}s.")
    sys.exit(1)


def _resolve_stadium(query: str, vocab: Dict[str, int]) -> Tuple[str, int]:
    """Resolve stadium name/token to (raw_token, encoded_id)."""
    q = _norm(query)
    # Direct token match (e.g. "NYY")
    q_upper = query.strip().upper()
    if q_upper in vocab:
        return q_upper, vocab[q_upper]
    # Alias lookup
    for token, aliases in STADIUM_ALIASES.items():
        if q in [_norm(a) for a in aliases] or q == token.lower():
            enc = vocab.get(token, 1)
            return token, enc
    # Partial alias match
    for token, aliases in STADIUM_ALIASES.items():
        if any(q in _norm(a) or _norm(a) in q for a in aliases):
            enc = vocab.get(token, 1)
            return token, enc
    print(f"[RESOLVER] WARNING: Stadium '{query}' not found. Defaulting to UNK (no park effect).")
    return query.upper(), 1


# ---------------------------------------------------------------------------
# Handedness inference from train split
# ---------------------------------------------------------------------------

def _infer_handedness(
    pre_dir: str,
    batter_raw_id: int,
    pitcher_raw_id: int,
) -> Tuple[str, str]:
    """
    Stream train.parquet (raw ids stored before encoding? No - train has encoded ids).
    Fall back to scanning source CSV for most common stand/p_throws.
    We check vocabs to decode: batter/pitcher in train are encoded; we need the raw csv.
    Instead, scan statcast_pitches.csv for stand/p_throws by MLBAM id.
    """
    # Try loading from a hand_cache within the preprocessed dir
    hcache_path = os.path.join(pre_dir, "handedness_cache.json")
    if os.path.exists(hcache_path):
        try:
            hc = json.load(open(hcache_path))
            b_hand = hc.get("batters", {}).get(str(batter_raw_id), None)
            p_hand = hc.get("pitchers", {}).get(str(pitcher_raw_id), None)
            if b_hand and p_hand:
                return b_hand, p_hand
        except Exception:
            pass

    csv_path = "statcast_pitches.csv"
    if not os.path.exists(csv_path):
        print("[HAND] statcast_pitches.csv not found; defaulting to R/R.")
        return "R", "R"

    from collections import Counter
    b_counter: Counter = Counter()
    p_counter: Counter = Counter()

    try:
        for chunk in pd.read_csv(csv_path, chunksize=500_000, low_memory=True,
                                  usecols=["batter_id", "pitcher_id", "stand", "p_throws"]):
            bm = chunk[chunk["batter_id"] == batter_raw_id]
            if not bm.empty:
                b_counter.update(bm["stand"].dropna().tolist())
            pm = chunk[chunk["pitcher_id"] == pitcher_raw_id]
            if not pm.empty:
                p_counter.update(pm["p_throws"].dropna().tolist())
    except Exception as e:
        print(f"[HAND] Warning: {e}; defaulting to R/R.")
        return "R", "R"

    b_hand = b_counter.most_common(1)[0][0] if b_counter else "R"
    p_hand = p_counter.most_common(1)[0][0] if p_counter else "R"

    # Save to cache
    try:
        hc = json.load(open(hcache_path)) if os.path.exists(hcache_path) else {"batters": {}, "pitchers": {}}
        hc["batters"][str(batter_raw_id)] = b_hand
        hc["pitchers"][str(pitcher_raw_id)] = p_hand
        with open(hcache_path, "w") as f:
            json.dump(hc, f)
    except Exception:
        pass

    return str(b_hand).upper().strip(), str(p_hand).upper().strip()


# ---------------------------------------------------------------------------
# Low sample-size warnings
# ---------------------------------------------------------------------------

def _compute_pa_counts(pre_dir: str) -> Tuple[Dict[int, int], Dict[int, int], float, float]:
    """
    Stream all splits to count PA rows per encoded batter/pitcher id.
    Returns (batter_counts, pitcher_counts, batter_p10_threshold, pitcher_p10_threshold).
    Note: split files have encoded ids; we use them as keys (integer).
    """
    from collections import defaultdict
    b_counts: Dict[int, int] = defaultdict(int)
    p_counts: Dict[int, int] = defaultdict(int)

    for split in ("train", "val", "test"):
        path = os.path.join(pre_dir, f"{split}.parquet")
        if not os.path.exists(path):
            path = os.path.join(pre_dir, f"{split}.csv.gz")
        if not os.path.exists(path):
            continue
        if path.endswith(".parquet"):
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=50_000, columns=["batter_id", "pitcher_id"]):
                df = batch.to_pandas()
                for bid, cnt in df["batter_id"].value_counts().items():
                    b_counts[int(bid)] += int(cnt)
                for pid, cnt in df["pitcher_id"].value_counts().items():
                    p_counts[int(pid)] += int(cnt)
        else:
            for chunk in pd.read_csv(path, chunksize=50_000, compression="gzip",
                                     usecols=["batter_id", "pitcher_id"]):
                for bid, cnt in chunk["batter_id"].value_counts().items():
                    b_counts[int(bid)] += int(cnt)
                for pid, cnt in chunk["pitcher_id"].value_counts().items():
                    p_counts[int(pid)] += int(cnt)

    b_vals = np.array(list(b_counts.values()), dtype=float) if b_counts else np.array([0.0])
    p_vals = np.array(list(p_counts.values()), dtype=float) if p_counts else np.array([0.0])
    b_p10 = float(np.percentile(b_vals, 10))
    p_p10 = float(np.percentile(p_vals, 10))
    return dict(b_counts), dict(p_counts), b_p10, p_p10


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _load_six_vector_stats(pre_dir: str) -> dict:
    files = {
        "batter_overall":   "six_vector_batter_overall.parquet",
        "pitcher_overall":  "six_vector_pitcher_overall.parquet",
        "stadium_rates":    "six_vector_stadium_rates.parquet",
        "batter_platoon":   "six_vector_batter_platoon.parquet",
        "pitcher_platoon":  "six_vector_pitcher_platoon.parquet",
        "batter_pitch_type":"six_vector_batter_pitch_type.parquet",
        "pitcher_mix":      "six_vector_pitcher_mix.parquet",
    }
    stats = {}
    for key, fname in files.items():
        p = os.path.join(pre_dir, fname)
        if os.path.exists(p):
            stats[key] = pd.read_parquet(p)
        else:
            stats[key] = pd.DataFrame()
    return stats


def _safe_logit(p: float, eps: float = 1e-6) -> float:
    """Clamp and compute log-odds: log(p / (1-p))."""
    p_clamped = max(eps, min(p, 1.0 - eps))
    return math.log(p_clamped / (1.0 - p_clamped))


def _load_league_logodds(pre_dir: str) -> List[float]:
    path = os.path.join(pre_dir, "league_logodds.json")
    if os.path.exists(path):
        d = json.load(open(path))
        return list(d["logodds"])
    # Fallback: compute from league_rates
    lr = _load_base_rates(pre_dir)
    return [_safe_logit(r) for r in lr]


def _compute_six_vectors_single(
    batter_raw_id: int,
    pitcher_raw_id: int,
    stadium_raw_id: str,
    stand: str,
    p_throws: str,
    stats: dict,
    league_rates: List[float],
    league_logodds: List[float],
) -> Dict[str, float]:
    """Compute the 42 six-vector features + pitcher_fatigue for one PA."""
    lr = league_rates
    feats: Dict[str, float] = {}

    def _get_row(df: pd.DataFrame, id_col: str, key) -> Optional[pd.Series]:
        if df.empty or id_col not in df.columns:
            return None
        sub = df[df[id_col] == key]
        return sub.iloc[0] if not sub.empty else None

    batter_raw_id_int = int(batter_raw_id)
    pitcher_raw_id_int = int(pitcher_raw_id)

    bo = _get_row(stats["batter_overall"], "batter_id", batter_raw_id_int)
    po = _get_row(stats["pitcher_overall"], "pitcher_id", pitcher_raw_id_int)
    sr = _get_row(stats["stadium_rates"], "stadium_id", stadium_raw_id)
    bp = _get_row(stats["batter_platoon"], "batter_id", batter_raw_id_int)
    pp = _get_row(stats["pitcher_platoon"], "pitcher_id", pitcher_raw_id_int)
    bpt = _get_row(stats["batter_pitch_type"], "batter_id", batter_raw_id_int)
    pm = _get_row(stats["pitcher_mix"], "pitcher_id", pitcher_raw_id_int)

    # V1: Batter overall
    for i, oc in enumerate(OUTCOMES):
        raw_rate = float(bo[f"batter_{oc}_rate"]) if bo is not None and f"batter_{oc}_rate" in bo.index else lr[i]
        feats[f"v1_batter_{oc}"] = _safe_logit(raw_rate) - league_logodds[i]
    feats["v1_batter_log_n"] = math.log1p(float(bo["batter_n_pa"])) if bo is not None else 0.0

    # V2: Pitcher overall
    for i, oc in enumerate(OUTCOMES):
        raw_rate = float(po[f"pitcher_{oc}_rate"]) if po is not None and f"pitcher_{oc}_rate" in po.index else lr[i]
        feats[f"v2_pitcher_{oc}"] = _safe_logit(raw_rate) - league_logodds[i]
    feats["v2_pitcher_log_n"] = math.log1p(float(po["pitcher_n_pa"])) if po is not None else 0.0

    # V3: Stadium
    for i, oc in enumerate(OUTCOMES):
        raw_rate = float(sr[f"stadium_{oc}_rate"]) if sr is not None and f"stadium_{oc}_rate" in sr.index else lr[i]
        feats[f"v3_stadium_{oc}"] = _safe_logit(raw_rate) - league_logodds[i]
    feats["v3_stadium_log_n"] = math.log1p(float(sr["stadium_n_pa"])) if sr is not None else 0.0

    # V4: Batter platoon (vs pitcher hand)
    for i, oc in enumerate(OUTCOMES):
        if bp is not None:
            col = f"batter_{oc}_vs_L" if p_throws == "L" else f"batter_{oc}_vs_R"
            val = float(bp[col]) if col in bp.index and not pd.isna(bp[col]) else lr[i]
        else:
            val = lr[i]
        feats[f"v4_batter_platoon_{oc}"] = _safe_logit(val) - league_logodds[i]
    if bp is not None:
        n_col = "batter_n_vs_L" if p_throws == "L" else "batter_n_vs_R"
        n_val = float(bp[n_col]) if n_col in bp.index and not pd.isna(bp[n_col]) else 0.0
    else:
        n_val = 0.0
    feats["v4_batter_platoon_log_n"] = math.log1p(n_val)

    # V5: Pitcher platoon (vs batter hand)
    for i, oc in enumerate(OUTCOMES):
        if pp is not None:
            col = f"pitcher_{oc}_vs_L" if stand == "L" else f"pitcher_{oc}_vs_R"
            val = float(pp[col]) if col in pp.index and not pd.isna(pp[col]) else lr[i]
        else:
            val = lr[i]
        feats[f"v5_pitcher_platoon_{oc}"] = _safe_logit(val) - league_logodds[i]
    if pp is not None:
        n_col = "pitcher_n_vs_L" if stand == "L" else "pitcher_n_vs_R"
        n_val = float(pp[n_col]) if n_col in pp.index and not pd.isna(pp[n_col]) else 0.0
    else:
        n_val = 0.0
    feats["v5_pitcher_platoon_log_n"] = math.log1p(n_val)

    # V6: Pitch mix interaction
    for i, oc in enumerate(OUTCOMES):
        interaction = 0.0
        for pt in PITCH_TYPES:
            mix_col = f"pitcher_mix_{pt}"
            p_pitch = float(pm[mix_col]) if pm is not None and mix_col in pm.index and not pd.isna(pm[mix_col]) else 0.0
            batter_col = f"batter_{oc}_vs_{pt}"
            p_outcome = float(bpt[batter_col]) if bpt is not None and batter_col in bpt.index and not pd.isna(bpt[batter_col]) else lr[i]
            interaction += p_pitch * p_outcome
        feats[f"v6_mix_{oc}"] = _safe_logit(interaction) - league_logodds[i]

    if pm is not None and "pitcher_n_pitches" in pm.index and not pd.isna(pm["pitcher_n_pitches"]):
        pitcher_n = float(pm["pitcher_n_pitches"])
    else:
        pitcher_n = 0.0
    batter_n = float(bo["batter_n_pa"]) if bo is not None else 0.0
    feats["v6_mix_log_n"] = math.log1p(min(pitcher_n, batter_n))

    # Extremeness features
    _vec_prefixes = ["v1_batter", "v2_pitcher", "v3_stadium",
                     "v4_batter_platoon", "v5_pitcher_platoon", "v6_mix"]
    all_devs = [feats[f"{prefix}_{oc}"] for oc in OUTCOMES for prefix in _vec_prefixes]
    feats["ext_max_abs_dev"] = max(abs(d) for d in all_devs)
    feats["ext_mean_abs_dev"] = sum(abs(d) for d in all_devs) / len(all_devs)
    disagreements = []
    for oc in OUTCOMES:
        preds = [feats[f"{prefix}_{oc}"] for prefix in _vec_prefixes]
        mean_p = sum(preds) / len(preds)
        var_p = sum((p - mean_p) ** 2 for p in preds) / len(preds)
        disagreements.append(var_p ** 0.5)
    feats["ext_vector_disagreement"] = sum(disagreements) / len(disagreements)

    return feats


def _merge_pt_statcast(
    batter_raw_id: int,
    pitcher_raw_id: int,
    pre_dir: str,
    pt_statcast_cols: List[str],
) -> Dict[str, float]:
    """Load pitchtype statcast wide tables and extract rows for this matchup."""
    pitcher_path = os.path.join(pre_dir, "pitchtype_statcast_pitcher.parquet")
    batter_path  = os.path.join(pre_dir, "pitchtype_statcast_batter.parquet")

    feats: Dict[str, float] = {c: 0.0 for c in pt_statcast_cols}

    if os.path.exists(pitcher_path):
        pw = pd.read_parquet(pitcher_path)
        pw = pw.set_index("pitcher_id")
        if pitcher_raw_id in pw.index:
            row = pw.loc[pitcher_raw_id]
            for col in pw.columns:
                if col in feats:
                    val = row[col]
                    feats[col] = 0.0 if pd.isna(val) else float(val)

    if os.path.exists(batter_path):
        bw = pd.read_parquet(batter_path)
        bw = bw.set_index("batter_id")
        if batter_raw_id in bw.index:
            row = bw.loc[batter_raw_id]
            for col in bw.columns:
                if col in feats:
                    val = row[col]
                    feats[col] = 0.0 if pd.isna(val) else float(val)

    return feats


# ---------------------------------------------------------------------------
# Model reconstruction (mirrors evaluate_checklist_modelG.py logic)
# ---------------------------------------------------------------------------

def _detect_pt_statcast_cols(num_cols: List[str]) -> List[str]:
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


def _load_model(pre_dir: str, art_dir: str):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import train_model2 as tm

    meta = json.load(open(os.path.join(pre_dir, "metadata.json")))
    cfg  = json.load(open(os.path.join(art_dir, "train_config.json")))

    feature_order = meta["features"]["feature_list_ordered"]
    cat_cols  = meta["features"].get("categorical_features", [])
    num_cols  = [c for c in feature_order if c not in set(cat_cols)]
    vocab_sizes = meta["features"].get("categorical_vocab_sizes", {})
    n_classes = len(meta["labels"]["label_to_id"])
    model_class = cfg.get("resolved", {}).get("model_class", "HybridModel")
    hidden_dims = cfg.get("HIDDEN_DIMS", [256, 128])
    dropout = cfg.get("DROPOUT", 0.2)
    pt_cols = _detect_pt_statcast_cols(num_cols)
    num_pt = len(pt_cols)
    # Load league logodds: prefer train_config.json, then file, then fallback
    league_logodds = cfg.get("league_logodds") or _load_league_logodds(pre_dir)

    if model_class == "ContextualGateLogitHybridModel" and num_pt > 0:
        model = tm.ContextualGateLogitHybridModel(
            cat_cols=cat_cols, num_numeric=len(num_cols),
            vocab_sizes=vocab_sizes, num_classes=n_classes,
            hidden_dims=hidden_dims, dropout=dropout, num_pt_statcast_cols=num_pt,
            league_logodds=league_logodds,
        )
    elif model_class == "PerClassGateLogitHybridModel" and num_pt > 0:
        model = tm.PerClassGateLogitHybridModel(
            cat_cols=cat_cols, num_numeric=len(num_cols),
            vocab_sizes=vocab_sizes, num_classes=n_classes,
            hidden_dims=hidden_dims, dropout=dropout, num_pt_statcast_cols=num_pt,
            league_logodds=league_logodds,
        )
    elif model_class == "StatcastLogitHybridModel" and num_pt > 0:
        model = tm.StatcastLogitHybridModel(
            cat_cols=cat_cols, num_numeric=len(num_cols),
            vocab_sizes=vocab_sizes, num_classes=n_classes,
            hidden_dims=hidden_dims, dropout=dropout, num_pt_statcast_cols=num_pt,
            league_logodds=league_logodds,
        )
    else:
        model = tm.HybridModel(
            cat_cols=cat_cols, num_numeric=len(num_cols),
            vocab_sizes=vocab_sizes, num_classes=n_classes,
            hidden_dims=hidden_dims, dropout=dropout,
            league_logodds=league_logodds,
        )

    state = torch.load(os.path.join(art_dir, "model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, meta, cat_cols, num_cols, pt_cols


# ---------------------------------------------------------------------------
# Base-rate loader
# ---------------------------------------------------------------------------

def _load_base_rates(pre_dir: str) -> List[float]:
    lr_path = os.path.join(pre_dir, "league_rates.json")
    if os.path.exists(lr_path):
        d = json.load(open(lr_path))
        return list(d["rates"])
    # Fallback: compute from train split y distribution
    path = os.path.join(pre_dir, "train.parquet")
    if os.path.exists(path):
        import pyarrow.parquet as pq
        counts = np.zeros(6, dtype=float)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=["y"]):
            for yi in batch.to_pandas()["y"]:
                counts[int(yi)] += 1
        return (counts / counts.sum()).tolist()
    return [1/6] * 6


# ---------------------------------------------------------------------------
# Name map loader (builds once from CSV, then cached)
# ---------------------------------------------------------------------------

def _get_player_maps(
    pre_dir: str,
    csv_path: str = "statcast_pitches.csv",
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, str], Dict[str, str], dict, str]:
    """
    Returns (batter_map, pitcher_map, batter_raw, pitcher_raw, cache, cache_path).
    batter_map/pitcher_map: norm_name -> mlbam_id
    batter_raw/pitcher_raw: norm_name -> display_name
    """
    cache_path = os.path.join(pre_dir, CACHE_FILENAME)
    cache = _load_cache(cache_path)

    if cache.get("built"):
        batter_map  = {k: v["id"] for k, v in cache.get("batters", {}).items()}
        batter_raw  = {k: v.get("name", k) for k, v in cache.get("batters", {}).items()}
        pitcher_map = {k: v["id"] for k, v in cache.get("pitchers", {}).items()}
        pitcher_raw = {k: v.get("name", k) for k, v in cache.get("pitchers", {}).items()}
        return batter_map, pitcher_map, batter_raw, pitcher_raw, cache, cache_path

    # Build from CSV
    bm_raw: Dict[str, int] = {}
    pm_raw: Dict[str, int] = {}
    br: Dict[str, str] = {}
    pr: Dict[str, str] = {}

    if os.path.exists(csv_path):
        print(f"[RESOLVER] Scanning {csv_path} to build name->ID cache (one-time)...")
        seen_b: set = set()
        seen_p: set = set()
        for chunk in pd.read_csv(csv_path, chunksize=500_000, low_memory=True,
                                  usecols=["batter", "batter_id", "pitcher", "pitcher_id"]):
            for _, row in chunk.drop_duplicates(subset=["batter_id"]).iterrows():
                bid = int(row["batter_id"])
                if bid in seen_b:
                    continue
                seen_b.add(bid)
                bname = str(row["batter"])
                if not bname.replace(".", "").replace(" ", "").isdigit():
                    key = _norm(bname)
                    bm_raw[key] = bid
                    br[key] = bname
            for _, row in chunk.drop_duplicates(subset=["pitcher_id"]).iterrows():
                pid = int(row["pitcher_id"])
                if pid in seen_p:
                    continue
                seen_p.add(pid)
                pname = str(row["pitcher"])
                if not pname.replace(".", "").replace(" ", "").isdigit():
                    key = _norm(pname)
                    pm_raw[key] = pid
                    pr[key] = pname

    cache["batters"]  = {k: {"id": v, "name": br.get(k, k)}  for k, v in bm_raw.items()}
    cache["pitchers"] = {k: {"id": v, "name": pr.get(k, k)} for k, v in pm_raw.items()}
    cache["built"] = True
    _save_cache(cache, cache_path)
    print(f"[RESOLVER] Cached {len(bm_raw)} batters, {len(pm_raw)} pitchers -> {cache_path}")

    return bm_raw, pm_raw, br, pr, cache, cache_path


# ---------------------------------------------------------------------------
# List helpers
# ---------------------------------------------------------------------------

def _list_players(query: str, player_map: Dict[str, int], raw: Dict[str, str], vocab: Dict[str, int], role: str) -> None:
    q = _norm(query)
    results = [(raw.get(k, k), v) for k, v in player_map.items() if q in k]
    results.sort(key=lambda x: x[0])
    if not results:
        print(f"No {role}s matching '{query}'.")
        return
    print(f"\nMatching {role}s for '{query}':")
    print(f"  {'Name':<30s} {'MLBAM ID':>10s}  In Model")
    print("  " + "-" * 50)
    for name, pid in results[:50]:
        in_v = "yes" if str(pid) in vocab else "UNK"
        print(f"  {name:<30s} {pid:>10d}  {in_v}")


def _list_stadiums(query: str) -> None:
    q = _norm(query)
    print(f"\nMatching stadiums for '{query}':")
    print(f"  {'Token':<6s}  {'Stadium / Team'}")
    print("  " + "-" * 40)
    for token, aliases in sorted(STADIUM_ALIASES.items()):
        if any(q in _norm(a) for a in aliases) or q in token.lower():
            print(f"  {token:<6s}  {aliases[0].title()}")


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def run_matchup(
    batter_query: str,
    pitcher_query: str,
    stadium_query: str,
    pre_dir: str,
    art_dir: str,
    interactive: bool = True,
) -> None:
    # -- Load artifacts --
    print("\n[Loading model and artifacts...]")
    model, meta, cat_cols, num_cols, pt_cols = _load_model(pre_dir, art_dir)
    feature_order = meta["features"]["feature_list_ordered"]
    label_set     = meta["labels"]["label_set_ordered"]
    vocabs        = json.load(open(os.path.join(pre_dir, "vocabs.json")))
    base_rates     = _load_base_rates(pre_dir)
    league_logodds = _load_league_logodds(pre_dir)
    six_stats      = _load_six_vector_stats(pre_dir)

    batter_vocab  = vocabs["batter_id"]   # str(raw_id) -> encoded_int
    pitcher_vocab = vocabs["pitcher_id"]
    stadium_vocab = vocabs["stadium_id"]  # raw_token -> encoded_int

    # Stand/p_throws: hardcoded (L=2, R=3) - vocabs.json only has PAD/UNK
    stand_vocab   = {"L": 2, "R": 3}
    p_throws_vocab = {"L": 2, "R": 3}

    # -- Resolve names --
    batter_map, pitcher_map, batter_raw, pitcher_raw, cache, cache_path = _get_player_maps(pre_dir)

    batter_raw_id, batter_enc, batter_disp, batter_in_vocab = _resolve_player(
        batter_query, batter_map, batter_raw, "batter", batter_vocab, cache, cache_path, interactive)
    pitcher_raw_id, pitcher_enc, pitcher_disp, pitcher_in_vocab = _resolve_player(
        pitcher_query, pitcher_map, pitcher_raw, "pitcher", pitcher_vocab, cache, cache_path, interactive)
    stadium_raw, stadium_enc = _resolve_stadium(stadium_query, stadium_vocab)

    # -- Infer handedness --
    stand_str, p_throws_str = _infer_handedness(pre_dir, batter_raw_id, pitcher_raw_id)
    stand_enc    = stand_vocab.get(stand_str, 1)
    p_throws_enc = p_throws_vocab.get(p_throws_str, 1)

    # -- Low sample-size warnings --
    print("[Computing sample-size thresholds...]")
    b_counts, p_counts, b_p10, p_p10 = _compute_pa_counts(pre_dir)
    b_count = b_counts.get(batter_enc, 0)
    p_count = p_counts.get(pitcher_enc, 0)

    # -- Build features --
    six_feats = _compute_six_vectors_single(
        batter_raw_id, pitcher_raw_id, stadium_raw,
        stand_str, p_throws_str, six_stats, base_rates, league_logodds,
    )
    pt_feats = _merge_pt_statcast(batter_raw_id, pitcher_raw_id, pre_dir, pt_cols)

    # Assemble in exact feature_order
    cat_set = set(cat_cols)
    cat_values = {
        "batter_id":  batter_enc,
        "pitcher_id": pitcher_enc,
        "stadium_id": stadium_enc,
        "stand":      stand_enc,
        "p_throws":   p_throws_enc,
    }
    num_feats: Dict[str, float] = {"pitcher_fatigue": 0.0}
    num_feats.update(six_feats)
    num_feats.update(pt_feats)

    # Build tensors in strict feature_order
    x_cat_list = [cat_values[c] for c in cat_cols]
    num_cols_ordered = [c for c in feature_order if c not in cat_set]
    x_num_list = []
    missing_feats = []
    for c in num_cols_ordered:
        if c in num_feats:
            x_num_list.append(float(num_feats[c]))
        else:
            x_num_list.append(0.0)
            missing_feats.append(c)

    if missing_feats:
        print(f"[WARN] {len(missing_feats)} numeric features defaulted to 0.0: {missing_feats[:5]}{'...' if len(missing_feats)>5 else ''}")

    x_cat = torch.tensor([x_cat_list], dtype=torch.long)
    x_num = torch.tensor([x_num_list], dtype=torch.float32)

    # -- Inference --
    with torch.no_grad():
        logits = model(x_cat, x_num)
        probs = F.softmax(logits, dim=1).squeeze(0).numpy()

    # -- Output --
    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"  MATCHUP: {batter_disp} vs {pitcher_disp} @ {stadium_raw}")
    print(SEP)
    print(f"  Batter:  {batter_disp:<28s} MLBAM={batter_raw_id}  enc={batter_enc}  stand={stand_str}")
    print(f"  Pitcher: {pitcher_disp:<28s} MLBAM={pitcher_raw_id}  enc={pitcher_enc}  hand={p_throws_str}")
    print(f"  Stadium: {stadium_raw}")
    print(f"  (pitcher_fatigue defaulted to 0.0 - live mode)")

    print(f"\n  {'Outcome':<8s} {'Model':>8s}  {'Typical':>8s}  {'Diff vs Typical':>18s}")
    print("  " + "-" * 50)
    for i, oc in enumerate(label_set):
        p_model   = probs[i]
        p_typical = base_rates[i]
        diff_pct  = (p_model - p_typical) / max(p_typical, 1e-12) * 100
        sign = f"+{diff_pct:.1f}" if diff_pct >= 0 else f"{diff_pct:.1f}"
        print(f"  {oc:<8s} {p_model*100:>7.2f}%  {p_typical*100:>7.2f}%  {sign:>7s}% vs typical")

    print(f"\n  {'WARNINGS':}")
    warned = False
    if not batter_in_vocab:
        print(f"  !!  Batter '{batter_disp}' (id={batter_raw_id}) not in training vocab -> UNK embedding.")
        warned = True
    if not pitcher_in_vocab:
        print(f"  !!  Pitcher '{pitcher_disp}' (id={pitcher_raw_id}) not in training vocab -> UNK embedding.")
        warned = True
    if b_count <= b_p10:
        print(f"  !!  Batter has only {b_count} PAs in dataset (10th pct threshold = {b_p10:.0f}) -> low data.")
        warned = True
    if p_count <= p_p10:
        print(f"  !!  Pitcher has only {p_count} BFs in dataset (10th pct threshold = {p_p10:.0f}) -> low data.")
        warned = True
    if not warned:
        print(f"  None.")
    print(SEP + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Live matchup PA probability predictor")
    parser.add_argument("--preprocessed_dir", default=DEFAULT_PREPROCESSED_DIR)
    parser.add_argument("--artifact_dir",     default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--batter",   default=None, help="Batter name (e.g. 'Juan Soto')")
    parser.add_argument("--pitcher",  default=None, help="Pitcher name (e.g. 'Gerrit Cole')")
    parser.add_argument("--stadium",  default=None, help="Stadium name or token (e.g. 'Yankee Stadium' or 'NYY')")
    parser.add_argument("--list_batters",  default=None, metavar="QUERY")
    parser.add_argument("--list_pitchers", default=None, metavar="QUERY")
    parser.add_argument("--list_stadiums", default=None, metavar="QUERY")
    args = parser.parse_args()

    pre_dir = args.preprocessed_dir
    art_dir = args.artifact_dir

    # -- List helpers (don't need full model load) --
    if args.list_stadiums is not None:
        _list_stadiums(args.list_stadiums)
        return

    if args.list_batters is not None or args.list_pitchers is not None:
        vocabs = json.load(open(os.path.join(pre_dir, "vocabs.json")))
        batter_map, pitcher_map, batter_raw, pitcher_raw, _, _ = _get_player_maps(pre_dir)
        if args.list_batters is not None:
            _list_players(args.list_batters, batter_map, batter_raw, vocabs["batter_id"], "batter")
        if args.list_pitchers is not None:
            _list_players(args.list_pitchers, pitcher_map, pitcher_raw, vocabs["pitcher_id"], "pitcher")
        return

    # -- Interactive or CLI matchup --
    batter  = args.batter
    pitcher = args.pitcher
    stadium = args.stadium

    if batter is None or pitcher is None or stadium is None:
        print("Interactive mode (press Ctrl-C to quit)")
        if batter  is None: batter  = input("Batter name  : ").strip()
        if pitcher is None: pitcher = input("Pitcher name : ").strip()
        if stadium is None: stadium = input("Stadium      : ").strip()

    run_matchup(batter, pitcher, stadium, pre_dir, art_dir, interactive=True)


if __name__ == "__main__":
    main()
