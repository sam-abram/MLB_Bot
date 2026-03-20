"""
app/search.py — Player and stadium search logic.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

# Stadium metadata: token -> (display name, team name)
STADIUM_INFO: Dict[str, tuple] = {
    "ATH": ("Sutter Health Park",        "Athletics"),
    "ATL": ("Truist Park",               "Braves"),
    "AZ":  ("Chase Field",               "Diamondbacks"),
    "BAL": ("Camden Yards",              "Orioles"),
    "BOS": ("Fenway Park",               "Red Sox"),
    "CHC": ("Wrigley Field",             "Cubs"),
    "CIN": ("Great American Ball Park",  "Reds"),
    "CLE": ("Progressive Field",         "Guardians"),
    "COL": ("Coors Field",               "Rockies"),
    "CWS": ("Guaranteed Rate Field",     "White Sox"),
    "DET": ("Comerica Park",             "Tigers"),
    "HOU": ("Minute Maid Park",          "Astros"),
    "KC":  ("Kauffman Stadium",          "Royals"),
    "LAA": ("Angel Stadium",             "Angels"),
    "LAD": ("Dodger Stadium",            "Dodgers"),
    "MIA": ("loanDepot Park",            "Marlins"),
    "MIL": ("American Family Field",     "Brewers"),
    "MIN": ("Target Field",              "Twins"),
    "NYM": ("Citi Field",                "Mets"),
    "NYY": ("Yankee Stadium",            "Yankees"),
    "PHI": ("Citizens Bank Park",        "Phillies"),
    "PIT": ("PNC Park",                  "Pirates"),
    "SD":  ("Petco Park",                "Padres"),
    "SEA": ("T-Mobile Park",             "Mariners"),
    "SF":  ("Oracle Park",               "Giants"),
    "STL": ("Busch Stadium",             "Cardinals"),
    "TB":  ("Tropicana Field",           "Rays"),
    "TEX": ("Globe Life Field",          "Rangers"),
    "TOR": ("Rogers Centre",             "Blue Jays"),
    "WSH": ("Nationals Park",            "Nationals"),
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def get_stadiums() -> List[dict]:
    return [
        {"token": token, "name": name, "team": team}
        for token, (name, team) in STADIUM_INFO.items()
    ]


def search_players(
    query: str,
    player_index: Dict[str, dict],  # norm_name -> {id, name}
    vocab: Dict[str, int],          # str(mlbam_id) -> encoded_int
    pa_counts: Dict[int, float],    # mlbam_id -> pa_count
    min_pa: float = 0.0,
    limit: int = 10,
) -> List[dict]:
    """
    Substring search on normalized names. Returns top `limit` matches.
    Each result: {name, mlbam_id, in_vocab, pa_count}.
    """
    if len(query.strip()) < 2:
        return []

    q = _norm(query)
    results = []

    for norm_name, entry in player_index.items():
        if q not in norm_name:
            continue
        mlbam_id = int(entry["id"])
        pa = pa_counts.get(mlbam_id, 0.0)
        results.append({
            "name": entry["name"],
            "mlbam_id": mlbam_id,
            "in_vocab": str(mlbam_id) in vocab,
            "pa_count": pa,
        })

    # Filter by min_pa; if nothing passes, fall back to showing all matches
    filtered = [r for r in results if r["pa_count"] >= min_pa]
    if not filtered:
        filtered = results

    # Sort: in-vocab first, then by pa_count descending, then name
    filtered.sort(key=lambda r: (-int(r["in_vocab"]), -r["pa_count"], r["name"]))

    return filtered[:limit]
