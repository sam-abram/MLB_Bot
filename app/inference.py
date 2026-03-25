"""
app/inference.py — Pure inference logic, ported from livematchup.py.
No file I/O per request; all data passed as arguments from AppData.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F

OUTCOMES = ["K", "BIPO", "BB", "1B", "XBH", "HR"]
PITCH_TYPES = ["FF", "SI", "FC", "SL", "CU", "CH", "FS", "ST", "SV", "OTHER"]
HAND_VOCAB = {"L": 2, "R": 3}


def _safe_logit(p: float, eps: float = 1e-6) -> float:
    p_clamped = max(eps, min(p, 1.0 - eps))
    return math.log(p_clamped / (1.0 - p_clamped))


def _get_row(df: pd.DataFrame, id_col: str, key) -> Optional[pd.Series]:
    if df is None or df.empty or id_col not in df.columns:
        return None
    sub = df[df[id_col] == key]
    return sub.iloc[0] if not sub.empty else None


def compute_six_vectors_single(
    batter_raw_id: int,
    pitcher_raw_id: int,
    stadium_raw_id: str,
    stand: str,
    p_throws: str,
    sv_batter_overall: pd.DataFrame,
    sv_pitcher_overall: pd.DataFrame,
    sv_stadium_rates: pd.DataFrame,
    sv_batter_platoon: pd.DataFrame,
    sv_pitcher_platoon: pd.DataFrame,
    sv_batter_pitch_type: pd.DataFrame,
    sv_pitcher_mix: pd.DataFrame,
    league_rates: List[float],
    league_logodds: List[float],
) -> Dict[str, float]:
    """Compute 42 six-vector features + 3 extremeness features for one PA."""
    lr = league_rates
    feats: Dict[str, float] = {}

    bid = int(batter_raw_id)
    pid = int(pitcher_raw_id)

    bo  = _get_row(sv_batter_overall,    "batter_id",  bid)
    po  = _get_row(sv_pitcher_overall,   "pitcher_id", pid)
    sr  = _get_row(sv_stadium_rates,     "stadium_id", stadium_raw_id)
    bp  = _get_row(sv_batter_platoon,    "batter_id",  bid)
    pp  = _get_row(sv_pitcher_platoon,   "pitcher_id", pid)
    bpt = _get_row(sv_batter_pitch_type, "batter_id",  bid)
    pm  = _get_row(sv_pitcher_mix,       "pitcher_id", pid)

    # V1: Batter overall
    for i, oc in enumerate(OUTCOMES):
        raw = float(bo[f"batter_{oc}_rate"]) if bo is not None and f"batter_{oc}_rate" in bo.index else lr[i]
        feats[f"v1_batter_{oc}"] = _safe_logit(raw) - league_logodds[i]
    feats["v1_batter_log_n"] = math.log1p(float(bo["batter_n_pa"])) if bo is not None else 0.0

    # V2: Pitcher overall
    for i, oc in enumerate(OUTCOMES):
        raw = float(po[f"pitcher_{oc}_rate"]) if po is not None and f"pitcher_{oc}_rate" in po.index else lr[i]
        feats[f"v2_pitcher_{oc}"] = _safe_logit(raw) - league_logodds[i]
    feats["v2_pitcher_log_n"] = math.log1p(float(po["pitcher_n_pa"])) if po is not None else 0.0

    # V3: Stadium
    for i, oc in enumerate(OUTCOMES):
        raw = float(sr[f"stadium_{oc}_rate"]) if sr is not None and f"stadium_{oc}_rate" in sr.index else lr[i]
        feats[f"v3_stadium_{oc}"] = _safe_logit(raw) - league_logodds[i]
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
    pitcher_n = float(pm["pitcher_n_pitches"]) if pm is not None and "pitcher_n_pitches" in pm.index and not pd.isna(pm["pitcher_n_pitches"]) else 0.0
    batter_n = float(bo["batter_n_pa"]) if bo is not None else 0.0
    feats["v6_mix_log_n"] = math.log1p(min(pitcher_n, batter_n))

    # Extremeness features
    _vec_prefixes = ["v1_batter", "v2_pitcher", "v3_stadium", "v4_batter_platoon", "v5_pitcher_platoon", "v6_mix"]
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


def merge_pt_statcast(
    batter_raw_id: int,
    pitcher_raw_id: int,
    pt_pitcher_df: Optional[pd.DataFrame],
    pt_batter_df: Optional[pd.DataFrame],
    pt_statcast_cols: List[str],
) -> Dict[str, float]:
    """Extract pitchtype statcast features for this matchup from indexed DataFrames."""
    feats: Dict[str, float] = {c: 0.0 for c in pt_statcast_cols}

    if pt_pitcher_df is not None and pitcher_raw_id in pt_pitcher_df.index:
        row = pt_pitcher_df.loc[pitcher_raw_id]
        for col in pt_pitcher_df.columns:
            if col in feats:
                val = row[col]
                feats[col] = 0.0 if pd.isna(val) else float(val)

    if pt_batter_df is not None and batter_raw_id in pt_batter_df.index:
        row = pt_batter_df.loc[batter_raw_id]
        for col in pt_batter_df.columns:
            if col in feats:
                val = row[col]
                feats[col] = 0.0 if pd.isna(val) else float(val)

    return feats


def run_prediction(
    batter_raw_id: int,
    pitcher_raw_id: int,
    stadium_token: str,
    ad,  # AppData instance
) -> Dict[str, float]:
    """
    Full inference pipeline. Returns {label: probability} dict.
    ad is the AppData singleton.
    """
    # Handedness
    stand = ad.batter_hand.get(str(batter_raw_id), "R")
    p_throws = ad.pitcher_hand.get(str(pitcher_raw_id), "R")

    # Switch hitter logic: if batter appears with both hands, use opposite of pitcher
    # (The handedness cache stores the MOST COMMON stance; for true switch hitters
    #  this may be ambiguous — we apply the standard convention.)
    # Simple implementation: if stand == "S" treat as switch hitter
    if stand == "S":
        stand = "L" if p_throws == "R" else "R"

    # Encode categorical IDs
    batter_enc  = ad.batter_vocab.get(str(batter_raw_id), 1)
    pitcher_enc = ad.pitcher_vocab.get(str(pitcher_raw_id), 1)
    stadium_enc = ad.stadium_vocab.get(stadium_token, 1)
    stand_enc   = HAND_VOCAB.get(stand, 1)
    p_throws_enc = HAND_VOCAB.get(p_throws, 1)

    # Six-vector features
    six_feats = compute_six_vectors_single(
        batter_raw_id, pitcher_raw_id, stadium_token,
        stand, p_throws,
        ad.sv_batter_overall, ad.sv_pitcher_overall, ad.sv_stadium_rates,
        ad.sv_batter_platoon, ad.sv_pitcher_platoon,
        ad.sv_batter_pitch_type, ad.sv_pitcher_mix,
        ad.league_rates, ad.league_logodds,
    )

    # PT statcast features
    pt_feats = merge_pt_statcast(
        batter_raw_id, pitcher_raw_id,
        ad.pt_pitcher_df, ad.pt_batter_df, ad.pt_cols,
    )

    # Assemble feature dict
    cat_set = set(ad.cat_cols)
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

    # Build tensors in exact feature_list_ordered order
    x_cat_list = [cat_values[c] for c in ad.cat_cols]
    num_cols_ordered = [c for c in ad.feature_list_ordered if c not in cat_set]
    x_num_list = [float(num_feats.get(c, 0.0)) for c in num_cols_ordered]

    x_cat = torch.tensor([x_cat_list], dtype=torch.long)
    x_num = torch.tensor([x_num_list], dtype=torch.float32)

    with torch.no_grad():
        logits = ad.model(x_cat, x_num)
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()

    # Batter and pitcher overall rates (shrinkage-smoothed)
    bo = _get_row(ad.sv_batter_overall,  "batter_id",  int(batter_raw_id))
    po = _get_row(ad.sv_pitcher_overall, "pitcher_id", int(pitcher_raw_id))
    batter_rates  = {oc: float(bo[f"batter_{oc}_rate"])  if bo  is not None and f"batter_{oc}_rate"  in bo.index  else ad.league_rates[i] for i, oc in enumerate(OUTCOMES)}
    pitcher_rates = {oc: float(po[f"pitcher_{oc}_rate"]) if po  is not None and f"pitcher_{oc}_rate" in po.index else ad.league_rates[i] for i, oc in enumerate(OUTCOMES)}

    return {
        "probs": {label: float(p) for label, p in zip(ad.label_set, probs)},
        "batter_rates":  batter_rates,
        "pitcher_rates": pitcher_rates,
        "stand": stand,
        "p_throws": p_throws,
        "batter_in_vocab": str(batter_raw_id) in ad.batter_vocab,
        "pitcher_in_vocab": str(pitcher_raw_id) in ad.pitcher_vocab,
    }
