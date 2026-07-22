"""
app/data.py — In-memory application state, loaded once at startup.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import torch

PREPROCESSED_DIR = os.environ.get("PREPROCESSED_DIR", "preprocessed")
MODEL_DIR = os.environ.get("MODEL_DIR", "model_artifacts")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class AppData:
    # Model
    model: object = None
    # Feature configuration
    feature_list_ordered: List[str] = field(default_factory=list)
    cat_cols: List[str] = field(default_factory=list)
    num_cols: List[str] = field(default_factory=list)
    pt_cols: List[str] = field(default_factory=list)
    label_set: List[str] = field(default_factory=list)
    # Vocabs (raw_id_str -> encoded_int)
    batter_vocab: Dict[str, int] = field(default_factory=dict)
    pitcher_vocab: Dict[str, int] = field(default_factory=dict)
    stadium_vocab: Dict[str, int] = field(default_factory=dict)
    # Six-vector DataFrames indexed by ID
    sv_batter_overall: Optional[pd.DataFrame] = None
    sv_pitcher_overall: Optional[pd.DataFrame] = None
    sv_stadium_rates: Optional[pd.DataFrame] = None
    sv_batter_platoon: Optional[pd.DataFrame] = None
    sv_pitcher_platoon: Optional[pd.DataFrame] = None
    sv_batter_pitch_type: Optional[pd.DataFrame] = None
    sv_pitcher_mix: Optional[pd.DataFrame] = None
    # PT statcast DataFrames indexed by player_id
    pt_pitcher_df: Optional[pd.DataFrame] = None
    pt_batter_df: Optional[pd.DataFrame] = None
    # League stats
    league_rates: List[float] = field(default_factory=list)
    league_logodds: List[float] = field(default_factory=list)
    # Player name index: norm_name -> {id, name}
    batter_index: Dict[str, dict] = field(default_factory=dict)
    pitcher_index: Dict[str, dict] = field(default_factory=dict)
    # Handedness: str(mlbam_id) -> "L"/"R"
    batter_hand: Dict[str, str] = field(default_factory=dict)
    pitcher_hand: Dict[str, str] = field(default_factory=dict)
    # PA counts for search filtering: mlbam_id (int) -> count
    batter_pa_counts: Dict[int, float] = field(default_factory=dict)
    pitcher_pa_counts: Dict[int, float] = field(default_factory=dict)


# Global singleton
_app_data: Optional[AppData] = None


def get_app_data() -> AppData:
    if _app_data is None:
        raise RuntimeError("AppData not loaded. Call load_all() at startup.")
    return _app_data


def load_all() -> AppData:
    global _app_data

    pre = PREPROCESSED_DIR
    art = MODEL_DIR

    print(f"[startup] Loading data from preprocessed={pre!r}, artifacts={art!r}")

    # --- Metadata + config ---
    # metadata.json defines the feature order, categorical columns and vocab
    # sizes the model was built with, so it must come from the same bundle as
    # model.pt. Training copies it into model_artifacts/ next to the checkpoint,
    # and model_loader routes the S3 copy there too — preprocessed/metadata.json
    # is never refreshed by a sync and goes stale (it is whatever was baked into
    # the image). Prefer the artifact copy; fall back only for local dev runs
    # that never produced one.
    meta_path = os.path.join(art, "metadata.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(pre, "metadata.json")
    print(f"[startup] Reading metadata from {meta_path!r}")
    meta = json.load(open(meta_path))
    train_config = json.load(open(os.path.join(art, "train_config.json")))

    feature_list_ordered = meta["features"]["feature_list_ordered"]
    cat_cols = meta["features"].get("categorical_features", [])
    cat_set = set(cat_cols)
    num_cols = [c for c in feature_list_ordered if c not in cat_set]
    label_set = meta["labels"]["label_set_ordered"]

    # --- Vocabs ---
    vocabs = json.load(open(os.path.join(pre, "vocabs.json")))
    batter_vocab = {str(k): int(v) for k, v in vocabs["batter_id"].items()}
    pitcher_vocab = {str(k): int(v) for k, v in vocabs["pitcher_id"].items()}
    stadium_vocab = {str(k): int(v) for k, v in vocabs["stadium_id"].items()}

    # --- League stats ---
    lr_data = json.load(open(os.path.join(pre, "league_rates.json")))
    league_rates = list(lr_data["rates"])
    ll_data = json.load(open(os.path.join(pre, "league_logodds.json")))
    league_logodds = list(ll_data["logodds"])

    # --- Six-vector DataFrames (not indexed — lookup by column match) ---
    def _load_sv(fname):
        p = os.path.join(pre, fname)
        return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()

    sv_batter_overall = _load_sv("six_vector_batter_overall.parquet")
    sv_pitcher_overall = _load_sv("six_vector_pitcher_overall.parquet")
    sv_stadium_rates = _load_sv("six_vector_stadium_rates.parquet")
    sv_batter_platoon = _load_sv("six_vector_batter_platoon.parquet")
    sv_pitcher_platoon = _load_sv("six_vector_pitcher_platoon.parquet")
    sv_batter_pitch_type = _load_sv("six_vector_batter_pitch_type.parquet")
    sv_pitcher_mix = _load_sv("six_vector_pitcher_mix.parquet")

    # --- PT statcast DataFrames (indexed by player_id for fast .loc[] lookup) ---
    def _load_pt(fname, id_col):
        p = os.path.join(pre, fname)
        if not os.path.exists(p):
            return None
        df = pd.read_parquet(p)
        if id_col in df.columns:
            df = df.set_index(id_col)
        return df

    pt_pitcher_df = _load_pt("pitchtype_statcast_v2_pitcher.parquet", "pitcher_id")
    pt_batter_df = _load_pt("pitchtype_statcast_v2_batter.parquet", "batter_id")

    # --- PT statcast columns ---
    pt_prefixes = (
        "pitcher_pitch_rate_",
        "pitcher_FF_", "pitcher_SI_", "pitcher_FC_", "pitcher_SL_",
        "pitcher_CU_", "pitcher_CH_", "pitcher_FS_", "pitcher_ST_",
        "pitcher_SV_", "pitcher_OTHER_",
        "batter_FF_", "batter_SI_", "batter_FC_", "batter_SL_",
        "batter_CU_", "batter_CH_", "batter_FS_", "batter_ST_",
        "batter_SV_", "batter_OTHER_",
    )
    pt_cols = [c for c in num_cols if any(c.startswith(p) for p in pt_prefixes)]

    # --- PA counts from six-vector tables ---
    batter_pa_counts: Dict[int, float] = {}
    if not sv_batter_overall.empty and "batter_id" in sv_batter_overall.columns:
        for _, row in sv_batter_overall.iterrows():
            batter_pa_counts[int(row["batter_id"])] = float(row.get("batter_n_pa", 0))
    pitcher_pa_counts: Dict[int, float] = {}
    if not sv_pitcher_overall.empty and "pitcher_id" in sv_pitcher_overall.columns:
        for _, row in sv_pitcher_overall.iterrows():
            pitcher_pa_counts[int(row["pitcher_id"])] = float(row.get("pitcher_n_pa", 0))

    # --- Player name index ---
    cache_path = os.path.join(pre, "name_id_cache.json")
    batter_index: Dict[str, dict] = {}
    pitcher_index: Dict[str, dict] = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))
        batter_index = {k: v for k, v in cache.get("batters", {}).items()}
        pitcher_index = {k: v for k, v in cache.get("pitchers", {}).items()}
        print(f"[startup] Loaded {len(batter_index)} batters, {len(pitcher_index)} pitchers from name cache")
    else:
        print("[startup] WARNING: name_id_cache.json not found — player search will return no results")

    # --- Handedness cache ---
    batter_hand: Dict[str, str] = {}
    pitcher_hand: Dict[str, str] = {}
    hand_path = os.path.join(pre, "handedness_cache.json")
    if os.path.exists(hand_path):
        hc = json.load(open(hand_path))
        batter_hand = {str(k): v for k, v in hc.get("batters", {}).items()}
        pitcher_hand = {str(k): v for k, v in hc.get("pitchers", {}).items()}
        print(f"[startup] Loaded handedness for {len(batter_hand)} batters, {len(pitcher_hand)} pitchers")
    else:
        print("[startup] WARNING: handedness_cache.json not found — defaulting all to R")

    # --- Model ---
    # Add project root to sys.path so we can import train_model2
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    import train_model2 as tm  # noqa: imported here intentionally

    vocab_sizes = meta["features"].get("categorical_vocab_sizes", {})
    n_classes = len(meta["labels"]["label_to_id"])
    hidden_dims = train_config.get("HIDDEN_DIMS", [256, 128])
    dropout = train_config.get("DROPOUT", 0.2)
    num_pt = len(pt_cols)

    state = torch.load(os.path.join(art, "model.pt"), map_location="cpu", weights_only=True)

    # The checkpoint is the authority on embedding sizes: each categorical column
    # has an `embeddings.<col>.weight` of shape [vocab_size, emb_dim]. Sizing from
    # the tensors themselves means a retrain that adds players can never desync
    # the architecture from the weights, whatever metadata.json says.
    ckpt_vocab_sizes = {
        col: int(state[f"embeddings.{col}.weight"].shape[0])
        for col in cat_cols
        if f"embeddings.{col}.weight" in state
    }
    for col, size in ckpt_vocab_sizes.items():
        declared = vocab_sizes.get(col)
        if declared is not None and int(declared) != size:
            print(f"[startup] WARNING: metadata vocab size for {col!r} is {declared}, "
                  f"checkpoint has {size}; using the checkpoint.")
    vocab_sizes = {**vocab_sizes, **ckpt_vocab_sizes}

    missing = [c for c in cat_cols if c not in vocab_sizes]
    if missing:
        raise RuntimeError(
            f"No vocab size for categorical columns {missing} in either "
            f"{meta_path} or the checkpoint's embedding weights."
        )

    model = tm.PerClassGateLogitHybridModel(
        cat_cols=cat_cols,
        num_numeric=len(num_cols),
        vocab_sizes=vocab_sizes,
        num_classes=n_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_pt_statcast_cols=num_pt,
        league_logodds=league_logodds,
    )
    model.load_state_dict(state)
    model.eval()
    print("[startup] Model loaded successfully")

    _app_data = AppData(
        model=model,
        feature_list_ordered=feature_list_ordered,
        cat_cols=cat_cols,
        num_cols=num_cols,
        pt_cols=pt_cols,
        label_set=label_set,
        batter_vocab=batter_vocab,
        pitcher_vocab=pitcher_vocab,
        stadium_vocab=stadium_vocab,
        sv_batter_overall=sv_batter_overall,
        sv_pitcher_overall=sv_pitcher_overall,
        sv_stadium_rates=sv_stadium_rates,
        sv_batter_platoon=sv_batter_platoon,
        sv_pitcher_platoon=sv_pitcher_platoon,
        sv_batter_pitch_type=sv_batter_pitch_type,
        sv_pitcher_mix=sv_pitcher_mix,
        pt_pitcher_df=pt_pitcher_df,
        pt_batter_df=pt_batter_df,
        league_rates=league_rates,
        league_logodds=league_logodds,
        batter_index=batter_index,
        pitcher_index=pitcher_index,
        batter_hand=batter_hand,
        pitcher_hand=pitcher_hand,
        batter_pa_counts=batter_pa_counts,
        pitcher_pa_counts=pitcher_pa_counts,
    )
    print("[startup] All data loaded.")
    return _app_data
