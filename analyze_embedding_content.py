#!/usr/bin/env python3
"""
analyze_embedding_content.py

Test whether learned embeddings are just encoding outcome rates.

Method:
1. Load trained model and extract batter/pitcher embeddings
2. Load explicit rate vectors from preprocessing
3. Regress embeddings against rates
4. Compute R2 to measure how much of embedding variance is explained by rates
"""

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuration
MODEL_PATH = "model_artifacts/model.pt"
METADATA_PATH = "model_artifacts/metadata.json"
VOCABS_PATH = "preprocessed/vocabs.json"
BATTER_OVERALL_PATH = "preprocessed/six_vector_batter_overall.parquet"
PITCHER_OVERALL_PATH = "preprocessed/six_vector_pitcher_overall.parquet"
BATTER_PLATOON_PATH = "preprocessed/six_vector_batter_platoon.parquet"
PITCHER_PLATOON_PATH = "preprocessed/six_vector_pitcher_platoon.parquet"
BATTER_PITCH_TYPE_PATH = "preprocessed/six_vector_batter_pitch_type.parquet"

OUTCOMES = ["K", "BIPO", "BB", "1B", "XBH", "HR"]
PITCH_TYPES = ["FF", "SI", "FC", "SL", "CU", "CH", "FS", "ST", "SV", "OTHER"]


def load_model_embeddings(model_path: str, vocabs_path: str) -> Dict:
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    batter_emb = None
    pitcher_emb = None

    for key, value in state_dict.items():
        if "batter_id" in key and "weight" in key:
            batter_emb = value.cpu().numpy()
            print(f"  Found batter embeddings: {key}, shape {batter_emb.shape}")
        if "pitcher_id" in key and "weight" in key:
            pitcher_emb = value.cpu().numpy()
            print(f"  Found pitcher embeddings: {key}, shape {pitcher_emb.shape}")

    if batter_emb is None or pitcher_emb is None:
        raise ValueError("Could not find batter_id or pitcher_id embeddings in model")

    with open(vocabs_path, "r") as f:
        vocabs = json.load(f)

    return {
        "batter_embeddings": batter_emb,
        "pitcher_embeddings": pitcher_emb,
        "batter_vocab": vocabs.get("batter_id", {}),
        "pitcher_vocab": vocabs.get("pitcher_id", {}),
    }


def build_rate_feature_matrix(
    player_ids: List[int],
    overall_df: pd.DataFrame,
    platoon_df: pd.DataFrame,
    pitch_type_df: pd.DataFrame,
    player_type: str,
) -> Tuple[np.ndarray, List[str]]:
    id_col = f"{player_type}_id"

    overall_indexed = overall_df.set_index(id_col) if id_col in overall_df.columns else overall_df
    feature_names = []
    feature_columns = []

    # Overall rates
    for outcome in OUTCOMES:
        col = f"{player_type}_{outcome}_rate"
        if col in overall_indexed.columns:
            feature_names.append(col)
            feature_columns.append(overall_indexed.reindex(player_ids)[col].fillna(0).values)

    # Sample size
    n_col = f"{player_type}_n_pa"
    if n_col in overall_indexed.columns:
        feature_names.append(f"{player_type}_log_n")
        feature_columns.append(np.log1p(overall_indexed.reindex(player_ids)[n_col].fillna(0).values))

    # Platoon rates
    if not platoon_df.empty and id_col in platoon_df.columns:
        platoon_indexed = platoon_df.set_index(id_col)
        for hand in ["L", "R"]:
            for outcome in OUTCOMES:
                col = f"{player_type}_{outcome}_vs_{hand}"
                if col in platoon_indexed.columns:
                    feature_names.append(col)
                    feature_columns.append(platoon_indexed.reindex(player_ids)[col].fillna(0).values)
            n_col = f"{player_type}_n_vs_{hand}"
            if n_col in platoon_indexed.columns:
                feature_names.append(f"{player_type}_log_n_vs_{hand}")
                feature_columns.append(np.log1p(platoon_indexed.reindex(player_ids)[n_col].fillna(0).values))

    # Pitch type rates (batter only)
    if not pitch_type_df.empty and id_col in pitch_type_df.columns:
        pt_indexed = pitch_type_df.set_index(id_col)
        for pt in PITCH_TYPES:
            for outcome in OUTCOMES:
                col = f"{player_type}_{outcome}_vs_{pt}"
                if col in pt_indexed.columns:
                    feature_names.append(col)
                    feature_columns.append(pt_indexed.reindex(player_ids)[col].fillna(0).values)

    X = np.column_stack(feature_columns)
    return X, feature_names


def regress_embeddings_on_rates(
    embeddings: np.ndarray,
    rates: np.ndarray,
    alpha: float = 1.0,
) -> Dict[str, float]:
    n = min(len(embeddings), len(rates))
    emb = embeddings[:n]
    rate = rates[:n]

    valid_mask = ~(np.isnan(emb).any(axis=1) | np.isnan(rate).any(axis=1))
    # Also skip zero-norm embeddings (PAD/UNK/unused)
    valid_mask &= np.linalg.norm(emb, axis=1) > 1e-6
    emb = emb[valid_mask]
    rate = rate[valid_mask]

    if len(emb) < 10:
        return {"error": "Too few valid samples", "num_samples": int(valid_mask.sum())}

    rate_scaler = StandardScaler()
    rate_scaled = rate_scaler.fit_transform(rate)

    emb_scaler = StandardScaler()
    emb_scaled = emb_scaler.fit_transform(emb)

    ridge = Ridge(alpha=alpha)
    ridge.fit(rate_scaled, emb_scaled)
    emb_pred = ridge.predict(rate_scaled)

    ss_tot = np.sum((emb_scaled - emb_scaled.mean(axis=0)) ** 2)
    ss_res = np.sum((emb_scaled - emb_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    per_dim_r2 = []
    for d in range(emb_scaled.shape[1]):
        ss_tot_d = np.sum((emb_scaled[:, d] - emb_scaled[:, d].mean()) ** 2)
        ss_res_d = np.sum((emb_scaled[:, d] - emb_pred[:, d]) ** 2)
        r2_d = 1 - (ss_res_d / ss_tot_d) if ss_tot_d > 0 else 0
        per_dim_r2.append(r2_d)

    return {
        "overall_r2": float(r2),
        "mean_per_dim_r2": float(np.mean(per_dim_r2)),
        "median_per_dim_r2": float(np.median(per_dim_r2)),
        "min_per_dim_r2": float(np.min(per_dim_r2)),
        "max_per_dim_r2": float(np.max(per_dim_r2)),
        "num_samples": int(len(emb)),
        "num_embedding_dims": int(emb.shape[1]),
        "num_rate_features": int(rate.shape[1]),
    }


def analyze_embedding_structure(embeddings: np.ndarray, label: str) -> Dict:
    emb = embeddings.copy()
    nonzero_mask = np.linalg.norm(emb, axis=1) > 1e-6
    emb = emb[nonzero_mask]

    if len(emb) < 10:
        return {"error": "Too few non-zero embeddings"}

    stats = {
        "num_players": int(len(emb)),
        "embed_dim": int(emb.shape[1]),
        "mean_norm": float(np.linalg.norm(emb, axis=1).mean()),
        "std_norm": float(np.linalg.norm(emb, axis=1).std()),
    }

    pca = PCA()
    pca.fit(emb)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    stats["pca_dims_for_90pct"] = int(np.searchsorted(cumvar, 0.90) + 1)
    stats["pca_dims_for_95pct"] = int(np.searchsorted(cumvar, 0.95) + 1)
    stats["top_5_pca_var"] = [round(float(v), 4) for v in pca.explained_variance_ratio_[:5]]

    return stats


def main():
    print("=" * 70)
    print("EMBEDDING CONTENT ANALYSIS")
    print("Testing: Do embeddings just learn outcome rates?")
    print("=" * 70)

    # 1. Load embeddings
    print("\n[1] Loading trained model embeddings...")
    emb_data = load_model_embeddings(MODEL_PATH, VOCABS_PATH)

    batter_emb = emb_data["batter_embeddings"]
    pitcher_emb = emb_data["pitcher_embeddings"]
    batter_vocab = emb_data["batter_vocab"]
    pitcher_vocab = emb_data["pitcher_vocab"]

    print(f"  Batter embeddings: {batter_emb.shape}")
    print(f"  Pitcher embeddings: {pitcher_emb.shape}")

    # 2. Load rate vectors
    print("\n[2] Loading explicit rate vectors...")
    batter_overall = pd.read_parquet(BATTER_OVERALL_PATH)
    pitcher_overall = pd.read_parquet(PITCHER_OVERALL_PATH)
    batter_platoon = pd.read_parquet(BATTER_PLATOON_PATH) if os.path.exists(BATTER_PLATOON_PATH) else pd.DataFrame()
    pitcher_platoon = pd.read_parquet(PITCHER_PLATOON_PATH) if os.path.exists(PITCHER_PLATOON_PATH) else pd.DataFrame()
    batter_pitch_type = pd.read_parquet(BATTER_PITCH_TYPE_PATH) if os.path.exists(BATTER_PITCH_TYPE_PATH) else pd.DataFrame()

    print(f"  Batter overall: {len(batter_overall)} rows")
    print(f"  Pitcher overall: {len(pitcher_overall)} rows")
    print(f"  Batter platoon: {len(batter_platoon)} rows")
    print(f"  Pitcher platoon: {len(pitcher_platoon)} rows")
    print(f"  Batter pitch type: {len(batter_pitch_type)} rows")

    # 3. Map vocab indices to player IDs
    print("\n[3] Mapping vocab indices to player IDs...")

    max_b = batter_emb.shape[0]
    max_p = pitcher_emb.shape[0]

    batter_ids_ordered = [None] * max_b
    for pid_str, idx in batter_vocab.items():
        if pid_str in ("__PAD__", "__UNK__"):
            continue
        idx = int(idx)
        if idx < max_b:
            batter_ids_ordered[idx] = int(pid_str)

    pitcher_ids_ordered = [None] * max_p
    for pid_str, idx in pitcher_vocab.items():
        if pid_str in ("__PAD__", "__UNK__"):
            continue
        idx = int(idx)
        if idx < max_p:
            pitcher_ids_ordered[idx] = int(pid_str)

    valid_b_idx = [i for i, pid in enumerate(batter_ids_ordered) if pid is not None]
    valid_b_ids = [batter_ids_ordered[i] for i in valid_b_idx]
    valid_p_idx = [i for i, pid in enumerate(pitcher_ids_ordered) if pid is not None]
    valid_p_ids = [pitcher_ids_ordered[i] for i in valid_p_idx]

    print(f"  Valid batters: {len(valid_b_ids)}")
    print(f"  Valid pitchers: {len(valid_p_ids)}")

    # 4. Build rate matrices
    print("\n[4] Building rate feature matrices...")

    batter_rates, batter_feat_names = build_rate_feature_matrix(
        valid_b_ids, batter_overall, batter_platoon, batter_pitch_type, "batter"
    )
    print(f"  Batter rate features: {len(batter_feat_names)} ({batter_feat_names[:5]}...)")

    pitcher_rates, pitcher_feat_names = build_rate_feature_matrix(
        valid_p_ids, pitcher_overall, pitcher_platoon, pd.DataFrame(), "pitcher"
    )
    print(f"  Pitcher rate features: {len(pitcher_feat_names)} ({pitcher_feat_names[:5]}...)")

    batter_emb_valid = batter_emb[valid_b_idx]
    pitcher_emb_valid = pitcher_emb[valid_p_idx]

    # 5. Embedding structure
    print("\n[5] Embedding structure analysis...")
    print("\n  BATTER EMBEDDINGS:")
    for k, v in analyze_embedding_structure(batter_emb, "batter").items():
        print(f"    {k}: {v}")

    print("\n  PITCHER EMBEDDINGS:")
    for k, v in analyze_embedding_structure(pitcher_emb, "pitcher").items():
        print(f"    {k}: {v}")

    # 6. Regression
    print("\n[6] Regressing embeddings on rate features...")
    print("    (R2 = how much of embedding variance is explained by rates)")

    # Try multiple Ridge alphas
    for alpha in [0.1, 1.0, 10.0]:
        print(f"\n  --- Ridge alpha={alpha} ---")

        print(f"\n  BATTER (emb dim={batter_emb_valid.shape[1]}, rate features={batter_rates.shape[1]}):")
        br = regress_embeddings_on_rates(batter_emb_valid, batter_rates, alpha=alpha)
        for k, v in br.items():
            print(f"    {k}: {v}")

        print(f"\n  PITCHER (emb dim={pitcher_emb_valid.shape[1]}, rate features={pitcher_rates.shape[1]}):")
        pr = regress_embeddings_on_rates(pitcher_emb_valid, pitcher_rates, alpha=alpha)
        for k, v in pr.items():
            print(f"    {k}: {v}")

    # Use alpha=1.0 results for interpretation
    br_final = regress_embeddings_on_rates(batter_emb_valid, batter_rates, alpha=1.0)
    pr_final = regress_embeddings_on_rates(pitcher_emb_valid, pitcher_rates, alpha=1.0)

    # 7. Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    batter_r2 = br_final.get("overall_r2", 0)
    pitcher_r2 = pr_final.get("overall_r2", 0)
    avg_r2 = (batter_r2 + pitcher_r2) / 2

    print(f"\n  Batter R2:  {batter_r2:.3f}")
    print(f"  Pitcher R2: {pitcher_r2:.3f}")
    print(f"  Average R2: {avg_r2:.3f}")

    if avg_r2 > 0.7:
        print("\n  CONCLUSION: HIGH R2 (>0.7)")
        print("  Embeddings are primarily learning outcome rates.")
        print("  The performance gap is likely due to better regularization.")
    elif avg_r2 > 0.4:
        print("\n  CONCLUSION: MODERATE R2 (0.4-0.7)")
        print("  Embeddings partially capture rates but also encode other structure.")
    else:
        print("\n  CONCLUSION: LOW R2 (<0.4)")
        print("  Embeddings contain substantial information not in explicit rates.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
