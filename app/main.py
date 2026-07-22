"""
app/main.py — FastAPI application: startup, routes, static file serving.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import model_loader
from .data import get_app_data, load_all
from .inference import run_prediction
from .search import get_stadiums, search_players

STATIC_DIR = os.environ.get("STATIC_DIR", "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pull the latest model bundle from S3 (no-op in local dev) before loading.
    model_loader.sync_from_s3()
    load_all()
    yield


app = FastAPI(title="MLB At-Bat Outcome Predictor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/stadiums")
def api_stadiums():
    return get_stadiums()


@app.get("/api/search/batters")
def api_search_batters(q: str = Query(default="", min_length=0)):
    ad = get_app_data()
    results = search_players(
        query=q,
        player_index=ad.batter_index,
        vocab=ad.batter_vocab,
        pa_counts=ad.batter_pa_counts,
        min_pa=50,
        limit=10,
    )
    return results


@app.get("/api/search/pitchers")
def api_search_pitchers(q: str = Query(default="", min_length=0)):
    ad = get_app_data()
    results = search_players(
        query=q,
        player_index=ad.pitcher_index,
        vocab=ad.pitcher_vocab,
        pa_counts=ad.pitcher_pa_counts,
        min_pa=30,
        limit=10,
    )
    return results


class PredictRequest(BaseModel):
    batter_id: int
    pitcher_id: int
    stadium: str


@app.post("/api/predict")
def api_predict(req: PredictRequest):
    ad = get_app_data()

    # Validate stadium token
    from .search import STADIUM_INFO
    stadium_token = req.stadium.strip().upper()
    if stadium_token not in STADIUM_INFO and stadium_token not in ad.stadium_vocab:
        raise HTTPException(status_code=400, detail=f"Unknown stadium token: {req.stadium!r}")

    result = run_prediction(req.batter_id, req.pitcher_id, stadium_token, ad)
    probs          = result["probs"]
    batter_rates   = result["batter_rates"]
    pitcher_rates  = result["pitcher_rates"]

    outcome_labels = {
        "K":    "Strikeout",
        "BIPO": "Ball In Play Out",
        "BB":   "Walk / HBP",
        "1B":   "Single",
        "XBH":  "Double / Triple",
        "HR":   "Home Run",
    }

    outcomes = []
    for label, prob in probs.items():
        league_avg = ad.league_rates[ad.label_set.index(label)]
        vs_pct = (prob - league_avg) / max(league_avg, 1e-9) * 100
        outcomes.append({
            "code":         label,
            "name":         outcome_labels.get(label, label),
            "prob":         prob,
            "league_avg":   league_avg,
            "batter_avg":   batter_rates.get(label, league_avg),
            "pitcher_avg":  pitcher_rates.get(label, league_avg),
            "vs_pct":       vs_pct,
        })

    # Resolve display names from index
    def _display_name(player_index, mlbam_id):
        for entry in player_index.values():
            if int(entry["id"]) == mlbam_id:
                return entry["name"]
        return str(mlbam_id)

    batter_name  = _display_name(ad.batter_index, req.batter_id)
    pitcher_name = _display_name(ad.pitcher_index, req.pitcher_id)

    return {
        "outcomes": outcomes,
        "batter":  {"name": batter_name,  "mlbam_id": req.batter_id,  "stand":    result["stand"],    "in_vocab": result["batter_in_vocab"]},
        "pitcher": {"name": pitcher_name, "mlbam_id": req.pitcher_id, "p_throws": result["p_throws"], "in_vocab": result["pitcher_in_vocab"]},
        "stadium": stadium_token,
    }


# ── Static files / SPA catch-all ─────────────────────────────────────────────

if os.path.isdir(STATIC_DIR):
    # Mount assets (JS/CSS chunks) at /assets
    assets_dir = os.path.join(STATIC_DIR, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_catch_all(full_path: str):
        # Serve specific static files if they exist
        candidate = os.path.join(STATIC_DIR, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))
