# MLB At-Bat Predictor

Predicts the outcome of MLB at-bats using a neural network trained on Statcast pitch-level data. Given a batter, pitcher, and stadium, the model outputs probabilities across six outcome classes: K, BIPO, BB, 1B, XBH, and HR.

**Live app: [atbatpredictor.com](https://atbatpredictor.com)**

## How It Works

The model combines categorical embeddings (batter, pitcher, stadium, handedness) with a six-vector feature system that captures shrinkage-adjusted log-odds deviations from league averages — batter tendencies, pitcher tendencies, stadium effects, platoon splits, and pitch-mix interactions. A per-class contextual gate blends base-rate logits with the neural network's predictions.

## Pipeline

1. **Data collection** — `data_pipeline2.py` downloads pitch-level Statcast data
2. **Preprocessing** — `preprocessing2.py` builds six-vectors, pitch-type statcast features, and train/val/test splits
3. **Training** — `train_model2.py` trains the PerClassGateLogitHybridModel with early stopping
4. **Inference** — `livematchup.py` runs single batter-vs-pitcher predictions

## Web App

FastAPI backend (`app/`) with a React frontend (`frontend/`). Dockerized for deployment.

```
docker-compose up --build
```

## Requirements

Python 3.11+, Node 20+ (for frontend build). See `requirements.txt` for Python dependencies.
