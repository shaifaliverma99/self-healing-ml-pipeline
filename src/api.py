"""FastAPI serving layer: exposes the self-healing pipeline as a real HTTP service.

GET  /         -- service info and links (so the bare URL isn't a 404)
POST /predict   -- classify one sample, logging it for drift monitoring
GET  /status    -- current model version, drift-detector state, cumulative cost
POST /reset     -- reinitialize with a fresh synthetic stream (demo convenience)

Run locally with: uvicorn src.api:app --reload
"""
import threading
from collections import deque
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.model import OnlineModel
from src.drift_detectors import ADWIN
from src.cost_tracker import CostTracker
from src.stream_generator import sea_stream

app = FastAPI(title="Self-Healing ML Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

N_FEATURES = 3
INIT_SIZE = 500
BUFFER_SIZE = 500
COOLDOWN = 200

_state = {}
_state_lock = threading.Lock()


def _init_state():
    X, y, _ = sea_stream(n_samples=INIT_SIZE)
    model = OnlineModel(n_features=N_FEATURES)
    model.fit_initial(X, y)
    _state["model"] = model
    _state["detector"] = ADWIN()
    _state["cost"] = CostTracker()
    _state["buffer_X"] = deque(maxlen=BUFFER_SIZE)
    _state["buffer_y"] = deque(maxlen=BUFFER_SIZE)
    _state["n_seen"] = 0
    _state["last_retrain_step"] = -COOLDOWN
    _state["drift_events"] = []


_init_state()


class PredictRequest(BaseModel):
    features: List[float]
    true_label: int | None = None  # optional, enables online drift monitoring


class PredictResponse(BaseModel):
    prediction: int
    model_version: int
    drift_detected: bool


@app.get("/")
def root():
    return {
        "service": "Self-Healing ML Pipeline",
        "docs": "/docs",
        "endpoints": {
            "POST /predict": "classify one sample; pass true_label to enable drift monitoring",
            "GET /status": "model version, drift events, cumulative cost",
            "POST /reset": "reinitialize with a fresh synthetic stream",
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != N_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"expected {N_FEATURES} features, got {len(req.features)}",
        )
    x = np.array(req.features)

    with _state_lock:
        model: OnlineModel = _state["model"]
        pred = model.predict(x)
        _state["cost"].record_inference()
        _state["n_seen"] += 1

        drift = False
        if req.true_label is not None:
            correct = int(pred == req.true_label)
            _state["buffer_X"].append(x)
            _state["buffer_y"].append(req.true_label)
            _, drift = _state["detector"].update(correct)
            i = _state["n_seen"]
            if drift and (i - _state["last_retrain_step"]) > COOLDOWN and len(_state["buffer_X"]) >= 30:
                retrained = model.retrain(np.array(_state["buffer_X"]), np.array(_state["buffer_y"]))
                if retrained:
                    _state["cost"].record_retrain()
                    _state["last_retrain_step"] = i
                    _state["drift_events"].append(i)

        model_version = model.version

    return PredictResponse(prediction=pred, model_version=model_version, drift_detected=drift)


@app.get("/status")
def status():
    with _state_lock:
        return {
            "n_requests_seen": _state["n_seen"],
            "model_version": _state["model"].version,
            "drift_events": list(_state["drift_events"]),
            "cost": _state["cost"].summary(),
        }


@app.post("/reset")
def reset():
    with _state_lock:
        _init_state()
    return {"status": "reset"}
