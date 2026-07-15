# ============================================================
# FASTAPI REST ENDPOINT
# Makes your IDS queryable by any system
# Run: uvicorn api.rest_api:app --reload --port 8000
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import numpy as np
import torch
import pickle
import os
import sys
import uvicorn

sys.path.append(r"C:\ids_project")
sys.path.append(r"C:\ids_project\models")

from database.attack_logger import (
    get_recent_attacks, get_attack_summary,
    get_top_sources, get_stats, log_attack
)

DATA_DIR  = r"C:\ids_project\data"
MODEL_DIR = r"C:\ids_project\models"
RESULT_DIR = r"C:\ids_project\results"

# ── Load artifacts ───────────────────────────────────────────
with open(f"{DATA_DIR}/classes.txt",
          encoding="utf-8") as f:
    classes = f.read().strip().split("\n")

with open(f"{MODEL_DIR}/model_config.pkl", "rb") as f:
    config = pickle.load(f)

benign_idx = classes.index("BENIGN")
threshold  = config["threshold"]

# ── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title       = "AI-Powered IDS API",
    description = """
## 🛡️ AI-Powered Intrusion Detection System

REST API for querying the FL-based IDS system.

### Features:
- **Real-time classification** of network packets
- **Attack history** from SQLite database
- **FL training status** and metrics
- **SHAP feature importance** data
- **System health** monitoring

Built with: LSTM Autoencoder + Federated Learning +
Differential Privacy + SHAP/LIME
    """,
    version     = "1.0.0",
)

# Allow dashboard to query API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Request/Response Models ──────────────────────────────────
class PacketFeatures(BaseModel):
    features: list[float]
    src_ip:   Optional[str] = "0.0.0.0"
    dst_ip:   Optional[str] = "0.0.0.0"
    port:     Optional[int] = 0

class ClassificationResult(BaseModel):
    is_attack:       bool
    attack_type:     str
    confidence:      float
    recon_error:     float
    threshold:       float
    severity:        str
    timestamp:       str
    recommendation:  str

class SystemStatus(BaseModel):
    status:          str
    model_loaded:    bool
    threshold:       float
    total_classes:   int
    fl_rounds:       int
    dp_epsilon:      float
    timestamp:       str

# ── Load Model ───────────────────────────────────────────────
device     = torch.device("cpu")
model      = None
model_loaded = False

def load_model():
    global model, model_loaded

    class IDSNet(torch.nn.Module):
        def __init__(self, inp, out):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(inp, 256),
                torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(256, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, out)
            )
        def forward(self, x):
            return self.net(x)

    try:
        m = IDSNet(
            config["input_size"],
            len(classes)
        ).to(device)

        m.load_state_dict(torch.load(
            f"{MODEL_DIR}/fl_global_model.pth",
            map_location=device
        ))
        m.eval()

        # Verify it works
        with torch.no_grad():
            test = torch.zeros(1, config["input_size"])
            _    = m(test)

        model        = m
        model_loaded = True
        globals()["IDSNet"] = IDSNet
        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"⚠️ Model load warning: {e}")
        model_loaded = False

load_model()

def get_severity(attack_type):
    if attack_type in ["DDoS", "Bot", "Infiltration"]:
        return "CRITICAL"
    elif "DoS" in attack_type or \
         attack_type == "Heartbleed":
        return "HIGH"
    elif attack_type in ["PortScan", "FTP-Patator",
                          "SSH-Patator"]:
        return "MEDIUM"
    elif attack_type == "BENIGN":
        return "NONE"
    return "LOW"

def get_recommendation(attack_type, severity):
    recs = {
        "DDoS":        "Block source IP. Enable rate limiting. Contact ISP.",
        "Bot":         "Isolate infected machine. Run malware scan. Reset credentials.",
        "Infiltration":"Immediate network isolation. Forensic analysis required.",
        "PortScan":    "Monitor source IP. Check firewall rules.",
        "Heartbleed":  "Patch OpenSSL immediately. Rotate SSL certificates.",
        "BENIGN":      "No action required. Traffic is normal.",
    }
    return recs.get(
        attack_type,
        f"Investigate {attack_type}. Review logs."
    )

# ── ENDPOINTS ────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    """API welcome endpoint."""
    return {
        "message": "🛡️ AI-Powered IDS API",
        "version": "1.0.0",
        "docs":    "http://localhost:8000/docs",
        "status":  "online",
        "model":   "LSTM Autoencoder + FL + DP"
    }


@app.get("/api/status",
         response_model=SystemStatus,
         tags=["System"])
def get_status():
    """Get system health and model status."""
    fl_log = f"{RESULT_DIR}/fl_log.txt"
    fl_rounds = 0
    dp_epsilon = 0.0

    if os.path.exists(fl_log):
        import pandas as pd
        try:
            df = pd.read_csv(fl_log)
            fl_rounds  = len(df)
            dp_epsilon = float(
                df["DP_Epsilon"].iloc[-1]
            )
        except Exception:
            pass

    return SystemStatus(
        status        = "online",
        model_loaded  = model_loaded,
        threshold     = threshold,
        total_classes = len(classes),
        fl_rounds     = fl_rounds,
        dp_epsilon    = dp_epsilon,
        timestamp     = datetime.now().isoformat()
    )


@app.post("/api/classify",
          response_model=ClassificationResult,
          tags=["Detection"])
def classify_packet(packet: PacketFeatures):
    """
    Classify a network packet as normal or attack.

    Send 78 numerical features extracted from a
    network packet. Returns attack classification
    with confidence score and recommendations.
    """
    if not model_loaded:
        raise HTTPException(
            status_code = 503,
            detail      = "Model not loaded"
        )

    features = np.array(packet.features)

    if len(features) != config["input_size"]:
        raise HTTPException(
            status_code = 422,
            detail      = (
                f"Expected {config['input_size']} "
                f"features, got {len(features)}"
            )
        )

    # Create sequence
    seq_len  = config["seq_len"]
    sequence = np.tile(
        features, (seq_len, 1)
    ).reshape(1, seq_len, -1)

    # Run model — classifier approach
    features_tensor = torch.FloatTensor(
        features.reshape(1, -1)
    ).to(device)

    with torch.no_grad():
        output      = model(features_tensor)
        probs       = torch.softmax(output, dim=1)
        pred_class  = output.argmax(dim=1).item()
        confidence  = float(probs[0][pred_class]) * 100

    attack_type = classes[pred_class]
    is_attack   = attack_type != "BENIGN"
    error       = 1 - float(probs[0][pred_class])
    severity    = get_severity(attack_type)
    # Log to database
    if is_attack:
        log_attack(
            attack_type = attack_type,
            src_ip      = packet.src_ip,
            dst_ip      = packet.dst_ip,
            port        = packet.port,
            recon_error = error,
            confidence  = confidence,
        )

    return ClassificationResult(
        is_attack      = is_attack,
        attack_type    = attack_type,
        confidence     = round(confidence, 2),
        recon_error    = round(error, 6),
        threshold      = round(threshold, 6),
        severity       = severity,
        timestamp      = datetime.now().isoformat(),
        recommendation = get_recommendation(
            attack_type, severity
        )
    )


@app.get("/api/alerts",
         tags=["Detection"])
def get_alerts(limit: int = 50):
    """Get recent attack alerts from database."""
    try:
        df = get_recent_attacks(limit)
        return {
            "count":   len(df),
            "alerts":  df.to_dict(orient="records"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"count": 0, "alerts": [],
                "error": str(e)}


@app.get("/api/stats",
         tags=["Analytics"])
def get_statistics():
    """Get overall attack statistics."""
    try:
        stats = get_stats()
        return {
            **stats,
            "timestamp": datetime.now().isoformat(),
            "model": {
                "threshold":  threshold,
                "classes":    len(classes),
                "attack_types": [
                    c for c in classes
                    if c != "BENIGN"
                ]
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/stats/summary",
         tags=["Analytics"])
def get_summary(hours: int = 24):
    """Get attack summary for last N hours."""
    try:
        df = get_attack_summary(hours)
        return {
            "hours":   hours,
            "summary": df.to_dict(orient="records"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/stats/top-sources",
         tags=["Analytics"])
def get_sources(limit: int = 10):
    """Get top attacking IP addresses."""
    try:
        df = get_top_sources(limit)
        return {
            "top_sources": df.to_dict(
                orient="records"
            ),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/fl/progress",
         tags=["Federated Learning"])
def get_fl_progress():
    """Get FL training progress and metrics."""
    fl_log = f"{RESULT_DIR}/fl_log.txt"

    if not os.path.exists(fl_log):
        raise HTTPException(
            status_code = 404,
            detail      = "FL training not started"
        )

    try:
        import pandas as pd
        df = pd.read_csv(fl_log)
        return {
            "rounds_completed": len(df),
            "final_accuracy":   float(
                df["Accuracy"].iloc[-1]
            ),
            "final_f1":         float(
                df["F1"].iloc[-1]
            ),
            "dp_epsilon":       float(
                df["DP_Epsilon"].iloc[-1]
            ),
            "history":          df.to_dict(
                orient="records"
            ),
            "timestamp":        datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/xai/features",
         tags=["Explainability"])
def get_feature_importance():
    """Get SHAP feature importance scores."""
    summary_path = (
        f"{RESULT_DIR}/xai/feature_summary.pkl"
    )

    if not os.path.exists(summary_path):
        raise HTTPException(
            status_code = 404,
            detail      = "Run xai_explainer.py first"
        )

    with open(summary_path, "rb") as f:
        summary = pickle.load(f)

    return {
        "top_features": [
            {
                "feature":    feat,
                "importance": round(float(score), 6)
            }
            for feat, score in zip(
                summary["top_features"],
                summary["importance_scores"]
            )
        ],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/classes",
         tags=["General"])
def get_classes():
    """Get all attack types the model detects."""
    return {
        "total":   len(classes),
        "classes": [
            {
                "id":       i,
                "name":     c,
                "severity": get_severity(c),
                "is_attack": c != "BENIGN"
            }
            for i, c in enumerate(classes)
        ]
    }


if __name__ == "__main__":
    print("🚀 Starting IDS REST API...")
    print("📖 Docs: http://localhost:8000/docs")
    print("🔍 API:  http://localhost:8000/api/status")
    uvicorn.run(
        app, host="0.0.0.0",
        port=8000, reload=False
    )