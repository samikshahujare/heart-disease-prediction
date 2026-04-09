from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException


logger = logging.getLogger("heart-disease-api")


def _ensure_backend_on_path() -> None:
    backend_root = Path(__file__).resolve().parents[1]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))


_ensure_backend_on_path()

try:
    from model.predict import get_feature_metadata, predict_single
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"Failed to import prediction module: {e}") from e


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    try:
        _ = get_feature_metadata()
    except Exception as e:
        logger.exception("Startup failed due to missing/invalid artifacts.")
        raise RuntimeError(
            "Model artifacts are missing or invalid in backend/model/. "
            "Ensure backend/model/model.pkl and backend/model/scaler.pkl exist."
        ) from e


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    try:
        result = predict_single(features)
        return {
            "prediction": int(result["prediction"]),
            "probability": float(result["probability"]),
            "confidence": float(result.get("confidence", 0.0)),
            "interpretation": result.get(
                "interpretation",
                "Low Risk" if int(result.get("prediction", 0)) == 0 else "High Risk",
            ),
            "risk_level": result.get(
                "risk_level",
                "Low Risk" if int(result.get("prediction", 0)) == 0 else "High Risk",
            ),
            "top_features": result.get("top_features", []),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

