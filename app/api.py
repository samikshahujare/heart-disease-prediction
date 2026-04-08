from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException


logger = logging.getLogger("heart-disease-api")


def _ensure_project_on_path() -> None:
    # Ensure imports work regardless of how Uvicorn is launched.
    root = Path(__file__).resolve().parents[1]
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_project_on_path()

try:
    from model.predict import get_feature_metadata, predict_single
except Exception as e:  # pragma: no cover
    raise RuntimeError(f"Failed to import prediction module: {e}") from e


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    # Fail fast with a clear message if artifacts are missing.
    try:
        _ = get_feature_metadata()
    except Exception as e:
        logger.exception("Startup failed due to missing/invalid artifacts.")
        raise RuntimeError(
            "Model artifacts are missing or invalid. Run `python model/train_model.py` "
            "before starting the API."
        ) from e


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Request body must be a JSON object mapping every feature name to a value.
    Example:
      {
        "age": 63,
        "sex": 1,
        ...
      }
    """

    try:
        result = predict_single(features)
        # Ensure output only contains the expected fields.
        return {
            "prediction": int(result["prediction"]),
            "probability": float(result["probability"]),
            "confidence": float(result.get("confidence", 0.0)),
            "interpretation": result.get("interpretation", "Low Risk" if result.get("prediction", 0) == 0 else "High Risk"),
            "risk_level": result.get("risk_level", "Low Risk" if result.get("prediction", 0) == 0 else "High Risk"),
            "top_features": result.get("top_features", []),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e

