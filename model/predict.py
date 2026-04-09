from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

# Ensure project root is importable so joblib can unpickle custom transformers
# from `model.outliers` reliably.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd

from model.feature_engineering import engineer_features


def _project_root() -> Path:
    return _ROOT


@lru_cache(maxsize=1)
def load_artifacts(
    model_path: str | None = None,
    scaler_path: str | None = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads the trained XGBoost model and the preprocessing artifact.
    """

    root = _project_root()
    model_file = Path(model_path) if model_path else (root / "model" / "model.pkl")
    scaler_file = Path(scaler_path) if scaler_path else (root / "model" / "scaler.pkl")

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_file}. Run `python model/train_model.py` first."
        )
    if not scaler_file.exists():
        raise FileNotFoundError(
            f"Scaler artifact not found: {scaler_file}. Run `python model/train_model.py` first."
        )

    model = joblib.load(model_file)
    scaler_artifact = joblib.load(scaler_file)

    if "preprocessor" not in scaler_artifact:
        raise ValueError("Invalid scaler artifact format: missing 'preprocessor'.")

    return model, scaler_artifact


def get_feature_metadata() -> Dict[str, Any]:
    """
    Returns the input schema needed for the UI:
    feature column order + categorical options + numeric defaults.
    """

    _, scaler_artifact = load_artifacts()
    # UI should only ask for ORIGINAL dataset features.
    return {
        "user_feature_columns": scaler_artifact.get("user_feature_columns", scaler_artifact["feature_columns"]),
        "user_numeric_features": scaler_artifact.get("user_numeric_features", scaler_artifact["numeric_features"]),
        "user_categorical_features": scaler_artifact.get(
            "user_categorical_features", scaler_artifact["categorical_features"]
        ),
        "categorical_values": scaler_artifact.get("categorical_values", {}),
        "numeric_defaults": scaler_artifact.get("numeric_defaults", {}),
        "target_name": scaler_artifact.get("target_name", "target"),
        "threshold": scaler_artifact.get("threshold", 0.5),
        "risk_thresholds": scaler_artifact.get("risk_thresholds", {"low": 0.33, "medium": 0.67}),
        "feature_names_out": scaler_artifact.get("feature_names_out", []),
        "top_features_global": scaler_artifact.get("top_features_global", []),
    }


def _validate_and_build_dataframe(features: Dict[str, Any], user_feature_columns: List[str]) -> pd.DataFrame:
    if not isinstance(features, dict):
        raise ValueError("Input must be a JSON object mapping feature names to values.")

    missing = [c for c in user_feature_columns if c not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    row = {c: features[c] for c in user_feature_columns}
    return pd.DataFrame([row], columns=user_feature_columns)


def _risk_level(prob: float, risk_thresholds: Dict[str, float]) -> str:
    low = float(risk_thresholds.get("low", 0.33))
    medium = float(risk_thresholds.get("medium", 0.67))
    if prob < low:
        return "Low Risk"
    if prob < medium:
        return "Medium Risk"
    return "High Risk"


def _confidence(prob: float) -> float:
    p = float(prob)
    return max(p, 1.0 - p) * 100.0


def _safe_to_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


@lru_cache(maxsize=1)
def _maybe_load_shap():
    try:
        import shap  # type: ignore

        return shap
    except Exception:
        return None


def _get_top_features_shap(
    model: Any,
    preprocessor: Any,
    feature_names_out: List[str],
    X_transformed,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    shap = _maybe_load_shap()
    if shap is None or not feature_names_out:
        return []

    # If we have an ensemble, pick the first underlying estimator with tree-like importance.
    explain_model = model
    if hasattr(model, "fitted_estimators_"):
        explain_model = None
        for est in getattr(model, "fitted_estimators_", []):
            if hasattr(est, "feature_importances_"):
                explain_model = est
                break
        if explain_model is None:
            return []

    # TreeExplainer is best-effort; never crash.
    try:
        X_dense = _safe_to_dense(X_transformed)
        explainer = shap.TreeExplainer(explain_model)
        shap_values = explainer.shap_values(X_dense)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values
        sv = np.asarray(shap_vals)[0]
        idx = np.argsort(np.abs(sv))[::-1][:top_k]
        out: List[Dict[str, Any]] = []
        for i in idx:
            val = float(sv[i])
            direction = "increases risk" if val >= 0 else "decreases risk"
            out.append({"feature": feature_names_out[i], "direction": direction, "shap_value": val})
        return out
    except Exception:
        return []


def _get_top_features_fallback(
    top_features_global: List[Dict[str, Any]], top_k: int = 6
) -> List[Dict[str, Any]]:
    if not top_features_global:
        return []
    # Convert to a shorter schema for API/Streamlit.
    trimmed = top_features_global[:top_k]
    return [{"feature": d["feature"], "direction": "global", "shap_value": float(d.get("importance", 0.0))} for d in trimmed]


def predict_single(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predicts risk for a single input sample.

    Returns:
      {
        "prediction": 0 or 1,
        "probability": float,
        "interpretation": "Low Risk" or "High Risk"
      }
    """

    model, scaler_artifact = load_artifacts()
    preprocessor = scaler_artifact["preprocessor"]

    user_feature_columns = scaler_artifact.get("user_feature_columns", scaler_artifact["feature_columns"])
    feature_columns_all = scaler_artifact.get("feature_columns", user_feature_columns)
    threshold = float(scaler_artifact.get("threshold", 0.5))
    risk_thresholds = scaler_artifact.get("risk_thresholds", {"low": 0.33, "medium": 0.67})
    feature_names_out = scaler_artifact.get("feature_names_out", [])
    top_features_global = scaler_artifact.get("top_features_global", [])

    df_user = _validate_and_build_dataframe(features, user_feature_columns)

    # Add engineered features consistent with training.
    df_all = engineer_features(df_user)

    # Ensure the transformed column order matches training.
    missing_in_df_all = [c for c in feature_columns_all if c not in df_all.columns]
    if missing_in_df_all:
        raise ValueError(f"Missing engineered features after feature engineering: {missing_in_df_all}")
    df_all = df_all[feature_columns_all]

    try:
        X_trans = preprocessor.transform(df_all)
    except Exception as e:
        raise ValueError(f"Failed to preprocess input features: {e}") from e

    try:
        proba = model.predict_proba(X_trans)
        prob = float(np.asarray(proba)[:, 1].ravel()[0])
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}") from e

    pred = int(prob >= threshold)
    if prob < 0.3:
        risk_level = "Low Risk"
    elif prob < 0.7:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    interpretation = risk_level
    risk_level = _risk_level(prob, risk_thresholds)
    confidence = _confidence(prob)

    # Explainability: SHAP if possible, else global fallback.
    top_features = _get_top_features_shap(
        model=model,
        preprocessor=preprocessor,
        feature_names_out=feature_names_out,
        X_transformed=X_trans,
        top_k=6,
    )
    if not top_features:
        top_features = _get_top_features_fallback(top_features_global=top_features_global, top_k=6)

    return {
        "prediction": pred,
        "probability": prob,
        "confidence": confidence,
        "interpretation": interpretation,
        "risk_level": risk_level,
        "top_features": top_features,
    }

