from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy()

    def _get(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce")

    if "chol" in df.columns:
        df["log_chol"] = np.log1p(np.maximum(_get("chol"), 0))
    if "trestbps" in df.columns:
        df["log_trestbps"] = np.log1p(np.maximum(_get("trestbps"), 0))
    if "thalach" in df.columns:
        df["log_thalach"] = np.log1p(np.maximum(_get("thalach"), 0))
    if "oldpeak" in df.columns:
        df["log_oldpeak"] = np.log1p(np.maximum(_get("oldpeak"), 0))

    if "age" in df.columns and "chol" in df.columns:
        df["age_chol"] = _get("age") * _get("chol")
    if "age" in df.columns and "trestbps" in df.columns:
        df["age_trestbps"] = _get("age") * _get("trestbps")
    if "chol" in df.columns and "trestbps" in df.columns:
        df["chol_trestbps_ratio"] = _get("chol") / (_get("trestbps") + 1e-6)
    if "thalach" in df.columns and "oldpeak" in df.columns:
        df["thalach_oldpeak"] = _get("thalach") * _get("oldpeak")

    if "cp" in df.columns and "restecg" in df.columns:
        df["cp_restecg_interaction"] = _get("cp") * _get("restecg")
    if "slope" in df.columns and "oldpeak" in df.columns:
        df["slope_oldpeak"] = _get("slope") * _get("oldpeak")

    return df

