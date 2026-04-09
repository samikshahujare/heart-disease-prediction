from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

import sys


def _ensure_project_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_ensure_project_on_path()

import requests

API_URL = "https://your-render-url.onrender.com/predict" # noqa: E402


def _css() -> None:
    st.markdown(
        """
        <style>
        .title { font-size: 34px; font-weight: 700; color: #0f172a; margin-bottom: 6px; }
        .subtitle { color: #475569; margin-bottom: 18px; }
        .card { padding: 18px; border-radius: 12px; background: #ffffff; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12); }
        .stButton>button { width: 100%; border-radius: 10px; height: 42px; font-weight: 600; background: #2563eb; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_prob(prob: float) -> str:
    if prob < 0:
        prob = 0.0
    if prob > 1:
        prob = 1.0
    return f"{prob*100:.1f}%"


def _default_numeric(numeric_defaults: Dict[str, Any], col: str) -> float:
    val = numeric_defaults.get(col, 0.0)
    try:
        return float(val)
    except Exception:
        return 0.0


def main() -> None:
    st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
    _css()

    st.markdown('<div class="title">Heart Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter patient features to estimate risk with the trained model.</div>', unsafe_allow_html=True)

    try:
        meta = get_feature_metadata()
    except Exception as e:
        st.error(
            "Model artifacts are missing or invalid. "
            "Run `python model/train_model.py` first, then restart Streamlit.\n"
            f"Details: {e}"
        )
        st.stop()

    feature_columns: List[str] = meta["user_feature_columns"]
    numeric_defaults: Dict[str, Any] = meta.get("numeric_defaults", {})
    categorical_features: List[str] = meta["user_categorical_features"]
    categorical_values: Dict[str, List[Any]] = meta.get("categorical_values", {})
    threshold: float = float(meta.get("threshold", 0.5))
    risk_thresholds = meta.get("risk_thresholds", {"low": 0.33, "medium": 0.67})

    categorical_set = set(categorical_features)

    with st.form("predict_form"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Patient features")

        inputs: Dict[str, Any] = {}
        cols = st.columns(2)

        for i, col in enumerate(feature_columns):
            with cols[i % 2]:
                if col in categorical_set:
                    options = categorical_values.get(col)
                    if not options:
                        options = [0, 1]
                    # Keep options in their original types (typically ints).
                    inputs[col] = st.selectbox(col, options=options, index=0, format_func=str)
                else:
                    default_val = _default_numeric(numeric_defaults, col)
                    inputs[col] = st.number_input(col, value=float(default_val), format="%.4f")

        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            response = requests.post(API_URL, json=inputs)

            if response.status_code == 200:
                result = response.json()
            else:
                st.error(f"API Error: {response.text}")
                st.stop()
            pred = result["prediction"]
            prob = float(result["probability"])
            interpretation = result["interpretation"]

            st.markdown('<div class="card" style="margin-top: 14px;">', unsafe_allow_html=True)
            st.subheader("Result")
            st.write(
                {
                    "prediction": pred,
                    "probability": prob,
                    "probability_readable": _format_prob(prob),
                    "interpretation": interpretation,
                    "decision_threshold": threshold,
                }
            )

            # Nice high-level labels
            risk_level = result.get("risk_level", "Low Risk")
            confidence = float(result.get("confidence", 0.0))
            top_features = result.get("top_features", [])

            risk_color = "#ef4444" if risk_level.startswith("High") else ("#f59e0b" if risk_level.startswith("Medium") else "#16a34a")
            st.markdown(
                f"""
                <div style="font-size: 22px; font-weight: 700; color: {risk_color}; margin-top: 10px;">
                {risk_level}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div style='color:#334155;margin-top:8px;'>Confidence: <b>{confidence:.1f}%</b></div>",
                unsafe_allow_html=True,
            )

            if top_features:
                # Simple bar chart of top feature contributions.
                try:
                    import matplotlib.pyplot as plt
                    top_names = [d.get("feature", "") for d in top_features]
                    top_vals = [abs(float(d.get("shap_value", 0.0))) for d in top_features]
                    plt.figure(figsize=(9, 4))
                    plt.barh(top_names[::-1], top_vals[::-1])
                    plt.title("Top Feature Contributions (absolute)")
                    plt.xlabel("Contribution magnitude")
                    plt.tight_layout()
                    st.pyplot(plt.gcf())
                    plt.close()
                except Exception:
                    st.write(top_features)

            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.caption("This tool is for estimation only and not a medical diagnosis.")


if __name__ == "__main__":
    main()

