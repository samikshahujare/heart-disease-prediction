from __future__ import annotations

from typing import Any, Dict

import requests
import streamlit as st

# Replace with your deployed Render backend URL
API_URL = "https://your-render-url.onrender.com/predict"


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
    p = max(0.0, min(1.0, float(prob)))
    return f"{p * 100:.1f}%"


def main() -> None:
    st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
    _css()

    st.markdown('<div class="title">Heart Disease Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Frontend-only app. Predictions are fetched from backend API.</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.caption("Patient features")

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("age", min_value=1.0, max_value=120.0, value=54.0, step=1.0)
            cp = st.selectbox("cp", options=[0, 1, 2, 3], index=0)
            chol = st.number_input("chol", min_value=50.0, max_value=700.0, value=240.0, step=1.0)
            restecg = st.selectbox("restecg", options=[0, 1, 2], index=1)
            exang = st.selectbox("exang", options=[0, 1], index=0)
            slope = st.selectbox("slope", options=[0, 1, 2], index=1)
            thal = st.selectbox("thal", options=[0, 1, 2, 3], index=2)

        with c2:
            sex = st.selectbox("sex", options=[0, 1], index=1)
            trestbps = st.number_input("trestbps", min_value=70.0, max_value=250.0, value=130.0, step=1.0)
            fbs = st.selectbox("fbs", options=[0, 1], index=0)
            thalach = st.number_input("thalach", min_value=50.0, max_value=250.0, value=150.0, step=1.0)
            oldpeak = st.number_input("oldpeak", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            ca = st.selectbox("ca", options=[0, 1, 2, 3, 4], index=0)

        input_data: Dict[str, Any] = {
            "age": float(age),
            "sex": int(sex),
            "cp": int(cp),
            "trestbps": float(trestbps),
            "chol": float(chol),
            "fbs": int(fbs),
            "restecg": int(restecg),
            "thalach": float(thalach),
            "exang": int(exang),
            "oldpeak": float(oldpeak),
            "slope": int(slope),
            "ca": int(ca),
            "thal": int(thal),
        }

        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            response = requests.post(API_URL, json=input_data, timeout=30)
            if response.status_code != 200:
                st.error(f"API Error ({response.status_code}): {response.text}")
                return

            result = response.json()
            pred = int(result.get("prediction", 0))
            prob = float(result.get("probability", 0.0))
            interpretation = str(result.get("interpretation", "Unknown"))
            risk_level = str(result.get("risk_level", interpretation))
            confidence = float(result.get("confidence", 0.0))
            top_features = result.get("top_features", [])

            st.markdown('<div class="card" style="margin-top: 14px;">', unsafe_allow_html=True)
            st.subheader("Result")
            st.write(
                {
                    "prediction": pred,
                    "probability": prob,
                    "probability_readable": _format_prob(prob),
                    "confidence": confidence,
                    "risk_level": risk_level,
                    "interpretation": interpretation,
                }
            )

            risk_color = "#ef4444" if risk_level.startswith("High") else ("#f59e0b" if risk_level.startswith("Medium") else "#16a34a")
            st.markdown(
                f"""
                <div style="font-size: 22px; font-weight: 700; color: {risk_color}; margin-top: 10px;">
                {risk_level}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if top_features:
                st.markdown("**Top Features**")
                for item in top_features:
                    feature = item.get("feature", "unknown")
                    direction = item.get("direction", "global")
                    score = float(item.get("shap_value", 0.0))
                    st.write(f"- {feature}: {direction} ({score:.4f})")

            st.markdown("</div>", unsafe_allow_html=True)
        except requests.RequestException as e:
            st.error(f"Network/API request failed: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.caption("This tool is for estimation only and not a medical diagnosis.")


if __name__ == "__main__":
    main()

