# Heart Disease Prediction System

Production-grade Heart Disease prediction system with:
- Robust preprocessing (missing values, outlier capping, one-hot encoding, `StandardScaler`)
- Advanced feature engineering (log transforms + interaction/ratio features)
- Model comparison with hyperparameter tuning (XGBoost, Random Forest, Logistic Regression, optional LightGBM)
- Class imbalance handling (XGBoost `scale_pos_weight` and optional SMOTE comparison)
- Soft-voting style ensembling (probability averaging)
- FastAPI backend (`POST /predict`) + Streamlit frontend

## Project Structure

- `data/heart-disease.csv`: dataset
- `notebook/eda.ipynb`: EDA notebook
- `model/train_model.py`: training script
- `model/model.pkl`: trained best model / ensemble artifact (generated)
- `model/scaler.pkl`: preprocessing + schema + feature-engineering metadata (generated)
- `model/predict.py`: inference utilities (used by API + Streamlit)
- `app/api.py`: FastAPI backend
- `app/app.py`: Streamlit frontend

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Local Run (Mandatory)

### 1) Train the model

```bash
python model/train_model.py
```

Outputs:
- `model/model.pkl`
- `model/scaler.pkl`
- `model/metrics/metrics.json`
- `model/metrics/confusion_matrix.png`
- `model/metrics/roc_curve.png`
- `model/metrics/feature_importance.png` (and optionally `shap_summary.png`)

### 2) Run FastAPI backend

Run the API on port `10000`:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 10000
```

Test the endpoint:

```bash
curl -X POST "http://127.0.0.1:10000/predict" -H "Content-Type: application/json" -d "{ \"age\": 63, \"sex\": 1, \"cp\": 3, \"trestbps\": 145, \"chol\": 233, \"fbs\": 1, \"restecg\": 0, \"thalach\": 150, \"exang\": 0, \"oldpeak\": 2.3, \"slope\": 0, \"ca\": 0, \"thal\": 1 }"
```

### 3) Run Streamlit frontend

```bash
streamlit run app/app.py
```

## Model Comparison and Results

This project trains multiple models with cross-validation and hyperparameter tuning, including:
- XGBoost (`scale_pos_weight` imbalance strategy)
- XGBoost + SMOTE (comparison)
- Random Forest
- Logistic Regression (baseline tuned with elastic net)
- Optional LightGBM (if installed)

It then applies probability averaging ensembling and selects the best model using a combined objective emphasizing both:
- ROC-AUC
- F1 (after threshold optimization on a validation split)

### Performance Improvement Summary (Before vs After)

Baseline (previous model, ~80% range):
- Accuracy: 0.8033
- F1: 0.8333
- ROC-AUC: 0.8896

Current best model selected (smoke training with `--n-iter 1`):
- `Logistic Regression (elasticnet)`
- Accuracy: 0.8525
- F1: 0.8767
- ROC-AUC: 0.9167

### Model Comparison Table (test-set)

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| `Logistic Regression (elasticnet)` | 0.8525 | 0.8767 | 0.9167 |
| `Random Forest` | 0.8033 | 0.8378 | 0.8961 |
| `XGBoost (scale_pos_weight)` | 0.7869 | 0.8312 | 0.8799 |
| `Ensemble (probability averaging)` | 0.8361 | 0.8649 | 0.9113 |

### Final Model Justification

The final model was selected based on ROC-AUC + threshold-optimized F1 to improve positive-class detection without sacrificing ranking quality (generalization-focused).

### Feature Engineering (Applied Automatically)

The system automatically adds interaction and nonlinear features such as:
- `log_chol`, `log_trestbps`, `log_thalach`, `log_oldpeak`
- `age_chol`, `age_trestbps`, `chol_trestbps_ratio`
- `thalach_oldpeak`, `cp_restecg_interaction`, `slope_oldpeak`

## Deployment Options (Mandatory)

### A) Streamlit Cloud (Frontend)

1. Create a new Streamlit Cloud app from this repository.
2. Ensure `requirements.txt` is present.
3. Set the entry point to:
   - `app/app.py`
4. Make sure the deployed environment has the trained artifacts:
   - `model/model.pkl`
   - `model/scaler.pkl`

If you do not commit the generated `.pkl` files, the app will fail to load artifacts and you must either:
- run training during your deployment setup, or
- commit artifacts to the repository.

### B) Render (FastAPI Backend)

1. Create a new Render Web Service from the repository.
2. Set the build/start command to:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 10000
```

3. Ensure Render installs dependencies via `requirements.txt`.
4. Ensure the trained artifacts exist in the deployed container:
   - `model/model.pkl`
   - `model/scaler.pkl`

Practical approach:
- Train locally (`python model/train_model.py`)
- Commit `model/model.pkl` and `model/scaler.pkl` to the repo (or provide them via your deployment pipeline).

## Notes
- The API expects a JSON object containing **all original dataset feature columns** (the columns in `data/heart-disease.csv` excluding `target`).
- Feature engineering is applied automatically during inference.
- API output includes:
  - `prediction` (0 or 1)
  - `probability` (risk probability for class 1)
  - `confidence` (derived from probability)
  - `interpretation` (`Low Risk` or `High Risk`)
  - `risk_level` (`Low Risk`, `Medium Risk`, `High Risk`)
  - `top_features` (important features influencing the prediction)

