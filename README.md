# Heart Disease Prediction (Production Split)

This project is refactored for deployment reliability with strict separation between backend and frontend.

- Backend (Render): FastAPI + model inference using existing trained artifacts
- Frontend (Streamlit Cloud): UI only, calls backend API via HTTP

No model retraining is required. Existing model behavior and prediction pipeline are preserved.

## Final Structure

```text
heart-disease-prediction/
│
├── backend/
│   ├── app/
│   │   └── api.py
│   ├── model/
│   │   ├── model.pkl
│   │   ├── scaler.pkl
│   │   ├── predict.py
│   │   ├── feature_engineering.py
│   │   └── outliers.py
│   ├── data/
│   ├── requirements.txt
│   └── runtime.txt
│
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── runtime.txt
│
└── README.md
```

## Backend

### Files
- `backend/app/api.py`
- `backend/model/model.pkl`
- `backend/model/scaler.pkl`
- `backend/model/predict.py`

### backend/requirements.txt
Contains only:
- fastapi
- uvicorn
- pandas
- numpy
- scikit-learn
- xgboost
- joblib
- imbalanced-learn

### backend/runtime.txt
- `python-3.10.13`

### Run backend locally
From project root:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.api:app --host 0.0.0.0 --port 10000
```

Health check:
- `GET /health`

Prediction endpoint:
- `POST /predict`

## Frontend

### Key rule
Frontend does not import model code or ML libraries.
It sends patient features to backend API:

- `API_URL = "https://your-render-url.onrender.com/predict"`
- `requests.post(API_URL, json=input_data)`

### frontend/requirements.txt
Contains only:
- streamlit
- requests

### frontend/runtime.txt
- `python-3.10.13`

### Run frontend locally
From project root:

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

### Render (Backend)
- Root directory: `backend/`
- Build/Install: `pip install -r requirements.txt`
- Start command:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 10000
```

### Streamlit Cloud (Frontend)
- Root directory: `frontend/`
- Main file: `app.py`
- Ensure `API_URL` in `frontend/app.py` points to your deployed Render URL.

## Notes
- No retraining is performed in this refactor.
- Existing `.pkl` artifacts are reused.
- Prediction output structure remains the same (`prediction`, `probability`, `confidence`, `interpretation`, `risk_level`, `top_features`).
