from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sys

# Ensure project root is importable when running as a script:
# `python model/train_model.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


from model.outliers import OutlierCapper
from model.feature_engineering import engineer_features, list_engineered_feature_names

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    _IMBLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    _IMBLEARN_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    _LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover
    _LIGHTGBM_AVAILABLE = False



def _build_onehot_encoder() -> OneHotEncoder:
    """
    Create OneHotEncoder in a way that's compatible with multiple sklearn versions.
    """

    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _infer_target_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower() in {"target", "label", "output", "y", "class"}]
    if "target" in df.columns:
        return "target"
    if candidates:
        # If multiple candidates exist, prefer 'target' (already handled) then first.
        return candidates[0]
    # Fallback: last column
    return df.columns[-1]

def _infer_feature_types(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[List[str], List[str]]:
    """
    Infers categorical vs numeric columns in a data-driven way.

    - Object/category columns => categorical
    - Integer-like columns with small cardinality => categorical
    - Everything else => numeric
    """

    categorical: List[str] = []
    numeric: List[str] = []

    for col in feature_columns:
        s = df[col]
        if s.dtype == "object" or str(s.dtype).lower().startswith("category"):
            categorical.append(col)
            continue

        # Treat integer-like columns with low unique counts as categorical.
        # This works well for medical code features like sex/cp/restecg/etc.
        if pd.api.types.is_integer_dtype(s.dtype):
            nunique = s.nunique(dropna=True)
            if nunique <= 10:
                categorical.append(col)
            else:
                numeric.append(col)
            continue

        # Floats are assumed numeric by default.
        numeric.append(col)

    return numeric, categorical


def _jsonable_categories(series: pd.Series) -> List[Any]:
    vals = series.dropna().unique().tolist()
    # Convert numpy scalars to python scalars to keep joblib output clean.
    out: List[Any] = []
    for v in vals:
        if isinstance(v, (np.generic,)):
            out.append(v.item())
        else:
            out.append(v)
    return sorted(out)


def _build_onehot_encoder_dense(dense: bool) -> OneHotEncoder:
    """
    Build OneHotEncoder with deterministic sparse/dense output.
    """

    # sklearn >= 1.2
    if dense:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
    # sparse
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    dense: bool,
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("outlier_cap", OutlierCapper(iqr_factor=1.5)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _build_onehot_encoder_dense(dense=dense)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    model: Any,
    dense: bool,
) -> Pipeline:
    preprocessor = build_preprocessor(
        numeric_features=numeric_features, categorical_features=categorical_features, dense=dense
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


@dataclass(frozen=True)
class TrainingArtifacts:
    model_path: Path
    scaler_path: Path
    metrics_dir: Path


def _roc_auc_scorer(estimator, X_val, y_val) -> float:
    # Explicit callable scorer for sklearn stability across versions.
    y_proba = estimator.predict_proba(X_val)
    pos = np.asarray(y_proba)[:, 1]
    return float(roc_auc_score(y_val, pos))


def _choose_threshold_max_f1(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Pick a classification threshold that maximizes F1 on validation data.
    """

    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).ravel().astype(float)

    # Coarse grid is fast + robust; fine-tuning can be added later.
    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        y_pred = (y_proba_arr >= thr).astype(int)
        f1 = f1_score(y_true_arr, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
    return best_thr


def _compute_metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).ravel().astype(float)
    y_pred = (y_proba_arr >= float(threshold)).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_proba_arr)),
    }


def _maybe_import_shap():
    try:
        import shap  # type: ignore

        return shap
    except Exception:
        return None


def _top_features_from_importances(
    feature_names_out: List[str],
    importances: np.ndarray,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    imp = np.asarray(importances).ravel().astype(float)
    idx = np.argsort(np.abs(imp))[::-1][:top_k]
    out: List[Dict[str, Any]] = []
    for i in idx:
        out.append({"feature": feature_names_out[i], "importance": float(imp[i])})
    return out


def _plot_feature_importance_bar(
    feature_importances: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not feature_importances:
        return

    labels = [d["feature"] for d in feature_importances]
    values = [d["importance"] for d in feature_importances]

    plt.figure(figsize=(10, 6))
    # Use absolute importances for consistent bar direction.
    y = np.abs(values)
    plt.barh(labels[::-1], y[::-1])
    plt.title("Top Feature Importances (absolute)")
    plt.xlabel("Importance (absolute)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def train_advanced(
    data_path: Path,
    artifacts: TrainingArtifacts,
    test_size: float = 0.2,
    random_state: int = 42,
    n_iter: int = 60,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Dataset is empty.")

    # Persist a baseline snapshot (for README / resume metrics summary).
    before_metrics_path = artifacts.metrics_dir / "metrics.json"
    before_metrics = None
    if before_metrics_path.exists():
        try:
            before_metrics = json.loads(before_metrics_path.read_text(encoding="utf-8"))
        except Exception:
            before_metrics = None

    target_col = _infer_target_column(df)
    feature_user_columns = [c for c in df.columns if c != target_col]
    if not feature_user_columns:
        raise ValueError("No feature columns found.")

# ================== ✅ FIXED TARGET LOGIC ==================

    X_user = df[feature_user_columns].copy()

    # Correct target extraction
    y_raw = df[target_col].copy()

    # Force correct binary meaning
    # 0 = No Disease, 1 = Disease
    y = y_raw.astype(int)

    # Validation
    if set(y.unique()) != {0, 1}:
        raise ValueError("Target must contain exactly {0,1}")

    print("✅ Target distribution:\n", y.value_counts())

    # ==========================================================

    # Domain feature engineering (added columns computed from X_user)
    X_all = engineer_features(X_user)
    engineered_columns = [c for c in X_all.columns if c not in X_user.columns]
    feature_all_columns = X_all.columns.tolist()

    numeric_features, categorical_features = _infer_feature_types(X_all, feature_all_columns)
    if not numeric_features and not categorical_features:
        raise ValueError("Could not infer any features for modeling.")

    user_numeric_features, user_categorical_features = _infer_feature_types(X_user, feature_user_columns)
    # Safe defaults for UI widgets
    categorical_values = {c: _jsonable_categories(X_user[c]) for c in user_categorical_features}
    numeric_defaults = {c: float(X_user[c].median(skipna=True)) for c in user_numeric_features}

    y_np = np.asarray(y, dtype=int)
    n_pos = int((y_np == 1).sum())
    n_neg = int((y_np == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Target must contain both classes. Counts: n_pos={n_pos}, n_neg={n_neg}")
    scale_pos_weight = n_neg / n_pos

    # Train/test split for final evaluation.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all,
        y,
        test_size=float(test_size),
        stratify=y,
        random_state=int(random_state),
    )

    # Separate validation split for threshold selection.
    val_fraction = 0.2
    X_tune, X_val, y_tune, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_fraction,
        stratify=y_train_full,
        random_state=int(random_state),
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": _roc_auc_scorer,
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(random_state))

    results: List[Dict[str, Any]] = []

    def fit_and_evaluate(
        model_name: str,
        pipeline_builder,
        param_distributions: Dict[str, Any],
        use_smote: bool = False,
        smote_params: Dict[str, Any] | None = None,
        n_iter_override: int | None = None,
    ) -> Dict[str, Any]:
        """
        Runs RandomizedSearchCV on X_tune (cv=5), selects best params,
        then picks a threshold on X_val and evaluates on X_test.
        """

        # Build a "template" pipeline for RandomizedSearchCV.
        template = pipeline_builder(best_params=None)

        search = RandomizedSearchCV(
            estimator=template,
            param_distributions=param_distributions,
            n_iter=int(n_iter_override) if n_iter_override is not None else int(n_iter),
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            verbose=0,
            n_jobs=int(n_jobs),
            random_state=int(random_state),
            error_score="raise",
        )
        search.fit(X_tune, y_tune)
        best_params_local = search.best_params_

        # Fit on X_tune only to choose threshold on X_val (avoid val leakage).
        est_for_thr = pipeline_builder(best_params=best_params_local)
        est_for_thr.fit(X_tune, y_tune)
        val_proba = est_for_thr.predict_proba(X_val)[:, 1]
        best_thr = _choose_threshold_max_f1(y_val, val_proba)

        # Fit on full training set for final evaluation.
        est_final = pipeline_builder(best_params=best_params_local)
        est_final.fit(X_train_full, y_train_full)
        test_proba = est_final.predict_proba(X_test)[:, 1]
        metrics = _compute_metrics_at_threshold(y_test, test_proba, best_thr)

        out: Dict[str, Any] = {
            "name": model_name,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "scale_pos_weight": float(scale_pos_weight),
            "threshold": float(best_thr),
            **metrics,
            "best_params": best_params_local,
        }
        return out, est_final

    # XGBoost: scale_pos_weight variant
    def xgb_scale_pos_pipeline(best_params: Dict[str, Any] | None):
        base_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=int(random_state),
            n_jobs=-1,
            tree_method="hist",
            scale_pos_weight=float(scale_pos_weight),
            reg_alpha=0.0,
            reg_lambda=1.0,
        )
        if best_params:
            for k, v in best_params.items():
                if not k.startswith("model__"):
                    continue
            # We'll apply via set_params on the final pipeline.
        dense = False
        pre = build_preprocessor(numeric_features, categorical_features, dense=dense)
        model = base_model
        pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
        if best_params:
            pipe.set_params(**best_params)
        return pipe

    xgb_param_distributions = {
        "model__n_estimators": [300, 500, 700, 900, 1100, 1300],
        "model__max_depth": [2, 3, 4, 5, 6, 7, 8],
        "model__learning_rate": np.logspace(-3, -0.1, 20),
        "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 2, 3, 4, 5, 7, 10, 15],
        "model__gamma": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4],
        "model__reg_alpha": np.logspace(-8, 1, 18),
        "model__reg_lambda": np.logspace(-6, 4, 18),
    }

    n_iter_xgb_scale = max(3, int(n_iter))
    # Run both imbalance strategies but keep runtime reasonable.
    xgb_scale_result, xgb_scale_est = fit_and_evaluate(
        model_name="XGBoost (scale_pos_weight)",
        pipeline_builder=lambda best_params: xgb_scale_pos_pipeline(best_params),
        param_distributions=xgb_param_distributions,
        n_iter_override=n_iter_xgb_scale,
    )
    results.append(xgb_scale_result)

    # XGBoost: SMOTE variant
    if _IMBLEARN_AVAILABLE:
        def xgb_smote_pipeline(best_params: Dict[str, Any] | None):
            base_model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=int(random_state),
                n_jobs=-1,
                tree_method="hist",
                scale_pos_weight=1.0,
                reg_alpha=0.0,
                reg_lambda=1.0,
            )
            dense = True
            pre = build_preprocessor(numeric_features, categorical_features, dense=dense)
            from imblearn.over_sampling import SMOTE as _SMOTE

            steps = [
                ("preprocess", pre),
                ("smote", _SMOTE(random_state=int(random_state))),
                ("model", base_model),
            ]
            pipe = ImbPipeline(steps=steps)
            if best_params:
                pipe.set_params(**best_params)
            return pipe

        # Slightly smaller search for SMOTE to keep total runtime sane.
        n_iter_smote = max(3, int(n_iter_xgb_scale // 2))
        search_scoring = scoring
        smote_param_distributions = dict(xgb_param_distributions)

        # Inline fit with overridden n_iter for SMOTE.
        template = xgb_smote_pipeline(best_params=None)
        search = RandomizedSearchCV(
            estimator=template,
            param_distributions=smote_param_distributions,
            n_iter=int(n_iter_smote),
            scoring=search_scoring,
            refit="roc_auc",
            cv=cv,
            verbose=0,
            n_jobs=int(n_jobs),
            random_state=int(random_state),
            error_score="raise",
        )
        search.fit(X_tune, y_tune)
        best_params_smote = search.best_params_

        est_for_thr = xgb_smote_pipeline(best_params_smote)
        est_for_thr.fit(X_tune, y_tune)
        val_proba = est_for_thr.predict_proba(X_val)[:, 1]
        best_thr = _choose_threshold_max_f1(y_val, val_proba)

        est_final = xgb_smote_pipeline(best_params_smote)
        est_final.fit(X_train_full, y_train_full)
        test_proba = est_final.predict_proba(X_test)[:, 1]
        metrics = _compute_metrics_at_threshold(y_test, test_proba, best_thr)
        results.append(
            {
                "name": "XGBoost (SMOTE)",
                "n_pos": n_pos,
                "n_neg": n_neg,
                "scale_pos_weight": float(scale_pos_weight),
                "threshold": float(best_thr),
                **metrics,
                "best_params": best_params_smote,
            }
        )
        xgb_smote_est = est_final
    else:
        xgb_smote_est = None

    # Random Forest
    def rf_pipeline(best_params: Dict[str, Any] | None):
        model = RandomForestClassifier(
            random_state=int(random_state),
            n_jobs=-1,
            class_weight="balanced",
        )
        pre = build_preprocessor(numeric_features, categorical_features, dense=True)
        pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
        if best_params:
            pipe.set_params(**best_params)
        return pipe

    rf_param_distributions = {
        "model__n_estimators": [300, 500, 700, 900, 1200],
        "model__max_depth": [None, 3, 4, 5, 6, 8, 10, 14],
        "model__min_samples_split": [2, 5, 10, 15],
        "model__min_samples_leaf": [1, 2, 4, 6],
        "model__max_features": ["sqrt", "log2", 0.5, 0.7, 0.9],
    }
    rf_n_iter = max(3, int(n_iter // 2))
    rf_res, rf_est = fit_and_evaluate(
        model_name="Random Forest",
        pipeline_builder=lambda best_params: rf_pipeline(best_params),
        param_distributions=rf_param_distributions,
        n_iter_override=rf_n_iter,
    )
    results.append(rf_res)

    # Logistic Regression
    def lr_pipeline(best_params: Dict[str, Any] | None):
        model = LogisticRegression(
            solver="saga",
            max_iter=5000,
            random_state=int(random_state),
            class_weight="balanced",
            penalty="elasticnet",
            l1_ratio=0.5,
        )
        pre = build_preprocessor(numeric_features, categorical_features, dense=False)
        pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
        if best_params:
            pipe.set_params(**best_params)
        return pipe

    lr_param_distributions = {
        "model__C": np.logspace(-3, 3, 25),
        "model__l1_ratio": np.linspace(0.1, 0.9, 9),
        "model__tol": [1e-4, 5e-4, 1e-3, 2e-3],
    }
    lr_n_iter = max(3, int(n_iter // 3))
    lr_res, lr_est = fit_and_evaluate(
        model_name="Logistic Regression (elasticnet)",
        pipeline_builder=lambda best_params: lr_pipeline(best_params),
        param_distributions=lr_param_distributions,
        n_iter_override=lr_n_iter,
    )
    results.append(lr_res)

    # LightGBM (optional)
    lgbm_res = None
    lgbm_est = None
    if _LIGHTGBM_AVAILABLE:
        def lgbm_pipeline(best_params: Dict[str, Any] | None):
            model = LGBMClassifier(
                objective="binary",
                random_state=int(random_state),
                n_jobs=-1,
                class_weight="balanced",
            )
            pre = build_preprocessor(numeric_features, categorical_features, dense=False)
            pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
            if best_params:
                pipe.set_params(**best_params)
            return pipe

        lgbm_param_distributions = {
            "model__n_estimators": [300, 500, 700, 900, 1200],
            "model__learning_rate": np.logspace(-3, -0.1, 20),
            "model__num_leaves": [15, 31, 63, 127],
            "model__max_depth": [-1, 3, 4, 5, 6, 8],
            "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__min_child_samples": [10, 20, 30, 40],
            "model__reg_alpha": np.logspace(-8, 1, 18),
            "model__reg_lambda": np.logspace(-6, 4, 18),
        }

        lgb_n_iter = max(3, int(n_iter // 2))
        lgbm_res, lgbm_est = fit_and_evaluate(
            model_name="LightGBM",
            pipeline_builder=lambda best_params: lgbm_pipeline(best_params),
            param_distributions=lgbm_param_distributions,
            n_iter_override=lgb_n_iter,
        )
        results.append(lgbm_res)

    # Select best model based on ROC-AUC and F1 (weighted).
    def combined_score(r: Dict[str, Any]) -> float:
        return float(r["roc_auc"]) * 0.6 + float(r["f1"]) * 0.4

    # Merge results and keep the fitted estimators for each.
    fitted_by_name = {
        "XGBoost (scale_pos_weight)": xgb_scale_est,
        "Random Forest": rf_est,
        "Logistic Regression (elasticnet)": lr_est,
    }
    if xgb_smote_est is not None:
        fitted_by_name["XGBoost (SMOTE)"] = xgb_smote_est
    if lgbm_est is not None:
        fitted_by_name["LightGBM"] = lgbm_est

    results_sorted = sorted(results, key=combined_score, reverse=True)
    best_single = results_sorted[0]
    best_name = best_single["name"]

    best_pipeline_for_saving = fitted_by_name[best_name]

    # Train an ensemble on top models by averaging predict_proba.
    # We avoid sklearn VotingClassifier due to an sklearn/tag compatibility issue with XGBoost.
    from model.ensembles import ProbabilityAveragingEnsemble

    ensemble_candidates: List[str] = []
    for n in ["XGBoost (scale_pos_weight)", "LightGBM", "Logistic Regression (elasticnet)"]:
        if n in fitted_by_name:
            ensemble_candidates.append(n)
    # Limit to 3 members.
    ensemble_candidates = ensemble_candidates[:3]
    if len(ensemble_candidates) < 2:
        # Fall back to any other candidate if needed.
        ensemble_candidates = [best_name] + [n for n in fitted_by_name.keys() if n != best_name][:1]

    def build_ensemble_pipeline():
        pre = build_preprocessor(numeric_features, categorical_features, dense=False)
        estimators = []
        for member_name in ensemble_candidates:
            pipe_member = fitted_by_name[member_name]
            estimators.append(pipe_member.named_steps["model"])
        ensemble_model = ProbabilityAveragingEnsemble(estimators=estimators, weights=None)
        return Pipeline(steps=[("preprocess", pre), ("model", ensemble_model)])

    # Fit ensemble with threshold selection like other models.
    ensemble_pipeline_template = build_ensemble_pipeline()
    # No tuning for ensemble; it uses already-tuned best members.
    est_for_thr = ensemble_pipeline_template
    est_for_thr.fit(X_tune, y_tune)
    val_proba = est_for_thr.predict_proba(X_val)[:, 1]
    ensemble_thr = _choose_threshold_max_f1(y_val, val_proba)

    est_final = ensemble_pipeline_template
    est_final.fit(X_train_full, y_train_full)
    test_proba = est_final.predict_proba(X_test)[:, 1]
    ensemble_metrics = _compute_metrics_at_threshold(y_test, test_proba, ensemble_thr)
    ensemble_result = {
        "name": "Ensemble (soft voting)",
        "n_pos": n_pos,
        "n_neg": n_neg,
        "scale_pos_weight": float(scale_pos_weight),
        "threshold": float(ensemble_thr),
        **ensemble_metrics,
        "best_params": {},
    }

    # Choose between best single and ensemble using combined score.
    best_overall = ensemble_result if combined_score(ensemble_result) >= combined_score(best_single) else best_single
    best_overall_name = best_overall["name"]

    if best_overall_name == ensemble_result["name"]:
        best_pipeline_for_saving = est_final
        best_single_used_for_explain = None
    else:
        best_single_used_for_explain = best_name

    # Fit the chosen model again on full training to ensure it's fitted.
    # (Some models above are fitted already, but we keep it explicit.)
    if best_overall_name == ensemble_result["name"]:
        chosen_pipeline = best_pipeline_for_saving
    else:
        chosen_pipeline = best_pipeline_for_saving

    chosen_preprocessor = chosen_pipeline.named_steps["preprocess"]
    chosen_model = chosen_pipeline.named_steps["model"]

    # Feature names after preprocessing (needed for interpretability).
    feature_names_out = []
    try:
        feature_names_out = list(chosen_preprocessor.get_feature_names_out())
    except Exception:
        feature_names_out = []

    # Save confusion matrix + ROC curve for chosen model.
    chosen_test_proba = chosen_pipeline.predict_proba(X_test)[:, 1]
    chosen_y_pred = (chosen_test_proba >= float(best_overall["threshold"])).astype(int)

    plot_confusion_matrix(y_test, chosen_y_pred, artifacts.metrics_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, chosen_test_proba, artifacts.metrics_dir / "roc_curve.png")

    # Global feature importance plot (best effort)
    feature_importances_top: List[Dict[str, Any]] = []
    primary_explainer_name = None

    def _extract_importances_from_model(mdl) -> np.ndarray | None:
        if hasattr(mdl, "feature_importances_"):
            return np.asarray(mdl.feature_importances_)
        if hasattr(mdl, "coef_"):
            coef = np.asarray(mdl.coef_)
            # For binary LR, coef_ shape (1, n_features)
            return np.ravel(coef)
        return None

    if hasattr(chosen_model, "fitted_estimators_"):
        # ProbabilityAveragingEnsemble
        for est in getattr(chosen_model, "fitted_estimators_", []):
            if hasattr(est, "feature_importances_") or hasattr(est, "coef_"):
                primary_explainer_name = "primary_ensemble_member"
                importances = _extract_importances_from_model(est)
                if importances is not None and feature_names_out:
                    feature_importances_top = _top_features_from_importances(feature_names_out, importances, top_k=15)
                break
    else:
        importances = _extract_importances_from_model(chosen_model)
        if importances is not None and feature_names_out:
            feature_importances_top = _top_features_from_importances(feature_names_out, importances, top_k=15)

    _plot_feature_importance_bar(
        feature_importances=feature_importances_top,
        out_path=artifacts.metrics_dir / "feature_importance.png",
    )

    # SHAP summary plot (optional)
    shap = _maybe_import_shap()
    if shap is not None and feature_names_out:
        try:
            # Best effort: use an underlying tree model if available.
            explain_model = chosen_model
            if hasattr(chosen_model, "fitted_estimators_"):
                explain_model = None
                for est in getattr(chosen_model, "fitted_estimators_", []):
                    if est.__class__.__name__.lower().startswith("xgb"):
                        explain_model = est
                        break
                if explain_model is None:
                    for est in getattr(chosen_model, "fitted_estimators_", []):
                        if hasattr(est, "feature_importances_"):
                            explain_model = est
                            break
            if explain_model is not None and hasattr(explain_model, "predict_proba"):
                # Compute SHAP on a small sample for speed.
                X_trans = chosen_preprocessor.transform(X_train_full)
                if hasattr(X_trans, "toarray"):
                    X_trans = X_trans.toarray()
                if X_trans.shape[0] > 60:
                    X_trans = X_trans[:60]
                explainer = shap.TreeExplainer(explain_model)
                shap_values = explainer.shap_values(X_trans)
                # Binary classifiers might return list.
                if isinstance(shap_values, list):
                    shap_values_to_plot = shap_values[1]
                else:
                    shap_values_to_plot = shap_values
                shap.summary_plot(
                    shap_values_to_plot,
                    X_trans,
                    feature_names=feature_names_out,
                    show=False,
                    max_display=15,
                )
                import matplotlib.pyplot as plt

                plt.tight_layout()
                plt.savefig(artifacts.metrics_dir / "shap_summary.png", dpi=180)
                plt.close()
        except Exception:
            # SHAP is best-effort; never crash training.
            pass

    # Save model + preprocessor separately per requirements.
    artifacts.model_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.scaler_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(chosen_model, artifacts.model_path)

    scaler_artifact = {
        "preprocessor": chosen_preprocessor,
        "user_feature_columns": feature_user_columns,
        "feature_columns": feature_all_columns,
        "engineered_columns": engineered_columns,
        "target_name": target_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "user_numeric_features": user_numeric_features,
        "user_categorical_features": user_categorical_features,
        "categorical_values": categorical_values,
        "numeric_defaults": numeric_defaults,
        "threshold": float(best_overall["threshold"]),
        "risk_thresholds": {"low": 0.33, "medium": 0.67},
        "feature_names_out": feature_names_out,
        "top_features_global": feature_importances_top,
        "primary_explainer": best_single_used_for_explain,
        "chosen_model_name": best_overall_name,
    }

    joblib.dump(scaler_artifact, artifacts.scaler_path)

    # Final metrics payload (and model comparison table)
    final_payload: Dict[str, Any] = {
        "before_metrics": before_metrics,
        "final_model_name": best_overall_name,
        "final_metrics": {k: v for k, v in best_overall.items() if k not in {"name", "best_params"}},
        "model_comparison": results_sorted + [ensemble_result],
        "ensemble_result": ensemble_result,
    }

    metrics_path = artifacts.metrics_dir / "metrics.json"
    artifacts.metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")

    # For stdout convenience
    print("\nModel comparison (test-set, threshold-optimized F1):")
    for r in sorted(results + [ensemble_result], key=combined_score, reverse=True)[:5]:
        print(
            f"- {r['name']}: acc={r['accuracy']:.4f}, f1={r['f1']:.4f}, roc_auc={r['roc_auc']:.4f}"
        )

    return final_payload


def train(
    data_path: Path,
    artifacts: TrainingArtifacts,
    test_size: float = 0.2,
    random_state: int = 42,
    n_iter: int = 40,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    # Delegate to the production-grade training routine.
    return train_advanced(
        data_path=data_path,
        artifacts=artifacts,
        test_size=float(test_size),
        random_state=int(random_state),
        n_iter=int(n_iter),
        n_jobs=int(n_jobs),
    )

    # Legacy implementation (kept for reference) is intentionally unreachable.
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Dataset is empty.")

    target_col = _infer_target_column(df)
    if target_col not in df.columns:
        raise ValueError(f"Could not infer target column. Expected one of known names. Got: {df.columns.tolist()}")

    feature_columns = [c for c in df.columns if c != target_col]
    if not feature_columns:
        raise ValueError("No feature columns found.")

    X = df[feature_columns].copy()
    y_raw = df[target_col].copy()
    y, target_info = _coerce_binary_target(y_raw)

    numeric_features, categorical_features = _infer_feature_types(df, feature_columns)
    if not numeric_features and not categorical_features:
        raise ValueError("Could not infer any features.")

    # Basic class imbalance handling via scale_pos_weight.
    y_np = np.asarray(y, dtype=int)
    n_pos = int((y_np == 1).sum())
    n_neg = int((y_np == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Target must contain both classes. Counts: n_pos={n_pos}, n_neg={n_neg}")
    scale_pos_weight = n_neg / n_pos

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        stratify=y,
        random_state=int(random_state),
    )

    pipeline = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )

    # Hyperparameter search space. Tuned around typical XGBoost settings.
    param_distributions = {
        "model__n_estimators": [200, 300, 400, 500, 700, 900, 1100],
        "model__max_depth": [2, 3, 4, 5, 6, 8],
        "model__learning_rate": np.logspace(-3, -0.3, 20),  # ~0.001 .. 0.5
        "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 2, 3, 4, 5, 7, 10],
        "model__gamma": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "model__reg_alpha": np.logspace(-8, 1, 15),
        "model__reg_lambda": np.logspace(-4, 3, 15),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    # NOTE:
    # In some sklearn/xgboost combinations, sklearn's built-in string scorer for
    # ROC-AUC can incorrectly infer the estimator as a regressor and refuse to use
    # `predict_proba`. To guarantee stability, we use an explicit callable scorer.
    def roc_auc_scorer(estimator, X_val, y_val) -> float:
        y_proba = estimator.predict_proba(X_val)
        # y_proba is shape (n_samples, 2). Take positive class probability.
        pos = np.asarray(y_proba)[:, 1]
        return float(roc_auc_score(y_val, pos))

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": roc_auc_scorer,
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=int(n_iter),
        scoring=scoring,
        refit="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=int(n_jobs),
        random_state=int(random_state),
        error_score="raise",
    )

    search.fit(X_train, y_train)
    best_pipeline: Pipeline = search.best_estimator_

    best_preprocessor = best_pipeline.named_steps["preprocess"]
    best_model: XGBClassifier = best_pipeline.named_steps["model"]

    # Final evaluation on held-out test set.
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "scale_pos_weight": float(scale_pos_weight),
        "target_mapping": target_info.get("mapping", {}),
        "best_params": search.best_params_,
    }

    artifacts.metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = artifacts.metrics_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_confusion_matrix(y_test, y_pred, artifacts.metrics_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, artifacts.metrics_dir / "roc_curve.png")

    # Save model + preprocessor separately per requirements.
    artifacts.model_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.scaler_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, artifacts.model_path)

    categorical_values = {c: _jsonable_categories(df[c]) for c in categorical_features}
    numeric_defaults = {c: float(df[c].median(skipna=True)) for c in numeric_features}

    scaler_artifact = {
        "preprocessor": best_preprocessor,
        "feature_columns": feature_columns,
        "target_name": target_col,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categorical_values": categorical_values,
        "numeric_defaults": numeric_defaults,
        "threshold": 0.5,
    }
    joblib.dump(scaler_artifact, artifacts.scaler_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Heart Disease XGBoost model with preprocessing.")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=40, help="RandomizedSearchCV iterations.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data_path) if args.data_path else (project_root / "data" / "heart-disease.csv")

    artifacts = TrainingArtifacts(
        model_path=project_root / "model" / "model.pkl",
        scaler_path=project_root / "model" / "scaler.pkl",
        metrics_dir=project_root / "model" / "metrics",
    )

    payload = train(
        data_path=data_path,
        artifacts=artifacts,
        test_size=args.test_size,
        random_state=args.random_state,
        n_iter=args.n_iter,
        n_jobs=args.n_jobs,
    )

    # Print a compact summary to stdout for convenience.
    print("\nTraining completed. Test-set metrics:")
    final_metrics = payload.get("final_metrics", {})
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if k in final_metrics:
            print(f"- {k}: {final_metrics[k]:.4f}")
    if "scale_pos_weight" in final_metrics:
        print(f"- scale_pos_weight: {final_metrics['scale_pos_weight']:.4f}")
    if "threshold" in final_metrics:
        print(f"- threshold: {final_metrics['threshold']:.2f}")
    if "final_model_name" in payload:
        print(f"- final_model: {payload['final_model_name']}")
    print(f"- Confusion matrix saved to: {artifacts.metrics_dir / 'confusion_matrix.png'}")
    print(f"- ROC curve saved to: {artifacts.metrics_dir / 'roc_curve.png'}")


if __name__ == "__main__":
    # Make sure joblib pickles custom classes under a stable module path
    # (`model.train_model`) even when this file is executed directly.
    import sys

    sys.modules.setdefault("model.train_model", sys.modules[__name__])
    main()

