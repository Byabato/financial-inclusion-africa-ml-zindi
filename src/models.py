# ============================================================
# src/models.py
#
# Model training, K-Fold cross-validation, threshold tuning,
# and evaluation. Each function is self-contained and documented.
#
# USAGE:
#   from src.models import train_kfold, optimize_threshold
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, accuracy_score,
    classification_report, roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.config import (
    SEED, CV_FOLDS, THRESHOLD,
    XGB_DEFAULT_PARAMS, LGBM_DEFAULT_PARAMS,
    CATBOOST_DEFAULT_PARAMS, MODELS_DIR
)


# ------------------------------------------------------------------
# HELPER: Print a clean evaluation report
# ------------------------------------------------------------------
def print_evaluation(y_true, y_pred, y_proba, model_name="Model"):
    """Prints a comprehensive evaluation summary."""
    mae  = mean_absolute_error(y_true, y_pred)
    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_proba)

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")
    print(f"  MAE      : {mae:.4f}  ← Competition Metric (lower = better)")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}% correct)")
    print(f"  ROC-AUC  : {auc:.4f}  (1.0 = perfect, 0.5 = random)")
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                 target_names=["No Account", "Has Account"]))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"{'='*50}\n")

    return {"mae": mae, "accuracy": acc, "roc_auc": auc}


# ------------------------------------------------------------------
# CORE: Stratified K-Fold Training
#
# WHY Stratified?
#   With ~14% positive class, random folds could have very few
#   positives. Stratified ensures each fold mirrors the full ratio.
#
# WHAT ARE OOF PREDICTIONS?
#   Out-Of-Fold predictions: each sample is predicted by a model
#   that NEVER trained on it. This gives a true, unbiased estimate
#   of model performance — more reliable than train/val split.
# ------------------------------------------------------------------
def train_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    model_type: str = "xgb",
    params: dict = None,
    n_splits: int = CV_FOLDS,
    verbose: int = -1
) -> tuple:
    """
    Trains a model using Stratified K-Fold cross-validation.

    Parameters
    ----------
    X          : training features
    y          : training target (0/1)
    X_test     : test features
    model_type : one of "xgb", "lgbm", "catboost", "logreg"
    params     : model hyperparameters (uses defaults if None)
    n_splits   : number of CV folds
    verbose    : print fold-by-fold MAE

    Returns
    -------
    oof_preds  : out-of-fold probability predictions on train set
    test_preds : averaged probability predictions on test set
    models     : list of trained model objects (one per fold)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=SEED)

    oof_preds   = np.zeros(len(X))       # OOF probabilities
    test_preds  = np.zeros(len(X_test))  # Test probabilities (averaged)
    models      = []

    fold_maes = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # ---- Build model ----
        model = _build_model(model_type, params)

        # ---- Fit with early stopping where supported ----
        if model_type in ["xgb", "lgbm"]:
            eval_set = [(X_val, y_val)]
            model.fit(
                X_tr, y_tr,
                eval_set=eval_set,
                # verbose=-1
            )
        elif model_type == "catboost":
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=0
            )
        else:
            model.fit(X_tr, y_tr)

        # ---- Predict probabilities ----
        val_proba  = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        # ---- Store OOF predictions ----
        oof_preds[val_idx] = val_proba

        # ---- Accumulate test predictions (will average later) ----
        test_preds += test_proba / n_splits

        # ---- Fold evaluation ----
        fold_mae = mean_absolute_error(
            y_val, (val_proba >= THRESHOLD).astype(int)
        )
        fold_maes.append(fold_mae)
        models.append(model)

        if verbose:
            print(f"  Fold {fold}/{n_splits}  |  MAE: {fold_mae:.4f}")

    if verbose:
        print(f"\n  CV Mean MAE : {np.mean(fold_maes):.4f}")
        print(f"  CV Std  MAE : {np.std(fold_maes):.4f}")

    return oof_preds, test_preds, models


# ------------------------------------------------------------------
# HELPER: Build a model object by type
# ------------------------------------------------------------------
def _build_model(model_type: str, params: dict = None):
    """
    Returns an instantiated model.
    Uses provided params or falls back to config defaults.
    """
    if model_type == "xgb":
        p = params or XGB_DEFAULT_PARAMS
        return XGBClassifier(**p)

    elif model_type == "lgbm":
        p = params or LGBM_DEFAULT_PARAMS
        return LGBMClassifier(**p)

    elif model_type == "catboost":
        p = params or CATBOOST_DEFAULT_PARAMS
        return CatBoostClassifier(**p)

    elif model_type == "logreg":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
            C=params.get("C", 1.0) if params else 1.0
        )

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from: 'xgb', 'lgbm', 'catboost', 'logreg'"
        )


# ------------------------------------------------------------------
# THRESHOLD OPTIMIZATION
#
# WHY tune the threshold?
#   Model outputs probabilities (e.g., 0.42).
#   Default threshold = 0.5 → predict 1 if proba >= 0.5.
#   But with class imbalance, 0.5 causes too many false negatives.
#   We scan thresholds and pick the one with LOWEST MAE.
#
# This single step can improve MAE by 0.01–0.03.
# ------------------------------------------------------------------
def optimize_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    step: float = 0.01,
    verbose: int = 1
) -> float:
    """
    Finds the probability threshold that minimizes MAE on
    the OOF predictions.

    Parameters
    ----------
    y_true  : true binary labels
    y_proba : predicted probabilities (OOF)
    step    : threshold scan step size
    verbose : print results

    Returns
    -------
    optimal_threshold : float
    """
    thresholds = np.arange(0.05, 0.95, step)
    maes = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        maes.append(mean_absolute_error(y_true, preds))

    optimal_threshold = thresholds[np.argmin(maes)]
    best_mae = min(maes)

    if verbose:
        print(f"\n  Threshold Optimization:")
        print(f"  Default (0.50) MAE : "
              f"{mean_absolute_error(y_true, (y_proba>=0.5).astype(int)):.4f}")
        print(f"  Optimal threshold  : {optimal_threshold:.2f}")
        print(f"  Optimal MAE        : {best_mae:.4f}")

    return float(optimal_threshold)


# ------------------------------------------------------------------
# SAVE / LOAD MODELS
# ------------------------------------------------------------------
def save_model(model, name: str):
    """Saves a model to the models directory."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  ✓ Model saved → {path}")


def load_model(name: str):
    """Loads a model from the models directory."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    return joblib.load(path)
