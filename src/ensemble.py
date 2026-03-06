# ============================================================
# src/ensemble.py
#
# Stacking ensemble: combines predictions from multiple models
# using a meta-learner (Logistic Regression).
#
# HOW STACKING WORKS:
#   Layer 1: Base models (XGB, LGBM, CatBoost, RF) produce
#            Out-Of-Fold probability predictions.
#   Layer 2: A meta-learner trains on those OOF predictions
#            as features, learning HOW to best combine them.
#   Result:  The ensemble inherits strengths of all base models
#            and typically outperforms any single model.
#
# WHY Logistic Regression as meta-learner?
#   - Simple, fast, interpretable
#   - Regularized (won't overfit on just 3-4 features)
#   - Outputs calibrated probabilities
# ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score

from src.config import SEED
from src.models import optimize_threshold


def build_stacking_ensemble(
    oof_dict: dict,
    test_dict: dict,
    y_train: pd.Series,
    verbose: bool = True
) -> tuple:
    """
    Builds a stacking ensemble from OOF predictions.

    Parameters
    ----------
    oof_dict  : dict of {model_name: oof_probabilities_array}
                e.g. {"xgb": array([0.3, 0.7, ...]),
                      "lgbm": array([0.4, 0.6, ...]), ...}
    test_dict : dict of {model_name: test_probabilities_array}
    y_train   : true training labels

    Returns
    -------
    final_test_proba : final ensemble probabilities for test set
    optimal_threshold: tuned threshold
    meta_model       : fitted meta-learner
    """
    # ---- Build meta-feature matrices ----
    # Stack OOF predictions side-by-side as columns
    model_names = list(oof_dict.keys())

    X_meta_train = np.column_stack([oof_dict[m] for m in model_names])
    X_meta_test  = np.column_stack([test_dict[m] for m in model_names])

    if verbose:
        print("\n" + "="*50)
        print("  STACKING ENSEMBLE")
        print("="*50)
        print(f"  Base models  : {model_names}")
        print(f"  Meta-features: {X_meta_train.shape[1]} columns")
        print(f"  Meta-train   : {X_meta_train.shape[0]} rows")

        # Print individual model MAEs for comparison
        print("\n  Individual model OOF MAEs:")
        for name in model_names:
            oof_preds = (oof_dict[name] >= 0.5).astype(int)
            mae = mean_absolute_error(y_train, oof_preds)
            auc = roc_auc_score(y_train, oof_dict[name])
            print(f"    {name:<12} MAE={mae:.4f}  AUC={auc:.4f}")

    # ---- Train meta-learner ----
    meta_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED
    )
    meta_model.fit(X_meta_train, y_train)

    # ---- Meta-learner coefficients (interpret which model it trusts most) ----
    if verbose:
        print("\n  Meta-learner coefficients (trust weights):")
        for name, coef in zip(model_names, meta_model.coef_[0]):
            print(f"    {name:<12}: {coef:+.4f}")

    # ---- OOF predictions from ensemble ----
    ensemble_oof_proba = meta_model.predict_proba(X_meta_train)[:, 1]

    # ---- Tune threshold on ensemble OOF ----
    optimal_threshold = optimize_threshold(
        y_train.values, ensemble_oof_proba, verbose=verbose
    )

    # ---- Final test predictions ----
    final_test_proba = meta_model.predict_proba(X_meta_test)[:, 1]

    if verbose:
        ensemble_preds = (
            ensemble_oof_proba >= optimal_threshold
        ).astype(int)
        ens_mae = mean_absolute_error(y_train, ensemble_preds)
        ens_auc = roc_auc_score(y_train, ensemble_oof_proba)
        print(f"\n  Ensemble OOF MAE : {ens_mae:.4f}")
        print(f"  Ensemble OOF AUC : {ens_auc:.4f}")
        print(f"  Optimal threshold: {optimal_threshold:.2f}")
        print("="*50)

    return final_test_proba, optimal_threshold, meta_model


def simple_average_ensemble(
    oof_dict: dict,
    test_dict: dict,
    y_train: pd.Series,
    weights: dict = None,
    verbose: bool = True
) -> tuple:
    """
    Simple (weighted) average ensemble — sometimes beats stacking.

    Parameters
    ----------
    weights : dict of {model_name: weight} — if None, equal weights
    """
    model_names = list(oof_dict.keys())

    if weights is None:
        weights = {m: 1.0 / len(model_names) for m in model_names}

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {m: w / total_weight for m, w in weights.items()}

    oof_avg  = sum(oof_dict[m] * weights[m] for m in model_names)
    test_avg = sum(test_dict[m] * weights[m] for m in model_names)

    optimal_threshold = optimize_threshold(
        y_train.values, oof_avg, verbose=verbose
    )

    if verbose:
        print(f"\n  Blend weights: {weights}")
        blend_preds = (oof_avg >= optimal_threshold).astype(int)
        blend_mae = mean_absolute_error(y_train, blend_preds)
        blend_auc = roc_auc_score(y_train, oof_avg)
        print(f"  Blend OOF MAE: {blend_mae:.4f}")
        print(f"  Blend OOF AUC: {blend_auc:.4f}")

    return test_avg, optimal_threshold
