# ============================================================
# NOTEBOOK 04 — Hyperparameter Tuning with Optuna
# Financial Inclusion in Africa — Zindi Challenge
#
# PURPOSE:
#   Use Optuna (Bayesian optimization) to find optimal
#   hyperparameters for XGBoost and LightGBM.
#
# WHY OPTUNA over GridSearch?
#   - GridSearch: exhaustively tries all combinations
#     → O(n^k) time, very slow
#   - RandomSearch: random sampling, no memory
#   - Optuna: Bayesian (Tree Parzen Estimator) — learns
#     which regions of hyperparameter space are good
#     → finds better params in fewer trials
#   - Optuna is the industry standard for AutoML tuning
#
# EXPECTED IMPROVEMENT: 0.003–0.010 MAE reduction
#
# RUN AFTER: 03_modeling.py
# NOTE: This can take 15–45 min. Set n_trials lower
#       (e.g., 30) for a quick run.
# ============================================================

# %% [markdown]
# # Notebook 04: Hyperparameter Tuning with Optuna
#
# We search for better hyperparameters than the defaults.
# Optuna's Bayesian search learns from previous trials.

# %% CELL 1 — Setup
import numpy as np
import pandas as pd
import optuna
import warnings
import sys
import os

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
sys.path.append("..")

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import SEED, CV_FOLDS, THRESHOLD
from src.models import train_kfold, optimize_threshold, save_model
from src.ensemble import build_stacking_ensemble

np.random.seed(SEED)
print("✓ Setup complete")

# %% CELL 2 — Load data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
X_test  = pd.read_csv("data/processed/X_test.csv")
test_ids = pd.read_csv("data/processed/test_ids.csv").squeeze()

common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test  = X_test[common_cols]

print(f"✓ Loaded data: {X_train.shape}")

# %% CELL 3 — Helper: CV evaluation
# -------------------------------------------------------
# We use cross_val with MAE as the objective to minimize.
# This helper is called inside each Optuna trial.
# -------------------------------------------------------

def evaluate_with_cv(model, X, y, n_splits=5):
    """
    Runs stratified K-fold CV and returns mean OOF MAE.
    Used by Optuna as the objective metric.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                           random_state=SEED)
    oof = np.zeros(len(X))

    for tr_idx, val_idx in skf.split(X, y):
        Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[tr_idx], y.iloc[val_idx]

        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        oof[val_idx] = model.predict_proba(Xval)[:, 1]

    # Optimize threshold within CV
    best_threshold = optimize_threshold(y.values, oof, verbose=False)
    preds = (oof >= best_threshold).astype(int)
    return mean_absolute_error(y, preds)


# %% CELL 4 — Optuna Objective: XGBoost
# -------------------------------------------------------
# The objective function defines the search space.
# trial.suggest_* methods define what Optuna can try:
#   - suggest_int: integer range
#   - suggest_float: float range (can be log-uniform)
#   - suggest_categorical: discrete choices
# -------------------------------------------------------

def xgb_objective(trial):
    params = {
        # Tree structure
        "max_depth"       : trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma"           : trial.suggest_float("gamma", 0.0, 1.0),

        # Learning
        "n_estimators"    : trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),

        # Regularization (key for avoiding overfitting)
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),

        # Sampling (variance reduction)
        "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),

        # Class imbalance (range based on ~6:1 ratio in our data)
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 8.0),

        # Fixed
        "eval_metric"     : "logloss",
        "random_state"    : SEED,
        "n_jobs"          : -1,
        "verbosity"       : 0,
    }
    model = XGBClassifier(**params)
    return evaluate_with_cv(model, X_train, y_train)


# %% CELL 5 — Run XGBoost Optuna study
print("="*55)
print("  OPTUNA: XGBoost Hyperparameter Search")
print("="*55)
print("  n_trials=50 (change to 100 for better results)")
print("  Each trial = 1 full 5-fold CV")
print("  ETA: ~10–20 minutes\n")

xgb_study = optuna.create_study(
    direction="minimize",
    study_name="xgb_financial_inclusion",
    sampler=optuna.samplers.TPESampler(seed=SEED)
    # TPE = Tree-structured Parzen Estimator = Bayesian search
)

# Track progress
class ProgressCallback:
    def __init__(self, total):
        self.total = total
        self.best = float("inf")

    def __call__(self, study, trial):
        if trial.value < self.best:
            self.best = trial.value
            print(f"  Trial {trial.number:3d}/{self.total} | "
                  f"MAE={trial.value:.4f} ← New best! 🎯")
        elif trial.number % 10 == 0:
            print(f"  Trial {trial.number:3d}/{self.total} | "
                  f"Best so far: {study.best_value:.4f}")

N_TRIALS_XGB = 50  # Increase to 100 for better optimization
xgb_study.optimize(
    xgb_objective,
    n_trials=N_TRIALS_XGB,
    callbacks=[ProgressCallback(N_TRIALS_XGB)]
)

print(f"\n  ✓ XGBoost best MAE  : {xgb_study.best_value:.4f}")
print(f"  Best params:")
for k, v in xgb_study.best_params.items():
    print(f"    {k:<25}: {v}")


# %% CELL 6 — Optuna Objective: LightGBM
def lgbm_objective(trial):
    params = {
        "n_estimators"    : trial.suggest_int("n_estimators", 200, 1000),
        "max_depth"       : trial.suggest_int("max_depth", 3, 9),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves"      : trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "class_weight"    : "balanced",
        "random_state"    : SEED,
        "n_jobs"          : -1,
        "verbose"         : -1,
    }

    from lightgbm import LGBMClassifier
    model = LGBMClassifier(**params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_train))
    for tr_idx, val_idx in skf.split(X_train, y_train):
        Xtr = X_train.iloc[tr_idx]; Xval = X_train.iloc[val_idx]
        ytr = y_train.iloc[tr_idx]; yval = y_train.iloc[val_idx]
        model.fit(Xtr, ytr,
                  eval_set=[(Xval, yval)],
                  callbacks=[])
        oof[val_idx] = model.predict_proba(Xval)[:, 1]

    best_t = optimize_threshold(y_train.values, oof, verbose=False)
    return mean_absolute_error(y_train, (oof >= best_t).astype(int))


print("\n" + "="*55)
print("  OPTUNA: LightGBM Hyperparameter Search")
print("="*55)

N_TRIALS_LGBM = 50
lgbm_study = optuna.create_study(
    direction="minimize",
    study_name="lgbm_financial_inclusion",
    sampler=optuna.samplers.TPESampler(seed=SEED)
)
lgbm_study.optimize(
    lgbm_objective,
    n_trials=N_TRIALS_LGBM,
    callbacks=[ProgressCallback(N_TRIALS_LGBM)]
)

print(f"\n  ✓ LightGBM best MAE  : {lgbm_study.best_value:.4f}")
print(f"  Best params:")
for k, v in lgbm_study.best_params.items():
    print(f"    {k:<25}: {v}")


# %% CELL 7 — Train final models with tuned params
print("\n" + "="*55)
print("  TRAINING FINAL MODELS WITH TUNED PARAMS")
print("="*55)

# XGBoost with best params
xgb_best_params = {
    **xgb_study.best_params,
    "eval_metric": "logloss",
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}
print("\nTraining tuned XGBoost...")
xgb_oof_tuned, xgb_test_tuned, xgb_models_tuned = train_kfold(
    X_train, y_train, X_test,
    model_type="xgb",
    params=xgb_best_params,
    verbose=True
)
xgb_thresh_tuned = optimize_threshold(y_train.values, xgb_oof_tuned)
xgb_mae_tuned = mean_absolute_error(
    y_train, (xgb_oof_tuned >= xgb_thresh_tuned).astype(int)
)
print(f"\n  Tuned XGBoost MAE: {xgb_mae_tuned:.4f}")
save_model(xgb_models_tuned[0], "xgboost_tuned")

# LightGBM with best params
lgbm_best_params = {
    **lgbm_study.best_params,
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}
print("\nTraining tuned LightGBM...")
lgbm_oof_tuned, lgbm_test_tuned, lgbm_models_tuned = train_kfold(
    X_train, y_train, X_test,
    model_type="lgbm",
    params=lgbm_best_params,
    verbose=True
)
lgbm_thresh_tuned = optimize_threshold(y_train.values, lgbm_oof_tuned)
lgbm_mae_tuned = mean_absolute_error(
    y_train, (lgbm_oof_tuned >= lgbm_thresh_tuned).astype(int)
)
print(f"\n  Tuned LightGBM MAE: {lgbm_mae_tuned:.4f}")
save_model(lgbm_models_tuned[0], "lightgbm_tuned")


# %% CELL 8 — Re-stack with tuned models
print("\n" + "="*55)
print("  FINAL ENSEMBLE (with tuned models)")
print("="*55)

# Load CatBoost OOF from previous notebook
cat_oof  = np.load("models/cat_oof.npy")
cat_test = np.load("models/cat_test.npy")

oof_dict_tuned = {
    "xgb_tuned"  : xgb_oof_tuned,
    "lgbm_tuned" : lgbm_oof_tuned,
    "catboost"   : cat_oof,
}
test_dict_tuned = {
    "xgb_tuned"  : xgb_test_tuned,
    "lgbm_tuned" : lgbm_test_tuned,
    "catboost"   : cat_test,
}

final_proba_tuned, final_threshold_tuned, meta_tuned = build_stacking_ensemble(
    oof_dict=oof_dict_tuned,
    test_dict=test_dict_tuned,
    y_train=y_train,
    verbose=True
)

final_preds_tuned = (final_proba_tuned >= final_threshold_tuned).astype(int)

# %% CELL 9 — Save updated submission
import json

submission_tuned = pd.DataFrame({
    "unique_id"   : test_ids,
    "bank_account": final_preds_tuned
})
submission_tuned.to_csv("outputs/submission_tuned.csv", index=False)
print(f"\n✓ Tuned submission saved: outputs/submission_tuned.csv")
print(f"  Predicted banked: {final_preds_tuned.sum():,} "
      f"({final_preds_tuned.mean()*100:.1f}%)")

# Save best params for reproducibility
best_params_record = {
    "xgb_best_params" : xgb_best_params,
    "lgbm_best_params": lgbm_best_params,
    "xgb_best_mae"    : float(xgb_mae_tuned),
    "lgbm_best_mae"   : float(lgbm_mae_tuned),
}
with open("models/best_params.json", "w") as f:
    json.dump(best_params_record, f, indent=2)
print("✓ Best params saved: models/best_params.json")

# Save tuned OOF arrays
np.save("models/xgb_oof_tuned.npy",  xgb_oof_tuned)
np.save("models/lgbm_oof_tuned.npy", lgbm_oof_tuned)
np.save("models/xgb_test_tuned.npy",  xgb_test_tuned)
np.save("models/lgbm_test_tuned.npy", lgbm_test_tuned)
np.save("models/final_proba_tuned.npy", final_proba_tuned)

print("\n✓ Ready for Notebook 05 — SHAP Explainability")
