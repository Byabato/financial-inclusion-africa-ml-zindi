# ============================================================
# NOTEBOOK 03 — Model Training: Baseline → Advanced → Ensemble
# Financial Inclusion in Africa — Zindi Challenge
#
# PURPOSE:
#   Train all models in sequence:
#   1. Logistic Regression (interpretable baseline)
#   2. XGBoost (best single model)
#   3. LightGBM (speed + balanced class handling)
#   4. CatBoost (research-validated for this domain)
#   5. Stacking Ensemble (combines all)
#   6. Threshold optimization (competition-critical step)
#
# OUTPUT: submission.csv ready for Zindi upload
#
# RUN AFTER: 02_feature_engineering.py
# ============================================================

# %% [markdown]
# # Notebook 03: Model Training & Evaluation
#
# We train from simple to complex, comparing at each step.
# The final submission uses the stacked ensemble.

# %% CELL 1 — Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import sys
import os
import json
import warnings


warnings.filterwarnings("ignore")
# Robust src import regardless of working directory
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models import (
    train_kfold, optimize_threshold,
    print_evaluation, save_model
)
from src.ensemble import build_stacking_ensemble, simple_average_ensemble
from src.config import SEED, OUTPUTS_DIR, SUBMISSION_PATH

np.random.seed(SEED)
print("✓ Setup complete")

# %% CELL 2 — Load processed data

from src.config import DATA_PROCESSED, OUTPUTS_DIR, SUBMISSION_PATH
X_train = pd.read_csv(os.path.join(DATA_PROCESSED, "X_train.csv"))
y_train = pd.read_csv(os.path.join(DATA_PROCESSED, "y_train.csv")).squeeze()
X_test  = pd.read_csv(os.path.join(DATA_PROCESSED, "X_test.csv"))
test_ids = pd.read_csv(os.path.join(DATA_PROCESSED, "test_ids.csv")).squeeze()

print(f"✓ X_train: {X_train.shape}")
print(f"✓ y_train: {y_train.shape} | positive rate: {y_train.mean():.3f}")
print(f"✓ X_test : {X_test.shape}")
print(f"✓ test_ids: {len(test_ids)} rows")

# Align columns — test must match train exactly
common_cols = [c for c in X_train.columns if c in X_test.columns]
X_train = X_train[common_cols]
X_test  = X_test[common_cols]
print(f"✓ Aligned to {len(common_cols)} common features")

# %% CELL 3 — STEP 1: Logistic Regression Baseline
# -------------------------------------------------------
# WHY start with logistic regression?
#   - Fast to train
#   - Fully interpretable (coefficients)
#   - Sets a performance FLOOR
#   - Any good model must beat this convincingly
#
# Expected performance: ~0.13-0.17 MAE
# -------------------------------------------------------

print("\n" + "="*55)
print("  STEP 1: LOGISTIC REGRESSION BASELINE")
print("="*55)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Scale features (required for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

lr_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=SEED,
    C=1.0
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_scores = cross_val_score(
    lr_model, X_train_scaled, y_train,
    cv=skf, scoring="neg_mean_absolute_error"
)

lr_cv_mae = -cv_scores.mean()
print(f"  Logistic Regression CV MAE: {lr_cv_mae:.4f} ± {cv_scores.std():.4f}")
print(f"  → Baseline to beat: {lr_cv_mae:.4f}")

# Fit on full train for OOF predictions
lr_oof    = np.zeros(len(X_train))
lr_test   = np.zeros(len(X_test))
lr_models = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train), 1):
    Xtr, Xval = X_train_scaled[tr_idx], X_train_scaled[val_idx]
    ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    lr = LogisticRegression(class_weight="balanced",
                             max_iter=1000, random_state=SEED)
    lr.fit(Xtr, ytr)

    lr_oof[val_idx] = lr.predict_proba(Xval)[:, 1]
    lr_test += lr.predict_proba(X_test_scaled)[:, 1] / 5
    lr_models.append(lr)

lr_threshold = optimize_threshold(y_train.values, lr_oof, verbose=True)
lr_preds = (lr_oof >= lr_threshold).astype(int)
lr_mae = mean_absolute_error(y_train, lr_preds)
print(f"\n  LR Final OOF MAE (tuned threshold): {lr_mae:.4f}")
save_model(lr_models[-1], "logreg_baseline")

# %% CELL 4 — STEP 2: XGBoost
# -------------------------------------------------------
# XGBoost is THE proven best model for this dataset.
# Key advantages:
#   - Handles mixed data types
#   - Built-in regularization (prevents overfitting)
#   - scale_pos_weight handles class imbalance natively
#   - Fast with early stopping
# -------------------------------------------------------

print("\n" + "="*55)
print("  STEP 2: XGBOOST")
print("="*55)

from src.config import XGB_DEFAULT_PARAMS

xgb_oof, xgb_test, xgb_models = train_kfold(
    X_train, y_train, X_test,
    model_type="xgb",
    params=XGB_DEFAULT_PARAMS,
    verbose=True
)

xgb_threshold = optimize_threshold(y_train.values, xgb_oof)
xgb_preds = (xgb_oof >= xgb_threshold).astype(int)
xgb_mae = mean_absolute_error(y_train, xgb_preds)
print(f"\n  XGBoost Final OOF MAE: {xgb_mae:.4f}")
print(f"  Improvement over baseline: "
      f"{lr_mae - xgb_mae:.4f} points")

save_model(xgb_models[0], "xgboost_fold1")

# %% CELL 5 — STEP 3: LightGBM
# -------------------------------------------------------
# LightGBM advantages:
#   - Faster training than XGBoost
#   - class_weight='balanced' handles imbalance
#   - Good on datasets with many features
#   - Often very close to XGBoost performance
# -------------------------------------------------------

print("\n" + "="*55)
print("  STEP 3: LIGHTGBM")
print("="*55)

from src.config import LGBM_DEFAULT_PARAMS

lgbm_oof, lgbm_test, lgbm_models = train_kfold(
    X_train, y_train, X_test,
    model_type="lgbm",
    params=LGBM_DEFAULT_PARAMS,
    verbose=True
)

lgbm_threshold = optimize_threshold(y_train.values, lgbm_oof)
lgbm_preds = (lgbm_oof >= lgbm_threshold).astype(int)
lgbm_mae = mean_absolute_error(y_train, lgbm_preds)
print(f"\n  LightGBM Final OOF MAE: {lgbm_mae:.4f}")

save_model(lgbm_models[0], "lightgbm_fold1")

# %% CELL 6 — STEP 4: CatBoost
# -------------------------------------------------------
# CatBoost advantages:
#   - Auto-handles categorical features (no encoding needed,
#     but we pre-encoded for consistency)
#   - Research shows it's strong for African financial data
#   - auto_class_weights='Balanced' handles imbalance
#   - Often best when categories have many levels
# -------------------------------------------------------

print("\n" + "="*55)
print("  STEP 4: CATBOOST")
print("="*55)

from src.config import CATBOOST_DEFAULT_PARAMS

cat_oof, cat_test, cat_models = train_kfold(
    X_train, y_train, X_test,
    model_type="catboost",
    params=CATBOOST_DEFAULT_PARAMS,
    verbose=True
)

cat_threshold = optimize_threshold(y_train.values, cat_oof)
cat_preds = (cat_oof >= cat_threshold).astype(int)
cat_mae = mean_absolute_error(y_train, cat_preds)
print(f"\n  CatBoost Final OOF MAE: {cat_mae:.4f}")

save_model(cat_models[0], "catboost_fold1")

# %% CELL 7 — STEP 5: Stacking Ensemble
# -------------------------------------------------------
# Stack the OOF predictions of all 3 tree models.
# A meta-Logistic Regression learns the optimal combination.
# -------------------------------------------------------

print("\n" + "="*55)
print("  STEP 5: STACKING ENSEMBLE")
print("="*55)

oof_dict  = {"xgb": xgb_oof,  "lgbm": lgbm_oof,  "catboost": cat_oof}
test_dict = {"xgb": xgb_test, "lgbm": lgbm_test, "catboost": cat_test}

final_test_proba, ensemble_threshold, meta_model = build_stacking_ensemble(
    oof_dict=oof_dict,
    test_dict=test_dict,
    y_train=y_train,
    verbose=True
)

final_predictions = (final_test_proba >= ensemble_threshold).astype(int)
print(f"\n  Test prediction distribution:")
print(f"    Predicted banked  : {final_predictions.sum():,} "
      f"({final_predictions.mean()*100:.1f}%)")
print(f"    Predicted unbanked: {(1-final_predictions).sum():,} "
      f"({(1-final_predictions).mean()*100:.1f}%)")

# %% CELL 8 — Model comparison chart
# -------------------------------------------------------

model_results = {
    "Logistic Regression": lr_mae,
    "XGBoost"            : xgb_mae,
    "LightGBM"           : lgbm_mae,
    "CatBoost"           : cat_mae,
}

# Ensemble OOF MAE
from sklearn.metrics import mean_absolute_error as mae_fn
from sklearn.linear_model import LogisticRegression as LR

oof_stack  = np.column_stack([xgb_oof, lgbm_oof, cat_oof])
ens_oof    = meta_model.predict_proba(oof_stack)[:, 1]
ens_preds  = (ens_oof >= ensemble_threshold).astype(int)
ens_mae    = mae_fn(y_train, ens_preds)
model_results["Stacking Ensemble"] = ens_mae

fig, ax = plt.subplots(figsize=(10, 5))
names  = list(model_results.keys())
values = list(model_results.values())
colors = ["#95A5A6", "#3498DB", "#2ECC71", "#E67E22", "#E74C3C"]

bars = ax.bar(names, values, color=colors, edgecolor="white", width=0.55)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")

# Best model annotation
best_idx = np.argmin(values)
bars[best_idx].set_edgecolor("gold")
bars[best_idx].set_linewidth(3)
ax.annotate("★ Best", xy=(best_idx, values[best_idx]),
            xytext=(best_idx + 0.1, values[best_idx] + 0.003),
            fontsize=10, color="goldenrod", fontweight="bold")

ax.set_ylabel("Mean Absolute Error (lower = better)", fontsize=11)
ax.set_title("Model Performance Comparison — OOF MAE\n"
             "(Lower MAE = Better Competition Score)",
             fontsize=13, pad=12)
ax.set_ylim(0, max(values) + 0.02)
ax.tick_params(axis="x", rotation=10)
plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}/12_model_comparison.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved: 12_model_comparison.png")

print("\n  Model Ranking (OOF MAE):")
for i, (name, val) in enumerate(sorted(model_results.items(),
                                        key=lambda x: x[1]), 1):
    print(f"  {i}. {name:<25}: {val:.4f}")

# %% CELL 9 — XGBoost Feature Importance (Built-in)
# -------------------------------------------------------
# Quick built-in feature importance from best XGBoost model.
# We'll do a deeper SHAP analysis in Notebook 04.
# -------------------------------------------------------

best_xgb = xgb_models[0]  # First fold model
importances = pd.Series(
    best_xgb.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
top_n = 20
colors_imp = plt.cm.Blues(
    np.linspace(0.4, 0.9, top_n)
)[::-1]
ax.barh(importances.index[:top_n][::-1],
        importances.values[:top_n][::-1],
        color=colors_imp, edgecolor="white", height=0.7)
ax.set_xlabel("Feature Importance Score", fontsize=11)
ax.set_title(f"Top {top_n} Features — XGBoost Importance\n"
             "(See Notebook 04 for deeper SHAP analysis)",
             fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}/13_xgb_feature_importance.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved: 13_xgb_feature_importance.png")

# %% CELL 10 — Generate submission file
# -------------------------------------------------------
# FORMAT: uniqueid + " x " + country → bank_account (0 or 1)
# -------------------------------------------------------

print("\n" + "="*55)
print("  GENERATING SUBMISSION FILE")
print("="*55)

submission = pd.DataFrame({
    "unique_id"   : test_ids,
    "bank_account": final_predictions
})

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"✓ Submission saved: {SUBMISSION_PATH}")
print(f"\n  Preview:")
print(submission.head(10).to_string(index=False))
print(f"\n  Submission shape: {submission.shape}")
print(f"  Predicted banked  : {submission['bank_account'].sum():,} "
      f"({submission['bank_account'].mean()*100:.1f}%)")
print(f"  Predicted unbanked: {(submission['bank_account']==0).sum():,} "
      f"({(1-submission['bank_account'].mean())*100:.1f}%)")

# %% CELL 11 — Save OOF predictions for stacking/analysis
from src.config import MODELS_DIR
np.save(os.path.join(MODELS_DIR, "xgb_oof.npy"),  xgb_oof)
np.save(os.path.join(MODELS_DIR, "lgbm_oof.npy"), lgbm_oof)
np.save(os.path.join(MODELS_DIR, "cat_oof.npy"),  cat_oof)
np.save(os.path.join(MODELS_DIR, "xgb_test.npy"),  xgb_test)
np.save(os.path.join(MODELS_DIR, "lgbm_test.npy"), lgbm_test)
np.save(os.path.join(MODELS_DIR, "cat_test.npy"),  cat_test)
np.save(os.path.join(MODELS_DIR, "final_test_proba.npy"), final_test_proba)

print("\n✓ OOF and test probability arrays saved to models/")
print("\n✓ Ready for Notebook 04 — SHAP & Explainability")
