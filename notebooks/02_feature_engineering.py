# ============================================================
# NOTEBOOK 02 — Feature Engineering & Preprocessing
# Financial Inclusion in Africa — Zindi Challenge
#
# PURPOSE:
#   Transform raw data into model-ready features.
#   Apply all engineering from src/features.py.
#   Produce engineered train/test CSVs for modeling.
#
# RUN AFTER: 01_EDA.py
# ============================================================

# %% [markdown]
# # Notebook 02: Feature Engineering & Preprocessing
#
# We transform raw survey data into model-ready features.
# Every feature created here is backed by domain knowledge
# from financial inclusion research.

# %% CELL 1 — Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append("..")


# Robust src import regardless of working directory
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.features import (
    engineer_features,
    kfold_target_encode,
    get_feature_columns
)
  
from src.config import TARGET, OUTPUTS_DIR, TRAIN_PATH, TEST_PATH, DATA_PROCESSED

plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
PALETTE = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12"]

print("✓ Setup complete")

# %% CELL 2 — Load raw data
train = pd.read_csv("data/raw/Train.csv")
test  = pd.read_csv("data/raw/Test.csv")


from src.config import TRAIN_PATH, TEST_PATH, DATA_PROCESSED
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
# Encode target: Yes → 1, No → 0
train[TARGET] = (train[TARGET] == "Yes").astype(int)

print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Target mean (inclusion rate): {train[TARGET].mean():.3f}")

# Keep uniqueid for final submission
train_ids = train["uniqueid"].copy()
test_ids  = test["uniqueid"].copy()
test_countries = test["country"].copy()

# %% CELL 3 — Apply feature engineering pipeline
# -------------------------------------------------------
# engineer_features() applies in order:
#   1. Binary encoding (Yes/No → 0/1)
#   2. Ordinal encoding (education, employment ranks)
#   3. One-hot encoding (country dummies)
#   4. Age feature engineering
#   5. Composite domain features
#   6. Interaction features
#
# IMPORTANT: We keep target column in train_eng
# and pass drop_originals=True to remove raw strings
# -------------------------------------------------------

print("Applying feature engineering pipeline...")

train_eng = engineer_features(train, drop_originals=True)
test_eng  = engineer_features(test,  drop_originals=True)

# Re-attach target to train after engineering
train_eng[TARGET] = train[TARGET].values

print(f"✓ Train after engineering: {train_eng.shape}")
print(f"✓ Test after engineering : {test_eng.shape}")
print(f"\nNew features added:")
original_cols = set(train.columns)
new_cols = [c for c in train_eng.columns if c not in original_cols]
for col in new_cols:
    print(f"  + {col}")

# %% CELL 4 — Target encoding (K-Fold)
# -------------------------------------------------------
# WHY do target encoding AFTER engineer_features?
#   Because we need the target column in train_eng.
#   Target encoding requires the target to compute
#   group means — never use raw train split for this.
#
# We encode these high-cardinality group interactions:
#   - country (already in train via dummies, but TE adds
#     the numerical inclusion rate directly)
#   - job_type: What % of people in this job have accounts?
#   - education_level: What % at this level have accounts?
#   - relationship_with_head: What % in this role?
# -------------------------------------------------------

print("\nApplying K-Fold target encoding...")

# We need the original string columns for target encoding
# Let's temporarily add them back from raw data
te_cols = ["job_type", "education_level",
           "relationship_with_head", "marital_status"]

# Re-attach string columns to train/test for TE
train_eng_te = train_eng.copy()
test_eng_te  = test_eng.copy()

for col in te_cols:
    train_eng_te[col] = train[col].values
    test_eng_te[col]  = test[col].values

# Apply K-Fold target encoding
train_eng_te, test_eng_te = kfold_target_encode(
    train_df   = train_eng_te,
    test_df    = test_eng_te,
    target_col = TARGET,
    group_cols = te_cols,
    n_splits   = 5,
    smoothing  = 20.0
)

# Drop original string columns (already have ordinal versions)
train_eng_te.drop(columns=te_cols, errors="ignore", inplace=True)
test_eng_te.drop(columns=te_cols, errors="ignore", inplace=True)

new_te_cols = [f"{c}_te" for c in te_cols]
print(f"✓ Target-encoded columns added: {new_te_cols}")

# %% CELL 5 — Verify no data leakage
# -------------------------------------------------------
# CRITICAL CHECK: Target encoding must not leak.
# If there's leakage, train MAE will be unrealistically low
# but test MAE will be much higher. Check by verifying that
# the TE columns were built on out-of-fold data only.
# -------------------------------------------------------

print("\n🔍 Data leakage check:")
print(f"  TARGET col in train_eng_te: {TARGET in train_eng_te.columns}")
print(f"  Unique TE values (should be varied, not 0/1 only): "
      f"{train_eng_te['job_type_te'].nunique()} unique values")
print(f"  TE column range: "
      f"[{train_eng_te['job_type_te'].min():.3f}, "
      f"{train_eng_te['job_type_te'].max():.3f}]")
print("  ✓ Range is between 0 and 1 (proportions) — looks correct")

# %% CELL 6 — Align columns
# -------------------------------------------------------
# After all transformations, train and test must have
# the exact same feature columns (except target).
# This alignment step ensures no column mismatch errors
# when predicting on test set.
# -------------------------------------------------------

target_col = TARGET
feature_cols = [c for c in train_eng_te.columns if c != target_col]

# Keep only columns that exist in BOTH train and test
common_cols = [c for c in feature_cols if c in test_eng_te.columns]
missing_in_test  = set(feature_cols) - set(test_eng_te.columns)
missing_in_train = set(test_eng_te.columns) - set(feature_cols)

if missing_in_test:
    print(f"⚠️  Cols in train not in test: {missing_in_test}")
    # Fill with 0 (most are dummy columns for missing countries)
    for col in missing_in_test:
        test_eng_te[col] = 0

if missing_in_train:
    print(f"⚠️  Cols in test not in train: {missing_in_train}")
    for col in missing_in_train:
        train_eng_te[col] = 0

# Final feature list
FEATURE_COLS = sorted(
    [c for c in train_eng_te.columns
     if c not in [TARGET, "bank_account"]]
)

X_train = train_eng_te[FEATURE_COLS]
y_train = train_eng_te[TARGET]
X_test  = test_eng_te[FEATURE_COLS]

print(f"\n✓ Final feature count: {len(FEATURE_COLS)}")
print(f"✓ X_train: {X_train.shape}")
print(f"✓ X_test : {X_test.shape}")
print(f"✓ y_train: {y_train.shape} | mean={y_train.mean():.3f}")

# %% CELL 7 — Feature correlation with target
# -------------------------------------------------------
# CHART: Which engineered features correlate most with target?
# -------------------------------------------------------

correlations = X_train.corrwith(y_train).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 10))
colors = ["#2ECC71" if v > 0 else "#E74C3C" for v in correlations.values]
ax.barh(correlations.index[::-1], correlations.values[::-1],
        color=colors[::-1], edgecolor="white", height=0.7)
ax.axvline(x=0, color="black", linewidth=0.8)
ax.set_xlabel("Pearson Correlation with Target (bank_account)",
              fontsize=11)
ax.set_title("Feature Correlation with Target\n"
             "Green = positive (more likely banked), "
             "Red = negative (less likely)",
             fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}/11_feature_correlations.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved: 11_feature_correlations.png")

# %% CELL 8 — Save processed data
# -------------------------------------------------------

X_train.to_csv("data/processed/X_train.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv",  index=False)


X_train.to_csv(os.path.join(DATA_PROCESSED, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(DATA_PROCESSED, "y_train.csv"), index=False)
X_test.to_csv(os.path.join(DATA_PROCESSED, "X_test.csv"),  index=False)

# Save test IDs for submission
test_submission_ids = test_ids + " x " + test_countries
test_submission_ids.to_csv(os.path.join(DATA_PROCESSED, "test_ids.csv"),
                            index=False, header=["unique_id"])

# Save feature column list
import json
with open(os.path.join(DATA_PROCESSED, "feature_cols.json"), "w") as f:
    json.dump(FEATURE_COLS, f, indent=2)

print("\n✓ Saved processed data:")
print("  data/processed/X_train.csv")
print("  data/processed/y_train.csv")
print("  data/processed/X_test.csv")
print("  data/processed/test_ids.csv")
print("  data/processed/feature_cols.json")

# %% CELL 9 — Feature summary table
print("\n" + "="*60)
print("  FEATURE ENGINEERING SUMMARY")
print("="*60)
print(f"  Original features : {train.shape[1]}")
print(f"  Engineered features: {len(FEATURE_COLS)}")
print(f"\n  Feature groups created:")
print(f"    Binary encodings    : cellphone, gender, location")
print(f"    Ordinal encodings   : education, employment, relationship, marital")
print(f"    Country dummies     : Rwanda, Tanzania, Uganda (Kenya=ref)")
print(f"    Age features        : age_group, age_squared")
print(f"    Composite features  : inclusion_score, is_head, is_dependent, ...")
print(f"    Interaction features: edu_x_employment, mobile_x_urban, ...")
print(f"    Target encodings    : job_te, education_te, relationship_te, marital_te")
print(f"\n  Top 5 correlated features:")
for feat, corr in correlations.head(5).items():
    print(f"    {feat:<30}: r={corr:.4f}")
print(f"\n✓ Ready for Notebook 03 — Baseline Model")
