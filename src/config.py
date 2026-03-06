# ============================================================
# src/config.py
# Central configuration — all constants live here.
# If you need to change a path, threshold, or mapping,
# change it ONCE here and it propagates everywhere.
# ============================================================

import os

import sys
from pathlib import Path

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

TRAIN_PATH = os.path.join(DATA_RAW, "Train.csv")
TEST_PATH = os.path.join(DATA_RAW, "Test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_RAW, "SampleSubmission.csv")
SUBMISSION_PATH = os.path.join(OUTPUTS_DIR, "submission.csv")

# Ensure directories exist
for path in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(path, exist_ok=True)

# ------------------------------------------------------------------
# RANDOM SEED — used everywhere for reproducibility
# ------------------------------------------------------------------
SEED = 42

# ------------------------------------------------------------------
# TARGET COLUMN
# ------------------------------------------------------------------
TARGET = "bank_account"

# ------------------------------------------------------------------
# ORDINAL ENCODING MAPS
# These encode categorical features as ordered integers,
# reflecting real-world hierarchy (e.g. more educated = higher #)
# ------------------------------------------------------------------

EDUCATION_MAP = {
    "No formal education"            : 0,
    "Other/Dont know/RTA"            : 0,
    "Primary education"              : 1,
    "Secondary education"            : 2,
    "Vocational/Specialised training": 3,
    "Tertiary education"             : 4,
}

EMPLOYMENT_MAP = {
    "No Income"                      : 0,
    "Dont Know/Refuse to answer"     : 0,
    "Government Dependent"           : 1,
    "Remittance Dependent"           : 1,
    "Farming and Fishing"            : 2,
    "Informally employed"            : 2,
    "Other Income"                   : 2,
    "Self employed"                  : 3,
    "Formally employed Private"      : 4,
    "Formally employed Government"   : 5,
}

RELATIONSHIP_MAP = {
    "Dont know"          : 0,
    "Other non-relatives": 1,
    "Other relative"     : 2,
    "Child"              : 3,
    "Parent"             : 3,
    "Spouse"             : 4,
    "Head of Household"  : 5,
}

MARITAL_MAP = {
    "Dont know"                  : 0,
    "Divorced/Seperated"         : 1,
    "Widowed"                    : 2,
    "Single/Never Married"       : 3,
    "Married/Living together"    : 4,
}

# ------------------------------------------------------------------
# BINARY ENCODING MAPS
# ------------------------------------------------------------------
BINARY_MAP = {
    "Yes": 1, "No": 0,
    "Male": 1, "Female": 0,
    "Urban": 1, "Rural": 0,
}

# ------------------------------------------------------------------
# COUNTRY ONE-HOT (reference category = Kenya, dropped to avoid
# dummy variable trap in linear models)
# ------------------------------------------------------------------
COUNTRIES = ["Kenya", "Rwanda", "Tanzania", "Uganda"]

# ------------------------------------------------------------------
# AGE BINS — based on lifecycle financial behaviour research
# 0  = minor (<18)
# 1  = young adult (18-25)
# 2  = early career (26-35)
# 3  = prime earning (36-50)
# 4  = late career (51-65)
# 5  = elderly (65+)
# ------------------------------------------------------------------
AGE_BINS   = [0, 18, 25, 35, 50, 65, 120]
AGE_LABELS = [0, 1, 2, 3, 4, 5]

# ------------------------------------------------------------------
# MODEL CONFIG
# ------------------------------------------------------------------
CV_FOLDS = 5          # Stratified K-Fold
THRESHOLD = 0.35      # Starting threshold; will be tuned per model

# XGBoost default params (will be overridden by Optuna)
XGB_DEFAULT_PARAMS = {
    "n_estimators"     : 500,
    "max_depth"        : 6,
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "scale_pos_weight" : 3,   # ~ratio of negatives/positives; handles imbalance
    "eval_metric"      : "logloss",
    "use_label_encoder": False,
    "random_state"     : SEED,
    "n_jobs"           : -1,
}

LGBM_DEFAULT_PARAMS = {
    "n_estimators"    : 500,
    "max_depth"       : 6,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "class_weight"    : "balanced",
    "random_state"    : SEED,
    "n_jobs"          : -1,
}

CATBOOST_DEFAULT_PARAMS = {
    "iterations"  : 500,
    "depth"       : 6,
    "learning_rate": 0.05,
    "auto_class_weights": "Balanced",
    "random_seed" : SEED,
    "verbose"     : 0,
}
