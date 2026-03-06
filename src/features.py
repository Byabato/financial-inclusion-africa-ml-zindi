# ============================================================
# src/features.py
#
# ALL feature engineering lives here as pure functions.
# Why pure functions?
#   - Easy to test independently
#   - No hidden state / side effects
#   - Can be applied identically to train AND test
#
# USAGE:
#   from src.features import engineer_features
#   df_train = engineer_features(df_train)
#   df_test  = engineer_features(df_test)
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.config import (
    EDUCATION_MAP, EMPLOYMENT_MAP, RELATIONSHIP_MAP,
    MARITAL_MAP, BINARY_MAP, AGE_BINS, AGE_LABELS, SEED
)


# ------------------------------------------------------------------
# STEP 1 — BINARY ENCODING
# Simple Yes/No, Male/Female, Urban/Rural → 0 or 1
# ------------------------------------------------------------------
def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes binary categorical columns to 0/1.
    Acts on a copy — never mutates original.
    """
    df = df.copy()

    df["cellphone_access"] = df["cellphone_access"].map(BINARY_MAP)
    df["gender_of_respondent"] = df["gender_of_respondent"].map(BINARY_MAP)
    df["location_type"] = df["location_type"].map(BINARY_MAP)

    return df


# ------------------------------------------------------------------
# STEP 2 — ORDINAL ENCODING
# Maps categorical strings to ordered integers based on domain
# knowledge (not just alphabetical order like LabelEncoder would).
# ------------------------------------------------------------------
def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-knowledge-based ordinal encoding.

    WHY ordinal not one-hot here?
      - education and employment have a REAL ORDER
      - one-hot would lose that ordering information
      - tree models can use ordinal splits effectively
    """
    df = df.copy()

    df["education_rank"]    = df["education_level"].map(EDUCATION_MAP)
    df["employment_rank"]   = df["job_type"].map(EMPLOYMENT_MAP)
    df["relationship_rank"] = df["relationship_with_head"].map(RELATIONSHIP_MAP)
    df["marital_rank"]      = df["marital_status"].map(MARITAL_MAP)

    # Fill any unmapped values (rare/new categories) with median
    for col in ["education_rank", "employment_rank",
                "relationship_rank", "marital_rank"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    return df


# ------------------------------------------------------------------
# STEP 3 — ONE-HOT ENCODING
# For nominal categoricals with no meaningful order: country.
# relationship_with_head and marital_status are already ordinal-
# encoded above but we also keep dummies for tree model diversity.
# ------------------------------------------------------------------
def encode_onehot_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes country (nominal, no order).
    drop_first=True drops Kenya to avoid dummy variable trap.
    """
    df = df.copy()

    country_dummies = pd.get_dummies(
        df["country"], prefix="country", drop_first=True
    )
    df = pd.concat([df, country_dummies], axis=1)

    return df


# ------------------------------------------------------------------
# STEP 4 — AGE FEATURE ENGINEERING
# Raw age is useful but lifecycle stages add richer signal.
# Young adults and elderly behave differently from prime earners.
# ------------------------------------------------------------------
def engineer_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates:
      - age_group: lifecycle bucket (0–5)
      - age_squared: captures non-linear relationship
        (very young AND very old less likely to be banked)
    """
    df = df.copy()

    df["age_group"] = pd.cut(
        df["age_of_respondent"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True
    ).astype(float)

    # Non-linear age: banking likelihood peaks mid-life
    df["age_squared"] = df["age_of_respondent"] ** 2

    return df


# ------------------------------------------------------------------
# STEP 5 — DOMAIN COMPOSITE FEATURES
# These are the "secret sauce" — features that encode domain
# knowledge into single powerful signals.
# ------------------------------------------------------------------
def engineer_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates high-signal composite and interaction features.

    EACH feature is justified below:
    """
    df = df.copy()

    # --- Is head of household? ---
    # Financial decision-maker → likely account holder
    df["is_household_head"] = (
        df["relationship_with_head"] == "Head of Household"
    ).astype(int)

    # --- Is economically dependent? ---
    # Remittance/Govt dependent, No Income → structural barrier to banking
    df["is_dependent"] = df["job_type"].isin([
        "Remittance Dependent", "Government Dependent", "No Income",
        "Dont Know/Refuse to answer"
    ]).astype(int)

    # --- Is formally employed? ---
    # Most reliable income → highest bank account probability
    df["is_formal_employed"] = df["job_type"].isin([
        "Formally employed Government", "Formally employed Private"
    ]).astype(int)

    # --- Is married? ---
    # Married = joint finances, more likely dual/primary account
    df["is_married"] = (
        df["marital_status"] == "Married/Living together"
    ).astype(int)

    # --- Inclusion Score (composite) ---
    # Weighted domain knowledge score. Not just for the model —
    # also used in the recommender system.
    df["inclusion_score"] = (
        df["employment_rank"]   * 0.30 +
        df["education_rank"]    * 0.25 +
        df["cellphone_access"]  * 0.20 +
        df["location_type"]     * 0.15 +   # Urban=1
        (1 - df["is_dependent"]) * 0.10
    )

    # --- Household size per person (crowding proxy) ---
    # Very large households may share one account or none
    df["household_size_log"] = np.log1p(df["household_size"])

    return df


# ------------------------------------------------------------------
# STEP 6 — INTERACTION FEATURES
# Multiplicative features that capture synergies.
# "Educated AND employed" is much stronger than either alone.
# ------------------------------------------------------------------
def engineer_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Key interaction terms backed by financial inclusion research:
    - Education × Employment: double advantage
    - Mobile × Urban: digital-first pathway
    - Age × Education: mature educated adults
    - Head × Formal job: financially responsible + income
    """
    df = df.copy()

    df["edu_x_employment"]  = df["education_rank"] * df["employment_rank"]
    df["mobile_x_urban"]    = df["cellphone_access"] * df["location_type"]
    df["age_x_education"]   = df["age_of_respondent"] * df["education_rank"]
    df["head_x_formal"]     = df["is_household_head"] * df["is_formal_employed"]
    df["edu_x_mobile"]      = df["education_rank"] * df["cellphone_access"]
    df["formal_x_urban"]    = df["is_formal_employed"] * df["location_type"]
    df["married_x_head"]    = df["is_married"] * df["is_household_head"]

    return df


# ------------------------------------------------------------------
# STEP 7 — TARGET ENCODING (with K-Fold to prevent data leakage)
#
# WHY target encoding?
#   Country × Job and Country × Education group means carry
#   powerful signal: "What % of people in Kenya with formal
#   govt jobs have bank accounts?" is a direct predictor.
#
# WHY K-Fold target encoding?
#   Naive target encoding leaks the target into features.
#   K-Fold solution: encode each row using the mean from
#   the OTHER folds only. Test set uses full-train means.
# ------------------------------------------------------------------
def kfold_target_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    group_cols: list,
    n_splits: int = 5,
    smoothing: float = 20.0
) -> tuple:
    """
    K-Fold target encoding for a list of group columns.

    Parameters
    ----------
    train_df   : training DataFrame (must contain target_col)
    test_df    : test DataFrame (no target_col)
    target_col : name of the binary target column
    group_cols : list of column names to encode
                 e.g. ["country", "job_type"]
    n_splits   : number of folds (default 5)
    smoothing  : shrinks group means toward global mean
                 to handle small groups (avoid overfitting)

    Returns
    -------
    (train_df, test_df) with new *_te columns added
    """
    train_df = train_df.copy()
    test_df  = test_df.copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    global_mean = train_df[target_col].mean()

    for col in group_cols:
        # Column name for the new encoded feature
        new_col = f"{col}_te"

        # Initialize with global mean (fallback for unseen categories)
        train_df[new_col] = global_mean
        test_df[new_col]  = global_mean

        # K-Fold encoding for train set
        for train_idx, val_idx in kf.split(train_df):
            # Compute group means from training folds
            group_means = (
                train_df.iloc[train_idx]
                .groupby(col)[target_col]
                .agg(["mean", "count"])
            )

            # Smoothing formula:
            # smoothed = (count * mean + smoothing * global_mean) / (count + smoothing)
            # When count is large → mean dominates
            # When count is small → global_mean dominates (less overfit)
            group_means["smoothed"] = (
                (group_means["count"] * group_means["mean"]
                 + smoothing * global_mean)
                / (group_means["count"] + smoothing)
            )

            # Map smoothed means to validation fold
            train_df.loc[
                train_df.index[val_idx], new_col
            ] = train_df.iloc[val_idx][col].map(
                group_means["smoothed"]
            ).fillna(global_mean).values

        # For test set: use means from full training data
        full_group_means = (
            train_df.groupby(col)[target_col]
            .agg(["mean", "count"])
        )
        full_group_means["smoothed"] = (
            (full_group_means["count"] * full_group_means["mean"]
             + smoothing * global_mean)
            / (full_group_means["count"] + smoothing)
        )
        test_df[new_col] = test_df[col].map(
            full_group_means["smoothed"]
        ).fillna(global_mean)

    return train_df, test_df


# ------------------------------------------------------------------
# MASTER FUNCTION — call this on both train and test
# ------------------------------------------------------------------
def engineer_features(
    df: pd.DataFrame,
    drop_originals: bool = True
) -> pd.DataFrame:
    """
    Master feature engineering pipeline.
    Apply this to BOTH train and test (before target encoding).

    Parameters
    ----------
    df              : raw DataFrame
    drop_originals  : whether to drop original string columns
                      after encoding (True for modeling)

    Returns
    -------
    df with all engineered features added
    """
    df = encode_binary_features(df)
    df = encode_ordinal_features(df)
    df = encode_onehot_features(df)
    df = engineer_age_features(df)
    df = engineer_composite_features(df)
    df = engineer_interaction_features(df)

    if drop_originals:
        # Drop raw string columns — we have numeric equivalents
        cols_to_drop = [
            "education_level",
            "job_type",
            "relationship_with_head",
            "marital_status",
            "country",      # kept as dummies
            "uniqueid",     # identifier — never a feature
            "year",         # low variance; 2016-2018 only
        ]
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df = df.drop(columns=cols_to_drop)

    return df


# ------------------------------------------------------------------
# FEATURE COLUMNS (used in model training)
# Returns the final list of feature column names after engineering
# ------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Returns list of feature columns (excludes target and id columns).
    Call after engineer_features() has been applied.
    """
    exclude = {"bank_account", "uniqueid", "year"}
    return [c for c in df.columns if c not in exclude]
