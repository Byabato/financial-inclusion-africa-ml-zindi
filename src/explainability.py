# ============================================================
# src/explainability.py
#
# SHAP-based model explainability.
#
# WHY SHAP over LIME for this project?
#   1. TreeExplainer is FAST — optimized for XGB/LGBM/CatBoost
#   2. SHAP gives BOTH global (all data) AND local (one person)
#      explanations — LIME is local only
#   3. SHAP values are mathematically consistent (game theory)
#   4. SHAP works natively with XGBoost — no workarounds needed
#
# What we produce:
#   - Summary plot: global feature importance
#   - Bar plot: mean absolute SHAP importance
#   - Dependence plots: how each top feature affects prediction
#   - Waterfall plot: why ONE specific person was predicted unbanked
#   - Policy insights: which features are highest-impact barriers
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving

import shap
import os

from src.config import OUTPUTS_DIR


def compute_shap_values(model, X: pd.DataFrame) -> tuple:
    """
    Computes SHAP values using TreeExplainer (fast, exact for trees).

    Parameters
    ----------
    model : trained XGBoost / LightGBM / CatBoost model
    X     : feature DataFrame to explain

    Returns
    -------
    explainer   : shap.TreeExplainer object
    shap_values : numpy array of shape (n_samples, n_features)
    """
    print("  Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # CatBoost / some LightGBM return list [class0, class1]
    # We want class 1 (has account)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print(f"  ✓ SHAP values computed — shape: {shap_values.shape}")
    return explainer, shap_values


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 20,
    save: bool = True
):
    """
    Beeswarm summary plot.
    - Each dot = one person
    - X-axis = SHAP value (positive → pushes toward "has account")
    - Color = feature value (red = high, blue = low)

    This is the most informative single SHAP plot — shows
    both direction AND magnitude of each feature's impact.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X,
        max_display=max_display,
        show=False,
        plot_size=(10, 8)
    )
    plt.title("SHAP Feature Impact — Financial Inclusion Model\n"
              "Red = high feature value, Blue = low feature value",
              fontsize=13, pad=15)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, "shap_summary_beeswarm.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {path}")
    plt.close()


def plot_shap_bar(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    max_display: int = 15,
    save: bool = True
):
    """
    Mean absolute SHAP values as bar chart.
    Simple to communicate to non-technical stakeholders:
    'These are the TOP factors driving bank account ownership.'
    """
    mean_shap = np.abs(shap_values).mean(axis=0)
    feature_names = X.columns.tolist()

    # Sort by importance
    idx = np.argsort(mean_shap)[::-1][:max_display]
    top_features = [feature_names[i] for i in idx]
    top_values   = [mean_shap[i] for i in idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax.barh(top_features[::-1], top_values[::-1],
                   color=colors, edgecolor="white", height=0.7)

    ax.set_xlabel("Mean |SHAP Value| — Average Impact on Prediction",
                  fontsize=11)
    ax.set_title("Top Features Driving Bank Account Prediction\n"
                 "(Financial Inclusion — East Africa)",
                 fontsize=13, pad=15)

    # Add value labels
    for bar, val in zip(bars, top_values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", ha="left", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, "shap_bar_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {path}")
    plt.close()

    # Return as DataFrame for further use
    return pd.DataFrame({
        "feature": top_features,
        "mean_shap": top_values
    })


def plot_shap_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str = None,
    save: bool = True
):
    """
    Dependence plot: how a single feature's value relates to
    its SHAP impact. Reveals non-linearities.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feature, shap_values, X,
        interaction_index=interaction_feature,
        ax=ax, show=False
    )
    plt.title(f"SHAP Dependence: {feature}", fontsize=12)
    plt.tight_layout()

    if save:
        fname = f"shap_dependence_{feature.replace('/', '_')}.png"
        path = os.path.join(OUTPUTS_DIR, fname)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {path}")
    plt.close()


def explain_single_prediction(
    explainer,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    idx: int,
    predicted_class: int,
    save: bool = True
):
    """
    Explains WHY the model predicted a specific individual
    as banked or unbanked.

    This is the core of the recommender system:
    - Shows which features pushed TOWARD banking (positive SHAP)
    - Shows which features pushed AWAY from banking (negative SHAP)
    - Returns the top barriers (for recommendation generation)
    """
    person = X.iloc[idx]
    person_shap = shap_values[idx]

    # Top barriers (features pushing prediction toward 0 = unbanked)
    shap_series = pd.Series(person_shap, index=X.columns)
    barriers    = shap_series.nsmallest(5)
    enablers    = shap_series.nlargest(5)

    status = "BANKED ✓" if predicted_class == 1 else "UNBANKED ✗"

    print(f"\n  Person {idx} — Predicted: {status}")
    print(f"  {'─'*45}")
    print(f"  Top BARRIERS (pushing toward unbanked):")
    for feat, val in barriers.items():
        print(f"    {feat:<30} SHAP={val:+.4f}  value={person[feat]:.2f}")
    print(f"\n  Top ENABLERS (pushing toward banked):")
    for feat, val in enablers.items():
        print(f"    {feat:<30} SHAP={val:+.4f}  value={person[feat]:.2f}")

    return {
        "barriers": barriers.to_dict(),
        "enablers": enablers.to_dict()
    }


def get_feature_importance_df(
    shap_values: np.ndarray,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns a clean DataFrame of features ranked by mean |SHAP|.
    Useful for reports, policy insights, and further analysis.
    """
    mean_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_shap,
        "mean_shap": shap_values.mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    df["rank"] = df.index + 1
    df["impact_direction"] = df["mean_shap"].apply(
        lambda x: "↑ Increases banking" if x > 0 else "↓ Decreases banking"
    )

    return df
