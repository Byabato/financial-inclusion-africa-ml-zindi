# ============================================================
# src/recommender.py
#
# INNOVATION LAYER: Financial Inclusion Recommender System
#
# This goes beyond prediction. Given a person predicted as
# UNBANKED, we explain WHY and recommend the most actionable
# pathway to financial inclusion.
#
# Also includes:
#   - Population segmentation of unbanked groups
#   - Country-level policy scorecard
#   - Intervention impact estimator
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

from src.config import OUTPUTS_DIR

# ------------------------------------------------------------------
# RECOMMENDATION RULES
# Maps a top barrier feature → actionable recommendation
# ------------------------------------------------------------------
RECOMMENDATION_RULES = {
    "employment_rank": {
        "message": "💼 Employment pathway",
        "detail": (
            "This person's job type is a major barrier. "
            "Micro-enterprise registration or vocational upskilling "
            "could transition them to formal/self-employment, "
            "unlocking access to business banking and credit."
        ),
        "sdg": "SDG 8: Decent Work & Economic Growth"
    },
    "education_rank": {
        "message": "📚 Education pathway",
        "detail": (
            "Low education is limiting financial literacy and access. "
            "Adult financial literacy programs or vocational training "
            "could significantly improve inclusion probability."
        ),
        "sdg": "SDG 4: Quality Education"
    },
    "cellphone_access": {
        "message": "📱 Mobile money bridge",
        "detail": (
            "No cellphone access is a critical gap. "
            "Mobile money (M-Pesa, MTN Mobile Money, Airtel Money) "
            "is the fastest and most affordable first step to "
            "formal financial services in East Africa."
        ),
        "sdg": "SDG 9: Industry, Innovation & Infrastructure"
    },
    "location_type": {
        "message": "🏦 Agent banking expansion",
        "detail": (
            "Rural location creates distance and cost barriers. "
            "Agent banking networks (bank-on-wheels, village agents) "
            "eliminate the need to travel to urban branches."
        ),
        "sdg": "SDG 10: Reduced Inequalities"
    },
    "is_dependent": {
        "message": "🤝 Savings group enrollment",
        "detail": (
            "Economic dependency reduces individual account incentive. "
            "Community savings groups (SACCOs, VSLAs) provide "
            "collective financial services as an entry point."
        ),
        "sdg": "SDG 1: No Poverty"
    },
    "inclusion_score": {
        "message": "📊 Multi-factor intervention needed",
        "detail": (
            "Low composite inclusion score suggests multiple overlapping "
            "barriers. A bundled intervention (mobile + literacy + "
            "savings group) is recommended."
        ),
        "sdg": "SDG 1, 4, 8, 10"
    },
    "is_formal_employed": {
        "message": "💼 Formal employment support",
        "detail": (
            "Informal or no employment is limiting banking access. "
            "Connecting to formal employment opportunities or "
            "supporting business formalization would help."
        ),
        "sdg": "SDG 8: Decent Work"
    },
    "edu_x_employment": {
        "message": "🎓 Dual education + employment gap",
        "detail": (
            "Both education and employment are barriers simultaneously. "
            "Integrated programs combining skills training with "
            "job placement have the highest impact."
        ),
        "sdg": "SDG 4 + SDG 8"
    },
}

# Default recommendation when feature not in rules
DEFAULT_RECOMMENDATION = {
    "message": "🌍 General financial inclusion program",
    "detail": (
        "Consider enrolling in a local financial literacy program "
        "and opening a basic mobile money account as a first step."
    ),
    "sdg": "SDG 1: No Poverty"
}


def generate_recommendation(
    person_features: pd.Series,
    shap_values_for_person: np.ndarray,
    feature_names: list,
    n_barriers: int = 3
) -> dict:
    """
    Generates personalized recommendations for an unbanked individual.

    Parameters
    ----------
    person_features        : feature values for one person
    shap_values_for_person : SHAP values for that person
    feature_names          : list of feature column names
    n_barriers             : number of top barriers to address

    Returns
    -------
    recommendation dict with barriers, actions, and SDG alignment
    """
    # Find the top N features with the most negative SHAP values
    # (most negative = biggest barrier to being banked)
    shap_series = pd.Series(shap_values_for_person, index=feature_names)
    top_barriers = shap_series.nsmallest(n_barriers)

    actions = []
    for barrier_feature, shap_val in top_barriers.items():
        # Match to recommendation rule
        rule = RECOMMENDATION_RULES.get(
            barrier_feature, DEFAULT_RECOMMENDATION
        )
        actions.append({
            "barrier_feature": barrier_feature,
            "feature_value"  : float(person_features.get(barrier_feature, 0)),
            "shap_impact"    : float(shap_val),
            "action"         : rule["message"],
            "detail"         : rule["detail"],
            "sdg_alignment"  : rule["sdg"]
        })

    return {
        "n_barriers_identified": len(actions),
        "recommended_actions"  : actions,
        "primary_action"       : actions[0]["action"] if actions else None
    }


def generate_batch_recommendations(
    X_unbanked: pd.DataFrame,
    shap_values_unbanked: np.ndarray,
    unbanked_indices: list,
    original_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates recommendations for all predicted-unbanked individuals.
    Returns a DataFrame summarizing the most common barriers.
    """
    all_recs = []

    for i, (df_idx, shap_row) in enumerate(
        zip(unbanked_indices, shap_values_unbanked)
    ):
        rec = generate_recommendation(
            X_unbanked.iloc[i],
            shap_row,
            X_unbanked.columns.tolist()
        )
        primary = rec["recommended_actions"][0] if rec["recommended_actions"] else {}
        all_recs.append({
            "idx"            : df_idx,
            "primary_barrier": primary.get("barrier_feature", "unknown"),
            "primary_action" : primary.get("action", "General program"),
            "sdg_alignment"  : primary.get("sdg_alignment", "SDG 1")
        })

    return pd.DataFrame(all_recs)


def generate_country_scorecard(
    df: pd.DataFrame,
    predictions: np.ndarray,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    country_col: str = "country"
) -> pd.DataFrame:
    """
    Country-level policy scorecard.
    Shows inclusion rate, top barriers, and recommendations per country.

    NOTE: Pass the original df (with country column), not engineered df.
    """
    results = []

    # Map country dummies back to names if needed
    # This function expects original df with 'country' column

    for country in df[country_col].unique():
        mask = df[country_col] == country
        country_preds = predictions[mask]
        country_shap  = shap_values[mask]
        country_X     = X[mask.values]

        inclusion_rate = country_preds.mean()
        n_total        = len(country_preds)
        n_banked       = country_preds.sum()

        # Top 3 barriers (mean SHAP of unbanked subgroup)
        unbanked_mask = country_preds == 0
        if unbanked_mask.sum() > 0:
            unbanked_shap = country_shap[unbanked_mask]
            mean_shap = pd.Series(
                unbanked_shap.mean(axis=0),
                index=country_X.columns
            )
            top_barriers = mean_shap.nsmallest(3).index.tolist()
        else:
            top_barriers = ["N/A"]

        results.append({
            "country"       : country,
            "total_surveyed": n_total,
            "predicted_banked": int(n_banked),
            "inclusion_rate": f"{inclusion_rate*100:.1f}%",
            "top_barrier_1" : top_barriers[0] if len(top_barriers) > 0 else "N/A",
            "top_barrier_2" : top_barriers[1] if len(top_barriers) > 1 else "N/A",
            "top_barrier_3" : top_barriers[2] if len(top_barriers) > 2 else "N/A",
        })

    return pd.DataFrame(results).sort_values(
        "inclusion_rate", ascending=True
    ).reset_index(drop=True)


def plot_intervention_simulator(
    baseline_rates: dict,
    intervention_impact: float = 0.08,
    save: bool = True
):
    """
    'What If' Intervention Simulator.
    Shows how much inclusion rate improves if mobile access
    is given to all rural unbanked people.

    baseline_rates: dict of {country: current_inclusion_rate}
    intervention_impact: estimated % uplift from mobile access
    """
    countries = list(baseline_rates.keys())
    baseline  = [baseline_rates[c] for c in countries]
    improved  = [min(b + intervention_impact, 1.0) for b in baseline]
    uplift    = [i - b for i, b in zip(improved, baseline)]

    x = np.arange(len(countries))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, [b*100 for b in baseline],
                   width, label="Current Inclusion Rate",
                   color="#E74C3C", edgecolor="white")
    bars2 = ax.bar(x + width/2, [i*100 for i in improved],
                   width, label="After Mobile Access Intervention",
                   color="#2ECC71", edgecolor="white")

    # Uplift annotations
    for i, (b, imp) in enumerate(zip(baseline, improved)):
        ax.annotate(f"+{(imp-b)*100:.1f}pp",
                    xy=(x[i] + width/2, imp*100 + 0.5),
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#27AE60")

    ax.set_xlabel("Country", fontsize=12)
    ax.set_ylabel("Financial Inclusion Rate (%)", fontsize=12)
    ax.set_title(
        "Intervention Simulator: Impact of Universal Mobile Access\n"
        "on Bank Account Ownership (East Africa)",
        fontsize=13, pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=12)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max([i*100 for i in improved]) + 10)

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUTS_DIR, "intervention_simulator.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: {path}")
    plt.close()
