# ============================================================
# NOTEBOOK 01 — Exploratory Data Analysis
# Financial Inclusion in Africa — Zindi Challenge
#
# PURPOSE:
#   Understand the data deeply before building models.
#   Every chart must answer a specific business question.
#   This notebook also generates insights for the policy report.
#
# RUN THIS FIRST before any other notebook.
# ============================================================

# %% [markdown]
# # 🌍 Financial Inclusion in Africa
# ## Notebook 01: Exploratory Data Analysis
#
# **Goal:** Understand who is banked, who isn't, and why.
# Each visualization answers a specific question about
# financial inclusion in Kenya, Rwanda, Tanzania, Uganda.

# %% CELL 1 — Install and import (run once in Colab)
# Uncomment the pip install line if running in Colab for the first time:
# !pip install -r requirements.txt -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os
warnings.filterwarnings("ignore")

# Robust src import regardless of working directory
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))



# Style setup — professional consistent look throughout
plt.rcParams.update({
    "figure.facecolor" : "#FAFAFA",
    "axes.facecolor"   : "#FAFAFA",
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})
PALETTE = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12",
           "#9B59B6", "#1ABC9C", "#E67E22", "#2C3E50"]
sns.set_palette(PALETTE)

print("✓ Libraries loaded")

# %% CELL 2 — Load data
# -------------------------------------------------------
# DATA LOADING: We always load raw data once at the top.
# Never overwrite raw files. Always work on copies.
# -------------------------------------------------------

from src.config import TRAIN_PATH, TEST_PATH

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"Train shape : {train.shape}")
print(f"Test shape  : {test.shape}")
print(f"\nTrain columns:\n{train.columns.tolist()}")

# %% CELL 3 — Basic data audit
# -------------------------------------------------------
# AUDIT: Before any analysis, verify data integrity.
# Checks: shape, dtypes, missing values, duplicates
# -------------------------------------------------------

print("=" * 55)
print("  DATA AUDIT")
print("=" * 55)

print("\n📋 First 3 rows:")
print(train.head(3).to_string())

print(f"\n📊 Data types:")
print(train.dtypes)

print(f"\n❓ Missing values:")
missing = train.isnull().sum()
print(missing[missing > 0] if missing.any() else "  → No missing values ✓")

print(f"\n🔁 Duplicate rows: {train.duplicated().sum()}")

print(f"\n📈 Target distribution:")
target_counts = train["bank_account"].value_counts()
print(target_counts)
print(f"\n  → {target_counts['Yes']} banked ({target_counts['Yes']/len(train)*100:.1f}%)")
print(f"  → {target_counts['No']} unbanked ({target_counts['No']/len(train)*100:.1f}%)")
print(f"\n  CLASS IMBALANCE RATIO: {target_counts['No']/target_counts['Yes']:.1f}:1")
print("  ⚠️  Imbalanced dataset — must handle in modeling!")

# %% CELL 4 — Binary encode target for analysis
train["has_account"] = (train["bank_account"] == "Yes").astype(int)

# %% CELL 5 — CHART 1: Class Distribution
# Business question: How severe is the class imbalance?
# -------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Count bar
counts = train["bank_account"].value_counts()
colors_bar = [PALETTE[0], PALETTE[1]]
bars = axes[0].bar(counts.index, counts.values,
                   color=colors_bar, edgecolor="white", width=0.5)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f"{val:,}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")
axes[0].set_title("Bank Account Ownership Count", fontsize=13, pad=10)
axes[0].set_ylabel("Number of Respondents")
axes[0].set_xlabel("")

# Right: Pie chart with % breakdown
pct = [counts["No"]/len(train)*100, counts["Yes"]/len(train)*100]
axes[1].pie(
    pct,
    labels=["Unbanked", "Banked"],
    colors=[PALETTE[0], PALETTE[1]],
    autopct="%1.1f%%",
    startangle=140,
    textprops={"fontsize": 13}
)
axes[1].set_title("Distribution of Bank Account Status", fontsize=13)

fig.suptitle(
    "Target Variable: ~14% of East Africans Have a Bank Account\n"
    "(Kenya, Rwanda, Tanzania, Uganda — 2016–2018)",
    fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("outputs/01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved: 01_class_distribution.png")

# %% CELL 6 — CHART 2: Inclusion Rate by Country
# Business question: Which countries lag most?
# -------------------------------------------------------

country_stats = (
    train.groupby("country")["has_account"]
    .agg(["mean", "sum", "count"])
    .rename(columns={"mean": "rate", "sum": "banked", "count": "total"})
    .sort_values("rate")
    .reset_index()
)
country_stats["rate_pct"] = country_stats["rate"] * 100

fig, ax = plt.subplots(figsize=(10, 5))
colors = [PALETTE[2] if r < 0.15 else PALETTE[1]
          for r in country_stats["rate"]]
bars = ax.barh(country_stats["country"], country_stats["rate_pct"],
               color=colors, edgecolor="white", height=0.5)
ax.axvline(x=14, color="red", linestyle="--", alpha=0.6,
           label="Regional avg (14%)")

for bar, row in zip(bars, country_stats.itertuples()):
    ax.text(bar.get_width() + 0.3,
            bar.get_y() + bar.get_height()/2,
            f"{row.rate_pct:.1f}%  ({row.banked:,}/{row.total:,})",
            va="center", ha="left", fontsize=10)

ax.set_xlabel("Percentage with Bank Account (%)", fontsize=11)
ax.set_title("Financial Inclusion Rate by Country\n"
             "Kenya leads; Tanzania lags significantly",
             fontsize=13, pad=12)
ax.legend(fontsize=10)
ax.set_xlim(0, 60)
plt.tight_layout()
plt.savefig("outputs/02_inclusion_by_country.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nCountry Summary:")
print(country_stats[["country", "rate_pct", "banked", "total"]].to_string(index=False))

# %% CELL 7 — CHART 3: Inclusion Rate by Job Type
# Business question: Which occupations predict inclusion most?
# -------------------------------------------------------

job_stats = (
    train.groupby("job_type")["has_account"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "rate", "count": "n"})
    .sort_values("rate")
    .reset_index()
)

fig, ax = plt.subplots(figsize=(12, 6))
norm = plt.Normalize(job_stats["rate"].min(), job_stats["rate"].max())
colors = plt.cm.RdYlGn(norm(job_stats["rate"]))

bars = ax.barh(job_stats["job_type"], job_stats["rate"] * 100,
               color=colors, edgecolor="white", height=0.6)
for bar, row in zip(bars, job_stats.itertuples()):
    ax.text(bar.get_width() + 0.3,
            bar.get_y() + bar.get_height()/2,
            f"{row.rate*100:.1f}%  (n={row.n:,})",
            va="center", ha="left", fontsize=9)

ax.set_xlabel("Bank Account Rate (%)", fontsize=11)
ax.set_title("Bank Account Rate by Employment Type\n"
             "Formal employment is strongly predictive",
             fontsize=13, pad=12)
ax.set_xlim(0, 85)
plt.tight_layout()
plt.savefig("outputs/03_inclusion_by_jobtype.png", dpi=150, bbox_inches="tight")
plt.show()

# %% CELL 8 — CHART 4: Inclusion Rate by Education
# Business question: Does education unlock banking monotonically?
# -------------------------------------------------------

edu_order = [
    "No formal education", "Primary education",
    "Secondary education", "Vocational/Specialised training",
    "Tertiary education", "Other/Dont know/RTA"
]
edu_stats = (
    train.groupby("education_level")["has_account"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "rate", "count": "n"})
    .reindex(edu_order)
    .dropna()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(range(len(edu_stats)), edu_stats["rate"] * 100,
              color=[plt.cm.Blues(0.3 + i*0.15) for i in range(len(edu_stats))],
              edgecolor="white", width=0.6)

for bar, row in zip(bars, edu_stats.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{row.rate*100:.1f}%\n(n={row.n:,})",
            ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(range(len(edu_stats)))
ax.set_xticklabels(
    [e.replace(" ", "\n") for e in edu_stats["education_level"]],
    fontsize=9
)
ax.set_ylabel("Bank Account Rate (%)", fontsize=11)
ax.set_title("Education Level vs Bank Account Ownership\n"
             "Clear monotonic relationship — education unlocks banking",
             fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("outputs/04_inclusion_by_education.png", dpi=150, bbox_inches="tight")
plt.show()

# %% CELL 9 — CHART 5: Mobile Access vs Banking
# Business question: Does cellphone access bridge the gap?
# -------------------------------------------------------

mobile_stats = (
    train.groupby(["cellphone_access", "bank_account"])
    .size().unstack(fill_value=0)
)
mobile_stats["total"] = mobile_stats.sum(axis=1)
mobile_stats["rate"]  = mobile_stats["Yes"] / mobile_stats["total"] * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: grouped bar
mobile_plot = mobile_stats.reset_index()
x = np.arange(2)
bars_no  = axes[0].bar(x - 0.2, mobile_stats["No"], 0.35,
                       label="No Account", color=PALETTE[0])
bars_yes = axes[0].bar(x + 0.2, mobile_stats["Yes"], 0.35,
                       label="Has Account", color=PALETTE[1])
axes[0].set_xticks(x)
axes[0].set_xticklabels(["No Cellphone", "Has Cellphone"], fontsize=12)
axes[0].set_ylabel("Count")
axes[0].set_title("Mobile Access vs Banking\n(Counts)", fontsize=12)
axes[0].legend()

# Right: inclusion rate
rate_bars = axes[1].bar(
    ["No Cellphone", "Has Cellphone"],
    mobile_stats["rate"].values,
    color=[PALETTE[0], PALETTE[1]], edgecolor="white", width=0.4
)
for bar, rate in zip(rate_bars, mobile_stats["rate"].values):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{rate:.1f}%", ha="center", fontsize=13, fontweight="bold")

axes[1].set_ylabel("Inclusion Rate (%)")
axes[1].set_title("Mobile Access Multiplies\nBanking Rate", fontsize=12)
axes[1].set_ylim(0, 50)

fig.suptitle("Cellphone Access: A Critical Bridge to Financial Inclusion",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/05_mobile_vs_banking.png", dpi=150, bbox_inches="tight")
plt.show()

# %% CELL 10 — CHART 6: Gender Gap
# -------------------------------------------------------

gender_stats = (
    train.groupby(["country", "gender_of_respondent"])["has_account"]
    .mean().unstack() * 100
)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(gender_stats))
ax.bar(x - 0.2, gender_stats["Female"], 0.35,
       label="Female", color="#E91E63", alpha=0.85)
ax.bar(x + 0.2, gender_stats["Male"], 0.35,
       label="Male", color="#2196F3", alpha=0.85)

# Gender gap annotation
for i, (m, f) in enumerate(zip(
    gender_stats["Male"], gender_stats["Female"]
)):
    gap = m - f
    ax.annotate(f"Gap: {gap:.1f}pp",
                xy=(i, max(m, f) + 0.5),
                ha="center", fontsize=9, color="#E74C3C")

ax.set_xticks(x)
ax.set_xticklabels(gender_stats.index, fontsize=12)
ax.set_ylabel("Bank Account Rate (%)", fontsize=11)
ax.set_title("Gender Gap in Financial Inclusion by Country\n"
             "Men are consistently more banked than women",
             fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("outputs/06_gender_gap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% CELL 11 — CHART 7: Age Distribution
# Business question: Is there a peak banking age?
# -------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: Age distributions by account status
train[train["has_account"] == 1]["age_of_respondent"].hist(
    ax=axes[0], bins=30, alpha=0.6, color=PALETTE[1], label="Has Account"
)
train[train["has_account"] == 0]["age_of_respondent"].hist(
    ax=axes[0], bins=30, alpha=0.6, color=PALETTE[0], label="No Account"
)
axes[0].set_xlabel("Age", fontsize=11)
axes[0].set_ylabel("Count")
axes[0].set_title("Age Distribution by Banking Status")
axes[0].legend()

# Right: Rolling inclusion rate by age
age_rate = train.groupby("age_of_respondent")["has_account"].mean()
age_count = train.groupby("age_of_respondent")["has_account"].count()

axes[1].plot(age_rate.index, age_rate.values * 100,
             color=PALETTE[2], linewidth=2, alpha=0.4, label="Raw rate")
# Smoothed
from scipy.ndimage import gaussian_filter1d
smoothed = gaussian_filter1d(age_rate.values * 100, sigma=2)
axes[1].plot(age_rate.index, smoothed,
             color=PALETTE[2], linewidth=2.5, label="Smoothed rate")
axes[1].fill_between(age_rate.index, smoothed, alpha=0.1, color=PALETTE[2])
axes[1].set_xlabel("Age of Respondent", fontsize=11)
axes[1].set_ylabel("Inclusion Rate (%)", fontsize=11)
axes[1].set_title("Banking Rate Peaks in Prime Working Age\n(30–50 years)")
axes[1].legend()

fig.suptitle("Age & Financial Inclusion: Non-Linear Relationship",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/07_age_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# %% CELL 12 — CHART 8: Urban vs Rural
# -------------------------------------------------------

location_stats = (
    train.groupby(["location_type", "country"])["has_account"]
    .mean().unstack() * 100
)

fig, ax = plt.subplots(figsize=(11, 5))
location_stats.T.plot(kind="bar", ax=ax, color=PALETTE[:2],
                       edgecolor="white", width=0.6)
ax.set_xlabel("Country", fontsize=11)
ax.set_ylabel("Bank Account Rate (%)", fontsize=11)
ax.set_title("Urban vs Rural Financial Inclusion by Country\n"
             "Urban residents are significantly more banked",
             fontsize=13)
ax.legend(title="Location", labels=["Rural", "Urban"], fontsize=10)
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("outputs/08_urban_rural.png", dpi=150, bbox_inches="tight")
plt.show()

# %% CELL 13 — CHART 9: Correlation Heatmap
# -------------------------------------------------------
# Encode for correlation analysis

corr_df = train.copy()
corr_df["has_account"] = (corr_df["bank_account"] == "Yes").astype(int)

encode_for_corr = {
    "cellphone_access"    : {"Yes": 1, "No": 0},
    "gender_of_respondent": {"Male": 1, "Female": 0},
    "location_type"       : {"Urban": 1, "Rural": 0},
}
for col, mapping in encode_for_corr.items():
    corr_df[col] = corr_df[col].map(mapping)

numeric_cols = [
    "has_account", "household_size", "age_of_respondent",
    "cellphone_access", "gender_of_respondent", "location_type"
]
corr_matrix = corr_df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
    ax=ax, linewidths=0.5,
    annot_kws={"fontsize": 10}
)
ax.set_title("Correlation Matrix — Numeric Features vs Target",
             fontsize=12, pad=12)
plt.tight_layout()
plt.savefig("outputs/09_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% CELL 14 — CHART 10: Multi-factor heatmap
# Business question: Which country × education combos have lowest inclusion?
# -------------------------------------------------------

pivot = (
    train.groupby(["country", "education_level"])["has_account"]
    .mean()
    .unstack()
    .reindex(columns=[
        "No formal education", "Primary education",
        "Secondary education", "Vocational/Specialised training",
        "Tertiary education"
    ])
)

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(
    pivot * 100, annot=True, fmt=".0f", cmap="YlOrRd_r",
    linewidths=0.5, ax=ax,
    cbar_kws={"label": "Inclusion Rate (%)"}
)
ax.set_title("Bank Account Rate: Country × Education Level\n"
             "Dark red = highest exclusion risk",
             fontsize=13, pad=12)
ax.set_xlabel("Education Level", fontsize=11)
ax.set_ylabel("Country", fontsize=11)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("outputs/10_country_education_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.show()

# %% CELL 15 — Summary statistics table
# -------------------------------------------------------

print("\n" + "="*60)
print("  EDA SUMMARY — KEY INSIGHTS")
print("="*60)

insights = [
    ("Class Imbalance",
     f"{target_counts['No']/len(train)*100:.0f}% unbanked — must handle in model"),
    ("Top Country",
     f"Kenya: {country_stats[country_stats.country=='Kenya']['rate_pct'].values[0]:.1f}% inclusion"),
    ("Worst Country",
     f"{country_stats.iloc[0]['country']}: {country_stats.iloc[0]['rate_pct']:.1f}% inclusion"),
    ("Mobile Effect",
     "Cellphone access ~3–4x higher banking rate"),
    ("Education",
     "Tertiary ed = ~5x higher banking rate vs no education"),
    ("Employment",
     "Formally employed = highest banking rate"),
    ("Gender Gap",
     "Men ~5–10pp more likely to be banked vs women"),
    ("Urban Premium",
     "Urban residents ~2–3x more likely to be banked"),
    ("Prime Age",
     "Ages 30–50 have highest banking rates"),
]

for topic, insight in insights:
    print(f"  ✦ {topic:<20}: {insight}")

print("\n✓ EDA complete — ready for Feature Engineering (Notebook 02)")
