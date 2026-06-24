"""
analysis.py  —  Jigsaw Identification: Proof-of-Concept

Expected files in the same folder:
  - psam_p06.csv   (Census PUMS Person data, any state)
  - salaries.csv   (Kaggle "Data Science Salaries" dataset)
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration ─────────────────────────────────────────────────────────────
PUMS_FILE    = "psam_p06.csv"
SALARY_FILE  = "salaries.csv"
FIGURE_OUT   = "figure1_kanonymity.png"

AGE_BINS   = [0, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ["<25", "25-35", "35-45", "45-55", "55-65", "65+"]

WAGE_BINS   = [0, 30_000, 60_000, 90_000, 120_000, 10_000_000]
WAGE_LABELS = ["<30k", "30-60k", "60-90k", "90-120k", "120k+"]

# ── Helper ─────────────────────────────────────────────────────────────────────
def uniqueness_stats(df, qi_cols):
    grp = df.groupby(qi_cols, observed=True).size().reset_index(name="grp_size")
    merged = df.merge(grp, on=qi_cols)
    n_unique = (merged["grp_size"] == 1).sum()
    pct = n_unique / len(merged) * 100
    return len(merged), n_unique, pct

# ── Step 1: Load Census PUMS ───────────────────────────────────────────────────
try:
    pums_raw = pd.read_csv(PUMS_FILE, low_memory=False, nrows=0)
except FileNotFoundError:
    sys.exit(
        f"\n[ERROR] Could not find '{PUMS_FILE}'.\n"
        "Download the ACS 1-Year PUMS Person CSV from:\n"
        "  https://census.gov/programs-surveys/acs/microdata/access.html\n"
        "and save it in the same folder as this script."
    )

all_cols = {c.upper(): c for c in pums_raw.columns}

def find_col(candidates):
    for c in candidates:
        if c.upper() in all_cols:
            return all_cols[c.upper()]
    return None

col_age  = find_col(["AGEP", "AGE", "age"])
col_sex  = find_col(["SEX", "sex", "GENDER"])
col_occ  = find_col(["OCCP", "OCC", "SOCP", "occ"])
col_puma = find_col(["PUMA", "PUMA00", "puma"])
col_wage = find_col(["WAGP", "WAGE", "PINCP", "wage"])

missing = [name for name, col in
           [("AGE(AGEP)", col_age), ("SEX", col_sex),
            ("OCCUPATION(OCCP)", col_occ), ("PUMA", col_puma)]
           if col is None]
if missing:
    sys.exit(f"[ERROR] Could not find required columns: {missing}")

use_cols = [c for c in [col_age, col_sex, col_occ, col_puma, col_wage] if c]
pums = pd.read_csv(PUMS_FILE, usecols=use_cols, low_memory=False)

rename = {}
if col_age  != "AGEP":  rename[col_age]  = "AGEP"
if col_sex  != "SEX":   rename[col_sex]  = "SEX"
if col_occ  != "OCCP":  rename[col_occ]  = "OCCP"
if col_puma != "PUMA":  rename[col_puma] = "PUMA"
if col_wage and col_wage != "WAGP": rename[col_wage] = "WAGP"
if rename:
    pums = pums.rename(columns=rename)

pums = pums.dropna(subset=["AGEP", "SEX", "OCCP", "PUMA"])
pums["AGE_BRACKET"]  = pd.cut(pums["AGEP"], bins=AGE_BINS, labels=AGE_LABELS)
pums["WAGE_BRACKET"] = pd.cut(
    pums["WAGP"].fillna(0) if "WAGP" in pums.columns else pd.Series([0]*len(pums)),
    bins=WAGE_BINS, labels=WAGE_LABELS
)

# ── Step 2: Baseline k-anonymity ───────────────────────────────────────────────
QI_BASELINE = ["AGE_BRACKET", "SEX", "OCCP", "PUMA"]
n_base, n_unique_base, pct_base = uniqueness_stats(pums, QI_BASELINE)

# ── Step 3: Load salary auxiliary dataset ─────────────────────────────────────
try:
    sal = pd.read_csv(SALARY_FILE)
except FileNotFoundError:
    sys.exit(
        f"\n[ERROR] Could not find '{SALARY_FILE}'.\n"
        "Download 'Data Science Salaries' from Kaggle:\n"
        "  https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries\n"
        "and save it as 'salaries.csv' in the same folder."
    )

salary_col = next(
    (c for c in sal.columns if c.lower() in ("salary_in_usd", "salary", "total_pay")),
    None,
)
if salary_col is None:
    sys.exit("Could not auto-detect salary column. Set 'salary_col' manually.")

sal["WAGE_BRACKET"] = pd.cut(sal[salary_col].fillna(0), bins=WAGE_BINS, labels=WAGE_LABELS)

# ── Step 4: Auxiliary join ─────────────────────────────────────────────────────
sal_agg = (
    sal.groupby("WAGE_BRACKET", observed=True)
    .size()
    .reset_index(name="sal_count")
)
pums_joined = pums.merge(sal_agg, on="WAGE_BRACKET", how="inner")

# ── Step 5: Post-join k-anonymity ─────────────────────────────────────────────
QI_JOINED = ["AGE_BRACKET", "SEX", "OCCP", "PUMA", "WAGE_BRACKET"]
n_join, n_unique_join, pct_join = uniqueness_stats(pums_joined, QI_JOINED)
delta = pct_join - pct_base

print(f"Baseline uniqueness rate : {pct_base:.1f}%")
print(f"Post-join uniqueness rate: {pct_join:.1f}%")
print(f"Increase                 : +{delta:.1f} pp")

# ── Step 6: Figure 1 (black and white) ────────────────────────────────────────
hatches = ["///", ""]
greys   = ["#666666", "#cccccc"]

fig, ax = plt.subplots(figsize=(7, 5))
bar_labels = ["PUMS alone\n(baseline)", "After auxiliary\ndata join"]
values = [pct_base, pct_join]

bars = ax.bar(bar_labels, values, color=greys, hatch=hatches,
              width=0.5, edgecolor="black", linewidth=0.8)

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}%",
        ha="center", va="bottom", fontsize=13, fontweight="bold",
    )

ax.annotate(
    f"+{delta:.1f} pp",
    xy=(1, pct_join), xytext=(1.25, (pct_base + pct_join) / 2),
    fontsize=11, color="black", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="black"),
)

ax.set_ylabel("Fraction of Records Unique (k=1)", fontsize=12)
ax.set_ylim(0, max(values) * 1.25)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.grid(axis="y", linestyle="--", alpha=0.5, color="black")

legend_patches = [
    mpatches.Patch(facecolor=greys[0], hatch=hatches[0], edgecolor="black",
                   label="Baseline (PUMS quasi-identifiers only)"),
    mpatches.Patch(facecolor=greys[1], hatch=hatches[1], edgecolor="black",
                   label="After auxiliary salary join"),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(FIGURE_OUT, dpi=300)
print(f"Saved → {FIGURE_OUT}"), PUMS-alone result)")
print(f"  Post-join uniqueness rate  : {pct_join:.1f}%  (Section 5, post-join result)")
print(f"  Increase                   : +{delta:.1f} pp  (headline finding)")
print("=" * 60)
