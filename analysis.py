"""
analysis.py  —  Jigsaw Identification: Proof-of-Concept
========================================================
Paper: "Re-identification and the Death of Anonymity"

What this script does:
  1. Loads Census PUMS microdata (your "anonymized" dataset)
  2. Computes baseline k-anonymity across quasi-identifiers
  3. Joins in a salary survey as auxiliary data (simulating a linkage attack)
  4. Shows how uniqueness (k=1 records) increases after the join
  5. Saves Figure 1: k-anonymity degradation bar chart

Usage:
  python analysis.py

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
    """Return (n_records, n_unique, pct_unique) for a set of quasi-identifiers."""
    grp = df.groupby(qi_cols, observed=True).size().reset_index(name="grp_size")
    merged = df.merge(grp, on=qi_cols)
    n_unique = (merged["grp_size"] == 1).sum()
    pct = n_unique / len(merged) * 100
    return len(merged), n_unique, pct

# ── Step 1: Load Census PUMS ───────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading Census PUMS data")
print("=" * 60)

try:
    # Read all columns first so we can detect the right names
    pums_raw = pd.read_csv(PUMS_FILE, low_memory=False, nrows=0)
except FileNotFoundError:
    sys.exit(
        f"\n[ERROR] Could not find '{PUMS_FILE}'.\n"
        "Download the ACS 1-Year PUMS Person CSV from:\n"
        "  https://census.gov/programs-surveys/acs/microdata/access.html\n"
        "and save it in the same folder as this script."
    )

# Normalise to uppercase for matching
all_cols = {c.upper(): c for c in pums_raw.columns}
print(f"  Columns detected (sample): {list(all_cols.keys())[:10]} …")

# Map the logical names we need → actual column names in the file
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
    print(f"\n[ERROR] Could not find these required columns: {missing}")
    print(f"  All columns in your file: {list(pums_raw.columns)}")
    sys.exit("  Please check you downloaded the PERSON file (not the housing file).")

use_cols = [c for c in [col_age, col_sex, col_occ, col_puma, col_wage] if c]
pums = pd.read_csv(PUMS_FILE, usecols=use_cols, low_memory=False)

# Rename to standard names the rest of the script uses
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
pums["WAGE_BRACKET"] = pd.cut(pums["WAGP"].fillna(0) if "WAGP" in pums.columns else pd.Series([0]*len(pums)),
                               bins=WAGE_BINS, labels=WAGE_LABELS)

print(f"  Records loaded : {len(pums):,}")
print(f"  Columns mapped : AGE={col_age}, SEX={col_sex}, OCC={col_occ}, PUMA={col_puma}, WAGE={col_wage}")

# ── Step 2: Baseline k-anonymity (PUMS alone) ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Baseline k-anonymity (PUMS alone)")
print("=" * 60)

QI_BASELINE = ["AGE_BRACKET", "SEX", "OCCP", "PUMA"]
n_base, n_unique_base, pct_base = uniqueness_stats(pums, QI_BASELINE)

print(f"  Quasi-identifiers : {QI_BASELINE}")
print(f"  Total records     : {n_base:,}")
print(f"  Unique (k=1)      : {n_unique_base:,}")
print(f"  Uniqueness rate   : {pct_base:.1f}%")

# ── Step 3: Load salary auxiliary dataset ─────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Loading salary auxiliary dataset")
print("=" * 60)

try:
    sal = pd.read_csv(SALARY_FILE)
except FileNotFoundError:
    sys.exit(
        f"\n[ERROR] Could not find '{SALARY_FILE}'.\n"
        "Download 'Data Science Salaries' from Kaggle:\n"
        "  https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries\n"
        "and save it as 'salaries.csv' in the same folder."
    )

print(f"  Columns found: {list(sal.columns)}")

# Detect the salary column (common names across Kaggle variants)
salary_col = next(
    (c for c in sal.columns if c.lower() in ("salary_in_usd", "salary", "total_pay")),
    None,
)
if salary_col is None:
    print("\n  [WARNING] Could not auto-detect salary column.")
    print("  Available columns:", list(sal.columns))
    sys.exit("  Set 'salary_col' manually in the script and re-run.")

sal["WAGE_BRACKET"] = pd.cut(
    sal[salary_col].fillna(0),
    bins=WAGE_BINS,
    labels=WAGE_LABELS,
)
print(f"  Salary column used : '{salary_col}'")
print(f"  Records loaded     : {len(sal):,}")

# ── Step 4: Linkage attack — join on WAGE_BRACKET ─────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Performing auxiliary join (linkage attack)")
print("=" * 60)

# Aggregate auxiliary data to wage bracket level
sal_agg = (
    sal.groupby("WAGE_BRACKET", observed=True)
    .size()
    .reset_index(name="sal_count")
)

pums_joined = pums.merge(sal_agg, on="WAGE_BRACKET", how="inner")
print(f"  PUMS records after join : {len(pums_joined):,}")

# ── Step 5: Post-join k-anonymity ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Post-join k-anonymity")
print("=" * 60)

QI_JOINED = ["AGE_BRACKET", "SEX", "OCCP", "PUMA", "WAGE_BRACKET"]
n_join, n_unique_join, pct_join = uniqueness_stats(pums_joined, QI_JOINED)

print(f"  Quasi-identifiers : {QI_JOINED}")
print(f"  Total records     : {n_join:,}")
print(f"  Unique (k=1)      : {n_unique_join:,}")
print(f"  Uniqueness rate   : {pct_join:.1f}%")

delta = pct_join - pct_base
print(f"\n  >>> Uniqueness increased by +{delta:.1f} percentage points after auxiliary join")

# ── Step 6: Save Figure 1 ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Generating Figure 1")
print("=" * 60)

labels = ["PUMS alone\n(baseline)", "After auxiliary\ndata join"]
values = [pct_base, pct_join]
colors = ["#4472C4", "#C0504D"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)

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
    fontsize=11, color="#C0504D", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#C0504D"),
)

ax.set_ylabel("Fraction of Records Unique (k=1)", fontsize=12)
ax.set_title(
    "Figure 1: K-Anonymity Degradation After Auxiliary Join\n"
    "Jigsaw Identification — Proof of Concept",
    fontsize=12,
)
ax.set_ylim(0, max(values) * 1.25)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.grid(axis="y", linestyle="--", alpha=0.5)

legend_patches = [
    mpatches.Patch(color=colors[0], label="Baseline (PUMS quasi-identifiers only)"),
    mpatches.Patch(color=colors[1], label="After auxiliary salary join"),
]
ax.legend(handles=legend_patches, loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(FIGURE_OUT, dpi=300)
plt.show()
print(f"  Saved → {FIGURE_OUT}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY — Numbers to insert into your paper")
print("=" * 60)
print(f"  Baseline uniqueness rate   : {pct_base:.1f}%  (Section 5, PUMS-alone result)")
print(f"  Post-join uniqueness rate  : {pct_join:.1f}%  (Section 5, post-join result)")
print(f"  Increase                   : +{delta:.1f} pp  (headline finding)")
print("=" * 60)
