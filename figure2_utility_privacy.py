"""
figure2_utility_privacy.py  —  Utility vs. Privacy Trade-off
=============================================================
Paper: "Re-identification and the Death of Anonymity"

What this script does:
  Generates Figure 2: a curve showing how utility (data usefulness) drops
  as privacy protection (k in k-anonymity, or ε in differential privacy)
  increases. Uses the PUMS data to compute real utility loss at different
  k thresholds, then overlays a theoretical differential privacy curve.

Usage:
  python figure2_utility_privacy.py

Expected files:
  - psam_p06.csv  (same Census PUMS file used in analysis.py)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PUMS_FILE  = "psam_p06.csv"
FIGURE_OUT = "figure2_utility_privacy.png"

AGE_BINS   = [0, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ["<25", "25-35", "35-45", "45-55", "55-65", "65+"]

# ── Load PUMS ─────────────────────────────────────────────────────────────────
print("Loading PUMS data …")
try:
    pums = pd.read_csv(
        PUMS_FILE,
        usecols=["AGEP", "SEX", "OCCP", "PUMA", "WAGP"],
        low_memory=False,
    )
except FileNotFoundError:
    sys.exit(f"[ERROR] '{PUMS_FILE}' not found. Run analysis.py first to confirm the file is in place.")

pums = pums.dropna(subset=["AGEP", "SEX", "OCCP", "PUMA"])
pums["AGE_BRACKET"] = pd.cut(pums["AGEP"], bins=AGE_BINS, labels=AGE_LABELS)
QI = ["AGE_BRACKET", "SEX", "OCCP", "PUMA"]

total_wage = pums["WAGP"].fillna(0).sum()  # baseline for utility measurement

# ── Compute utility loss at each k threshold ───────────────────────────────────
# "Utility" = fraction of wage data retained after suppressing groups with size < k
# (Generalisation/suppression is the standard k-anonymity enforcement mechanism)
print("Computing utility curve across k values …")

k_values   = list(range(1, 31))
utility_k  = []
privacy_k  = []   # re-identification risk = 1 / k (for the worst case)

grp = pums.groupby(QI, observed=True)["WAGP"].agg(["sum", "count"]).reset_index()
grp.columns = list(grp.columns[:-2]) + ["wage_sum", "count"]

for k in k_values:
    kept_wage = grp.loc[grp["count"] >= k, "wage_sum"].sum()
    util = kept_wage / total_wage * 100 if total_wage > 0 else 0
    utility_k.append(util)
    privacy_k.append(1 / k * 100)  # worst-case re-id risk as %

# ── Differential privacy curve (theoretical) ──────────────────────────────────
# As ε decreases, privacy improves but utility (query accuracy) degrades.
# We model utility loss ∝ σ² = Δf²/ε² (Laplace mechanism, global sensitivity Δf=1).
# Normalise so utility=100% at ε=10 (loose privacy) for comparison.
print("Computing differential privacy curve …")

epsilon_vals  = np.linspace(0.1, 10, 200)
dp_noise_var  = 1 / epsilon_vals ** 2          # proportional to noise variance
dp_utility    = 100 * (1 - dp_noise_var / dp_noise_var.max())  # normalised utility
dp_privacy    = (1 - epsilon_vals / epsilon_vals.max()) * 100  # higher privacy at low ε

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(
    "Figure 2: Utility–Privacy Trade-off\n"
    "K-Anonymity (empirical) vs. Differential Privacy (theoretical)",
    fontsize=13, fontweight="bold", y=0.98,
)
fig.subplots_adjust(top=0.88)

# — Left panel: k-anonymity (empirical) —
ax = axes[0]
ax.plot(k_values, utility_k,  color="#4472C4", lw=2.5, marker="o", ms=5, label="Data utility retained")
ax.plot(k_values, privacy_k,  color="#C0504D", lw=2.5, marker="s", ms=5, linestyle="--", label="Max re-id risk (1/k)")

ax.fill_between(k_values, utility_k, privacy_k, alpha=0.08, color="grey")

ax.set_xlabel("k  (minimum group size required)", fontsize=11)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.set_title("A.  K-Anonymity  (Census PUMS empirical)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.35, linestyle="--")
ax.set_xlim(1, 30)
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

# Annotate the sweet-spot knee (k=5 is common default)
k_ann = 5
ax.annotate(
    f"k={k_ann}\nutility={utility_k[k_ann-1]:.0f}%\nrisk={privacy_k[k_ann-1]:.0f}%",
    xy=(k_ann, utility_k[k_ann - 1]),
    xytext=(k_ann + 4, utility_k[k_ann - 1] - 15),
    fontsize=8.5, color="#555",
    arrowprops=dict(arrowstyle="->", color="#555", lw=1),
)

# — Right panel: differential privacy (theoretical) —
ax2 = axes[1]
ax2.plot(epsilon_vals, dp_utility,  color="#4472C4", lw=2.5, label="Utility (query accuracy)")
ax2.plot(epsilon_vals, dp_privacy,  color="#C0504D", lw=2.5, linestyle="--", label="Privacy strength (1 − ε/ε_max)")

ax2.fill_between(epsilon_vals, dp_utility, dp_privacy, alpha=0.08, color="grey")

ax2.set_xlabel("ε  (privacy budget — lower = stronger privacy)", fontsize=11)
ax2.set_ylabel("Normalised score (%)", fontsize=11)
ax2.set_title("B.  Differential Privacy  (theoretical, Laplace mechanism)", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.35, linestyle="--")
ax2.set_xlim(0.1, 10)
ax2.set_ylim(-5, 105)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

# Annotate a practical ε recommendation
ax2.axvline(x=1, color="green", lw=1.2, linestyle=":", alpha=0.7)
ax2.text(1.1, 5, "ε=1\n(practical\nrecommendation)", fontsize=8, color="green")

plt.savefig(FIGURE_OUT, dpi=300)
plt.show()
print(f"Saved → {FIGURE_OUT}")
