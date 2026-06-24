"""
figure2_utility_privacy.py  —  Utility vs. Privacy Trade-off
 
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
try:
    pums = pd.read_csv(PUMS_FILE, usecols=["AGEP", "SEX", "OCCP", "PUMA", "WAGP"], low_memory=False)
except FileNotFoundError:
    sys.exit(f"[ERROR] '{PUMS_FILE}' not found.")
 
pums = pums.dropna(subset=["AGEP", "SEX", "OCCP", "PUMA"])
pums["AGE_BRACKET"] = pd.cut(pums["AGEP"], bins=AGE_BINS, labels=AGE_LABELS)
QI = ["AGE_BRACKET", "SEX", "OCCP", "PUMA"]
 
total_wage = pums["WAGP"].fillna(0).sum()
 
# ── Utility curve across k thresholds ─────────────────────────────────────────
k_values  = list(range(1, 31))
utility_k = []
privacy_k = []
 
grp = pums.groupby(QI, observed=True)["WAGP"].agg(["sum", "count"]).reset_index()
grp.columns = list(grp.columns[:-2]) + ["wage_sum", "count"]
 
for k in k_values:
    kept_wage = grp.loc[grp["count"] >= k, "wage_sum"].sum()
    util = kept_wage / total_wage * 100 if total_wage > 0 else 0
    utility_k.append(util)
    privacy_k.append(1 / k * 100)
 
# ── Differential privacy curve (theoretical) ──────────────────────────────────
epsilon_vals = np.linspace(0.1, 10, 200)
dp_noise_var = 1 / epsilon_vals ** 2
dp_utility   = 100 * (1 - dp_noise_var / dp_noise_var.max())
dp_privacy   = (1 - epsilon_vals / epsilon_vals.max()) * 100
 
# ── Plot (black and white) ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
 
# Left panel: k-anonymity
ax = axes[0]
ax.plot(k_values, utility_k, color="black",  lw=2.5, marker="o", ms=5,
        label="Data utility retained")
ax.plot(k_values, privacy_k, color="black",  lw=2.5, marker="s", ms=5,
        linestyle="--", label="Max re-id risk (1/k)")
ax.fill_between(k_values, utility_k, privacy_k, alpha=0.12, color="black")
ax.set_xlabel("k  (minimum group size required)", fontsize=11)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.set_title("A.  K-Anonymity  (Census PUMS empirical)", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.35, linestyle="--", color="black")
ax.set_xlim(1, 30)
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
ax.annotate(
    f"k=5\nutility={utility_k[4]:.0f}%\nrisk={privacy_k[4]:.0f}%",
    xy=(5, utility_k[4]),
    xytext=(9, utility_k[4] - 15),
    fontsize=8.5, color="#333",
    arrowprops=dict(arrowstyle="->", color="#333", lw=1),
)
 
# Right panel: differential privacy
ax2 = axes[1]
ax2.plot(epsilon_vals, dp_utility, color="black", lw=2.5,
         label="Utility (query accuracy)")
ax2.plot(epsilon_vals, dp_privacy, color="black", lw=2.5, linestyle="--",
         label="Privacy strength (1 \u2212 \u03b5/\u03b5_max)")
ax2.fill_between(epsilon_vals, dp_utility, dp_privacy, alpha=0.12, color="black")
ax2.set_xlabel("\u03b5  (privacy budget \u2014 lower = stronger privacy)", fontsize=11)
ax2.set_ylabel("Normalised score (%)", fontsize=11)
ax2.set_title("B.  Differential Privacy  (theoretical, Laplace mechanism)", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.35, linestyle="--", color="black")
ax2.set_xlim(0.1, 10)
ax2.set_ylim(-5, 105)
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
ax2.axvline(x=1, color="black", lw=1.2, linestyle=":", alpha=0.7)
ax2.text(1.1, 5, "\u03b5=1\n(practical\nrecommendation)", fontsize=8, color="black")
 
plt.tight_layout()
plt.savefig(FIGURE_OUT, dpi=300)
print(f"Saved \u2192 {FIGURE_OUT}")

