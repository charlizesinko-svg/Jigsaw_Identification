"""
figure3_timeline.py  —  Timeline of Landmark Re-identification Breaches
========================================================================
Paper: "Re-identification and the Death of Anonymity"

What this script does:
  Generates Figure 3: a horizontal timeline of the landmark re-identification
  events cited in the paper (Sections 3, 5). No external data file needed —
  all events are hardcoded from the literature.

Usage:
  python figure3_timeline.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIGURE_OUT = "figure3_timeline.png"

# ── Event data ────────────────────────────────────────────────────────────────
# (year, short_label, detail, category)
# Categories: foundational | commercial | government | social_media
EVENTS = [
    (1997, "Sweeney\nVoter+Medical", "87% of US pop. uniquely\nidentifiable by ZIP+DOB+sex", "foundational"),
    (2000, "Sweeney\nWeld Re-id",   "Gov. Weld's medical record\nlinked from 'anonymous' data", "foundational"),
    (2006, "AOL Search\nLogs",       "657k users' raw queries\npublicly released; re-id'd", "commercial"),
    (2008, "Netflix Prize\nAttack",  "Narayanan & Shmatikov:\nIMDb + sparse ratings", "commercial"),
    (2009, "Twitter+Flickr\nDe-anon","Social graph topology\nde-anonymises 70–80%", "social_media"),
    (2013, "4 GPS Points",           "De Montjoye: 4 points\nuniquely identify 95%", "government"),
    (2014, "NYC Taxi\nData",         "License plates re-id'd\nfrom 'blurred' taxi logs", "commercial"),
    (2018, "Cambridge\nAnalytica",   "87M Facebook profiles\nused for political profiling", "social_media"),
    (2019, "Location Data\nBrokers", "NYT: precise GPS sold\nby brokers; officers re-id'd", "commercial"),
    (2023, "LLM-Assisted\nRe-id",    "Semantic matching of\npseudonymous posts via GPT", "social_media"),
]

COLORS = {
    "foundational": "#4472C4",
    "commercial":   "#C0504D",
    "government":   "#70AD47",
    "social_media": "#ED7D31",
}
LABELS = {
    "foundational": "Foundational research",
    "commercial":   "Commercial data breach",
    "government":   "Government / open data",
    "social_media": "Social media / modern",
}

years  = [e[0] for e in EVENTS]
labels = [e[1] for e in EVENTS]
details= [e[2] for e in EVENTS]
cats   = [e[3] for e in EVENTS]

# ── Layout: alternate above/below the axis ─────────────────────────────────────
n = len(EVENTS)
y_pos  = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
y_text = [1.6 if i % 2 == 0 else -1.7  for i in range(n)]

fig, ax = plt.subplots(figsize=(15, 6))
ax.set_xlim(1994, 2026)
ax.set_ylim(-3.5, 3.5)
ax.axis("off")

# Horizontal baseline
ax.axhline(0, color="#333", lw=1.5, zorder=1)

# Decade tick marks
for yr in range(1995, 2026, 5):
    ax.plot([yr, yr], [-0.15, 0.15], color="#333", lw=1, zorder=2)
    ax.text(yr, -0.35, str(yr), ha="center", va="top", fontsize=8.5, color="#555")

# Events
for i, (yr, lbl, det, cat) in enumerate(EVENTS):
    color = COLORS[cat]
    yp    = y_pos[i]
    yt    = y_text[i]

    # Stem line
    ax.plot([yr, yr], [0, yp * 0.85], color=color, lw=1.8, zorder=3)

    # Dot
    ax.scatter([yr], [yp * 0.85], color=color, s=80, zorder=4, edgecolors="white", linewidths=1)

    # Year label on stem
    ax.text(yr, yp * 0.45, str(yr), ha="center", va="center",
            fontsize=7.5, color=color, fontweight="bold")

    # Event box
    box_y = yt
    ax.text(yr, box_y, lbl, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none", alpha=0.92))

    # Detail text
    sign = 1 if yp > 0 else -1
    ax.text(yr, box_y + sign * 0.7, det, ha="center", va="center",
            fontsize=7.2, color="#333", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.7, lw=0.8))

# Legend
patches = [mpatches.Patch(color=v, label=LABELS[k]) for k, v in COLORS.items()]
ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.02),
          ncol=4, fontsize=9, framealpha=0.9)

ax.set_title(
    "Figure 3: Timeline of Landmark Re-identification Events (1997–2023)\n"
    "\"Re-identification and the Death of Anonymity\"",
    fontsize=12, fontweight="bold", pad=12,
)

plt.tight_layout()
plt.savefig(FIGURE_OUT, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved → {FIGURE_OUT}")
