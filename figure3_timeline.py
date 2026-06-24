"""
figure3_timeline.py  —  Timeline of Landmark Re-identification Events
"""
 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
FIGURE_OUT = "figure3_timeline.png"
 
# (year, short_label, detail, category)
EVENTS = [
    (1997, "Sweeney\nVoter+Medical", "87% of US pop. uniquely\nidentifiable by ZIP+DOB+sex", "foundational"),
    (2000, "Sweeney\nWeld Re-id",    "Gov. Weld's medical record\nlinked from 'anonymous' data", "foundational"),
    (2006, "AOL Search\nLogs",        "657k users' raw queries\npublicly released; re-id'd", "commercial"),
    (2008, "Netflix Prize\nAttack",   "Narayanan & Shmatikov:\nIMDb + sparse ratings", "commercial"),
    (2009, "Twitter+Flickr\nDe-anon", "Social graph topology\nde-anonymises 70-80%", "social_media"),
    (2013, "4 GPS Points",            "De Montjoye: 4 points\nuniquely identify 95%", "government"),
    (2014, "NYC Taxi\nData",          "License plates re-id'd\nfrom 'blurred' taxi logs", "commercial"),
    (2018, "Cambridge\nAnalytica",    "87M Facebook profiles\nused for political profiling", "social_media"),
    (2019, "Location Data\nBrokers",  "NYT: precise GPS sold\nby brokers; officers re-id'd", "commercial"),
    (2023, "LLM-Assisted\nRe-id",     "Semantic matching of\npseudonymous posts via GPT", "social_media"),
]
 
# B&W: use greyscale fills + different hatches per category
STYLES = {
    "foundational": {"facecolor": "#1a1a1a", "hatch": "",    "textcolor": "white"},
    "commercial":   {"facecolor": "#666666", "hatch": "///", "textcolor": "white"},
    "government":   {"facecolor": "#aaaaaa", "hatch": "...", "textcolor": "black"},
    "social_media": {"facecolor": "#dddddd", "hatch": "xxx", "textcolor": "black"},
}
LABELS = {
    "foundational": "Foundational research",
    "commercial":   "Commercial data breach",
    "government":   "Government / open data",
    "social_media": "Social media / modern",
}
 
years  = [e[0] for e in EVENTS]
n = len(EVENTS)
y_pos  = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
y_text = [1.6 if i % 2 == 0 else -1.7  for i in range(n)]
 
fig, ax = plt.subplots(figsize=(15, 6))
ax.set_xlim(1994, 2026)
ax.set_ylim(-3.5, 3.5)
ax.axis("off")
 
ax.axhline(0, color="black", lw=1.5, zorder=1)
 
for yr in range(1995, 2026, 5):
    ax.plot([yr, yr], [-0.15, 0.15], color="black", lw=1, zorder=2)
    ax.text(yr, -0.35, str(yr), ha="center", va="top", fontsize=8.5, color="#333")
 
for i, (yr, lbl, det, cat) in enumerate(EVENTS):
    s  = STYLES[cat]
    yp = y_pos[i]
    yt = y_text[i]
 
    ax.plot([yr, yr], [0, yp * 0.85], color="black", lw=1.8, zorder=3)
    ax.scatter([yr], [yp * 0.85], color=s["facecolor"], s=80, zorder=4,
               edgecolors="black", linewidths=1)
    ax.text(yr, yp * 0.45, str(yr), ha="center", va="center",
            fontsize=7.5, color="black", fontweight="bold")
 
    ax.text(yr, yt, lbl, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=s["textcolor"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=s["facecolor"],
                      hatch=s["hatch"], edgecolor="black", alpha=0.92))
 
    sign = 1 if yp > 0 else -1
    ax.text(yr, yt + sign * 0.7, det, ha="center", va="center",
            fontsize=7.2, color="#333", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="black", alpha=0.7, lw=0.8))
 
patches = [
    mpatches.Patch(facecolor=STYLES[k]["facecolor"], hatch=STYLES[k]["hatch"],
                   edgecolor="black", label=LABELS[k])
    for k in STYLES
]
ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.02),
          ncol=4, fontsize=9, framealpha=0.9)
 
plt.tight_layout()
plt.savefig(FIGURE_OUT, dpi=300, bbox_inches="tight")
print(f"Saved -> {FIGURE_OUT}")
 

