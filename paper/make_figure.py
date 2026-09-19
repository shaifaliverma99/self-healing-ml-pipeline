"""Builds the paper's headline figure: detection completeness vs. cost/false-alarms
trade-off across the four drift detectors, aggregated over all three streams.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent
df = pd.read_csv(R / "results" / "detector_comparison.csv")

agg = df.groupby("detector").agg(
    detected=("detected_drifts", "sum"),
    true_total=("true_drifts", "sum"),
    false_alarms=("false_alarms", "sum"),
    mean_accuracy=("overall_accuracy", "mean"),
    mean_cost=("total_cost_usd", "mean"),
).reindex(["DDM", "EDDM", "ADWIN", "KSWIN"])

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), dpi=200)

ax = axes[0]
bars = ax.bar(agg.index, agg["detected"], color=colors, width=0.6)
ax.axhline(agg["true_total"].iloc[0], color="gray", linestyle="--", linewidth=0.8)
ax.set_ylim(0, 9)
ax.set_ylabel("Drifts detected (of 8)")
ax.set_title("(a) Detection completeness", fontsize=9)
for b, v in zip(bars, agg["detected"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(int(v)), ha="center", fontsize=8)
ax.tick_params(labelsize=8)

ax = axes[1]
bars = ax.bar(agg.index, agg["mean_cost"], color=colors, width=0.6)
ax.set_ylabel("Mean cost per run (USD)")
ax.set_title("(b) Retraining cost vs. false alarms", fontsize=9)
for b, fa in zip(bars, agg["false_alarms"]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
             f"{int(fa)} FA", ha="center", fontsize=8)
ax.tick_params(labelsize=8)

for ax in axes:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.tight_layout()
out = R / "paper" / "figs" / "fig_detector_tradeoff.png"
fig.savefig(out, bbox_inches="tight")
print(f"Saved {out}")
