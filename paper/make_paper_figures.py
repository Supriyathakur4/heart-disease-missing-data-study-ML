"""Compact figures for the 4-page paper (fig_performance, fig_failure_audit), drawn only from the saved result files in ../results."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# Times-like font so the figures match the paper text (falls back silently if not installed)
for f in fm.findSystemFonts():
    if "texgyretermes" in f.lower() or "liberationserif" in f.lower():
        fm.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, RED, GREY = "#1f5fa8", "#c0392b", "#6b6b6b"
STY = {("LR", "median"): (BLUE, "-", "o"), ("LR", "knn"): (BLUE, "--", "s"),
       ("RF", "median"): (RED, "-", "o"), ("RF", "knn"): (RED, "--", "s")}
NAME = {("LR", "median"): "LR + median", ("LR", "knn"): "LR + KNN", ("RF", "median"): "RF + median", ("RF", "knn"): "RF + KNN"}

perf = pd.read_csv(os.path.join(R, "summary_performance_matched.csv"))
sel = pd.read_csv(os.path.join(R, "summary_selective.csv"))
selr = pd.read_csv(os.path.join(R, "selective_folds.csv"))
X = [0, 10, 20, 30, 40, 50]

# ---------------- Figure 1: performance vs missing data ----------------
fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.46))
for ax, (met, ttl, ylim) in zip(axes, [("bal_acc", "Balanced accuracy", (0.45, 0.92)), ("auc", "ROC-AUC", (0.45, 0.97)),
                                       ("brier", "Brier score (lower is better)", (0.08, 0.27))]):
    g = perf[perf.model == "Majority"].sort_values("level")
    ax.plot(g.level * 100, g[met + "_mean"], color=GREY, ls=":", lw=1.2, label="Majority class")
    for key, (c, ls, mk) in STY.items():
        g = perf[(perf.model == key[0]) & (perf.imputer == key[1])].sort_values("level")
        ax.plot(g.level * 100, g[met + "_mean"], color=c, ls=ls, marker=mk, ms=2.6, label=NAME[key])
        ax.fill_between(g.level * 100, g[met + "_mean"] - g[met + "_sd"], g[met + "_mean"] + g[met + "_sd"], color=c, alpha=0.05, lw=0)
    ax.set_title(ttl); ax.set_xlabel("Cells missing (%)"); ax.set_ylim(*ylim); ax.set_xticks(X); ax.grid(alpha=.25, lw=.4)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, frameon=False, handlelength=2.0, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.4, w_pad=0.8, rect=(0, 0, 1, 0.87))
fig.savefig(os.path.join(OUT, "fig_performance.pdf")); plt.close(fig)

# ---------------- Figure 3: failure-mode audit ----------------
k = ["model", "imputer", "level", "repeat", "fold"]
mm = selr[selr.scenario == "matched"].set_index(k); cc = selr[selr.scenario == "clean_train"].set_index(k)
fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.42))
for key, (c, ls, mk) in STY.items():
    g = sel[(sel.scenario == "clean_train") & (sel.model == key[0]) & (sel.imputer == key[1])].sort_values("level")
    axes[0].plot(g.level * 100, g.coverage_mean, color=c, ls=ls, marker=mk, ms=2.6, label=NAME[key])
axes[0].axhline(0.8, color="k", lw=0.6, ls=":")
axes[0].set_title("Coverage with a threshold set on clean data"); axes[0].set_ylim(0.62, 0.9)
axes[0].set_xlabel("Cells missing in the test data (%)"); axes[0].set_ylabel("Share answered"); axes[0].set_xticks(X)
d = (cc.rc_80 - mm.rc_80)
for key, (c, ls, mk) in STY.items():
    dd = d.xs((key[0], key[1]), level=("model", "imputer"))
    g = dd.groupby("level").agg(["mean", "std"])
    axes[1].errorbar(g.index * 100 + {"LR": -0.6, "RF": 0.6}[key[0]], g["mean"], yerr=g["std"] / np.sqrt(15),
                     color=c, ls=ls, marker=mk, ms=2.6, lw=1.0, capsize=1.5, elinewidth=0.6)
axes[1].axhline(0, color="k", lw=0.6)
axes[1].set_title("Extra error at equal 80% coverage"); axes[1].set_xlabel("Cells missing in the test data (%)")
axes[1].set_ylabel("Error difference"); axes[1].set_xticks(X)
for ax in axes: ax.grid(alpha=.25, lw=.4)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, handlelength=2.0, columnspacing=1.4, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.4, w_pad=1.0, rect=(0, 0, 1, 0.88))
fig.savefig(os.path.join(OUT, "fig_failure_audit.pdf")); plt.close(fig)
print("figures written to", OUT)
