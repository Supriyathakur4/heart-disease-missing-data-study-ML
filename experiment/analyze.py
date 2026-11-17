"""Summaries (CSV + markdown) and figures from the fold-level results. No modelling here."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

perf = pd.read_csv(os.path.join(RESULTS_DIR, "perf_folds.csv"))
sel = pd.read_csv(os.path.join(RESULTS_DIR, "selective_folds.csv"))
pred = pd.read_csv(os.path.join(RESULTS_DIR, "predictions_pooled.csv.gz"))
N_RUNS = N_SPLITS * N_REPEATS

COL = {"LR": "#1f77b4", "RF": "#d62728", "Majority": "#7f7f7f"}
LS = {"median": "-", "knn": "--", "-": ":"}
MK = {"median": "o", "knn": "s", "-": None}
def label(m, i): return m if m == "Majority" else f"{m} + {i}"

def mean_sd(df, keys, cols):
    g = df.groupby(keys)[cols]
    m, s = g.mean(), g.std(ddof=1)
    m.columns = [c + "_mean" for c in cols]; s.columns = [c + "_sd" for c in cols]
    return pd.concat([m, s], axis=1).reset_index()

# ---------------------------------------------------------------- tables
perf_m = perf[perf.scenario == "matched"]
t_perf = mean_sd(perf_m, ["model", "imputer", "level"], ["bal_acc", "recall", "auc", "brier"])
t_perf.to_csv(os.path.join(RESULTS_DIR, "summary_performance_matched.csv"), index=False)

t_perf_clean = mean_sd(perf[perf.scenario == "clean_train"], ["model", "imputer", "level"],
                       ["bal_acc", "recall", "auc", "brier"])
t_perf_clean.to_csv(os.path.join(RESULTS_DIR, "summary_performance_clean_train.csv"), index=False)

sel_cols = ["coverage", "sel_risk", "risk_all", "brier_all", "brier_answered", "val_coverage", "rc_80"]
t_sel = mean_sd(sel, ["scenario", "model", "imputer", "level"], sel_cols)
t_sel["risk_reduction"] = t_sel["risk_all_mean"] - t_sel["sel_risk_mean"]
t_sel.to_csv(os.path.join(RESULTS_DIR, "summary_selective.csv"), index=False)

# paired RF - LR differences (same folds, same masks)
piv = perf_m[perf_m.model.isin(["LR", "RF"])].pivot_table(
    index=["imputer", "level", "repeat", "fold"], columns="model", values=["bal_acc", "auc", "brier"])
rows = []
for (imp, lvl), g in piv.groupby(level=["imputer", "level"]):
    r = dict(imputer=imp, level=lvl)
    for met in ["bal_acc", "auc", "brier"]:
        d = (g[(met, "RF")] - g[(met, "LR")]).to_numpy()
        r[f"{met}_RF_minus_LR_mean"] = d.mean(); r[f"{met}_RF_minus_LR_sd"] = d.std(ddof=1)
        r[f"{met}_share_folds_RF_better"] = (d > 0).mean() if met != "brier" else (d < 0).mean()
    rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "summary_paired_RF_vs_LR.csv"), index=False)

# paired failure-mode statistics: clean-threshold scenario minus matched scenario (same folds, same test masks)
_k = ["model", "imputer", "level", "repeat", "fold"]
_m = sel[sel.scenario == "matched"].set_index(_k); _c = sel[sel.scenario == "clean_train"].set_index(_k)
_d = pd.DataFrame({"cov_diff": _c.coverage - _m.coverage,      # coverage lost/gained by using the clean threshold
                   "rc80_diff": _c.rc_80 - _m.rc_80,           # error at equal 80% coverage (test-ranked, descriptive)
                   "sel_risk_diff": _c.sel_risk - _m.sel_risk}).groupby(["model", "imputer", "level"])
pd.DataFrame({"cov_diff_mean": _d.cov_diff.mean(), "cov_diff_sd": _d.cov_diff.std(ddof=1),
              "share_folds_coverage_lower": _d.cov_diff.apply(lambda x: (x < 0).mean()),
              "rc80_diff_mean": _d.rc80_diff.mean(), "rc80_diff_sd": _d.rc80_diff.std(ddof=1),
              "sel_risk_diff_mean": _d.sel_risk_diff.mean(), "sel_risk_diff_sd": _d.sel_risk_diff.std(ddof=1)}
             ).reset_index().to_csv(os.path.join(RESULTS_DIR, "summary_failure_mode_paired.csv"), index=False)

# sanity: at 0% injected missingness the two imputers see data that are complete except for the 6 original
# missing cells (4 in `ca`, 2 in `thal`), so their results must be (almost) identical.
z = perf_m[perf_m.level == 0.0].pivot_table(index=["model", "repeat", "fold"], columns="imputer", values="bal_acc")
z = z.loc[["LR", "RF"]]
d0 = (z["median"] - z["knn"]).abs()
assert d0.mean() < 0.01, "median and KNN should nearly agree at 0% injected missingness"
print("0%% injected missing: median vs KNN balanced accuracy identical in %.0f%% of runs, mean |diff| = %.4f" % (100 * (d0 < 1e-12).mean(), d0.mean()))

# tuning choices: how often each setting was selected by the inner CV
tl = pd.read_csv(os.path.join(RESULTS_DIR, "tuning_log.csv"))
tl["setting"] = np.where(tl.model == "LR", "C=" + tl.C.astype(str), "min_samples_leaf=" + tl.min_samples_leaf.round().astype("Int64").astype(str))
t_tune = (tl[tl.scenario == "matched"].groupby(["model", "purpose", "level", "setting"]).size()
          .groupby(level=["model", "purpose", "level"]).transform(lambda x: x / x.sum()).rename("share").reset_index())
t_tune.to_csv(os.path.join(RESULTS_DIR, "summary_tuning_choices.csv"), index=False)
t_tune_all = (tl.groupby(["model", "setting"]).size().rename("n").reset_index())
t_tune_all["share"] = t_tune_all.n / t_tune_all.groupby("model").n.transform("sum")

# ---------------------------------------------------------------- figure 2: performance vs missingness
fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
for ax, (met, name) in zip(axes, [("bal_acc", "Balanced accuracy"), ("recall", "Recall (disease)"),
                                  ("auc", "ROC-AUC"), ("brier", "Brier score (lower = better)")]):
    for (m, i), g in t_perf.groupby(["model", "imputer"]):
        g = g.sort_values("level")
        ax.errorbar(g.level * 100, g[met + "_mean"], yerr=g[met + "_sd"], color=COL[m], ls=LS[i],
                    marker=MK[i], ms=4, lw=1.5, capsize=2, alpha=.9, label=label(m, i))
    ax.set_xlabel("Cells missing (%)"); ax.set_title(name, fontsize=10); ax.grid(alpha=.3)
axes[0].legend(fontsize=7)
fig.suptitle("Performance vs. amount of missing data (mean ± 1 SD over %d train/test runs)" % N_RUNS, fontsize=10, y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig2_performance_vs_missing.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "fig2_performance_vs_missing.pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- figure 3: selective prediction (matched)
sm = t_sel[t_sel.scenario == "matched"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
for (m, i), g in sm.groupby(["model", "imputer"]):
    g = g.sort_values("level")
    axes[0].errorbar(g.level * 100, g.sel_risk_mean, yerr=g.sel_risk_sd, color=COL[m], ls=LS[i], marker=MK[i], ms=4, capsize=2, label=label(m, i))
    axes[1].errorbar(g.level * 100, g.risk_all_mean, yerr=g.risk_all_sd, color=COL[m], ls=LS[i], marker=MK[i], ms=4, capsize=2, label=label(m, i))
    axes[2].errorbar(g.level * 100, g.coverage_mean, yerr=g.coverage_sd, color=COL[m], ls=LS[i], marker=MK[i], ms=4, capsize=2, label=label(m, i))
axes[0].set_title("Error on answered cases\n(selective risk)", fontsize=10)
axes[1].set_title("Error if nothing is skipped", fontsize=10)
axes[2].set_title("Coverage (share answered)", fontsize=10)
axes[2].axhline(TARGET_COVERAGE, color="k", lw=.8, ls=":")
for a in axes: a.set_xlabel("Cells missing (%)"); a.grid(alpha=.3)
axes[0].set_ylim(0, None); axes[1].set_ylim(0, None); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig3_selective_vs_missing.png"), dpi=200)
fig.savefig(os.path.join(FIG_DIR, "fig3_selective_vs_missing.pdf")); plt.close(fig)

# ---------------------------------------------------------------- figure 4: risk-coverage curves
rc_cols = [f"rc_{int(round(c * 100))}" for c in COVERAGE_GRID]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
for ax, lvl in zip(axes, [0.0, 0.3, 0.5]):
    for (m, i), g in sel[(sel.scenario == "matched") & (sel.level == lvl)].groupby(["model", "imputer"]):
        ax.plot(COVERAGE_GRID * 100, g[rc_cols].mean().to_numpy(), color=COL[m], ls=LS[i], marker=MK[i], ms=3, label=label(m, i))
    ax.set_title(f"{int(lvl * 100)}% missing", fontsize=10); ax.set_xlabel("Coverage (%)"); ax.grid(alpha=.3)
axes[0].set_ylabel("Error among answered cases"); axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig4_risk_coverage.png"), dpi=200)
fig.savefig(os.path.join(FIG_DIR, "fig4_risk_coverage.pdf")); plt.close(fig)

# ---------------------------------------------------------------- figure 5: failure-mode audit
fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True)
for ci, m in enumerate(["LR", "RF"]):
    for i in IMPUTERS:
        for scen, color, nm in [("matched", "#2ca02c", "threshold chosen on data with the same missingness"),
                                ("clean_train", "#ff7f0e", "threshold chosen on clean data")]:
            g = t_sel[(t_sel.scenario == scen) & (t_sel.model == m) & (t_sel.imputer == i)].sort_values("level")
            axes[0, ci].errorbar(g.level * 100, g.coverage_mean, yerr=g.coverage_sd, color=color, ls=LS[i], marker=MK[i], ms=4, capsize=2, label=f"{nm} ({i})")
            axes[1, ci].errorbar(g.level * 100, g.sel_risk_mean, yerr=g.sel_risk_sd, color=color, ls=LS[i], marker=MK[i], ms=4, capsize=2)
    axes[0, ci].axhline(TARGET_COVERAGE, color="k", lw=.8, ls=":")
    axes[0, ci].set_title(m, fontsize=11); axes[1, ci].set_xlabel("Cells missing in the TEST data (%)")
    for r in (0, 1): axes[r, ci].grid(alpha=.3)
axes[0, 0].set_ylabel("Coverage (target 80%)"); axes[1, 0].set_ylabel("Error among answered cases")
axes[0, 0].legend(fontsize=6)
fig.suptitle("Failure-mode audit: does a threshold chosen on clean data survive missing test data?", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig5_failure_mode_audit.png"), dpi=200)
fig.savefig(os.path.join(FIG_DIR, "fig5_failure_mode_audit.pdf")); plt.close(fig)

# ---------------------------------------------------------------- figure 6: reliability diagrams
def reliability(p, y, bins=8):
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.digitize(p, q[1:-1]), 0, len(q) - 2)
    xs, ys = [], []
    for b in range(len(q) - 1):
        s = idx == b
        if s.sum() >= 5: xs.append(p[s].mean()); ys.append(y[s].mean())
    return xs, ys
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
for ax, lvl in zip(axes, [0.0, 0.3, 0.5]):
    ax.plot([0, 1], [0, 1], "k:", lw=.8)
    for m in ["LR", "RF"]:
        g = pred[(pred.model == m) & (pred.imputer == "median") & (pred.level == lvl)]
        xs, ys = reliability(g.p.to_numpy(), g.y.to_numpy())
        b = brier = ((g.p - g.y) ** 2).mean()
        ax.plot(xs, ys, color=COL[m], marker="o", ms=4, label=f"{m} (Brier {b:.3f})")
    ax.set_title(f"{int(lvl * 100)}% missing (median imputation)", fontsize=10)
    ax.set_xlabel("Predicted probability of disease"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("Observed share with disease")
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig6_reliability.png"), dpi=200)
fig.savefig(os.path.join(FIG_DIR, "fig6_reliability.pdf")); plt.close(fig)

# ---------------------------------------------------------------- markdown tables
def fmt(df, cols, names):
    out = ["| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(c(r) if callable(c) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)
pm = lambda mc, sc: (lambda r: f"{r[mc]:.3f} ± {r[sc]:.3f}")
md = ["# Auto-generated result tables", "",
      f"Mean ± SD over {N_RUNS} runs (5-fold stratified CV x {N_REPEATS} repeats). Missing level = % of feature cells removed at random.", ""]
md += ["## Performance, matched scenario (train and test both incomplete)", ""]
tp = t_perf.copy(); tp["level"] = (tp.level * 100).astype(int).astype(str) + "%"
tp["model"] = tp.model + np.where(tp.imputer == "-", "", " + " + tp.imputer)
md.append(fmt(tp, ["model", "level", pm("bal_acc_mean", "bal_acc_sd"), pm("recall_mean", "recall_sd"), pm("auc_mean", "auc_sd"), pm("brier_mean", "brier_sd")],
              ["Model", "Missing", "Balanced acc.", "Recall", "ROC-AUC", "Brier"]))
for scen, title in [("matched", "Selective prediction, matched scenario"), ("clean_train", "Failure-mode audit: clean training/validation, incomplete test data")]:
    md += ["", f"## {title}", ""]
    ts = t_sel[t_sel.scenario == scen].copy(); ts["level"] = (ts.level * 100).astype(int).astype(str) + "%"
    ts["model"] = ts.model + " + " + ts.imputer
    md.append(fmt(ts, ["model", "level", pm("coverage_mean", "coverage_sd"), pm("sel_risk_mean", "sel_risk_sd"), pm("risk_all_mean", "risk_all_sd"),
                       lambda r: f"{r['risk_reduction']:.3f}", pm("brier_all_mean", "brier_all_sd")],
                  ["Model", "Missing", "Coverage", "Selective risk", "Risk w/o abstaining", "Risk reduction", "Brier (all)"]))
md += ["", "## Hyper-parameters chosen by the inner cross-validation (all tuning runs, all scenarios)", "",
       fmt(t_tune_all, ["model", "setting", "n", lambda r: f"{r['share']:.2f}"], ["Model", "Chosen setting", "Times chosen", "Share"])]
with open(os.path.join(RESULTS_DIR, "results_tables.md"), "w") as f:
    f.write("\n".join(md) + "\n")
print("analysis done")
