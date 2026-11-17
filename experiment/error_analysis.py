"""
Error analysis ("representative errors"): what kinds of mistakes are made, who gets skipped, and
which patients are persistently misclassified. Uses only the saved per-patient predictions of the
selective models (matched scenario, levels 0%, 30%, 50%); nothing is re-fitted here.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

LEVELS = [0.0, 0.3, 0.5]
COL = {"LR": "#1f77b4", "RF": "#d62728"}
LS = {"median": "-", "knn": "--"}
MK = {"median": "o", "knn": "s"}

sp = pd.read_csv(os.path.join(RESULTS_DIR, "selective_predictions.csv.gz"))
X, _ = load_data()
sp["pred"] = (sp.p >= 0.5).astype(int)
sp["err"] = (sp.pred != sp.y).astype(int)
sp["abstained"] = 1 - sp.accepted
N_RUNS = N_SPLITS * N_REPEATS

# ------------------------------------------------------------------ 1. error types and who is skipped
def type_row(g):
    pos, neg = g[g.y == 1], g[g.y == 0]
    ans = g[g.accepted == 1]
    fn = ((g.y == 1) & (g.pred == 0)).sum(); fp = ((g.y == 0) & (g.pred == 1)).sum()
    return pd.Series(dict(
        patient_runs=len(g),
        miss_rate_FN=(pos.pred == 0).mean(),                       # diseased patients predicted healthy
        false_alarm_rate_FP=(neg.pred == 1).mean(),                # healthy patients predicted diseased
        miss_rate_FN_answered=(ans[ans.y == 1].pred == 0).mean(),
        false_alarm_rate_FP_answered=(ans[ans.y == 0].pred == 1).mean(),
        share_of_errors_that_are_FN=fn / max(fn + fp, 1),
        abstain_rate_on_wrong=g[g.err == 1].abstained.mean(),
        abstain_rate_on_correct=g[g.err == 0].abstained.mean(),
        error_rate_answered=ans.err.mean(),
        error_rate_abstained=g[g.accepted == 0].err.mean(),
    ))
types = sp.groupby(["model", "imputer", "level"]).apply(type_row, include_groups=False).reset_index()
types.to_csv(os.path.join(RESULTS_DIR, "error_analysis_types.csv"), index=False)

# ------------------------------------------------------------------ 2. error / abstention rate by patient group
feat = X.copy()
feat["AgeBand"] = pd.cut(feat.Age, [0, 49, 59, 200], labels=["<50", "50-59", ">=60"])
feat["AnyVesselsColoured"] = np.where(feat.NumVessels.isna(), "unknown", np.where(feat.NumVessels > 0, ">=1", "0"))
GROUPS = ["ChestPainType", "Sex", "ExerciseAngina", "AgeBand", "Thal", "AnyVesselsColoured"]
j = sp.join(feat[GROUPS], on="row")
rows = []
for gcol in GROUPS:
    for (m, i, lvl, val), g in j.groupby(["model", "imputer", "level", gcol], observed=True):
        rows.append(dict(model=m, imputer=i, level=lvl, group=gcol, value=str(val),
                         patients=g.row.nunique(), patient_runs=len(g), error_rate=g.err.mean(),
                         abstain_rate=g.abstained.mean(), error_rate_answered=g[g.accepted == 1].err.mean()))
groups = pd.DataFrame(rows)
groups.to_csv(os.path.join(RESULTS_DIR, "error_analysis_by_group.csv"), index=False)

# ------------------------------------------------------------------ 3. persistently misclassified patients
per = (sp.groupby(["model", "imputer", "level", "row"])
         .agg(runs=("err", "size"), wrong=("err", "sum"), mean_p=("p", "mean"), mean_conf=("conf", "mean"),
              abstained=("abstained", "sum"), y=("y", "first")).reset_index())
per["always_wrong"] = per.wrong == per.runs
pers = (per.groupby(["model", "imputer", "level"])
          .apply(lambda g: pd.Series(dict(
              patients_always_wrong=int(g.always_wrong.sum()),
              share_of_patients=g.always_wrong.mean(),
              never_wrong=int((g.wrong == 0).sum()),
              share_abstained_among_always_wrong=(g[g.always_wrong].abstained.sum() / max(g[g.always_wrong].runs.sum(), 1)),
              share_abstained_among_never_wrong=(g[g.wrong == 0].abstained.sum() / max(g[g.wrong == 0].runs.sum(), 1)),
              always_wrong_that_are_diseased=(g[g.always_wrong].y.mean() if g.always_wrong.any() else np.nan))),
                 include_groups=False).reset_index())
pers.to_csv(os.path.join(RESULTS_DIR, "error_analysis_persistent_counts.csv"), index=False)
lst = per[per.always_wrong].join(X, on="row")
lst.sort_values(["model", "imputer", "level", "mean_conf"], ascending=[True, True, True, False]).to_csv(
    os.path.join(RESULTS_DIR, "error_analysis_persistent_errors.csv"), index=False)

# ------------------------------------------------------------------ 4. figure
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for (m, i), g in types.groupby(["model", "imputer"]):
    g = g.sort_values("level")
    axes[0].plot(g.level * 100, g.miss_rate_FN, color=COL[m], ls=LS[i], marker=MK[i], ms=4, label=f"{m} + {i}")
    axes[1].plot(g.level * 100, g.false_alarm_rate_FP, color=COL[m], ls=LS[i], marker=MK[i], ms=4)
axes[0].set_title("Missed disease (false-negative rate)", fontsize=10)
axes[1].set_title("False alarms (false-positive rate)", fontsize=10)
for a in axes[:2]:
    a.set_xlabel("Cells missing (%)"); a.set_xticks([0, 30, 50]); a.grid(alpha=.3); a.set_ylim(0, None)
axes[0].legend(fontsize=7)
for m in ["LR", "RF"]:
    g = types[(types.model == m) & (types.imputer == "median")].sort_values("level")
    axes[2].plot(g.level * 100, g.abstain_rate_on_wrong, color=COL[m], ls="-", marker="o", ms=4, label=f"{m}: wrong predictions")
    axes[2].plot(g.level * 100, g.abstain_rate_on_correct, color=COL[m], ls=":", marker="o", ms=4, mfc="white", label=f"{m}: correct predictions")
axes[2].set_title("Share skipped (median imputation)", fontsize=10); axes[2].set_xlabel("Cells missing (%)")
axes[2].set_xticks([0, 30, 50]); axes[2].grid(alpha=.3); axes[2].set_ylim(0, 1); axes[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "fig7_error_analysis.png"), dpi=200)
fig.savefig(os.path.join(FIG_DIR, "fig7_error_analysis.pdf")); plt.close(fig)

# ------------------------------------------------------------------ 5. markdown report (tables only; interpretation is in RESULTS_SUMMARY.md)
def md(df, fmts=None):
    fmts = fmts or {}
    out = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            cells.append(fmts.get(c, "{:.3f}").format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if pd.isna(v) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)

L = ["# Error analysis (auto-generated tables)", "",
     f"Pooled over {N_RUNS} runs; every patient is a test case once per repeat (3 times). Predictions come from the selective models "
     "(trained on 75% of each training fold), matched scenario. Positive class = heart disease.", "",
     "## 1. What kind of mistakes, and are they the ones that get skipped?", ""]
t = types.copy(); t["level"] = (t.level * 100).astype(int).astype(str) + "%"
t["model"] = t.model + " + " + t.imputer
t = t[["model", "level", "miss_rate_FN", "false_alarm_rate_FP", "share_of_errors_that_are_FN", "abstain_rate_on_wrong",
       "abstain_rate_on_correct", "error_rate_answered", "error_rate_abstained"]]
t.columns = ["Model", "Missing", "Missed disease (FN rate)", "False alarm (FP rate)", "FN share of errors",
             "Skipped (wrong)", "Skipped (correct)", "Error (answered)", "Error (skipped)"]
L += [md(t), "", "## 2. By patient group (median imputation; 0% and 50% missing)", "",
      "Groups are small, so differences between groups are descriptive only.", ""]
g = groups[(groups.imputer == "median") & (groups.level.isin([0.0, 0.5]))].copy()
g["level"] = (g.level * 100).astype(int).astype(str) + "%"
g = g[["model", "level", "group", "value", "patients", "error_rate", "abstain_rate", "error_rate_answered"]]
g.columns = ["Model", "Missing", "Group", "Value", "Patients", "Error rate", "Skipped rate", "Error (answered)"]
L += [md(g.sort_values(["Model", "Missing", "Group", "Value"]), {"Patients": "{:.0f}"}), "",
      "## 3. Patients misclassified in every one of their 3 test appearances", ""]
pc = pers.copy(); pc["level"] = (pc.level * 100).astype(int).astype(str) + "%"; pc["model"] = pc.model + " + " + pc.imputer
pc = pc[["model", "level", "patients_always_wrong", "share_of_patients", "never_wrong", "share_abstained_among_always_wrong",
         "share_abstained_among_never_wrong", "always_wrong_that_are_diseased"]]
pc.columns = ["Model", "Missing", "Always wrong", "Share of 303", "Never wrong", "Skipped (always wrong)", "Skipped (never wrong)", "Share truly diseased"]
L += [md(pc, {"Always wrong": "{:.0f}", "Never wrong": "{:.0f}"}), "",
      "### Most confidently wrong patients at 0% missing (full list: `error_analysis_persistent_errors.csv`)", ""]
for m in ["LR", "RF"]:
    z = lst[(lst.model == m) & (lst.imputer == "median") & (lst.level == 0.0)].sort_values("mean_conf", ascending=False).head(8)
    z = z[["row", "y", "mean_p", "mean_conf", "abstained", "Age", "Sex", "ChestPainType", "Cholesterol", "MaxHR", "ExerciseAngina", "Oldpeak", "NumVessels", "Thal"]]
    z = z.rename(columns={"row": "Patient (row)", "y": "True", "mean_p": "Mean P(disease)", "mean_conf": "Mean confidence", "abstained": "Times skipped (of 3)"})
    L += [f"**{m} + median**", "", md(z, {"Patient (row)": "{:.0f}", "True": "{:.0f}", "Times skipped (of 3)": "{:.0f}", "Age": "{:.0f}", "Cholesterol": "{:.0f}", "MaxHR": "{:.0f}", "NumVessels": "{:.0f}"}), ""]
with open(os.path.join(RESULTS_DIR, "error_analysis.md"), "w") as f:
    f.write("\n".join(L) + "\n")
print("error analysis done")
