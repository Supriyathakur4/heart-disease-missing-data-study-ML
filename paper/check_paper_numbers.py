"""Checks that every number quoted in es678_paper.tex matches the saved result files (run from anywhere)."""
import os, re, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "..", "results")
sys.path.insert(0, os.path.join(HERE, "..", "experiment"))
tex = open(os.path.join(HERE, "es678_paper.tex"), encoding="utf-8").read()
flat = re.sub(r"\s+", " ", tex).replace("\\%", "%").replace("--", "–").replace("$", "")
res = []
def has(desc, s): res.append((desc, s in flat or s.replace("–", "--") in tex))
def chk(desc, cond): res.append((desc, bool(cond)))

perf = pd.read_csv(f"{R}/summary_performance_matched.csv"); sel = pd.read_csv(f"{R}/summary_selective.csv")
pr = pd.read_csv(f"{R}/summary_paired_RF_vs_LR.csv"); fm = pd.read_csv(f"{R}/summary_failure_mode_paired.csv")
et = pd.read_csv(f"{R}/error_analysis_types.csv"); pe = pd.read_csv(f"{R}/error_analysis_persistent_errors.csv"); pc = pd.read_csv(f"{R}/error_analysis_persistent_counts.csv")
aud = json.load(open(f"{R}/data_audit.json"))
P = lambda m, i, l, c: float(perf[(perf.model == m) & (perf.imputer == i) & (perf.level == l)][c + "_mean"].iloc[0])
S = lambda sc, m, i, l, c: float(sel[(sel.scenario == sc) & (sel.model == m) & (sel.imputer == i) & (sel.level == l)][c].iloc[0])

# data
chk("303 patients, 13 features", aud["rows"] == 303 and aud["n_features"] == 13); has("303 patients and 13 features", "303 patients and 13 features")
has("label counts", "164/55/36/35/13"); chk("139/164 split", aud["class_counts"] == {"no_disease(0)": 164, "disease(1)": 139})
has("majority 54.1%", "54.1%"); chk("majority share", round(aud["majority_class_share"] * 100, 1) == 54.1)
chk("6 missing cells ca4 thal2 (0.15%)", aud["native_missing_cells_by_feature"] == {"NumVessels": 4, "Thal": 2} and round(aud["native_missing_cell_rate"] * 100, 2) == 0.15)
has("0.15% of cells", "0.15% of cells")
from common import load_data, encode
X, y = load_data(); chk("22 encoded columns", encode(X).shape[1] == 22); has("22 columns", "22 columns")
# performance
chk("baseline 0.500/0.500/0.248", (P("Majority", "-", 0.0, "bal_acc"), P("Majority", "-", 0.0, "auc")) == (0.5, 0.5) and round(P("Majority", "-", 0.0, "brier"), 3) == 0.248)
has("baseline numbers", "0.500, AUC 0.500, Brier 0.248")
chk("LR median 0%: .844 .916 .116", [round(P("LR", "median", 0.0, c), 3) for c in ("bal_acc", "auc", "brier")] == [0.844, 0.916, 0.116]); has("LR 0% numbers", "0.844, 0.916, 0.116")
chk("LR median 50%: .771 .857 .156", [round(P("LR", "median", 0.5, c), 3) for c in ("bal_acc", "auc", "brier")] == [0.771, 0.857, 0.156]); has("LR 50% numbers", "0.771, 0.857 and 0.156")
has("abstract 0.84 to 0.77", "0.84 to 0.77")
d = pr.bal_acc_RF_minus_LR_mean; chk("RF-LR BA range -0.023..+0.005", round(d.min(), 3) == -0.023 and round(d.max(), 3) == 0.005); has("RF-LR range", "−0.023 and +0.005" if "−" in flat else "-0.023 and +0.005")
chk("RF better in 13-47% of splits", round(pr.bal_acc_share_folds_RF_better.min() * 100) == 13 and round(pr.bal_acc_share_folds_RF_better.max() * 100) == 47); has("13-47%", "13–47%")
chk("RF worse in AUC and Brier in all 12", (pr.auc_RF_minus_LR_mean < 0).all() and (pr.brier_RF_minus_LR_mean > 0).all() and len(pr) == 12)
pv = perf[perf.model.isin(["LR", "RF"])].pivot_table(index=["model", "level"], columns="imputer", values="bal_acc_mean")
chk("KNN-median max 0.033", round((pv["knn"] - pv["median"]).max(), 3) == 0.033); has("0.033", "0.033")
# selective
m = sel[sel.scenario == "matched"]
chk("abstention lowers error in all 24 settings", len(m) == 24 and (m.sel_risk_mean < m.risk_all_mean).all()); has("all 24", "all 24")
chk("LR median 0%: 0.158 -> 0.112", round(S("matched", "LR", "median", 0.0, "risk_all_mean"), 3) == 0.158 and round(S("matched", "LR", "median", 0.0, "sel_risk_mean"), 3) == 0.112); has("0.158 to 0.112", "0.158 to 0.112")
chk("reduction 0.034-0.073", round(m.risk_reduction.min(), 3) == 0.034 and round(m.risk_reduction.max(), 3) == 0.073); has("0.034-0.073", "0.034–0.073")
m5 = m[m.level == 0.5]; m0 = m[m.level == 0.0]
chk("sel risk at 50%: 0.155-0.205", round(m5.sel_risk_mean.min(), 3) == 0.155 and round(m5.sel_risk_mean.max(), 3) == 0.205); has("0.155-0.205", "0.155–0.205")
chk("complete-data error w/o abstention 0.157-0.166", round(m0.risk_all_mean.min(), 3) == 0.157 and round(m0.risk_all_mean.max(), 3) == 0.166); has("0.157-0.166", "0.157–0.166")
cnt = sum(S("matched", a, b, 0.5, "sel_risk_mean") > S("matched", a, b, 0.0, "risk_all_mean") for a in ("LR", "RF") for b in ("median", "knn")); chk("exceeds in 3 of 4 configs", cnt == 3); has("three of four", "three of four")
# table rows
for mod, imp in [("LR", "median"), ("LR", "knn"), ("RF", "median"), ("RF", "knn")]:
    row = [S("matched", mod, imp, 0.0, "coverage_mean"), S("matched", mod, imp, 0.0, "sel_risk_mean"), S("matched", mod, imp, 0.0, "risk_all_mean"),
           S("matched", mod, imp, 0.5, "coverage_mean"), S("matched", mod, imp, 0.5, "sel_risk_mean"), S("matched", mod, imp, 0.5, "risk_all_mean"), S("clean_train", mod, imp, 0.5, "coverage_mean")]
    fmt = [f"{row[0]:.2f}", f"{row[1]:.3f}", f"{row[2]:.3f}", f"{row[3]:.2f}", f"{row[4]:.3f}", f"{row[5]:.3f}", f"{row[6]:.2f}"]
    line = re.search(rf"{mod} \+ {'median' if imp=='median' else 'KNN'}\s*&([^\\]*)\\\\", tex).group(1)
    cells = [c.strip() for c in line.split("&")]
    chk(f"table row {mod}+{imp}", cells == fmt)
chk("SD ranges in caption", round(m[m.level.isin([0, .5])].coverage_sd.min(), 2) in (0.06, 0.07) and round(m[m.level.isin([0, .5])].coverage_sd.max(), 2) == 0.09 and round(m[m.level.isin([0, .5])].sel_risk_sd.max(), 2) == 0.07)
# error analysis
e = et[(et.model == "LR") & (et.imputer == "median")]
chk("FN rate .20 -> .30", round(float(e[e.level == 0.0].miss_rate_FN.iloc[0]), 2) == 0.20 and round(float(e[e.level == 0.5].miss_rate_FN.iloc[0]), 2) == 0.29); has("0.20 to 0.29", "0.20 to 0.29")
chk("FN share of errors 52-61%", round(et.share_of_errors_that_are_FN.min() * 100) == 52 and round(et.share_of_errors_that_are_FN.max() * 100) == 61); has("52-61%", "52–61%")
chk("skips 44% wrong / 19% correct", round(float(e[e.level == 0.0].abstain_rate_on_wrong.iloc[0]) * 100) == 44 and round(float(e[e.level == 0.0].abstain_rate_on_correct.iloc[0]) * 100) == 19); has("44% / 19%", "44% of its wrong but only 19%")
cnt33 = pc[(pc.model == "LR") & (pc.imputer == "median") & (pc.level == 0.0)].iloc[0]
chk("33 patients (10.9%)", int(cnt33.patients_always_wrong) == 33 and round(cnt33.share_of_patients * 100, 1) == 10.9); has("33 patients, 10.9%", "33 patients, 10.9%")
top = pe[(pe.model == "LR") & (pe.imputer == "median") & (pe.level == 0.0)].sort_values("mean_conf", ascending=False).head(8)
dis, hea = top[top.y == 1], top[top.y == 0]
chk("top-8: five diseased, three healthy", len(dis) == 5 and len(hea) == 3)
chk("diseased: p 0.04-0.14", round(dis.mean_p.min(), 2) == 0.04 and round(dis.mean_p.max(), 2) == 0.14)
chk("diseased look low-risk (no angina, 0 vessels, normal thal, oldpeak 0)", (dis.ExerciseAngina == "N").all() and (dis.NumVessels == 0).all() and (dis.Thal == "normal").all() and (dis.Oldpeak == 0).all())
chk("healthy: p 0.90-0.94", round(hea.mean_p.min(), 2) == 0.90 and round(hea.mean_p.max(), 2) == 0.94)
chk("healthy: reversible thal, 1-3 vessels", (hea.Thal == "reversible").all() and hea.NumVessels.between(1, 3).all())
chk("none ever skipped", (top.abstained == 0).all()); has("none was ever skipped", "none was ever skipped")
# audit
f5 = fm[fm.level == 0.5].set_index(["model", "imputer"])
chk("LR median clean coverage .78 -> .67, matched .79", round(S("clean_train", "LR", "median", 0.0, "coverage_mean"), 2) == 0.78 and round(S("clean_train", "LR", "median", 0.5, "coverage_mean"), 2) == 0.67 and round(S("matched", "LR", "median", 0.5, "coverage_mean"), 2) == 0.79); has("0.78 to 0.67", "falls from 0.78 to 0.67")
chk("lower in 14 of 15 splits", round(f5.loc[("LR", "median")].share_folds_coverage_lower * 15) == 14); has("14 of 15", "14 of 15")
chk("RF median .79 -> .73", round(S("clean_train", "RF", "median", 0.0, "coverage_mean"), 2) == 0.79 and round(S("clean_train", "RF", "median", 0.5, "coverage_mean"), 2) == 0.73)
chk("RF KNN 0.84 (+0.070)", round(S("clean_train", "RF", "knn", 0.5, "coverage_mean"), 2) == 0.84 and round(f5.loc[("RF", "knn")].cov_diff_mean, 3) == 0.070); has("0.84, +0.070", "0.84, +0.070")
chk("equal-coverage +0.030 / +0.038", round(f5.loc[("LR", "median")].rc80_diff_mean, 3) == 0.030 and round(f5.loc[("RF", "median")].rc80_diff_mean, 3) == 0.038); has("+0.030, RF +0.038", "+0.030, RF +0.038")
chk("12 percentage points (67% vs 79%)", round((S("matched", "LR", "median", 0.5, "coverage_mean") - S("clean_train", "LR", "median", 0.5, "coverage_mean")) * 100) == 12); has("12 percentage points", "12 percentage points"); has("67% instead of 79%", "67% instead of 79%")
# conclusion
chk("LR BA falls ~0.07, AUC ~0.06", round(P("LR", "median", 0.0, "bal_acc") - P("LR", "median", 0.5, "bal_acc"), 2) == 0.07 and round(P("LR", "median", 0.0, "auc") - P("LR", "median", 0.5, "auc"), 2) == 0.06)
w = max(len(d) for d, _ in res)
for d_, ok in res: print(("PASS  " if ok else "FAIL  ") + d_)
print(f"\n{sum(ok for _, ok in res)}/{len(res)} checks passed"); sys.exit(0 if all(ok for _, ok in res) else 1)
