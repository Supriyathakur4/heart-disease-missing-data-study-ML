# What the results show (read together with `results_tables.md` and `error_analysis.md`)

Data: Cleveland subset, 303 patients, 13 features (6 original missing cells). All numbers are means over 15 runs
(5-fold stratified CV × 3 repeats); "SD" is the spread across those runs. The 15 runs share training data, so SDs describe
variability, they are **not** independent-sample confidence intervals, and no significance tests were run.
"X% missing" = X% of the feature cells removed at random *in addition to* the 6 original missing cells.
Hyper-parameters were tuned by inner cross-validation inside every training fold (Logistic Regression chose C = 0.1 in 406 of 420
tuning runs; Random Forest spread over min_samples_leaf = 3 (193), 1 (141), 5 (81), 10 (5)).

## 1. Missing data lowers reliability gradually, and the models stay well above the baseline
* Majority-class baseline: balanced accuracy 0.500, recall 0.000 (it always predicts "no disease"), ROC-AUC 0.500, Brier 0.248.
* Logistic Regression at 0% missing: balanced accuracy 0.844 ± 0.054, ROC-AUC 0.916 ± 0.034, Brier 0.116 (median imputation).
  At 50% missing: balanced accuracy 0.771 (median) / 0.766 (KNN), ROC-AUC 0.857 / 0.853, Brier 0.156 / 0.157.
* Random Forest: balanced accuracy 0.830 → 0.759 (median) / 0.831 → 0.764 (KNN); ROC-AUC 0.913 → 0.836 / 0.915 → 0.842.
* Recall for disease falls from about 0.78–0.80 to about 0.71–0.74 at 50% missing.
* The decline is gradual, with no collapse in the 0–50% range tested (for Logistic Regression the largest single step is 20% → 30%).

## 2. Random Forest did not beat Logistic Regression
Paired RF − LR difference in balanced accuracy (same folds, same masks) is between −0.023 and +0.005 at every level and imputer
(fold-to-fold SD 0.025–0.039); RF is better in only 13–47% of runs (`summary_paired_RF_vs_LR.csv`). RF is slightly worse on average in ROC-AUC
(by 0.001–0.021) and in Brier score (by 0.006–0.013) in all 12 level/imputer combinations, but these gaps are small relative to the noise.
So there is no evidence that the more flexible model helps here.

## 3. Median vs. KNN imputation: small differences
KNN was nominally ahead in balanced accuracy in 10 of the 12 model/level comparisons, by 0.001–0.033 (largest: RF at 30%: 0.801 vs 0.768),
always smaller than the run-to-run SD (about 0.04–0.06). At 0% injected missingness the two imputers still differ slightly (identical in 63% of runs,
mean |difference| 0.0065) because 6 cells are genuinely missing. The data do not establish that one imputer is better.

## 4. Selective prediction reduces errors, and the benefit holds across missing-data levels
* At ~80% target coverage the error among answered cases is lower than the error without abstaining in all 24 model/imputer/level settings.
  Examples (0% missing): LR + median 0.158 → 0.112 at 0.773 mean test coverage; RF + median 0.166 → 0.113 at 0.788.
  (50% missing): LR + median 0.231 → 0.173; LR + KNN 0.219 → 0.155; RF + median 0.260 → 0.205; RF + KNN 0.244 → 0.180.
* The mean reduction is 0.034–0.073 and stays roughly constant (about 0.05–0.06) from 0% to 50% missing; it is positive in 73–100% of the 15 runs.
* Abstention does not fully compensate for missing data: at 50% missing it brings LR + KNN to about the error of answering everyone on complete data
  (0.155 vs 0.157, while skipping ~21% of patients), but the other three configurations stay above that level (0.173–0.205 vs 0.158–0.166).
* Realised test coverage averages 0.767–0.814 around the 0.80 target (mean validation coverage 0.804); individual test folds (~61 patients) vary by an SD of 0.05–0.10.
* The majority-class baseline is not shown for selective prediction (its confidence is constant).

## 5. Graduate depth analysis — failure-mode audit
Question: if training, tuning and the confidence threshold use clean data, does the system still behave as planned when incoming data are incomplete?
(Test masks are identical to the matched scenario, so differences are paired.)
* **Median imputation: coverage drops below the planned 80%.** LR coverage falls from 0.785 (0% missing) to 0.671 at 50%; the paired difference to the matched
  threshold is −0.120 ± 0.113 at 50% (lower in 14 of 15 runs). RF: 0.789 → 0.735 (−0.055 ± 0.143, lower in 10 of 15 runs).
* **The direction depends on model and imputer.** With KNN imputation LR shows no consistent change (−0.014 at 50%), while RF + KNN *over*-answers:
  coverage 0.842 at 50% (+0.070 ± 0.075; lower in only 1 of 15 runs).
* **Error at equal 80% coverage:** for median imputation the clean-trained models are somewhat worse at 50% (LR +0.030, RF +0.038; SD 0.035–0.050; worse in 73% of runs);
  for KNN the difference is within noise (+0.008 LR, −0.015 RF).
  Error without abstaining at 50% missing tells a similar story: clean training is worse for median imputation (LR 0.260 vs 0.231, RF 0.290 vs 0.260) but not worse with KNN
  (LR 0.209 vs 0.219, RF 0.220 vs 0.244; differences within noise).
* So a threshold set on clean data is not reliable under missing data: with median imputation the system silently answers fewer patients than planned and is a little less accurate, and
  with RF + KNN it answers more than planned.
* Possible explanation (hypothesis, not tested): median-imputed values pull predicted probabilities toward the middle, lowering confidence.

## 6. Probability reliability (Fig. 6, pooled test predictions, median imputation)
Brier score rises with missingness (LR 0.116 → 0.148 → 0.156 at 0/30/50%; RF 0.124 → 0.159 → 0.168). See Fig. 6 for the reliability curves; with 303 patients the bins are noisy,
so treat them as descriptive.

## 7. Error analysis — representative errors (Fig. 7, `error_analysis.md`)
Pooled over the 15 runs, matched scenario, selective models (trained on 75% of each training fold).
* **Most errors are missed disease, and they increase with missing data.** For LR + median the false-negative rate (diseased patients predicted healthy) is 0.204 at 0%, 0.276 at 30% and 0.295 at 50%;
  the false-positive rate is 0.120, 0.148 and 0.177. Missed disease makes up 52–61% of all errors across the four configurations.
* **Abstention preferentially skips wrong predictions, but misses most of them.** LR + median at 0% missing skips 44% of its wrong predictions versus 19% of its correct ones
  (RF: 46% vs 16%); at 50% the figures are 40% vs 15% (RF: 38% vs 15%). Still, more than half of the wrong predictions are answered. Error rate among skipped cases is 0.31 (0%) to 0.44 (50%)
  for LR + median, against 0.11–0.18 among answered cases.
* **Some patients are wrong every time.** At 0% missing, 33 patients (10.9%) are misclassified in all 3 of their test appearances by LR + median (RF + median: 37, 12.2%);
  at 50% missing 23 (7.6%) and 21 (6.9%). About 64% of the always-wrong patients truly have the disease. They are skipped in only 24% (LR) / 31% (RF) of their appearances,
  versus 14% / 13% for patients that are never wrong: the persistent mistakes are mostly **confident** mistakes that abstention does not catch.
* **What the confident mistakes look like (LR + median, 0% missing, 8 most confident).** Five are patients with disease who look low-risk on every recorded feature (no exercise angina,
  no coloured vessels, normal thal, Oldpeak 0; predicted probability of disease 0.04–0.14 for all five); three are healthy patients with a reversible thal defect and 1–3 coloured vessels
  (predicted probability 0.90–0.94). None of the eight was ever skipped.
* **By group (descriptive; groups are small).** Typical-angina patients (TA, n = 23) have the highest error rate (0.36 at 0% for LR + median, versus 0.12–0.16 for the other chest-pain types);
  male patients have a higher error rate than female patients (0.19 vs 0.10 at 0%; 0.26 vs 0.17 at 50%); error rates rise with age band at 0% (0.12 under 50, 0.17 for 50–59, 0.18 for 60+).
  With these sample sizes none of these differences should be over-interpreted.

## Limitations to state in the paper
* Small data set (303 patients, ~61 per test fold); results vary by fold. Repeated CV runs are not independent, and SDs are not confidence intervals.
* Missingness is simulated, completely at random (MCAR). Real clinical missingness is usually not random, so conclusions about MCAR need not transfer.
* Cleveland is a single-site sample with no external validation. The binary target merges disease severities (original labels 0–4).
* Only two model families and two imputers; tuning grids are small (4 settings each). Categorical variables are one-hot encoded with NaN propagated.
* The selective-prediction models are fitted on 75% of each training fold (25% held out to set the threshold), so they use less data than the models in Section 1.
* The error analysis pools repeated predictions of the same patients, and its group comparisons rest on small groups.
* Research exercise only, not a clinical decision tool.
