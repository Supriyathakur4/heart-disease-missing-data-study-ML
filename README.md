# Robustness to Missing Data and Selective Prediction — Heart Disease (ES 678)

**Research question.** How does missing patient information at different levels affect the
reliability of machine-learning models for heart-disease prediction, and can selective
prediction (abstaining on uncertain cases) reduce errors while keeping useful coverage?

Author: Supriya Thakur (26210171). The study lives in `experiment/`. The repository also still contains
the original Streamlit/Flask demo app (last section), which is **not** part of the study.

## Reproduce everything (≈20 min, CPU only)

```bash
pip install -r requirements-experiment.txt     # numpy, pandas, scikit-learn, scipy, matplotlib, joblib
bash run_all.sh                                # tests -> data audit -> experiments -> tables/figures -> error analysis
```
The notebook `study_walkthrough.ipynb` shows the saved results without re-running anything (needs `pip install notebook` or the VS Code Jupyter extension).
Windows without bash: run the five commands inside `run_all.sh` one by one from the `experiment/` folder.
Google Colab: upload the folder, `!pip install -r requirements-experiment.txt`, `!bash run_all.sh`.
The experiment step caches every finished outer fold (`results/_cache`), so it can be interrupted and resumed.
Tested with Python 3.12 and the pinned package versions. Re-running an outer fold reproduces the stored results to floating-point precision
(largest difference 1e-16) with identical tuning choices; the full tuned run took about 16 minutes on one CPU core.

## Problem framing and data notes
**Question and claim.** How reliable are heart-disease predictions when patient information is missing, and can abstaining on
uncertain cases cut errors without leaving too many patients unanswered? Claims are limited to this data set and to simulated,
completely-at-random missingness.

**Unit and target.** One row = one patient (303 rows). Target: angiographic disease status. The UCI documentation defines the original
label `num` as 0 = no presence of heart disease and 1–4 = presence; the study uses 1 = presence (139 patients) vs 0 = absence (164 patients).

**When a prediction would be made and what is known then.** The setting is a patient assessed for suspected coronary artery disease.
The prediction is assumed to be made once the following information exists, and without the angiogram that defines the label:

| Stage | Features |
|---|---|
| Basic and resting measurements | Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG |
| Exercise stress test | MaxHR, ExerciseAngina, Oldpeak, ST_Slope |
| Imaging tests | NumVessels (`ca`, vessels coloured by fluoroscopy), Thal (`thal`: normal / fixed defect / reversible defect) |

Because the last two groups come from tests, the model is meant as decision support after this work-up, not for screening people who
have had no tests. That all three groups are available at prediction time is an assumption of this study.

**What "missing" means here.** A measurement that was not done, not recorded or not yet available. The experiment mimics this by removing
feature values at random (0–50% of cells). Real missingness is rarely random, which limits what the results can claim.

**Limits of the claim.** Single data set (Cleveland Clinic Foundation, documentation dated 1988), 303 patients, no external validation;
simulated MCAR missingness; the binary target merges disease severities; a research exercise, not a clinical tool.

**Source, access and credit.** Public UCI Machine Learning Repository data (https://archive.ics.uci.edu/dataset/45/heart+disease), free to
download. The creators (Janosi, Steinbrunn, Pfisterer, Detrano) ask that publications using the data credit the investigators; the standard citation is
Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new
probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64, 304–310. The documentation states that patient names and social
security numbers were replaced with dummy values. Donor: David W. Aha (1988).

**Data-quality checks done** (`experiment/data_audit.py`): 303 rows, 13 features; 6 missing cells in 6 patients (`ca`: 4, `thal`: 2), kept as missing;
no impossible zero values for cholesterol or blood pressure; class balance 164 / 139; original severity counts 164 / 55 / 36 / 35 / 13.

## Data
`data_raw/processed.cleveland.data` = Cleveland subset of the UCI Heart Disease data
(https://archive.ics.uci.edu/dataset/45/heart+disease): **303 patients**, one row = one patient,
**13 clinical features**, binary target (1 = disease present, original label > 0; the original label has
severity levels 0–4 with counts 164/55/36/35/13). 164 patients without and 139 with disease (majority share 54.1%).
Codings follow the UCI documentation (`heart-disease.names`).
The file has **6 genuinely missing cells** (`ca`: 4, `thal`: 2; 0.15% of all cells, in 6 patients). They are kept as missing and
handled by the imputers like every other missing value.
`heart.csv` is an 11-feature copy (no `ca`, `thal`) used only by the demo app; the study does not read it.

## Design (what the code does)
| Item | Choice |
|---|---|
| Models | Majority-class baseline; Logistic Regression; Random Forest (300 trees). |
| Imputation | Median; KNN (k=5) |
| Pipeline | StandardScaler → imputer → classifier, fitted on training rows only. The scaler is first so the KNN imputer measures distances on comparable scales (the scaler ignores NaN). |
| Encoding | Fixed, label-free: binary → 0/1, `ca` numeric, 4 multi-level columns → one-hot (22 numeric columns); a missing categorical value is NaN in all of its one-hot columns. |
| Tuning (inside training folds) | Inner stratified 3-fold CV on the training rows only, criterion = mean Brier score. LR: C ∈ {0.01, 0.1, 1, 10}; RF: min_samples_leaf ∈ {10, 5, 3, 1} (150 trees while searching, then refitted with 300). Ties keep the simpler setting. Scaler and imputer are re-fitted inside every inner fold. |
| Missing data | MCAR: each of the 13 feature cells is removed independently with probability 0, 10, 20, 30, 40, 50% *in addition to* the 6 original missing cells. Injected **after** the train/test split with **independent** random masks for training and test rows. |
| Validation | Stratified 5-fold CV repeated 3 times (15 runs), `MASTER_SEED=42`; per-cell mask seeds derived from (seed, repeat, fold, level, train/test). |
| Metrics | Balanced accuracy, recall (disease class), ROC-AUC, Brier score; for selective prediction: coverage, selective risk (error among answered cases), error with no abstention (same model), Brier on all/answered cases. |
| Selective prediction | Confidence = max(p, 1−p). Inside each outer-training fold 25% is held out (stratified) as validation; hyper-parameters are tuned and the model fitted on the other 75%; threshold τ = 20th percentile of validation confidence (≈80% validation coverage). The test fold is never used to choose τ or any hyper-parameter. |
| Scenarios | **matched**: training, validation and test all have the same missing rate (tuning and threshold are redone at every level). **clean_train** (failure-mode audit): training, tuning and validation are clean, only the test rows are masked; same test masks as *matched*, so the comparison is paired. |
| Depth analysis | Failure-mode audit (*clean_train*) + reliability diagrams / Brier score (`fig6`) + error analysis (`fig7`). |

## Leakage controls (tested in `experiment/tests/test_leakage.py`)
1. The data load correctly (303 × 13, exactly 6 original missing cells).
2. Masks have the requested rate; training and test masks come from different random streams.
3. Scaler/imputer statistics equal the training-row statistics and do not change when predicting on test rows.
4. End-to-end: scrambling the labels and features of the test fold and re-running a whole outer fold
   leaves every abstention threshold τ **and every tuned hyper-parameter** unchanged.
5. Tuning is deterministic and chooses only from the declared grid.

## Layout
```
experiment/common.py            settings, data loading/encoding, mask injection, pipelines, tuning, selective-prediction helpers
experiment/data_audit.py        Step 1   (-> results/data_audit.json, figures/fig1)
experiment/run_experiments.py   Steps 2-4 (-> results/perf_folds.csv, selective_folds.csv, tuning_log.csv, predictions_pooled.csv.gz, selective_predictions.csv.gz, run_metadata.json)
experiment/analyze.py           summaries + figures 2-6 (-> results/summary_*.csv, results_tables.md)
experiment/error_analysis.py    error types, who is skipped, persistent errors (-> results/error_analysis*.{md,csv}, figures/fig7)
experiment/tests/test_leakage.py
paper/es678_paper.tex (+ .pdf)    the 4-page ES 678 paper in the course LaTeX template (neurips_2026.sty is the unmodified course style file)
paper/make_paper_figures.py     redraws the two paper figures from results/ ;  paper/check_paper_numbers.py checks every number quoted in the paper against results/
study_walkthrough.ipynb         notebook (Jupyter / VS Code / Colab): loads the saved results, tables and figures; last cell re-runs everything
results/RESULTS_SUMMARY.md      what the numbers show + limitations (read this first)
results/results_tables.md       full tables (mean ± SD over 15 runs)
results/error_analysis.md       error-analysis tables
figures/                        fig1 data audit · fig2 performance vs missingness · fig3 selective prediction · fig4 risk-coverage
                                fig5 failure-mode audit · fig6 reliability · fig7 error analysis (PNG + PDF)
```

## Paper
`paper/` holds the ES 678 paper. Compile with `pdflatex es678_paper.tex` (twice) or upload the folder to Overleaf with `es678_paper.tex` as the main file.
`python paper/check_paper_numbers.py` confirms that every number in the paper matches the saved results (69 checks).

## Original demo app (unchanged code)
`streamlit run streamlit_app.py` (needs the packages in `requirements.txt`; `artifacts/model.joblib` was retrained on the
11-feature `heart.csv`). The app is not used by any experiment and its accuracy number is not a result of the study.
