"""
Shared code for the missing-data / selective-prediction study (ES 678).

Everything that touches data goes through the functions below so that the
leakage controls are explicit and testable:

  * missingness is injected AFTER the train/test split, with independent random
    masks for the training part and the test part (add_missing + mask_rng);
  * scaling and imputation live inside an sklearn Pipeline, so they are fitted
    on training rows only (make_pipeline);
  * hyper-parameters are tuned by an inner cross-validation on the training
    rows only (tune), and the abstention threshold is chosen from validation
    rows only (choose_threshold); both are then applied, unchanged, to the test rows.
"""
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

# ----------------------------------------------------------------------------
# Global settings (all documented in README.md)
# ----------------------------------------------------------------------------
MASTER_SEED = 42
N_SPLITS = 5            # stratified K-fold
N_REPEATS = 3           # repeats of the whole K-fold procedure
MISSING_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # fraction of feature cells removed
VAL_FRACTION = 0.25     # share of each outer-training fold used as validation
TARGET_COVERAGE = 0.80  # validation coverage used to choose the abstention threshold
KNN_NEIGHBORS = 5
TUNE_FOLDS = 3          # inner stratified CV used for hyper-parameter tuning (training rows only)
TUNE_RF_TREES = 150     # trees used while searching; the chosen model is refitted with 300
# Candidate settings. Listed from the simplest / most regularised to the most flexible;
# ties in the inner-CV score are resolved in favour of the first (simplest) entry.
TUNE_GRIDS = {
    "LR": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "RF": [{"min_samples_leaf": m} for m in (10, 5, 3, 1)],
}
COVERAGE_GRID = np.round(np.arange(0.20, 1.0001, 0.05), 2)  # risk-coverage curves

MODELS = ["LR", "RF"]
IMPUTERS = ["median", "knn"]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT, "data_raw", "processed.cleveland.data")   # study data: all 13 Cleveland features
APP_CSV = os.path.join(ROOT, "heart.csv")                                  # 11-feature copy used only by the demo app
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figures")

# ----------------------------------------------------------------------------
# Data and fixed (label-free) encoding
# ----------------------------------------------------------------------------
TARGET = "HeartDisease"   # 1 = disease present (original label > 0)
RAW_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
               "exang", "oldpeak", "slope", "ca", "thal", "num"]          # UCI column order
NUMERIC = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "NumVessels"]
BINARY = {"Sex": {"F": 0, "M": 1}, "ExerciseAngina": {"N": 0, "Y": 1}}
BINARY_NUMERIC = ["FastingBS"]  # already 0/1
CATEGORICAL = {
    "ChestPainType": ["ASY", "ATA", "NAP", "TA"],
    "RestingECG": ["LVH", "Normal", "ST"],
    "ST_Slope": ["Down", "Flat", "Up"],
    "Thal": ["fixed", "normal", "reversible"],
}
RAW_FEATURES = NUMERIC + list(BINARY) + BINARY_NUMERIC + list(CATEGORICAL)  # 13 columns


def load_data(path=DATA_PATH):
    """Cleveland subset of the UCI Heart Disease data: 303 patients, 13 features.

    Coding follows the UCI documentation (heart-disease.names). '?' entries are real
    missing values (4 in `ca`, 2 in `thal`) and are kept as NaN. Target: 1 if num > 0.
    """
    raw = pd.read_csv(path, header=None, names=RAW_COLUMNS, na_values="?")
    X = pd.DataFrame({
        "Age": raw.age.astype(float),
        "Sex": raw.sex.map({1: "M", 0: "F"}),
        "ChestPainType": raw.cp.map({1: "TA", 2: "ATA", 3: "NAP", 4: "ASY"}),
        "RestingBP": raw.trestbps.astype(float),
        "Cholesterol": raw.chol.astype(float),
        "FastingBS": raw.fbs.astype(float),
        "RestingECG": raw.restecg.map({0: "Normal", 1: "ST", 2: "LVH"}),
        "MaxHR": raw.thalach.astype(float),
        "ExerciseAngina": raw.exang.map({0: "N", 1: "Y"}),
        "Oldpeak": raw.oldpeak.astype(float),
        "ST_Slope": raw.slope.map({1: "Up", 2: "Flat", 3: "Down"}),
        "NumVessels": raw.ca.astype(float),                       # `ca`: major vessels coloured by fluoroscopy (0-3)
        "Thal": raw.thal.map({3.0: "normal", 6.0: "fixed", 7.0: "reversible"}),
    })[RAW_FEATURES]
    y = (raw.num > 0).astype(int).to_numpy()
    return X, y


def encode(X):
    """Fixed mapping raw columns -> numeric matrix. NaN stays NaN.

    The category lists are part of the data schema (not learned from labels or
    from any particular fold), so this step cannot leak information.
    A missing categorical value becomes NaN in *all* of its one-hot columns.
    """
    out = {}
    for c in NUMERIC:
        out[c] = X[c].astype(float).to_numpy()
    for c, mapping in BINARY.items():
        out[c] = X[c].map(mapping).astype(float).to_numpy()
    for c in BINARY_NUMERIC:
        out[c] = X[c].astype(float).to_numpy()
    for c, cats in CATEGORICAL.items():
        miss = X[c].isna().to_numpy()
        for k in cats:
            v = (X[c] == k).astype(float).to_numpy()
            v[miss] = np.nan
            out[f"{c}_{k}"] = v
    return pd.DataFrame(out, index=X.index)


# ----------------------------------------------------------------------------
# Missing-data simulation (MCAR)
# ----------------------------------------------------------------------------
def mask_rng(*keys):
    """Independent, reproducible random stream for a given tuple of keys."""
    return np.random.default_rng(np.random.SeedSequence([MASTER_SEED, *keys]))


def add_missing(X, rate, rng):
    """Set each cell of the RAW feature table to NaN independently with prob. `rate`.
    Cells that are already missing in the original data stay missing."""
    X = X.copy()
    if rate <= 0:
        return X
    mask = rng.random(X.shape) < rate
    return X.mask(mask)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def make_pipeline(model, imputer, seed=MASTER_SEED, params=None, n_trees=300):
    """scaler -> imputer -> classifier, all fitted on training rows only.

    The scaler comes first (it ignores NaN when fitting) so that the KNN
    imputer measures distances on comparable scales.
    `params` holds the tuned hyper-parameters (LR: C; RF: min_samples_leaf).
    """
    params = params or {}
    imp = SimpleImputer(strategy="median") if imputer == "median" else KNNImputer(n_neighbors=KNN_NEIGHBORS)
    if model == "LR":
        clf = LogisticRegression(C=params.get("C", 1.0), max_iter=2000)
    elif model == "RF":
        clf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=params.get("min_samples_leaf", 3),
                                     random_state=seed, n_jobs=1)
    else:
        raise ValueError(model)
    return Pipeline([("scale", StandardScaler()), ("impute", imp), ("clf", clf)])


def tune(model, imputer, X_enc, y, seed):
    """Choose hyper-parameters by an inner stratified CV on the given (training) rows ONLY.

    Criterion: mean Brier score on the inner validation folds (a proper scoring rule, so it
    rewards good probabilities, which the confidence-based abstention relies on). The scaler
    and imputer are re-fitted inside every inner fold because they are part of the pipeline.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import brier_score_loss
    skf = StratifiedKFold(TUNE_FOLDS, shuffle=True, random_state=seed)
    best_score, best_params = None, None
    for params in TUNE_GRIDS[model]:
        scores = []
        for a, b in skf.split(X_enc, y):
            pipe = make_pipeline(model, imputer, params=params, n_trees=TUNE_RF_TREES).fit(X_enc.iloc[a], y[a])
            scores.append(brier_score_loss(y[b], pipe.predict_proba(X_enc.iloc[b])[:, 1]))
        score = float(np.mean(scores))
        if best_score is None or score < best_score - 1e-12:      # strict: ties keep the simpler setting
            best_score, best_params = score, params
    return best_params


def make_majority():
    """Majority-class baseline (predicts the most common training label)."""
    return DummyClassifier(strategy="prior")


# ----------------------------------------------------------------------------
# Selective prediction helpers
# ----------------------------------------------------------------------------
def confidence(p1):
    """Confidence in the predicted class: max(p, 1-p)."""
    return np.maximum(p1, 1.0 - p1)


def choose_threshold(conf_val, target_coverage=TARGET_COVERAGE):
    """Abstention threshold using VALIDATION confidences only.

    tau is the (1 - target_coverage) quantile, so about `target_coverage` of the
    validation cases have confidence >= tau.
    """
    return float(np.quantile(conf_val, 1.0 - target_coverage))


def risk_coverage_curve(conf, errors, grid=COVERAGE_GRID):
    """Selective risk when only the most confident `c` fraction is answered (descriptive)."""
    order = np.argsort(-conf, kind="stable")
    e = np.asarray(errors, dtype=float)[order]
    n = len(e)
    out = []
    for c in grid:
        k = max(1, int(np.ceil(c * n)))
        out.append(e[:k].mean())
    return np.array(out)
