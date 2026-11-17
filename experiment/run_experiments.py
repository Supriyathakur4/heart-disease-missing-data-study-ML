"""
Steps 2-4: run every experiment and save fold-level results.

For each of the 15 outer folds (5-fold stratified CV x 3 repeats) and each
missing-data level:

  matched scenario     training rows and test rows both get independent MCAR masks
  clean_train scenario training rows are clean, only the test rows get masked
                       (failure-mode audit: does a threshold chosen on clean data
                       still behave on incomplete test data?)

  perf       : hyper-parameters tuned by inner 3-fold CV on the outer-training fold,
               model refitted on the whole outer-training fold, scored on the test fold
  selective  : the outer-training fold is split 75% / 25%; hyper-parameters are tuned by inner CV on
               the 75% part only; the abstention threshold is chosen on the 25% validation part so that
               ~80% of validation cases are answered. The test fold is never used for any choice.

The run is resumable: every finished outer fold is cached in results/_cache, so it can be
interrupted and restarted (`--max-seconds N` stops cleanly after the fold in progress).
"""
import argparse, json, os, pickle, platform, shutil, sys, time
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score, brier_score_loss
from common import *

PRED_LEVELS = {0.0, 0.3, 0.5}      # levels whose per-patient predictions are saved (reliability + error analysis)
CACHE_DIR = os.path.join(RESULTS_DIR, "_cache")


def perf_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(
        bal_acc=balanced_accuracy_score(y, pred),
        recall=recall_score(y, pred, zero_division=0),
        auc=roc_auc_score(y, p),
        brier=brier_score_loss(y, p),
    )


def selective_metrics(p, y, tau):
    pred = (p >= 0.5).astype(int)
    err = (pred != y)
    conf = confidence(p)
    keep = conf >= tau
    rc = risk_coverage_curve(conf, err)
    row = dict(
        tau=tau,
        coverage=keep.mean(),
        sel_risk=err[keep].mean() if keep.any() else np.nan,   # error among answered cases
        risk_all=err.mean(),                                    # error if nothing is skipped (same model)
        brier_all=brier_score_loss(y, p),
        brier_answered=brier_score_loss(y[keep], p[keep]) if keep.any() else np.nan,
    )
    for c, r in zip(COVERAGE_GRID, rc):
        row[f"rc_{int(round(c * 100))}"] = r
    return row


def tune_seed(rep, fold, li, purpose):
    """Seed of the inner CV; purpose 0 = tuning for the 'perf' model, 1 = tuning for the 'selective' model."""
    return int(mask_rng(rep, fold, li, 10 + purpose).integers(0, 2**31 - 1))


def fit_tuned(model, imp, X_enc, y, seed, tune_log, meta):
    """Tune on (X_enc, y) only, refit with the chosen setting, and log the choice."""
    params = tune(model, imp, X_enc, y, seed)
    tune_log.append(dict(meta, C=params.get("C", np.nan), min_samples_leaf=params.get("min_samples_leaf", np.nan)))
    return make_pipeline(model, imp, params=params).fit(X_enc, y)


def run_fold(rep, fold, tr_idx, te_idx, X, y, levels=None):
    levels = MISSING_LEVELS if levels is None else levels
    perf_rows, sel_rows, pred_rows, selpred_rows, tune_log = [], [], [], [], []
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]

    # inner split of the outer-training fold: fit part / validation part
    inner_seed = int(mask_rng(rep, fold, 999).integers(0, 2**31 - 1))
    itr, iva = train_test_split(np.arange(len(tr_idx)), test_size=VAL_FRACTION,
                                stratify=y_tr, random_state=inner_seed)

    base = dict(repeat=rep, fold=fold)

    # ---- majority-class baseline (ignores the features, so the same at every level) ----
    maj = make_majority().fit(np.zeros((len(y_tr), 1)), y_tr)
    p_maj = maj.predict_proba(np.zeros((len(y_te), 1)))[:, 1]
    for rate in levels:
        perf_rows.append(dict(base, scenario="matched", model="Majority", imputer="-", level=rate,
                              **perf_metrics(y_te, p_maj)))

    # ---- clean-training fits (needed once per fold, reused for every test-missing level) ----
    clean = {}
    Xtr_enc_clean = encode(X_tr)
    for model in MODELS:
        for imp in IMPUTERS:
            meta = dict(base, scenario="clean_train", model=model, imputer=imp, level=0.0)
            full = fit_tuned(model, imp, Xtr_enc_clean, y_tr, tune_seed(rep, fold, 99, 0), tune_log, dict(meta, purpose="perf"))
            inner = fit_tuned(model, imp, Xtr_enc_clean.iloc[itr], y_tr[itr], tune_seed(rep, fold, 99, 1), tune_log, dict(meta, purpose="selective"))
            conf_val = confidence(inner.predict_proba(Xtr_enc_clean.iloc[iva])[:, 1])
            clean[(model, imp)] = (full, inner, choose_threshold(conf_val))

    for rate in levels:
        li = MISSING_LEVELS.index(rate)
        # independent random streams for the training mask and the test mask
        X_tr_m = add_missing(X_tr, rate, mask_rng(rep, fold, li, 1))
        X_te_m = add_missing(X_te, rate, mask_rng(rep, fold, li, 2))
        Xtr_enc, Xte_enc = encode(X_tr_m), encode(X_te_m)

        for model in MODELS:
            for imp in IMPUTERS:
                key = dict(base, model=model, imputer=imp, level=rate)

                # ----- matched: train and test both incomplete -----
                meta = dict(key, scenario="matched")
                full = fit_tuned(model, imp, Xtr_enc, y_tr, tune_seed(rep, fold, li, 0), tune_log, dict(meta, purpose="perf"))
                p = full.predict_proba(Xte_enc)[:, 1]
                perf_rows.append(dict(key, scenario="matched", **perf_metrics(y_te, p)))
                if rate in PRED_LEVELS:
                    for i, (yy, pp) in enumerate(zip(y_te, p)):
                        pred_rows.append(dict(key, row=int(te_idx[i]), y=int(yy), p=float(pp)))

                inner = fit_tuned(model, imp, Xtr_enc.iloc[itr], y_tr[itr], tune_seed(rep, fold, li, 1), tune_log, dict(meta, purpose="selective"))
                conf_val = confidence(inner.predict_proba(Xtr_enc.iloc[iva])[:, 1])
                tau = choose_threshold(conf_val)
                p_sel = inner.predict_proba(Xte_enc)[:, 1]
                sel_rows.append(dict(key, scenario="matched", val_coverage=float((conf_val >= tau).mean()),
                                     **selective_metrics(p_sel, y_te, tau)))
                if rate in PRED_LEVELS:
                    cf = confidence(p_sel)
                    for i in range(len(y_te)):
                        selpred_rows.append(dict(key, row=int(te_idx[i]), y=int(y_te[i]), p=float(p_sel[i]),
                                                 conf=float(cf[i]), accepted=int(cf[i] >= tau)))

                # ----- clean_train: clean training + clean validation threshold, incomplete test -----
                full_c, inner_c, tau_c = clean[(model, imp)]
                p_c = full_c.predict_proba(Xte_enc)[:, 1]
                perf_rows.append(dict(key, scenario="clean_train", **perf_metrics(y_te, p_c)))
                sel_rows.append(dict(key, scenario="clean_train", val_coverage=TARGET_COVERAGE,
                                     **selective_metrics(inner_c.predict_proba(Xte_enc)[:, 1], y_te, tau_c)))
    return perf_rows, sel_rows, pred_rows, selpred_rows, tune_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=None, help="stop cleanly after the fold in progress once this time is used (resume by re-running)")
    ap.add_argument("--fresh", action="store_true", help="ignore cached folds")
    args = ap.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.fresh and os.path.isdir(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    X, y = load_data()
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=MASTER_SEED)
    t0 = time.time()
    total = N_SPLITS * N_REPEATS
    outs = [[], [], [], [], []]
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        rep, fold = divmod(k, N_SPLITS)
        f = os.path.join(CACHE_DIR, f"fold_{k:02d}.pkl")
        if os.path.exists(f):
            res = pickle.load(open(f, "rb"))
            print(f"outer fold {k + 1}/{total} loaded from cache", flush=True)
        else:
            if args.max_seconds is not None and time.time() - t0 > args.max_seconds:
                print(f"time budget reached before fold {k + 1}; run again to resume", flush=True)
                sys.exit(2)
            t1 = time.time()
            res = run_fold(rep, fold, tr, te, X, y)
            pickle.dump(res, open(f, "wb"))
            print(f"outer fold {k + 1}/{total} done ({time.time() - t1:.0f}s)", flush=True)
        for o, r in zip(outs, res):
            o += r

    names = ["perf_folds.csv", "selective_folds.csv", "predictions_pooled.csv.gz", "selective_predictions.csv.gz", "tuning_log.csv"]
    for o, n in zip(outs, names):
        pd.DataFrame(o).to_csv(os.path.join(RESULTS_DIR, n), index=False)

    meta = dict(
        master_seed=MASTER_SEED, n_splits=N_SPLITS, n_repeats=N_REPEATS, missing_levels=MISSING_LEVELS,
        val_fraction=VAL_FRACTION, target_coverage=TARGET_COVERAGE, knn_neighbors=KNN_NEIGHBORS,
        n_features=len(RAW_FEATURES), tuning=dict(inner_folds=TUNE_FOLDS, criterion="mean Brier score (lower is better)",
                                                  search_trees_RF=TUNE_RF_TREES, grids=TUNE_GRIDS),
        models={"LR": "LogisticRegression(C tuned, max_iter=2000)",
                "RF": "RandomForestClassifier(n_estimators=300, min_samples_leaf tuned, random_state=42)",
                "Majority": "DummyClassifier(strategy='prior')"},
        python=sys.version.split()[0], platform=platform.platform(),
        numpy=np.__version__, pandas=pd.__version__, scikit_learn=sklearn.__version__,
    )
    with open(os.path.join(RESULTS_DIR, "run_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    print("finished; all %d folds assembled (%.0fs in this session)" % (total, time.time() - t0))


if __name__ == "__main__":
    main()
