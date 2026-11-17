"""
Run:  python experiment/tests/test_leakage.py
Plain asserts (no pytest needed). Checks the leakage controls the paper claims.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import *
import run_experiments as rx
from sklearn.model_selection import StratifiedKFold

X, y = load_data()

# 0. data: 303 patients, 13 features, the only missing cells are the 6 original "?" entries (ca: 4, thal: 2)
assert X.shape == (303, 13) and len(y) == 303 and set(y) == {0, 1}
assert X.isna().sum().sum() == 6 and X.NumVessels.isna().sum() == 4 and X.Thal.isna().sum() == 2
print("ok 0: data loaded correctly (303 x 13, 6 native missing cells)")

# 1. injected missingness has the requested rate and is independent between streams
big = X.dropna().iloc[:290]   # complete rows only, so the measured rate is the injected rate
for rate in (0.1, 0.3, 0.5):
    m = add_missing(big, rate, mask_rng(1, 2, 3)).isna().to_numpy().mean()
    assert abs(m - rate) < 0.03, (rate, m)
m_train = add_missing(big, 0.3, mask_rng(0, 0, 1, 1)).isna().to_numpy()
m_test = add_missing(big, 0.3, mask_rng(0, 0, 1, 2)).isna().to_numpy()
assert (m_train != m_test).any(), "train and test masks must be different random streams"
assert not add_missing(big, 0.0, mask_rng(0)).isna().any().any()
print("ok 1: masks have the right rate and independent streams")

# 2. a missing categorical becomes NaN in ALL of its one-hot columns; observed ones stay one-hot
Xm = X.iloc[:5].copy()
Xm.loc[Xm.index[0], "ChestPainType"] = np.nan
E = encode(Xm)
cp_cols = [c for c in E.columns if c.startswith("ChestPainType_")]
assert E.loc[Xm.index[0], cp_cols].isna().all()
assert (E.loc[Xm.index[1:], cp_cols].sum(axis=1) == 1).all()
print("ok 2: encoding keeps NaN and is label-free")

# 3. scaler / imputer statistics come from the training rows only
tr, te = next(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
Xtr = encode(add_missing(X.iloc[tr], 0.3, mask_rng(5)))
Xte = encode(add_missing(X.iloc[te], 0.3, mask_rng(6)))
pipe = make_pipeline("LR", "median").fit(Xtr, y[tr])
assert np.allclose(pipe.named_steps["scale"].mean_, np.nanmean(Xtr.to_numpy(), axis=0))
before = pipe.named_steps["scale"].mean_.copy()
pipe.predict_proba(Xte)                      # predicting must not change learned statistics
assert np.array_equal(before, pipe.named_steps["scale"].mean_)
print("ok 3: preprocessing is fitted on training rows only")

# 4. END-TO-END: neither the abstention threshold NOR the tuned hyper-parameters depend on the test fold.
#    Scramble the test rows' labels and features, re-run one whole outer fold (one missing level, to keep it fast):
#    tau and every tuning choice must be identical (they may only depend on training/validation rows).
X2, y2 = X.copy(), y.copy()
rng = np.random.default_rng(0)
y2[te] = rng.permutation(y2[te])
X2.iloc[te] = X2.iloc[rng.permutation(te)].to_numpy()
_, sel_a, _, _, tune_a = rx.run_fold(0, 0, tr, te, X, y, levels=[0.3])
_, sel_b, _, _, tune_b = rx.run_fold(0, 0, tr, te, X2, y2, levels=[0.3])
assert [r["tau"] for r in sel_a] == [r["tau"] for r in sel_b], "tau changed when only the test fold changed -> leakage"
import pandas as pd
assert pd.DataFrame(tune_a).equals(pd.DataFrame(tune_b)), "tuning choices depend on the test fold -> leakage"
assert len(tune_a) == len(tune_b) == 16
assert [r["coverage"] for r in sel_a] != [r["coverage"] for r in sel_b], "sanity: test-side results should change when the test data change"
print("ok 4: tau and tuned hyper-parameters identical when test data are scrambled (no test-set leakage)")

# 5. tuning picks from the declared grid and only sees the rows it is given
Xtr5 = encode(X.iloc[tr]); params = tune("LR", "median", Xtr5, y[tr], seed=1)
assert params in TUNE_GRIDS["LR"]
Xs = Xtr5.copy(); ys = y[tr].copy()
assert tune("LR", "median", Xs, ys, seed=1) == params      # deterministic given the same training rows
print("ok 5: tuning is deterministic and chooses from the declared grid")
print("ALL LEAKAGE TESTS PASSED")
