"""Step 1: data audit for the Cleveland data (all 13 features; no modelling here)."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

X, yv = load_data()
df = X.copy(); df[TARGET] = yv
y = df[TARGET]

# severity levels in the original label (num = 0..4); the study collapses them to 0 / 1
raw = pd.read_csv(DATA_PATH, header=None, na_values="?", names=RAW_COLUMNS)
native = X.isna().sum(); native = native[native > 0]

audit = {
    "source_file": "data_raw/processed.cleveland.data (UCI Heart Disease, Cleveland subset)",
    "rows": int(len(df)),
    "feature_columns": RAW_FEATURES,
    "n_features": len(RAW_FEATURES),
    "class_counts": {"no_disease(0)": int((y == 0).sum()), "disease(1)": int((y == 1).sum())},
    "original_severity_counts_num_0_to_4": {int(k): int(v) for k, v in raw.num.value_counts().sort_index().items()},
    "majority_class_share": float(max((y == 0).mean(), (y == 1).mean())),
    "native_missing_cells_by_feature": {k: int(v) for k, v in native.items()},
    "native_missing_cells_total": int(X.isna().sum().sum()),
    "native_missing_cell_rate": float(X.isna().to_numpy().mean()),
    "rows_with_any_native_missing": int(X.isna().any(axis=1).sum()),
    "zero_cholesterol_rows": int((df["Cholesterol"] == 0).sum()),
    "zero_restingbp_rows": int((df["RestingBP"] == 0).sum()),
    "numeric_summary": df[NUMERIC].describe().round(2).to_dict(),
}
with open(os.path.join(RESULTS_DIR, "data_audit.json"), "w") as f:
    json.dump(audit, f, indent=2)
print(json.dumps({k: v for k, v in audit.items() if k != "numeric_summary"}, indent=2))

# ---- Figure: class balance + feature distributions by class -----------------
fig, axes = plt.subplots(1, 4, figsize=(11, 2.7))
counts = y.value_counts().sort_index()
axes[0].bar(["No disease", "Disease"], counts.values, color=["#4C72B0", "#C44E52"])
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 2, str(v), ha="center", fontsize=9)
axes[0].set_title("Class balance (n=%d)" % len(df), fontsize=10)
axes[0].set_ylabel("Patients")
for ax, col in zip(axes[1:], ["Age", "MaxHR", "Oldpeak"]):
    for lab, color, name in [(0, "#4C72B0", "No disease"), (1, "#C44E52", "Disease")]:
        ax.hist(df.loc[y == lab, col], bins=15, alpha=0.6, color=color, label=name)
    ax.set_title(col, fontsize=10)
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_data_audit.png"), dpi=200)
fig.savefig(os.path.join(FIG_DIR, "fig1_data_audit.pdf"))
print("saved fig1")
