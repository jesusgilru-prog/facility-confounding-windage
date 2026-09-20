"""Robustness check: does weighting the fit by inverse reported uncertainty
(1/error_pct^2) change the LOFO conclusion? The corpus carries a per-point
error_pct column (median 5%, up to 83.1% on one Zheng2024 point) that every
other fit in this paper ignores (unweighted OLS). This is the check flagged
as "cheap, would close the objection" by external review (revisar-paper,
round 15, 2026-08-19), previously declared as unexecuted future work in
Limitations (8) -- executed here for real.

Weighted least squares in log space: minimize sum_i w_i * (y_i - Xb_i)^2,
w_i = 1/error_pct_i^2 (a point reported as 2x more uncertain gets 4x less
weight). Same structure M6, same strict leave-one-facility-out protocol,
same decision rule as the rest of this paper.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
FEATURES = ["Re_Omega", "Pi_gap", "Pi_confinement", "Pi_aspect_axial"]

d = df.dropna(subset=FEATURES + ["Cp", "source", "error_pct"]).copy()
d = d[d["Cp"] > 0]
w_all = 1.0 / (d["error_pct"].values ** 2)

facilities = sorted(d["source"].unique())
per_facility = {}
all_true, all_pred = [], []
for held in facilities:
    train = d[d["source"] != held]
    test = d[d["source"] == held]
    Xtr = np.column_stack([np.ones(len(train))] + [np.log(train[f]) for f in FEATURES])
    ytr = np.log(train["Cp"].values)
    w = 1.0 / (train["error_pct"].values ** 2)
    sw = np.sqrt(w)
    Xte = np.column_stack([np.ones(len(test))] + [np.log(test[f]) for f in FEATURES])
    yte = np.log(test["Cp"].values)

    def resid(beta):
        return sw * (Xtr @ beta - ytr)

    sol = least_squares(resid, np.zeros(Xtr.shape[1]))
    pred = Xte @ sol.x
    r2 = 1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)
    per_facility[held] = float(r2)
    all_true.append(yte)
    all_pred.append(pred)

y_all = np.concatenate(all_true)
p_all = np.concatenate(all_pred)
pooled_r2 = float(1 - np.sum((y_all - p_all) ** 2) / np.sum((y_all - y_all.mean()) ** 2))

# Also: LOFO excluding the single worst-uncertainty point (Zheng2024, 83.1%
# error) entirely, as a second, simpler robustness variant.
d_excl = d[d["error_pct"] < 83.0]
facilities2 = sorted(d_excl["source"].unique())
per_facility_excl = {}
all_true2, all_pred2 = [], []
for held in facilities2:
    train = d_excl[d_excl["source"] != held]
    test = d_excl[d_excl["source"] == held]
    Xtr = np.column_stack([np.ones(len(train))] + [np.log(train[f]) for f in FEATURES])
    ytr = np.log(train["Cp"].values)
    Xte = np.column_stack([np.ones(len(test))] + [np.log(test[f]) for f in FEATURES])
    yte = np.log(test["Cp"].values)

    def resid2(beta):
        return Xtr @ beta - ytr

    sol = least_squares(resid2, np.zeros(Xtr.shape[1]))
    pred = Xte @ sol.x
    r2 = 1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)
    per_facility_excl[held] = float(r2)
    all_true2.append(yte)
    all_pred2.append(pred)

y_all2 = np.concatenate(all_true2)
p_all2 = np.concatenate(all_pred2)
pooled_r2_excl = float(1 - np.sum((y_all2 - p_all2) ** 2) / np.sum((y_all2 - y_all2.mean()) ** 2))

results = {
    "n": int(len(d)),
    "weighted_1_over_error_pct_sq": {
        "per_facility_r2": per_facility,
        "pooled_r2": pooled_r2,
    },
    "unweighted_baseline_pooled_r2_for_reference": -0.885,
    "excluding_worst_uncertainty_point_error_pct_ge_83": {
        "n_excluded": int((d["error_pct"] >= 83.0).sum()),
        "per_facility_r2": per_facility_excl,
        "pooled_r2": pooled_r2_excl,
    },
}

out_path = _ROOT + "/results/weighted_by_uncertainty_lofo_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
