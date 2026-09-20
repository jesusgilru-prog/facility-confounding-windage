"""Sensitivity check: does the paper's headline LOFO failure survive if
restricted to rows that are direct physical measurements, excluding every
row that is a CFD simulation output (all 45 Guo2024 rows -- confirmed by
reading the source paper's Section 2.3/2.4 and Figure 7 caption: "We
simulated the torque resistance..."; CHIEF itself is not built) or
reconstructed from a fitted constant rather than a direct reading (5
Liu2024 rows, data_origin=="computed_from_fitted_constant")?

This leaves 64 rows across 3 facilities (Vrancik1968 n=41, Liu2024 n=15,
Zheng2024 n=8). With only 3 facilities, the same 4-predictor structure
(Re_Omega, Pi_gap, Pi_confinement, Pi_aspect_axial) cannot be fit with
LOFO the same way (one fold would use only 2 facilities' worth of
variation to fit 5 coefficients including the intercept, which is
underdetermined/unstable for some facility pairs); this script fits what
is actually identifiable and reports it honestly rather than forcing the
same structure and hiding instability.

Triggered by external review (revisar-paper, round 13, 2026-08-19)
flagging that "114 measurements" overstates how much of the corpus is a
direct physical reading.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
FEATURES = ["Re_Omega", "Pi_gap", "Pi_confinement", "Pi_aspect_axial"]

phys = df[(df["source"] != "Guo2024") & (df["data_origin"] != "computed_from_fitted_constant")].copy()
phys = phys.dropna(subset=FEATURES + ["Cp", "source"])
phys = phys[phys["Cp"] > 0]

results = {"n_total": int(len(phys)), "facility_counts": phys["source"].value_counts().to_dict()}

# Full 4-predictor LOFO across the 3 remaining facilities.
facilities = sorted(phys["source"].unique())
per_facility = {}
all_true, all_pred = [], []
rank_deficient = []
for held in facilities:
    train = phys[phys["source"] != held]
    test = phys[phys["source"] == held]
    Xtr = np.column_stack([np.ones(len(train))] + [np.log(train[f]) for f in FEATURES])
    ytr = np.log(train["Cp"].values)
    rank = np.linalg.matrix_rank(Xtr)
    if rank < Xtr.shape[1]:
        rank_deficient.append(held)
        continue
    Xte = np.column_stack([np.ones(len(test))] + [np.log(test[f]) for f in FEATURES])
    yte = np.log(test["Cp"].values)

    def resid(beta):
        return Xtr @ beta - ytr

    sol = least_squares(resid, np.zeros(Xtr.shape[1]))
    pred = Xte @ sol.x
    r2 = 1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)
    per_facility[held] = float(r2)
    all_true.append(yte)
    all_pred.append(pred)

results["four_predictor_lofo"] = {
    "rank_deficient_folds_skipped": rank_deficient,
    "per_facility_r2": per_facility,
}
if all_true:
    y_all = np.concatenate(all_true)
    p_all = np.concatenate(all_pred)
    results["four_predictor_lofo"]["pooled_r2"] = float(
        1 - np.sum((y_all - p_all) ** 2) / np.sum((y_all - y_all.mean()) ** 2)
    )

# Reynolds-only LOFO (identifiable in every fold, 1 predictor + intercept).
per_facility_re = {}
all_true_re, all_pred_re = [], []
for held in facilities:
    train = phys[phys["source"] != held]
    test = phys[phys["source"] == held]
    Xtr = np.column_stack([np.ones(len(train)), np.log(train["Re_Omega"])])
    ytr = np.log(train["Cp"].values)
    Xte = np.column_stack([np.ones(len(test)), np.log(test["Re_Omega"])])
    yte = np.log(test["Cp"].values)

    def resid(beta):
        return Xtr @ beta - ytr

    sol = least_squares(resid, np.zeros(2))
    pred = Xte @ sol.x
    r2 = 1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)
    per_facility_re[held] = float(r2)
    all_true_re.append(yte)
    all_pred_re.append(pred)

y_all_re = np.concatenate(all_true_re)
p_all_re = np.concatenate(all_pred_re)
results["reynolds_only_lofo"] = {
    "per_facility_r2": per_facility_re,
    "pooled_r2": float(1 - np.sum((y_all_re - p_all_re) ** 2) / np.sum((y_all_re - y_all_re.mean()) ** 2)),
}

out_path = _ROOT + "/results/physical_only_sensitivity_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
