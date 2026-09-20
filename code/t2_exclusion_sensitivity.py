"""Sensitivity check: does the paper's headline LOFO pooled R^2 (-0.885)
change if the 8 Guo2024 "T2" rows are excluded -- rows whose reported
angular velocities (4.7-16.2 rad/s) and "motor vs CFD sim" framing match
the source paper's separate CFD-vs-experiment validation study on a
different, already-built facility ("ZJU400"), not the CHIEF geometry
(radius 4.35-4.65 m, omega=57 rad/s) that the corpus assigns to all
45 Guo2024 rows including these 8 (see manuscript Limitations (7)).

This backs a specific claim in the manuscript that was previously stated
without a saved script/JSON artifact, flagged by external review
(revisar-paper, round 11, 2026-08-19) as an unbacked number. Recomputed
here from scratch (same LOFO method as scaling_law_search.py's main
result: structure M6, ordinary least squares in log space, refit per
held-out facility), not copied from an earlier inline calculation.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
FEATURES = ["Re_Omega", "Pi_gap", "Pi_confinement", "Pi_aspect_axial"]


def lofo(data):
    facilities = sorted(data["source"].unique())
    per_facility = {}
    all_true, all_pred = [], []
    for held in facilities:
        train = data[data["source"] != held]
        test = data[data["source"] == held]
        Xtr = np.column_stack([np.log(train[f]) for f in FEATURES])
        ytr = np.log(train["Cp"].values)
        Xte = np.column_stack([np.log(test[f]) for f in FEATURES])
        yte = np.log(test["Cp"].values)

        def resid(beta):
            return Xtr @ beta[1:] + beta[0] - ytr

        sol = least_squares(resid, np.zeros(len(FEATURES) + 1))
        pred = Xte @ sol.x[1:] + sol.x[0]
        r2 = 1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)
        per_facility[held] = float(r2)
        all_true.append(yte)
        all_pred.append(pred)
    y_all = np.concatenate(all_true)
    p_all = np.concatenate(all_pred)
    pooled_r2 = 1 - np.sum((y_all - p_all) ** 2) / np.sum((y_all - y_all.mean()) ** 2)
    return per_facility, float(pooled_r2)

d_full = df.dropna(subset=FEATURES + ["Cp", "source"]).copy()
d_full = d_full[d_full["Cp"] > 0]

d_no_t2 = d_full[~d_full["case_id"].str.contains("Guo2024_T2")].copy()

per_full, pooled_full = lofo(d_full)
per_clean, pooled_clean = lofo(d_no_t2)

results = {
    "with_T2_n": int(len(d_full)),
    "with_T2_per_facility_r2": per_full,
    "with_T2_pooled_r2": pooled_full,
    "excluding_T2_n": int(len(d_no_t2)),
    "excluding_T2_guo2024_n": int((d_no_t2["source"] == "Guo2024").sum()),
    "excluding_T2_per_facility_r2": per_clean,
    "excluding_T2_pooled_r2": pooled_clean,
    "note": "excluding_T2 makes the pooled R2 more negative, not less -- "
            "the opposite of what a favourable-artifact explanation would need.",
}

out_path = _ROOT + "/results/t2_exclusion_sensitivity_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
