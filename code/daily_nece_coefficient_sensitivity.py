"""Sensitivity check: does the Daily-Nece regime-IV median underprediction
factor (26.7x, reported throughout the manuscript using the leading
coefficient 0.051) change materially if the alternate coefficient value
0.0102 -- seen circulating in some secondary sources -- is used instead?

This backs a specific footnote-level claim in the manuscript
(Section on the Daily-Nece comparison method) that was previously stated
without a saved script/JSON artifact, flagged by external review
(revisar-paper, round 11, 2026-08-19) as an unbacked number. Recomputed
here from scratch against the real corpus, no numbers assumed.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)

Re = d["Re_Omega"].values
G = d["Pi_gap"].values
Cp_actual = d["Cp"].values

results = {}
for label, coef in [("coef_0.051", 0.051), ("coef_0.0102", 0.0102)]:
    Cp_pred = coef * (G ** (1 / 10)) * (Re ** (-0.2))
    ratio = Cp_actual / Cp_pred
    results[label] = {
        "coefficient": coef,
        "median_ratio": float(np.median(ratio)),
        "min_ratio": float(np.min(ratio)),
        "max_ratio": float(np.max(ratio)),
        "n": int(len(ratio)),
    }

results["ratio_of_medians_0.0102_vs_0.051"] = (
    results["coef_0.0102"]["median_ratio"] / results["coef_0.051"]["median_ratio"]
)

out_path = _ROOT + "/results/daily_nece_coefficient_sensitivity_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
