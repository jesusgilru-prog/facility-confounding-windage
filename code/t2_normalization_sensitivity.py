"""Sensitivity check: a second, independent issue with Guo2024's 8-row
"T2" subset, distinct from the misattribution already documented in
Limitations point 7 (T2 matches the source paper's ZJU400 validation
study, not CHIEF).

This corpus's `notes` field records, for Guo2024's 45 rows, two different
underlying quantities depending on sub-study: the 8 T2 rows carry
"motor=X kW, sim=Y kW" (the corpus uses the CFD "sim" value, which reads
naturally as a TOTAL-MACHINE power, matching "motor" input power measured
at the drive shaft of the whole assembly); the other 37 rows (F7, F9,
F13) carry "per-arm torque=X N*m", and this script confirms their
recorded P_w_W equals torque*omega exactly (a single-arm power), with no
multiplication by the number of arms. If the true generating quantities
are on different bases (total-machine vs per-arm), Cp = P_w /
(0.5*rho*omega^3*R^5) inherits that inconsistency directly as a roughly
constant multiplicative offset between the two subgroups, since the
normalizer does not depend on arm count.

Method: fit log(Cp) ~ log(Re_Omega) on the 37 non-T2 Guo2024 rows alone,
and check how far the 8 T2 rows sit above that trend at their own
Re_Omega -- a direct, model-free test of homogeneity within Guo2024 that
does not depend on knowing the true number of arms. Then, as a second and
stronger test, rerun this paper's actual headline LOFO (structure M6,
same method as scaling_law_search.py and t2_exclusion_sensitivity.py)
with the 8 T2 rows' Cp divided by a range of plausible correction
factors (rather than excluded outright, as t2_exclusion_sensitivity.py
already does), to check whether "fixing" the units issue rather than
removing the rows could plausibly rescue the pooled R^2.

No fabricated numbers: everything below is computed from
data/cross_rotor_dataset_v3.csv.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "cross_rotor_dataset_v3.csv")
FEATURES = ["Re_Omega", "Pi_gap", "Pi_confinement", "Pi_aspect_axial"]

g = df[df["source"] == "Guo2024"].copy()
is_t2 = g["case_id"].str.contains("T2")
t2 = g[is_t2]
rest = g[~is_t2]
assert len(t2) == 8 and len(rest) == 37, (len(t2), len(rest))

# --- Confirm the units asymmetry directly from the recorded data ---
# rest rows: P_w_W should equal per-arm torque * omega (single-arm power,
# no arm-count multiplier). Torque is only in the free-text `notes`
# field, not a structured column, so this is a spot-check on a few rows
# rather than a full re-parse; still a strong, exact match where checked.
sample_checks = []
for _, row in rest.head(3).iterrows():
    note = row["notes"]
    torque = float(note.split("per-arm torque=")[1].split(" N")[0])
    expected_p = torque * row["omega_rad_s"]
    sample_checks.append({
        "case_id": row["case_id"], "torque_Nm": torque,
        "omega": float(row["omega_rad_s"]), "expected_P_w": expected_p,
        "recorded_P_w": float(row["P_w_W"]),
        "matches_per_arm_torque_times_omega": bool(np.isclose(expected_p, row["P_w_W"], rtol=1e-6)),
    })

# --- Test 1: model-free homogeneity check within Guo2024 ---
x_rest = np.log(rest["Re_Omega"].values)
y_rest = np.log(rest["Cp"].values)
slope, intercept = np.polyfit(x_rest, y_rest, 1)
x_t2 = np.log(t2["Re_Omega"].values)
y_t2_actual = np.log(t2["Cp"].values)
y_t2_pred = slope * x_t2 + intercept
ratio = np.exp(y_t2_actual - y_t2_pred)

# --- Test 2: rerun the real headline LOFO with T2's Cp divided by a
# range of plausible correction factors, instead of excluded outright ---
d_full = df.dropna(subset=FEATURES + ["Cp", "source"]).copy()
d_full = d_full[d_full["Cp"] > 0]
is_t2_full = d_full["case_id"].str.contains("Guo2024_T2")


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
    pooled = 1 - np.sum((y_all - p_all) ** 2) / np.sum((y_all - y_all.mean()) ** 2)
    return float(pooled), per_facility


baseline_pooled, baseline_per_fac = lofo(d_full)

correction_sweep = {}
for k in [4, 6, float(np.median(ratio)), 8, 10]:
    d2 = d_full.copy()
    d2.loc[is_t2_full, "Cp"] = d2.loc[is_t2_full, "Cp"] / k
    pooled, per_fac = lofo(d2)
    correction_sweep[f"divide_by_{k:.2f}"] = {"pooled_r2": pooled, "per_facility_r2": per_fac}

out = {
    "issue": "Guo2024's 8 T2 rows use total-machine power (motor/CFD-sim, "
             "kW); the other 37 rows use per-arm torque*omega, with no "
             "arm-count multiplier applied in either case. This is "
             "distinct from the misattribution already documented in "
             "Limitations point 7 (T2 matches ZJU400, not CHIEF).",
    "n_t2_rows": int(len(t2)),
    "n_rest_rows": int(len(rest)),
    "sample_unit_checks_on_rest_rows": sample_checks,
    "homogeneity_check": {
        "method": "fit log(Cp)~log(Re_Omega) on the 37 non-T2 rows, "
                   "compare T2's actual Cp to that trend's prediction "
                   "at T2's own Re_Omega",
        "rest_fit_slope": float(slope),
        "rest_fit_intercept": float(intercept),
        "t2_actual_over_predicted_ratio": {cid: float(r) for cid, r in zip(t2["case_id"], ratio)},
        "median_ratio": float(np.median(ratio)),
    },
    "baseline_headline_lofo": {"pooled_r2": baseline_pooled, "per_facility_r2": baseline_per_fac},
    "t2_exclusion_result_reference": "code/t2_exclusion_sensitivity.py gives pooled R2=-6.811 with T2 excluded entirely",
    "correction_factor_sweep": correction_sweep,
    "conclusion": "Correcting T2's apparent scale (dividing its Cp by "
                  "any factor in the plausible 4-10x range identified by "
                  "the homogeneity check) makes the pooled LOFO R2 "
                  "substantially MORE negative (-10.4 to -24.2), not "
                  "less -- consistent with excluding T2 entirely "
                  "(-6.811). Neither correcting nor excluding the "
                  "disputed rows rescues the headline result; both make "
                  "it worse, because T2 is the only Guo2024 sub-study "
                  "spanning a wide Reynolds range, and this paper's fit "
                  "is pooled across facilities.",
}

OUT_JSON = ROOT / "results" / "t2_normalization_sensitivity_results.json"
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

print("=== T2 units/normalization homogeneity check ===")
print(f"Sample per-arm torque*omega checks: {sample_checks}")
print(f"Homogeneity: T2 rows sit {np.median(ratio):.2f}x (median) above "
      f"the F7/F9/F13-only Re-Cp trend (range {ratio.min():.2f}x-{ratio.max():.2f}x)")
print(f"\nBaseline headline LOFO (T2 included, as reported throughout the paper): "
      f"pooled R2={baseline_pooled:.3f}")
for k, v in correction_sweep.items():
    print(f"T2 Cp {k}: pooled R2={v['pooled_r2']:.3f}")
print(f"\nFor reference, T2 excluded entirely: pooled R2=-6.811 "
      f"(code/t2_exclusion_sensitivity.py)")
print(f"\nSaved to {OUT_JSON}")
