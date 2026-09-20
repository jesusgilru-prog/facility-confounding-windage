"""
Positive control for the paper's own LOFO decision rule, independent of
structure M6's fitted coefficients.

Motivation (raised independently by an external reviewer, DeepSeek, across
two rounds): the Monte Carlo simulation in angle_facility_power_analysis.py
generates its synthetic "true law" from M6's own fitted coefficients --- but
M6 is the same four-predictor structure already shown elsewhere in this
paper to be facility-confounded (three of its four predictors take only
1-8 distinct values per facility). That simulation therefore cannot show
that a GENUINELY facility-independent physical law would pass this paper's
LOFO decision rule; it only shows that this specific, already-suspect
structure would fail under a random facility offset. The manuscript's own
Limitations section (point 6) flags this explicitly as "a natural
extension we did not carry out here." This script carries it out.

Method: build a synthetic "ground truth" Cp from the REAL predictor values
in the actual 114-row corpus (same facility-clustering, collinearity, and
leverage structure as the real data -- no easier a covariate problem than
the real test), but with a generating law that is:
  (a) NOT fit to this corpus at all -- the single free parameter (the
      Reynolds exponent) is taken directly from Harmand et al. 2013's
      regime-IV Daily-Nece correlation (-0.2, already cited and discussed
      in Section 4/dn-cause of this paper), an external, independently
      published source;
  (b) genuinely facility-independent by construction -- no per-facility
      random offset is added (unlike the M6-based simulation), because a
      real universal law should not need one;
  (c) corrupted only by i.i.d. noise at the SAME within-facility residual
      scale already estimated from the real data
      (sigma2_hat = 0.198, from angle_facility_power_analysis.py Part A),
      so the noise-to-signal ratio is not more favourable than reality.

This paper's own strict LOFO decision rule (pooled R^2_log >= 0.3 AND
every held-out facility R^2 >= 0) is then applied to this synthetic
dataset, exactly as done throughout the paper, under two model
specifications:
  (1) correctly-specified: Cp ~ Re_Omega alone (the true generating
      predictor);
  (2) the paper's own M6 four-predictor structure (over-specified,
      includes three irrelevant/partially-degenerate predictors), to
      test whether the protocol still accepts a valid law even when
      offered the same, imperfect feature set used throughout the paper.

No fabricated numbers: everything below is computed from
data/cross_rotor_dataset_v3.csv.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "cross_rotor_dataset_v3.csv"
OUT_JSON = Path(__file__).resolve().parent.parent / "results" / "posctrl_results.json"

df = pd.read_csv(DATA)
facilities = sorted(df["source"].unique())
assert len(facilities) == 4, facilities

log_Re = np.log(df["Re_Omega"].values)
log_gap = np.log(df["Pi_gap"].values)
log_conf = np.log(df["Pi_confinement"].values)
log_asp = np.log(df["Pi_aspect_axial"].values)
n = len(df)

# --- Externally-sourced generating law (Harmand et al. 2013, regime-IV
# Daily-Nece exponent), NOT fit to this corpus. Intercept chosen only to
# center the synthetic log(Cp) near the real corpus's own range, which has
# no effect on R^2 (translation-invariant).
BETA_RE_TRUE = -0.2
INTERCEPT_TRUE = np.log(df["Cp"].values).mean() - BETA_RE_TRUE * log_Re.mean()

# Within-facility noise scale from the same corpus's own estimated
# variance component (angle_facility_power_analysis.py, Part A,
# sigma2_hat = MSW = 0.198; also cited in Limitations point 6).
SIGMA2_HAT = 0.198
SIGMA_HAT = np.sqrt(SIGMA2_HAT)

N_REPLICATES = 500


def decision_rule(pooled_r2, per_facility_r2):
    return (pooled_r2 >= 0.3) and all(r >= 0.0 for r in per_facility_r2)


def lofo_pooled_and_per_facility(y, X, fac_labels):
    """Strict LOFO: fit OLS on 3 facilities, predict held-out 4th, pool
    residuals across all held-out predictions for the pooled R^2, and also
    report each facility's own held-out R^2."""
    preds = np.full(n, np.nan)
    per_fac_r2 = {}
    for held_out in facilities:
        train_mask = fac_labels != held_out
        test_mask = ~train_mask
        Xtr = np.column_stack([np.ones(train_mask.sum()), X[train_mask]])
        ytr = y[train_mask]
        # least squares; guard against rank deficiency (constant columns
        # within the 3-facility training set, same as elsewhere in this
        # paper's code when the 4-predictor structure is used)
        coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        Xte = np.column_stack([np.ones(test_mask.sum()), X[test_mask]])
        yhat = Xte @ coef
        preds[test_mask] = yhat
        yte = y[test_mask]
        ss_res = np.sum((yte - yhat) ** 2)
        ss_tot = np.sum((yte - yte.mean()) ** 2)
        per_fac_r2[held_out] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    ss_res_pooled = np.sum((y - preds) ** 2)
    ss_tot_pooled = np.sum((y - y.mean()) ** 2)
    pooled_r2 = 1 - ss_res_pooled / ss_tot_pooled
    return pooled_r2, per_fac_r2


fac_labels = df["source"].values

rng = np.random.default_rng(0)
results = {"reynolds_only": [], "m6_four_predictor": []}

for rep in range(N_REPLICATES):
    noise = rng.normal(0.0, SIGMA_HAT, size=n)
    log_Cp_synth = INTERCEPT_TRUE + BETA_RE_TRUE * log_Re + noise

    # Specification 1: correctly-specified, Reynolds-only
    X1 = log_Re.reshape(-1, 1)
    pooled_r2_1, per_fac_1 = lofo_pooled_and_per_facility(log_Cp_synth, X1, fac_labels)
    results["reynolds_only"].append({
        "pooled_r2": pooled_r2_1,
        "per_facility_r2": per_fac_1,
        "passes_decision_rule": decision_rule(pooled_r2_1, list(per_fac_1.values())),
    })

    # Specification 2: this paper's own M6 four-predictor structure
    # (over-specified relative to the true law, exactly as used
    # throughout the rest of this paper)
    X2 = np.column_stack([log_Re, log_gap, log_conf, log_asp])
    pooled_r2_2, per_fac_2 = lofo_pooled_and_per_facility(log_Cp_synth, X2, fac_labels)
    results["m6_four_predictor"].append({
        "pooled_r2": pooled_r2_2,
        "per_facility_r2": per_fac_2,
        "passes_decision_rule": decision_rule(pooled_r2_2, list(per_fac_2.values())),
    })

summary = {}
for spec, recs in results.items():
    pooled = np.array([r["pooled_r2"] for r in recs])
    pass_rate = np.mean([r["passes_decision_rule"] for r in recs])
    per_fac_arrays = {f: np.array([r["per_facility_r2"][f] for r in recs]) for f in facilities}
    summary[spec] = {
        "median_pooled_r2": float(np.median(pooled)),
        "p05_pooled_r2": float(np.percentile(pooled, 5)),
        "p95_pooled_r2": float(np.percentile(pooled, 95)),
        "decision_rule_pass_rate": float(pass_rate),
        "median_per_facility_r2": {f: float(np.median(arr)) for f, arr in per_fac_arrays.items()},
        "fraction_per_facility_nonneg": {f: float(np.mean(arr >= 0)) for f, arr in per_fac_arrays.items()},
        "min_per_facility_r2_across_all_replicates": {f: float(np.min(arr)) for f, arr in per_fac_arrays.items()},
        "max_abs_per_facility_r2_across_all_replicates": {f: float(np.max(np.abs(arr))) for f, arr in per_fac_arrays.items()},
        "worst_single_per_facility_r2_overall": float(min(np.min(arr) for arr in per_fac_arrays.values())),
    }

out = {
    "method": "Independent synthetic positive control: Cp_true = "
              "Re_Omega^(-0.2) (Harmand et al. 2013 regime-IV exponent, "
              "not fit to this corpus), no per-facility offset, noise at "
              "this corpus's own estimated within-facility scale "
              "(sigma2_hat=0.198). Real predictor covariate structure "
              "(114 rows, 4 facilities) from cross_rotor_dataset_v3.csv.",
    "n_replicates": N_REPLICATES,
    "beta_re_true": BETA_RE_TRUE,
    "sigma2_hat": SIGMA2_HAT,
    "decision_rule": "pooled R2_log >= 0.3 AND every held-out facility R2 >= 0",
    "summary": summary,
}

OUT_JSON.parent.mkdir(exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

print("=== Independent synthetic positive control (500 replicates) ===")
for spec, s in summary.items():
    print(f"\n[{spec}]")
    print(f"  median pooled R2_log = {s['median_pooled_r2']:.3f} "
          f"(90% range {s['p05_pooled_r2']:.3f} to {s['p95_pooled_r2']:.3f})")
    print(f"  decision-rule pass rate = {s['decision_rule_pass_rate']*100:.1f}%")
    print(f"  worst single per-facility R2 across all {N_REPLICATES} reps = "
          f"{s['worst_single_per_facility_r2_overall']:.3f}")
    for f in facilities:
        print(f"    {f}: median R2={s['median_per_facility_r2'][f]:.3f}, "
              f"frac non-negative={s['fraction_per_facility_nonneg'][f]*100:.1f}%, "
              f"min across reps={s['min_per_facility_r2_across_all_replicates'][f]:.3f}")
print(f"\nSaved to {OUT_JSON}")
