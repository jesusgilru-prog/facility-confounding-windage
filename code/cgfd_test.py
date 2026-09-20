"""Step 1 of the debate's action plan: does a diagnostic that compares
LOFO R^2 distributions actually discriminate "structured collinearity
artifact" from "genuine facility confounding", or are they statistically
indistinguishable given this corpus's real covariate structure?

Two synthetic generating processes, SAME real predictor values (114 rows,
4 facilities, same collinearity structure as the actual corpus), SAME
fitting procedure (M6's 4-predictor OLS, same LOFO method as everywhere
else in this paper), differing in exactly one thing:

  Process 1 (pure collinearity artifact): true law = Re_Omega^(-0.2)
  only (Harmand et al. 2013 regime-IV exponent, external, not fit to
  this corpus), NO per-facility offset. This is code/posctrl.py's
  "m6_four_predictor" case, rerun here for a controlled side-by-side
  comparison.

  Process 2 (genuine facility confounding): same true law, SAME noise
  scale, PLUS a per-facility random offset b_f ~ N(0, tau2_hat), where
  tau2_hat=0.073 is this corpus's own estimated between-facility
  variance component (angle_facility_power_analysis.py, Part A) -- i.e.
  facility identity genuinely shifts the outcome beyond what the
  predictors capture, exactly what "confounding" means.

If a diagnostic (KS test on pooled LOFO R^2, or the per-facility failure
pattern) can reliably tell these two apart, the CGFD approach proposed
in the multi-model debate (2026-08-20, debate_disruptivo/ACTA.md) is
viable. If not, it is not, and this script says so honestly rather than
force a result.

No fabricated numbers: everything below is computed from
data/cross_rotor_dataset_v3.csv.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "cross_rotor_dataset_v3.csv")
FEATURES = ["Re_Omega", "Pi_gap", "Pi_confinement", "Pi_aspect_axial"]

facilities = sorted(df["source"].unique())
log_Re = np.log(df["Re_Omega"].values)
log_gap = np.log(df["Pi_gap"].values)
log_conf = np.log(df["Pi_confinement"].values)
log_asp = np.log(df["Pi_aspect_axial"].values)
n = len(df)
fac_labels = df["source"].values

BETA_RE_TRUE = -0.2
INTERCEPT_TRUE = np.log(df["Cp"].values).mean() - BETA_RE_TRUE * log_Re.mean()
SIGMA2_HAT = 0.198  # within-facility, angle_facility_power_analysis.py
TAU2_HAT = 0.073    # between-facility, angle_facility_power_analysis.py
SIGMA_HAT = np.sqrt(SIGMA2_HAT)
TAU_HAT = np.sqrt(TAU2_HAT)

N_REPLICATES = 500


def lofo_pooled_and_per_facility(y, X, fac_labels):
    preds = np.full(n, np.nan)
    per_fac_r2 = {}
    for held_out in facilities:
        train_mask = fac_labels != held_out
        test_mask = ~train_mask
        Xtr = np.column_stack([np.ones(train_mask.sum()), X[train_mask]])
        ytr = y[train_mask]
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


X4 = np.column_stack([log_Re, log_gap, log_conf, log_asp])
rng = np.random.default_rng(1)

pooled_collinearity = []
pooled_confounded = []
per_fac_collinearity = {f: [] for f in facilities}
per_fac_confounded = {f: [] for f in facilities}

for rep in range(N_REPLICATES):
    noise = rng.normal(0.0, SIGMA_HAT, size=n)

    # Process 1: pure collinearity artifact, no facility offset
    y1 = INTERCEPT_TRUE + BETA_RE_TRUE * log_Re + noise
    p1, pf1 = lofo_pooled_and_per_facility(y1, X4, fac_labels)
    pooled_collinearity.append(p1)
    for f in facilities:
        per_fac_collinearity[f].append(pf1[f])

    # Process 2: same law + genuine per-facility offset (real confounding)
    offsets = {f: rng.normal(0.0, TAU_HAT) for f in facilities}
    offset_vec = np.array([offsets[f] for f in fac_labels])
    y2 = INTERCEPT_TRUE + BETA_RE_TRUE * log_Re + offset_vec + noise
    p2, pf2 = lofo_pooled_and_per_facility(y2, X4, fac_labels)
    pooled_confounded.append(p2)
    for f in facilities:
        per_fac_confounded[f].append(pf2[f])

pooled_collinearity = np.array(pooled_collinearity)
pooled_confounded = np.array(pooled_confounded)

ks_stat, ks_p = stats.ks_2samp(pooled_collinearity, pooled_confounded)

per_facility_ks = {}
for f in facilities:
    a = np.array(per_fac_collinearity[f])
    b = np.array(per_fac_confounded[f])
    s, p = stats.ks_2samp(a, b)
    per_facility_ks[f] = {"ks_stat": float(s), "ks_p": float(p),
                           "median_collinearity": float(np.median(a)),
                           "median_confounded": float(np.median(b))}

# Where does the REAL data's pooled R2 and per-facility pattern fall
# relative to both synthetic distributions?
real_pooled_r2 = -0.8848004392997477  # scaling_law_search_results.json, LOFO M6 pooled
real_per_facility = {
    "Guo2024": -8.986612059551168,
    "Liu2024": -118.63018606035955,
    "Vrancik1968": -1.8542073187456043,
    "Zheng2024": -553.7587044467027,
}  # exact values, scaling_law_search_results.json lofo_cv.per_facility

pct_real_le_collinearity = float(np.mean(pooled_collinearity <= real_pooled_r2)) * 100
pct_real_le_confounded = float(np.mean(pooled_confounded <= real_pooled_r2)) * 100

# Proper single-observation diagnostic: this paper has exactly ONE real
# dataset, not repeated draws, so a two-sample KS test between 500
# synthetic replicates and a single real number is not the right tool.
# Instead: for each facility, find the percentile of the real R2 within
# each synthetic null distribution (how "typical" is the real result
# under each hypothesis), and a log-density-ratio proxy via a simple
# kernel density estimate on the log-scale (since these R2 values span
# many orders of magnitude and are strongly left-skewed).
from scipy.stats import gaussian_kde

single_obs_diagnostic = {}
for f in facilities:
    a = np.array(per_fac_collinearity[f])  # collinearity-only
    b = np.array(per_fac_confounded[f])    # genuine confounding
    real_val = real_per_facility[f]
    pct_in_collinearity = float(np.mean(a <= real_val)) * 100
    pct_in_confounded = float(np.mean(b <= real_val)) * 100
    # KDE on a signed-log transform to handle the huge negative range
    def signed_log(x):
        return np.sign(x) * np.log1p(np.abs(x))
    try:
        kde_a = gaussian_kde(signed_log(a))
        kde_b = gaussian_kde(signed_log(b))
        dens_a = float(kde_a(signed_log(np.array([real_val])))[0])
        dens_b = float(kde_b(signed_log(np.array([real_val])))[0])
        log_lr = float(np.log(dens_b + 1e-300) - np.log(dens_a + 1e-300))
    except Exception:
        dens_a = dens_b = log_lr = None

    # CALIBRATION GUARD (added 2026-08-25 after self-audit).
    # A Gaussian KDE evaluated far outside the support of its own sample is
    # pure extrapolation, and its ratio against another such extrapolation is
    # not a likelihood ratio in any usable sense. Worse, when one density
    # underflows to exactly 0, the epsilon floor above turns the "ratio" into
    # a function of that arbitrary constant: Liu2024 originally reported
    # log-LR = +585, which is exactly log(1.45e-46) - log(1e-300) and carries
    # no evidential content at all. Flag which ratios are actually calibrated:
    # both densities must be evaluated inside the simulated support.
    within_a = bool(a.min() <= real_val <= a.max())
    within_b = bool(b.min() <= real_val <= b.max())
    calibrated = bool(within_a and within_b)

    single_obs_diagnostic[f] = {
        "real_r2": real_val,
        "percentile_under_collinearity_only": pct_in_collinearity,
        "percentile_under_genuine_confounding": pct_in_confounded,
        "density_at_real_value_collinearity_only": dens_a,
        "density_at_real_value_genuine_confounding": dens_b,
        "log_likelihood_ratio_confounded_over_collinearity": log_lr,
        "real_within_support_collinearity": within_a,
        "real_within_support_confounded": within_b,
        "log_lr_is_calibrated": calibrated,
        "log_lr_usable": log_lr if calibrated else None,
    }

out = {
    "purpose": "Test whether pooled-LOFO-R2 distributions from a pure "
               "collinearity artifact (no facility offset) and genuine "
               "facility confounding (same law + per-facility offset "
               "N(0,tau2_hat)) are statistically distinguishable, given "
               "this corpus's real covariate structure.",
    "n_replicates": N_REPLICATES,
    "beta_re_true": BETA_RE_TRUE, "sigma2_hat": SIGMA2_HAT, "tau2_hat": TAU2_HAT,
    "pooled_r2_ks_test": {"ks_statistic": float(ks_stat), "p_value": float(ks_p)},
    "pooled_r2_collinearity_only": {
        "median": float(np.median(pooled_collinearity)),
        "p05": float(np.percentile(pooled_collinearity, 5)),
        "p95": float(np.percentile(pooled_collinearity, 95)),
    },
    "pooled_r2_genuine_confounding": {
        "median": float(np.median(pooled_confounded)),
        "p05": float(np.percentile(pooled_confounded, 5)),
        "p95": float(np.percentile(pooled_confounded, 95)),
    },
    "per_facility_ks_tests": per_facility_ks,
    # Raw per-facility replicate distributions, persisted so the diagnostic
    # figure (code/make_cgfd_figure.py) is reproducible from this JSON alone
    # rather than requiring a rerun of the 500-replicate simulation.
    "raw_per_facility_collinearity": {f: [float(x) for x in per_fac_collinearity[f]] for f in facilities},
    "raw_per_facility_confounded": {f: [float(x) for x in per_fac_confounded[f]] for f in facilities},
    "real_pooled_r2_reference": real_pooled_r2,
    "pct_synthetic_collinearity_reps_as_bad_or_worse_than_real": pct_real_le_collinearity,
    "pct_synthetic_confounded_reps_as_bad_or_worse_than_real": pct_real_le_confounded,
    "single_observation_diagnostic": single_obs_diagnostic,
}

OUT_JSON = ROOT / "results" / "cgfd_test_results.json"
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

print("=== CGFD discrimination test: can pooled LOFO R2 tell collinearity from confounding apart? ===")
print(f"Pooled R2 KS test: stat={ks_stat:.3f}, p={ks_p:.4f}")
print(f"  collinearity-only:  median={np.median(pooled_collinearity):.3f} "
      f"(90% range {np.percentile(pooled_collinearity,5):.3f} to {np.percentile(pooled_collinearity,95):.3f})")
print(f"  genuine confounding: median={np.median(pooled_confounded):.3f} "
      f"(90% range {np.percentile(pooled_confounded,5):.3f} to {np.percentile(pooled_confounded,95):.3f})")
print(f"\nReal pooled R2 = {real_pooled_r2:.3f}")
print(f"  {pct_real_le_collinearity:.1f}% of collinearity-only reps are as bad or worse")
print(f"  {pct_real_le_confounded:.1f}% of genuine-confounding reps are as bad or worse")
print("\nPer-facility KS tests (distribution-level, NOT the real single-obs diagnostic):")
for f, v in per_facility_ks.items():
    print(f"  {f}: KS stat={v['ks_stat']:.3f}, p={v['ks_p']:.4f}, "
          f"median collinearity={v['median_collinearity']:.2f}, median confounded={v['median_confounded']:.2f}")

print("\n=== Single-observation diagnostic (the actually correct test for one real dataset) ===")
for f, v in single_obs_diagnostic.items():
    verdict = (f"log-LR={v['log_lr_usable']:+.2f} (CALIBRATED)" if v["log_lr_is_calibrated"]
               else "log-LR NOT calibrated (real value outside simulated support)")
    print(f"  {f}: real R2={v['real_r2']:.2f} | "
          f"pct collinearity={v['percentile_under_collinearity_only']:.1f}% | "
          f"pct confounding={v['percentile_under_genuine_confounding']:.1f}% | {verdict}")
n_cal = sum(1 for v in single_obs_diagnostic.values() if v["log_lr_is_calibrated"])
print(f"\n  -> {n_cal} of {len(single_obs_diagnostic)} facilities yield a calibrated likelihood ratio.")
print(f"\nSaved to {OUT_JSON}")
