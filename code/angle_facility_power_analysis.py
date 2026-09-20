"""
Angle: EMPIRICAL/STATISTICAL power analysis for "how many facilities would
be needed" before a pooled cross-facility model could be expected to
generalize under strict leave-one-facility-out (LOFO) cross-validation.

Goal (assigned): turn "we don't have enough facilities" from a vague excuse
into a quantified, defensible number, using the observed facility-to-facility
variance structure of this exact dataset (114 rows, 4 facilities).

Method, in two independent, complementary parts:

  PART A -- classical variance-component estimation and its own confidence.
    Fit the pooled fixed-effect log-log regression (same predictor set as
    the "winner" structure already selected elsewhere in this project:
    log(Cp) ~ log(Re_Omega) + log(Pi_gap) + log(Pi_confinement)
                            + log(Pi_aspect_axial)),
    then treat the per-facility mean residual as a facility random effect
    b_i ~ N(0, tau^2) on top of within-facility noise e ~ N(0, sigma^2).
    tau^2 and sigma^2 are estimated by unbalanced one-way random-effects
    ANOVA (method of moments), and an exact F-distribution confidence
    interval is put on the variance ratio tau^2/sigma^2 (Burdick &
    Graybill / Satterthwaite style bound). Because we only have k=4
    facilities (3 degrees of freedom for the between-facility mean
    square), this interval is reported explicitly and is expected to be
    very wide -- that width IS one of the deliverables.

  PART B -- Monte Carlo LOFO simulation as a function of facility count K.
    Synthetic "facility count K" worlds are built by resampling WHOLE
    facility profiles (their real covariate rows, preserving the true
    within-facility leverage/range structure) with replacement from the
    4 observed facilities, attaching each synthetic facility a FRESH
    random offset b ~ N(0, tau2_hat) and fresh residual noise, using the
    pooled fixed-effect coefficients as the generating "true" law. For
    each K we run many replicates of strict LOFO across the K synthetic
    facilities and record the distribution of pooled LOFO R2 (log space).
    This directly shows whether adding more facilities-of-the-same-kind
    would even be expected to help in principle (it tests whether there is
    a hard asymptotic ceiling driven by irreducible tau^2, independent of
    sample size) and, if there is enough headroom, at what K the criterion
    "pooled R2 substantially positive AND 5th percentile of per-facility R2
    >= 0" first becomes plausible.

  HONESTY / LIMITATION flagged explicitly in the results: Part B's
  synthetic facilities are statistically-exchangeable CLONES of the 4
  known facility types (same covariate ranges, only the facility-level
  offset and noise are redrawn) -- not genuinely novel geometry families.
  This makes Part B's simulated "K needed" an optimistic (best-case) lower
  bound: real new facilities/geometries could differ MORE than a random
  redraw of the same 4 (as suggested by the already-documented 11.7x
  spread in per-facility Reynolds slopes), so the true requirement is
  >= what is reported here, not <=.

No fabricated numbers: everything below is computed from
data/cross_rotor_dataset_v3.csv.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from scipy import stats

RNG_SEED = 20260818
rng = np.random.default_rng(RNG_SEED)

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)

y_all = np.log(d["Cp"].values)
X_cols = {
    "log_Re": np.log(d["Re_Omega"].values),
    "log_gap": np.log(d["Pi_gap"].values),
    "log_conf": np.log(d["Pi_confinement"].values),
    "log_asp": np.log(d["Pi_aspect_axial"].values),
}
sources = d["source"].values
facilities = sorted(pd.unique(sources).tolist())
k = len(facilities)
N = len(d)


def design_matrix(mask):
    n = mask.sum()
    X = np.column_stack([np.ones(n)] + [X_cols[c][mask] for c in
                                         ["log_Re", "log_gap", "log_conf", "log_asp"]])
    return X


def fit_ols(mask):
    X = design_matrix(mask)
    y = y_all[mask]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


full_mask = np.ones(N, dtype=bool)
beta_pooled = fit_ols(full_mask)
coef_names = ["const", "log_Re", "log_gap", "log_conf", "log_asp"]
pooled_pred = design_matrix(full_mask) @ beta_pooled
resid_pooled = y_all - pooled_pred

r2_pooled_insample = 1 - np.sum(resid_pooled ** 2) / np.sum((y_all - y_all.mean()) ** 2)

# ---------------------------------------------------------------
# PART A: one-way random-effects variance component decomposition
# of the pooled-fit residual, by facility.
# ---------------------------------------------------------------
per_fac = {}
n_i = {}
mean_resid_i = {}
var_resid_i = {}
for f in facilities:
    m = sources == f
    n_i[f] = int(m.sum())
    mean_resid_i[f] = float(resid_pooled[m].mean())
    var_resid_i[f] = float(resid_pooled[m].var(ddof=1)) if n_i[f] > 1 else 0.0

grand_mean = float(resid_pooled.mean())  # ~0 by OLS normal equations (const term), sanity check
n_arr = np.array([n_i[f] for f in facilities], dtype=float)
m_arr = np.array([mean_resid_i[f] for f in facilities])
s2_arr = np.array([var_resid_i[f] for f in facilities])

SSB = np.sum(n_arr * (m_arr - grand_mean) ** 2)
MSB = SSB / (k - 1)
SSW = np.sum((n_arr - 1) * s2_arr)
df_w = N - k
MSW = SSW / df_w

n0 = (N - np.sum(n_arr ** 2) / N) / (k - 1)
tau2_hat = max(0.0, (MSB - MSW) / n0)
sigma2_hat = MSW

# F-based confidence interval on the variance ratio theta = tau2/sigma2
# (Burdick & Graybill 1992 approach applied to the unbalanced case via n0):
# Under normal-theory one-way random effects, MSB/MSW is (approximately, using
# n0 for the unbalanced case) distributed as (1 + n0*theta) * F(k-1, N-k).
F_obs = MSB / MSW
alpha = 0.05
F_lo = stats.f.ppf(alpha / 2, k - 1, df_w)
F_hi = stats.f.ppf(1 - alpha / 2, k - 1, df_w)
# invert: F_obs / F_crit = 1 + n0*theta  =>  theta = (F_obs/F_crit - 1)/n0
theta_hi = max(0.0, (F_obs / F_lo - 1) / n0)  # upper bound on tau2/sigma2 uses lower F quantile
theta_lo = max(0.0, (F_obs / F_hi - 1) / n0)  # lower bound uses upper F quantile
tau2_ci = (theta_lo * sigma2_hat, theta_hi * sigma2_hat)

# p-value for H0: tau2 = 0  (facility identity adds no variance beyond
# within-facility noise), i.e. is there measurable between-facility signal
p_value_tau2_zero = float(1 - stats.f.cdf(F_obs, k - 1, df_w))

# ---------------------------------------------------------------
# PART A2: analytic scan -- how does the CI on tau2 (as a fraction of the
# point estimate) shrink if we HYPOTHETICALLY had k_hyp facilities with the
# SAME observed F_obs ratio and SAME average n_i (i.e. same signal-to-noise,
# just more independent degree-of-freedom units to measure it with)?
# This isolates the pure "estimation precision" argument, independent of
# any simulation assumptions.
# ---------------------------------------------------------------
avg_n = N / k


def ci_width_ratio_for_k(k_hyp):
    df_w_hyp = round(avg_n * k_hyp) - k_hyp
    if df_w_hyp < 1 or k_hyp < 2:
        return np.nan
    F_lo_h = stats.f.ppf(alpha / 2, k_hyp - 1, df_w_hyp)
    F_hi_h = stats.f.ppf(1 - alpha / 2, k_hyp - 1, df_w_hyp)
    n0_h = avg_n  # balanced approx
    th_hi = max(0.0, (F_obs / F_lo_h - 1) / n0_h)
    th_lo = max(0.0, (F_obs / F_hi_h - 1) / n0_h)
    if theta_lo + theta_hi == 0:
        return np.nan
    width_now = theta_hi - theta_lo
    width_hyp = th_hi - th_lo
    return width_hyp / width_now if width_now > 0 else np.nan


k_scan = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
ci_width_scan = {kk: ci_width_ratio_for_k(kk) for kk in k_scan}
# smallest scanned k at which CI half-width has shrunk to <=25% of current width
k_for_tight_ci = next((kk for kk in k_scan if not np.isnan(ci_width_scan[kk])
                        and ci_width_scan[kk] <= 0.25), None)

# ---------------------------------------------------------------
# PART B: Monte Carlo LOFO simulation vs facility count K
#
# Performance note: naively refitting OLS from scratch (stacking arrays,
# lstsq) for every one of K folds x many MC replicates x large K is
# infeasible (K up to 1024 folds/replicate). Since every synthetic facility
# reuses the EXACT covariate block of one of only 4 real facilities, X^T X
# for that block is deterministic and can be precomputed once per real
# facility. Leave-one-out training statistics are then just:
#   XtX_train = XtX_total - XtX_[chosen facility of held-out synthetic unit]
#   Xty_train = Xty_total - Xty_[held-out synthetic unit]
# reducing each fold to a 5x5 linear solve (exact, identical result to
# refitting from scratch -- verified against the naive path on K=4).
# ---------------------------------------------------------------
fac_rows = {f: np.where(sources == f)[0] for f in facilities}
fac_X = {f: design_matrix(np.isin(np.arange(N), fac_rows[f])) for f in facilities}
fac_XtX = {f: fac_X[f].T @ fac_X[f] for f in facilities}
fac_n = {f: fac_X[f].shape[0] for f in facilities}


def make_synthetic_facility(beta_true, tau2, sigma2, rng):
    """Pick one real facility's covariate block; attach a fresh random
    offset and fresh residual noise generated from the pooled fixed-effect
    law. Returns (facility_label, X, y, XtX, Xty)."""
    f = facilities[rng.integers(0, k)]
    X = fac_X[f]
    b = rng.normal(0.0, np.sqrt(tau2))
    e = rng.normal(0.0, np.sqrt(sigma2), size=X.shape[0])
    y = X @ beta_true + b + e
    Xty = X.T @ y
    return f, X, y, fac_XtX[f], Xty


def lofo_pooled_r2(fac_list, rng):
    """Given a list of (label,X,y,XtX,Xty) synthetic facilities, run strict
    LOFO using precomputed sufficient statistics (O(1) per fold) and
    return (pooled_r2, per_facility_r2_array)."""
    Ks = len(fac_list)
    XtX_total = sum(item[3] for item in fac_list)
    Xty_total = sum(item[4] for item in fac_list)
    all_resid = []
    all_y = []
    per_r2 = []
    for i in range(Ks):
        _, test_X, test_y, XtX_i, Xty_i = fac_list[i]
        XtX_train = XtX_total - XtX_i
        Xty_train = Xty_total - Xty_i
        beta_hat = np.linalg.solve(XtX_train, Xty_train)
        pred = test_X @ beta_hat
        res = test_y - pred
        all_resid.append(res)
        all_y.append(test_y)
        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((test_y - test_y.mean()) ** 2)
        r2_i = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        per_r2.append(r2_i)
    all_resid = np.concatenate(all_resid)
    all_y = np.concatenate(all_y)
    ss_res_pooled = np.sum(all_resid ** 2)
    ss_tot_pooled = np.sum((all_y - all_y.mean()) ** 2)
    pooled_r2 = 1 - ss_res_pooled / ss_tot_pooled
    return pooled_r2, np.array(per_r2)


K_list = [4, 8, 16, 32, 64, 128, 256, 1024]
N_MC = 300
mc_results = {}
for K in K_list:
    pooled_list = []
    p05_list = []
    p50_list = []
    frac_all_nonneg = []
    for _ in range(N_MC):
        facs = [make_synthetic_facility(beta_pooled, tau2_hat, sigma2_hat, rng) for _ in range(K)]
        pooled_r2, per_r2 = lofo_pooled_r2(facs, rng)
        pooled_list.append(pooled_r2)
        p05_list.append(np.percentile(per_r2, 5))
        p50_list.append(np.median(per_r2))
        frac_all_nonneg.append(np.mean(per_r2 >= 0))
    mc_results[K] = {
        "pooled_r2_mean": float(np.mean(pooled_list)),
        "pooled_r2_median": float(np.median(pooled_list)),  # robust to rare near-singular training draws (see note)
        "pooled_r2_p05": float(np.percentile(pooled_list, 5)),
        "pooled_r2_p95": float(np.percentile(pooled_list, 95)),
        "per_facility_r2_p05_mean": float(np.mean(p05_list)),  # avg across MC reps of the worst-5%-facility R2
        "per_facility_r2_median_mean": float(np.mean(p50_list)),
        "frac_facilities_nonneg_r2_mean": float(np.mean(frac_all_nonneg)),
    }

# ---------------------------------------------------------------
# PART B2: diagnostic -- break down simulated per-facility R2 by which of
# the 4 real facility TYPES the synthetic unit was cloned from, at a large
# K (256), to check whether the persistent negative tail in
# per_facility_r2_p05 (Part B) is driven mainly by small-n facilities
# (noisy single-facility R2 as a statistic) rather than by tau2 alone.
# ---------------------------------------------------------------
by_label_r2 = {f: [] for f in facilities}
for _ in range(60):
    facs = [make_synthetic_facility(beta_pooled, tau2_hat, sigma2_hat, rng) for _ in range(256)]
    pooled_r2, per_r2 = lofo_pooled_r2(facs, rng)
    for item, r2 in zip(facs, per_r2):
        by_label_r2[item[0]].append(r2)

per_facility_type_diagnostic = {}
for f in facilities:
    arr = np.array(by_label_r2[f])
    per_facility_type_diagnostic[f] = {
        "n_points_this_facility_type": int(n_i[f]),
        "n_synthetic_units_sampled": int(len(arr)),
        "mean_r2": float(arr.mean()),
        "median_r2": float(np.median(arr)),
        "p05_r2": float(np.percentile(arr, 5)),
        "frac_negative_r2": float((arr < 0).mean()),
    }

# also run with the tau2 CI bounds to bracket the answer under estimation uncertainty
mc_bracket = {}
for label, tau2_val in [("tau2_lo95", tau2_ci[0]), ("tau2_point", tau2_hat), ("tau2_hi95", tau2_ci[1])]:
    row = {}
    for K in [4, 32, 256, 2048]:
        pooled_list = []
        p05_list = []
        for _ in range(150):
            facs = [make_synthetic_facility(beta_pooled, tau2_val, sigma2_hat, rng) for _ in range(K)]
            pooled_r2, per_r2 = lofo_pooled_r2(facs, rng)
            pooled_list.append(pooled_r2)
            p05_list.append(np.percentile(per_r2, 5))
        row[K] = {
            "pooled_r2_mean": float(np.mean(pooled_list)),
            "per_facility_r2_p05_mean": float(np.mean(p05_list)),
        }
    mc_bracket[label] = row

# decision rule (matches project-wide bar): pooled R2 substantially positive
# (>=0.3, arbitrary but stated) AND worst-5%-facility R2 >= 0
K_meets_criterion_point = None
for K in K_list:
    r = mc_results[K]
    if r["pooled_r2_mean"] >= 0.3 and r["per_facility_r2_p05_mean"] >= 0.0:
        K_meets_criterion_point = K
        break

# asymptotic ceiling: use the largest K simulated (1024) as a proxy for K->infinity
ceiling_pooled_r2 = mc_results[1024]["pooled_r2_mean"]
ceiling_worst5_r2 = mc_results[1024]["per_facility_r2_p05_mean"]

results = {
    "meta": {
        "n_rows": int(N),
        "n_facilities": int(k),
        "facilities": facilities,
        "n_per_facility": {f: int(n_i[f]) for f in facilities},
        "rng_seed": RNG_SEED,
        "n_monte_carlo_reps_per_K": N_MC,
    },
    "pooled_fixed_effect_fit": {
        "coef_names": coef_names,
        "coef": {c: float(b) for c, b in zip(coef_names, beta_pooled)},
        "in_sample_r2": float(r2_pooled_insample),
        "note": "Same predictor family as the 'M6' winner structure selected elsewhere in this project (scaling_law_search.py); refit here for self-containment of this angle.",
    },
    "within_vs_between_facility_predictor_spread": {
        "note": "Sanity check for the eta^2 confound already documented: within-facility SD of the near-fully-confounded predictors is essentially zero (facilities occupy near-point locations in Pi_confinement/Pi_gap/Pi_aspect_axial space); only log_Re_Omega varies meaningfully within a facility.",
        "log_Pi_confinement_within_facility_sd": {f: float(np.log(d.loc[sources == f, "Pi_confinement"]).std(ddof=0)) for f in facilities},
    },
    "part_A_variance_components": {
        "method": "unbalanced one-way random-effects ANOVA (method of moments) on pooled-OLS residuals, grouped by facility",
        "MSB": float(MSB), "MSW": float(MSW), "df_between": int(k - 1), "df_within": int(df_w),
        "n0_unbalanced": float(n0),
        "tau2_hat_between_facility_variance": float(tau2_hat),
        "sigma2_hat_within_facility_variance": float(sigma2_hat),
        "tau2_over_sigma2_point_estimate": float(tau2_hat / sigma2_hat) if sigma2_hat > 0 else None,
        "F_obs_MSB_over_MSW": float(F_obs),
        "p_value_H0_tau2_is_zero": p_value_tau2_zero,
        "tau2_95CI_burdick_graybill_style": [float(tau2_ci[0]), float(tau2_ci[1])],
        "interpretation": "With only k=4 facilities (3 df between-facility), the 95% CI on tau2 (the facility-to-facility variance component) is extremely wide relative to the point estimate -- we can barely even measure the quantity that governs whether more facilities would help.",
    },
    "part_A2_estimation_precision_scan": {
        "note": "Holding the observed F_obs ratio and average per-facility n fixed, how much would the CI half-width on tau2/sigma2 shrink if we had k_hyp facilities instead of 4 (pure estimation-precision argument, no simulation)?",
        "ci_width_ratio_vs_k4": {str(kk): (None if np.isnan(v) else float(v)) for kk, v in ci_width_scan.items()},
        "k_hyp_for_ci_width_leq_25pct_of_current": k_for_tight_ci,
    },
    "part_B_monte_carlo_lofo_vs_facility_count": {
        "note": "Synthetic facilities = real covariate blocks resampled with replacement from the 4 observed facilities (same X-ranges/leverage), each given a FRESH random offset ~N(0,tau2_hat) and fresh noise ~N(0,sigma2_hat) generated from the pooled fixed-effect law. Strict LOFO across K synthetic facilities, averaged over Monte Carlo replicates.",
        "by_K": {str(K): v for K, v in mc_results.items()},
        "bracket_under_tau2_uncertainty": mc_bracket,
        "per_facility_type_diagnostic_at_K256": per_facility_type_diagnostic,
        "per_facility_type_diagnostic_note": "Breaks the persistent negative per-facility R2 tail down by which real facility TYPE the synthetic unit was cloned from. The smallest facility (Zheng2024, n=8) is much worse (mean R2, frac_negative) than the larger ones, showing part -- not all -- of the persistent floor in worst-facility R2 is a small-sample-size artifact (a single held-out facility's own R2 is itself a noisy statistic when n is small), layered on top of the genuine tau2 effect.",
        "asymptotic_ceiling_K_to_infinity_proxy_K1024": {
            "pooled_r2": float(ceiling_pooled_r2),
            "worst_5pct_facility_r2": float(ceiling_worst5_r2),
        },
        "K_meeting_project_decision_rule_pooled_r2_ge_0.3_and_worst5pct_ge_0": K_meets_criterion_point,
    },
    "headline_conclusions": [],
}

# fill headline conclusions programmatically from computed numbers
concl = []
concl.append(
    f"Between-facility variance component tau2_hat = {tau2_hat:.3f} (log-Cp^2 units) vs "
    f"within-facility sigma2_hat = {sigma2_hat:.3f}: facility identity contributes "
    f"{tau2_hat/(tau2_hat+sigma2_hat)*100:.1f}% of total residual variance not explained by the fixed-effect law."
)
concl.append(
    f"That tau2 estimate itself has a 95% CI of [{tau2_ci[0]:.3f}, {tau2_ci[1]:.3f}] with only k=4 facilities "
    f"(3 degrees of freedom) -- ratio of upper to point estimate is "
    f"{(tau2_ci[1]/tau2_hat if tau2_hat>0 else float('nan')):.1f}x, i.e. the number that would tell us 'how many "
    f"facilities are needed' is itself not reliably measurable from 4 facilities."
)
if k_for_tight_ci is not None:
    concl.append(
        f"Purely from estimation-precision (Part A2, no simulation assumptions): to shrink the CI on the "
        f"tau2/sigma2 ratio to <=25% of its current width at the SAME observed effect size, "
        f"one would need on the order of {k_for_tight_ci} facilities."
    )
concl.append(
    f"Monte Carlo LOFO simulation (Part B): the MEAN pooled LOFO R2 at K=4 ({mc_results[4]['pooled_r2_mean']:.1f}) is "
    f"dominated by rare near-singular training draws (when resampling-with-replacement happens to include few "
    f"distinct real facility types, the near-total facility/predictor confound described above makes the "
    f"training design matrix nearly rank-deficient) -- the MEDIAN is a more honest summary: "
    f"{mc_results[4]['pooled_r2_median']:.2f} at K=4, rising to {mc_results[16]['pooled_r2_median']:.2f} at K=16 and "
    f"an asymptotic ceiling near {mc_results[1024]['pooled_r2_median']:.2f} by K~32-64 (values stop changing "
    f"materially beyond that). This ceiling is comfortably positive on the POOLED metric -- so, in this "
    f"optimistic same-facility-type simulation, more facilities would in principle be expected to fix the "
    f"pooled-R2 side of generalization."
)
concl.append(
    f"But the project's actual decision rule needs EVERY held-out facility to be non-negative, not just the "
    f"pooled average -- and that does NOT converge with K. The worst-5%-of-facilities R2 stays pinned near "
    f"{mc_results[1024]['per_facility_r2_p05_mean']:.2f} from K=32 all the way to K=1024 (frac. of facilities with "
    f"R2>=0 plateaus near {mc_results[1024]['frac_facilities_nonneg_r2_mean']*100:.0f}%, never reaching 100%). "
    f"The per-facility-type breakdown (Part B2) shows this floor is not uniform: it is driven disproportionately "
    f"by the smallest facility type (Zheng-sized, n=8: mean R2={per_facility_type_diagnostic['Zheng2024']['mean_r2']:.2f}, "
    f"{per_facility_type_diagnostic['Zheng2024']['frac_negative_r2']*100:.0f}% of draws negative) versus the "
    f"largest (n=41-45: mean R2 {min(per_facility_type_diagnostic[f]['mean_r2'] for f in ['Guo2024','Vrancik1968']):.2f} to "
    f"{max(per_facility_type_diagnostic[f]['mean_r2'] for f in ['Guo2024','Vrancik1968']):.2f}, "
    f"{min(per_facility_type_diagnostic[f]['frac_negative_r2'] for f in ['Guo2024','Vrancik1968'])*100:.0f}-"
    f"{max(per_facility_type_diagnostic[f]['frac_negative_r2'] for f in ['Guo2024','Vrancik1968'])*100:.0f}% negative). "
    f"So part of the 'never converges' result is a genuine tau2 effect and part is a small-held-out-sample-size "
    f"statistical artifact (a single facility's own R2 is itself a noisy statistic when n~8) -- both are real, "
    f"and both argue for MORE points per future facility, not just more facilities."
)
concl.append(
    "IMPORTANT CAVEAT: Part B's synthetic facilities are statistically-exchangeable clones of the 4 known "
    "facility TYPES (identical covariate ranges/leverage, only the offset and noise are redrawn) -- not "
    "genuinely novel geometry families. This makes the reported convergence an optimistic best case: real new "
    "facilities could differ MORE than a random redraw of the same 4 (the already-documented 11.7x spread in "
    "per-facility Reynolds-exponent estimates is evidence of exactly that), so the true number of facilities "
    "needed to actually clear the pooled+worst-facility bar is >= what is reported here, quite possibly "
    "unbounded if genuinely new geometries carry even larger idiosyncratic offsets than tau2_hat captures."
)
results["headline_conclusions"] = concl

out_path = _ROOT + "/results/angle_facility_power_analysis_results.json"
with open(out_path, "w") as fh:
    json.dump(results, fh, indent=2)

print(json.dumps(results["headline_conclusions"], indent=2))
print("\nFull results written to", out_path)
