"""Angle (2026-08-18, sub-agent investigation): SHAP + per-facility causal-
discovery-style check on which physical variable(s) robustly matter across
ALL 4 facilities, even though the pooled quantitative predictive law does not
transfer (established: pooled LOFO R2_log in [-109, -0.34] across 16 methods).

Question asked: is there a variable whose SIGN / RANK-ORDER effect on
log(Cp) is consistent across all four facilities separately, even if the
magnitude/exponent is not (candidate from prior work: Re_Omega, whose
pooled-regression exponent survives a facility-cluster bootstrap with 95% CI
excluding zero, [-0.524, -0.203], per scaling_law_search_results.json)?

Important prior finding this script must NOT contradict or ignore:
rf_gbm_shap_diagnostics.json already found that even Re_Omega's SHAP
contribution in a pooled tree model is ~85% explained by facility identity
(eta2_shap_explained_by_source = 0.85 for both RF and GBM), and its
*mean* SHAP value flips in sign of magnitude ordering across facilities
(strongly negative mean SHAP in Guo2024, positive in the other three).
That diagnostic is about the GLOBAL/AVERAGE level of the SHAP contribution
per facility (a location-shift), which is a different question from "what
is the LOCAL slope of Cp vs Re_Omega within a given facility's own data".
Both can be true simultaneously: the model can use Re_Omega as a partial
facility-proxy for its overall level AND still show a locally decreasing
relationship within each facility. This script tests the LOCAL, within-
facility, sign/rank claim directly and honestly, with five independent
checks:

  (1) Per-facility SIMPLE (bivariate) log-log OLS slope of each predictor
      vs log(Cp), fit SEPARATELY on each facility's own rows only (no
      pooling, no cross-facility leakage of any kind).
  (2) Per-facility Spearman rank correlation (distribution-free, robust to
      the exact functional form) of each predictor vs log(Cp).
  (3) A facility-fixed-effects ("within") partial-correlation check on the
      pooled, facility-demeaned data: for each variable, remove the
      facility-specific mean from both y and x, then correlate the
      residuals -- a standard panel-data way of asking "does x correlate
      with y AFTER removing everything that is a pure facility offset",
      i.e. a real conditional-independence-style check of x _||_ y | facility.
  (4) A SHAP LOCAL-SLOPE check: fit one pooled (in-sample, diagnostic only)
      Gradient Boosting model, get SHAP values, and compute the within-
      facility Spearman correlation between each feature's SHAP value and
      the feature's own raw value. This is the tree-model analogue of (1)/(2)
      applied to the SHAP attribution itself, and it is a DIFFERENT quantity
      from the eta2-vs-source diagnostic already in rf_gbm_shap_diagnostics.json.
  (5) A held-out-facility PERMUTATION IMPORTANCE check under strict LOFO:
      train GBM on 3 facilities, and on the held-out 4th facility measure
      the drop in Spearman rank correlation between predicted and actual
      log(Cp) when each feature is permuted. This tests whether a feature's
      predictive *ranking* signal (not its quantitative law) transfers to an
      unseen facility -- the literal operationalization of the assigned
      angle's hypothesis.

KNOWN DATA CONSTRAINT (checked explicitly, must be reported, not hidden):
Pi_confinement, Pi_gap and Pi_aspect_axial are CONSTANT within Liu2024 (one
geometry, n=20) and within Zheng2024 (one geometry, n=8) -- there is zero
within-facility variance to test for these two facilities. So checks (1),
(2) and (3)-per-facility for the three Pi_* geometric ratios can only be
computed on 2 of the 4 facilities (Guo2024: 6 geometries, Vrancik1968: 4
geometries). Re_Omega (and M_tip) vary substantially within EVERY facility
(checked: n_unique Re_Omega per facility = 19, 20, 36, 8), so it is the only
variable whose per-facility sign/rank claim can be tested on all 4 facilities
independently. This asymmetry is reported explicitly, not glossed over.

Additionally: within Guo2024, Pi_confinement and Pi_gap are PERFECTLY
collinear (r = -1.000, an exact algebraic relation for that geometry family)
and within Vrancik1968 they are r = -0.976 -- so any *multiple*-regression
partial coefficient trying to separate confinement-effect from gap-effect
within a facility is numerically unstable/uninterpretable. This script uses
SIMPLE (one-predictor-at-a-time) per-facility slopes/correlations for the
per-facility check for exactly this reason, and reports the collinearity
so a multiple-regression version is not silently attempted and misread.

Environment note: sklearn/shap are not on the system Python (PEP 668
externally-managed); reused the project's existing venv at
paper_windage_power/.venv (already created for rf_gbm_shap_diagnostics.py).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
import shap

RNG_SEED = 42
DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT_PATH = _ROOT + "/results/angle_shap_causal_facility_results.json"

FEATURES = ["Re_Omega", "Pi_confinement", "Pi_gap", "Pi_aspect_axial"]

df = pd.read_csv(DATA_PATH)
d = df.dropna(subset=["Cp", "source"] + FEATURES).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
facilities = sorted(d["source"].unique())
n_total = len(d)

logy = np.log(d["Cp"].values)
logX = pd.DataFrame({f: np.log(d[f].values) for f in FEATURES})
d_log = logX.copy()
d_log["logCp"] = logy
d_log["source"] = d["source"].values

results = {"n_total": n_total, "facilities": facilities, "features": FEATURES}

# -------------------------------------------------------------------
# 0) Explicit within-facility variance check (must be reported honestly)
# -------------------------------------------------------------------
variance_check = {}
for src in facilities:
    g = d_log[d_log.source == src]
    variance_check[src] = {
        f: {"std": float(g[f].std()), "n_unique": int(g[f].round(10).nunique())}
        for f in FEATURES
    }
results["within_facility_variance_check"] = variance_check
TESTABLE_PER_FACILITY = {
    f: [src for src in facilities if variance_check[src][f]["std"] > 1e-8]
    for f in FEATURES
}
results["variables_with_nonzero_within_facility_variance_by_source"] = TESTABLE_PER_FACILITY

# Collinearity check within the two testable facilities (Guo2024, Vrancik1968)
collinearity = {}
for src in facilities:
    g = d_log[d_log.source == src]
    if g[FEATURES].std().min() > 1e-8:
        collinearity[src] = g[FEATURES].corr().round(4).to_dict()
    else:
        varying = [f for f in FEATURES if g[f].std() > 1e-8]
        collinearity[src] = g[varying].corr().round(4).to_dict() if len(varying) > 1 else "only Re_Omega/M_tip vary; Pi_* constant"
results["within_facility_pi_collinearity"] = collinearity

# -------------------------------------------------------------------
# 1) & 2) Per-facility SIMPLE log-log OLS slope + Spearman rank corr
# -------------------------------------------------------------------
per_facility_simple = {f: {} for f in FEATURES}
for f in FEATURES:
    for src in facilities:
        g = d_log[d_log.source == src]
        x = g[f].values
        y = g["logCp"].values
        n = len(g)
        if np.std(x) < 1e-8:
            per_facility_simple[f][src] = {
                "n": n, "testable": False,
                "reason": "zero within-facility variance (single geometry)",
            }
            continue
        slope, intercept, r, p, se = stats.linregress(x, y)
        rho, p_rho = stats.spearmanr(x, y)
        per_facility_simple[f][src] = {
            "n": n, "testable": True,
            "ols_slope": float(slope), "ols_slope_se": float(se),
            "ols_slope_ci95": [float(slope - 1.96 * se), float(slope + 1.96 * se)],
            "pearson_r": float(r), "pearson_r2": float(r ** 2), "p_value": float(p),
            "spearman_rho": float(rho), "spearman_p": float(p_rho),
        }
results["per_facility_simple_regression"] = per_facility_simple

# Sign-consistency summary per feature, over the facilities where testable
sign_consistency = {}
for f in FEATURES:
    entries = {src: v for src, v in per_facility_simple[f].items() if v.get("testable")}
    slopes = {src: v["ols_slope"] for src, v in entries.items()}
    signs = set(np.sign(s) for s in slopes.values())
    ci_excludes_zero = {
        src: not (v["ols_slope_ci95"][0] <= 0 <= v["ols_slope_ci95"][1])
        for src, v in entries.items()
    }
    sign_consistency[f] = {
        "n_facilities_testable": len(entries),
        "facilities_testable": list(entries.keys()),
        "slopes": slopes,
        "all_same_sign": len(signs) == 1 and len(entries) > 0,
        "all_negative": all(s < 0 for s in slopes.values()) if entries else None,
        "all_positive": all(s > 0 for s in slopes.values()) if entries else None,
        "ci_excludes_zero_per_facility": ci_excludes_zero,
        "all_ci_exclude_zero": all(ci_excludes_zero.values()) if entries else None,
    }
results["sign_consistency_summary"] = sign_consistency

# -------------------------------------------------------------------
# 3) Facility-fixed-effects ("within") partial correlation, pooled
# -------------------------------------------------------------------
demeaned = d_log.copy()
for col in FEATURES + ["logCp"]:
    fac_means = demeaned.groupby("source")[col].transform("mean")
    demeaned[col + "_dm"] = demeaned[col] - fac_means

fe_partial = {}
for f in FEATURES:
    x = demeaned[f + "_dm"].values
    y = demeaned["logCp_dm"].values
    if np.std(x) < 1e-8:
        fe_partial[f] = {"testable": False}
        continue
    r, p = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    slope, intercept, r2_lin, p2, se = stats.linregress(x, y)
    fe_partial[f] = {
        "testable": True,
        "within_facility_demeaned_pearson_r": float(r), "p_value": float(p),
        "within_facility_demeaned_spearman_rho": float(rho), "p_value_spearman": float(p_rho),
        "within_facility_demeaned_slope": float(slope), "slope_se": float(se),
        "note": "dominated by Guo2024/Vrancik1968 for Pi_* (zero within-variance rows contribute nothing)"
                if f != "Re_Omega" else "all 4 facilities contribute variance",
    }
results["facility_fixed_effects_partial_correlation"] = fe_partial

# -------------------------------------------------------------------
# 4) Pooled GBM (in-sample, diagnostic only) + SHAP local-slope-per-facility
# -------------------------------------------------------------------
X_full = logX.values
y_full = logy
gbm = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.9, random_state=RNG_SEED,
)
gbm.fit(X_full, y_full)
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(X_full)

shap_local_slope = {}
for i, f in enumerate(FEATURES):
    shap_local_slope[f] = {}
    for src in facilities:
        mask = (d["source"].values == src)
        x = X_full[mask, i]
        s = shap_values[mask, i]
        if np.std(x) < 1e-8:
            shap_local_slope[f][src] = {"testable": False, "n": int(mask.sum())}
            continue
        rho, p_rho = stats.spearmanr(x, s)
        slope, intercept, r_lin, p_lin, se = stats.linregress(x, s)
        shap_local_slope[f][src] = {
            "testable": True, "n": int(mask.sum()),
            "spearman_rho_shap_vs_feature": float(rho), "p_value": float(p_rho),
            "ols_slope_shap_vs_feature": float(slope),
        }
results["shap_local_slope_within_facility"] = shap_local_slope

shap_sign_consistency = {}
for f in FEATURES:
    entries = {src: v for src, v in shap_local_slope[f].items() if v.get("testable")}
    rhos = {src: v["spearman_rho_shap_vs_feature"] for src, v in entries.items()}
    signs = set(np.sign(r) for r in rhos.values())
    shap_sign_consistency[f] = {
        "n_facilities_testable": len(entries),
        "rhos": rhos,
        "all_same_sign": len(signs) == 1 and len(entries) > 0,
        "all_negative": all(r < 0 for r in rhos.values()) if entries else None,
    }
results["shap_local_slope_sign_consistency"] = shap_sign_consistency

# global mean|SHAP| ranking (in-sample, diagnostic only, cross-check vs prior script)
mean_abs_shap = {FEATURES[i]: float(np.mean(np.abs(shap_values[:, i]))) for i in range(len(FEATURES))}
results["pooled_insample_mean_abs_shap_ranking"] = sorted(mean_abs_shap.items(), key=lambda kv: -kv[1])

# -------------------------------------------------------------------
# 5) Strict LOFO permutation importance on RANK (Spearman) transfer
# -------------------------------------------------------------------
lofo_perm = {}
rng = np.random.default_rng(RNG_SEED)
N_PERM = 200
for held_out in facilities:
    train_mask = d["source"].values != held_out
    test_mask = ~train_mask
    Xtr, ytr = X_full[train_mask], y_full[train_mask]
    Xte, yte = X_full[test_mask], y_full[test_mask]
    m = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.9, random_state=RNG_SEED,
    )
    m.fit(Xtr, ytr)
    base_pred = m.predict(Xte)
    base_rho, base_p = stats.spearmanr(base_pred, yte) if len(yte) > 2 else (np.nan, np.nan)
    base_r2 = 1 - np.sum((yte - base_pred) ** 2) / np.sum((yte - yte.mean()) ** 2)

    feat_result = {}
    for i, f in enumerate(FEATURES):
        drops_rho = []
        drops_r2 = []
        for _ in range(N_PERM):
            Xte_perm = Xte.copy()
            perm_idx = rng.permutation(len(Xte_perm))
            Xte_perm[:, i] = Xte_perm[perm_idx, i]
            pred_perm = m.predict(Xte_perm)
            rho_perm, _ = stats.spearmanr(pred_perm, yte) if len(yte) > 2 else (np.nan, np.nan)
            r2_perm = 1 - np.sum((yte - pred_perm) ** 2) / np.sum((yte - yte.mean()) ** 2)
            drops_rho.append(base_rho - rho_perm if not np.isnan(base_rho) else np.nan)
            drops_r2.append(base_r2 - r2_perm)
        feat_result[f] = {
            "mean_spearman_drop_when_permuted": float(np.nanmean(drops_rho)),
            "mean_r2_drop_when_permuted": float(np.mean(drops_r2)),
        }
    lofo_perm[held_out] = {
        "n_test": int(test_mask.sum()), "n_train": int(train_mask.sum()),
        "base_held_out_spearman_pred_vs_actual": float(base_rho) if not np.isnan(base_rho) else None,
        "base_held_out_r2_log": float(base_r2),
        "feature_permutation_importance": feat_result,
    }
results["lofo_permutation_importance_rank_transfer"] = lofo_perm

# Consistency: which feature has a POSITIVE mean spearman-drop (i.e. actually
# helps held-out rank prediction) in ALL 4 held-out facilities?
perm_consistency = {}
for f in FEATURES:
    drops = {src: lofo_perm[src]["feature_permutation_importance"][f]["mean_spearman_drop_when_permuted"]
             for src in facilities}
    perm_consistency[f] = {
        "drops_by_held_out_facility": drops,
        "positive_in_all_4": all(v > 0 for v in drops.values()),
        "positive_in_n_of_4": sum(1 for v in drops.values() if v > 0),
    }
results["lofo_permutation_consistency_summary"] = perm_consistency

# -------------------------------------------------------------------
# Final honest verdict
# -------------------------------------------------------------------
re_sc = sign_consistency["Re_Omega"]
re_fe = fe_partial["Re_Omega"]
re_shap = shap_sign_consistency["Re_Omega"]
re_perm = perm_consistency["Re_Omega"]

verdict = {
    "claim_tested": "Re_Omega's sign/rank effect on log(Cp) is consistent across all 4 facilities even though the quantitative exponent/law does not transfer.",
    "check_1_per_facility_simple_slope_all_4_negative": re_sc["all_negative"],
    "check_1_slopes": re_sc["slopes"],
    "check_1_all_ci_exclude_zero": re_sc["all_ci_exclude_zero"],
    "check_3_fixed_effects_partial_corr_sign_negative": (re_fe["within_facility_demeaned_pearson_r"] < 0) if re_fe.get("testable") else None,
    "check_3_fixed_effects_partial_corr_r": re_fe.get("within_facility_demeaned_pearson_r"),
    "check_4_shap_local_slope_all_4_negative": re_shap["all_negative"],
    "check_4_shap_rhos": re_shap["rhos"],
    "check_5_lofo_permutation_positive_in_all_4_held_out": re_perm["positive_in_all_4"],
    "check_5_drops": re_perm["drops_by_held_out_facility"],
    "geometric_pi_variables_note": (
        "Pi_confinement, Pi_gap, Pi_aspect_axial have ZERO within-facility variance in "
        "Liu2024 and Zheng2024 (single geometry each), so their per-facility sign "
        "consistency can only be checked on 2 of 4 facilities (Guo2024, Vrancik1968), "
        "and within those two facilities Pi_confinement and Pi_gap are severely "
        "collinear (r=-1.00 in Guo2024, r=-0.976 in Vrancik1968), so no confident "
        "individual attribution between confinement-effect and gap-effect is possible "
        "even there."
    ),
    "pi_confinement_sign_in_2_testable_facilities": sign_consistency["Pi_confinement"]["slopes"],
    "pi_gap_sign_in_2_testable_facilities": sign_consistency["Pi_gap"]["slopes"],
    "pi_aspect_axial_sign_in_2_testable_facilities": sign_consistency["Pi_aspect_axial"]["slopes"],
}

all_checks_pass_for_re = (
    verdict["check_1_per_facility_simple_slope_all_4_negative"]
    and verdict["check_3_fixed_effects_partial_corr_sign_negative"]
    and verdict["check_4_shap_local_slope_all_4_negative"]
    and verdict["check_5_lofo_permutation_positive_in_all_4_held_out"]
)
verdict["ALL_FOUR_INDEPENDENT_CHECKS_AGREE_RE_OMEGA_NEGATIVE_AND_TRANSFERS"] = bool(all_checks_pass_for_re)
results["verdict"] = verdict

with open(OUT_PATH, "w") as fh:
    json.dump(results, fh, indent=2)

print(json.dumps(verdict, indent=2))
print("\nFull results written to", OUT_PATH)
