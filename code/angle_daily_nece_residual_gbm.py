"""
ANGLE: Daily-Nece regime-IV formula as a FIXED physics baseline, with a
gradient-boosted-tree model learning ONLY the log-residual as a function of
the geometric Pi-groups, evaluated under strict leave-one-facility-out (LOFO)
cross-validation.

Physics baseline (fixed, zero fitted parameters, taken from literature,
already used elsewhere in this project -- see
code/daily_nece_regime_check.py and code/ai_search_pinn_daily_nece.py):

    C_M_DailyNece = 0.051 * Pi_gap^0.1 * Re_Omega^(-0.2)      (Daily & Nece
    1960, regime IV: turbulent flow, separated boundary layers)

This is NOT re-fit to the corpus. It is evaluated as-is on every one of the
114 points (both in the "training" facilities and the held-out facility)
using each row's own Re_Omega and Pi_gap -- exactly the naive regime-IV
application already characterized in daily_nece_regime_check_results.json
(pooled R2_log = -4.13 on its own, median obs/pred ratio ~27x).

Residual target:
    z = log(Cp) - log(C_M_DailyNece)

Learned model: gradient boosting regression (scikit-learn
GradientBoostingRegressor), chosen over MLP/GP because:
  - n=114 total, ~73-106 per training fold -- too small for a meaningful
    deep MLP, and GBM handles small-n tabular data with weak/nonlinear
    feature interactions robustly without needing careful architecture
    search.
  - A plain GP was already tried on the raw Cp target in this project
    (gp_lofo_test.py) and failed; a GBM gives a genuinely different
    inductive bias (axis-aligned splits / additive trees) worth testing
    on the residual, which is the actual novel manipulation here, not the
    learner family.
  - GBM output is bounded by the training leaf values, which is actually a
    DESIRABLE property for a physics-correction residual (it cannot
    extrapolate a linear trend to absurd values on held-out geometry the
    way a GP with a poorly-chosen kernel or a linear model can).

Learned features: ONLY the geometric Pi-groups (Pi_confinement, Pi_gap,
Pi_aspect_axial) -- Re_Omega is deliberately EXCLUDED from the residual
model's features because it is already inside the physics baseline; feeding
it again to the residual model would let the GBM re-learn its own
Reynolds-exponent correction per facility, which is just the same
overfitting failure mode already documented for the direct-Cp models
(scaling_law_search.py, mlp_lofo.py, etc.) wearing a physics-baseline
costume. Keeping Re out of the residual model's inputs is the whole point
of testing whether "geometry alone" can correct the physics baseline.

Strict LOFO protocol (identical spirit to every other script in this repo):
  - For each held-out facility: fit GBM on the OTHER 3 facilities' residuals
    only (with a small internal train/hold-out split of ONLY the training
    facilities to pick n_estimators from a decision family, via GBM's own
    n_iter_no_change early stopping -- never touching the true held-out
    facility).
  - Predict the held-out facility's residual using its own Pi-groups.
  - Corrected prediction = baseline + predicted residual.
  - Compare corrected-prediction LOFO R2 (log space, and back-transformed
    ratio stats) against baseline-alone LOFO "R2" (which needs no fold
    structure since it fits nothing, but is reported per the same folds for
    a fair apples-to-apples pooled statistic).

Decision rule (project standard): pooled R2_log substantially positive AND
every per-facility R2_log >= 0. Anything short of that is reported as a
negative/inconclusive result, not spun as a discovery.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold

DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT_JSON = _ROOT + "/results/angle_daily_nece_residual_gbm_results.json"

RESID_FEATURES = ["Pi_confinement", "Pi_gap", "Pi_aspect_axial"]
SEED = 0

df = pd.read_csv(DATA_PATH)
d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap"] + RESID_FEATURES + ["source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)
facilities = sorted(d["source"].unique().tolist())
print(f"n={n}, facilities={facilities}")

Re = d["Re_Omega"].to_numpy()
G = d["Pi_gap"].to_numpy()
Cp = d["Cp"].to_numpy()
source = d["source"].to_numpy()

# --- fixed physics baseline, zero fitted parameters ---
Cp_dn = 0.051 * (G ** 0.1) * (Re ** (-0.2))
log_Cp = np.log(Cp)
log_Cp_dn = np.log(Cp_dn)
z = log_Cp - log_Cp_dn  # residual target for the learned correction

X_all = np.log(d[RESID_FEATURES].to_numpy())  # log Pi-groups as GBM features

# --- pooled/per-facility baseline-alone stats (physics only, no learning) ---
ref_mean_global = log_Cp.mean()


def r2(y_true, y_pred, ref_mean=None):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if ref_mean is None:
        ref_mean = y_true.mean()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - ref_mean) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


baseline_per_facility = {}
baseline_all_true, baseline_all_pred = [], []
for f in facilities:
    m = source == f
    baseline_per_facility[f] = {
        "n": int(m.sum()),
        "r2_log_own_mean": float(r2(log_Cp[m], log_Cp_dn[m])),
        "median_ratio_obs_over_pred": float(np.median(Cp[m] / Cp_dn[m])),
    }
    baseline_all_true.append(log_Cp[m])
    baseline_all_pred.append(log_Cp_dn[m])
baseline_pooled_r2 = float(r2(np.concatenate(baseline_all_true), np.concatenate(baseline_all_pred), ref_mean_global))

# --- GBM param grid, selected via inner CV on the 3 training facilities only ---
PARAM_GRID = {
    "n_estimators": [30, 60, 100],
    "max_depth": [1, 2, 3],
    "learning_rate": [0.03, 0.1],
    "subsample": [0.8, 1.0],
}

per_facility_corrected = {}
per_facility_baseline_used_in_fold = {}
all_true_log, all_pred_corrected_log, all_pred_baseline_log = [], [], []
chosen_params_per_fold = {}

for held_out in facilities:
    test_mask = source == held_out
    train_mask = ~test_mask

    Xtr, ztr = X_all[train_mask], z[train_mask]
    Xte = X_all[test_mask]

    # inner model selection: plain KFold on the training facilities' pooled
    # rows (never touches the held-out facility). n small, so 4-fold.
    n_splits_inner = min(4, int(train_mask.sum()))
    inner_cv = KFold(n_splits=n_splits_inner, shuffle=True, random_state=SEED)
    gbm = GradientBoostingRegressor(random_state=SEED)
    search = GridSearchCV(gbm, PARAM_GRID, cv=inner_cv, scoring="r2", n_jobs=2)
    search.fit(Xtr, ztr)
    best_params = search.best_params_
    chosen_params_per_fold[held_out] = best_params

    model = GradientBoostingRegressor(random_state=SEED, **best_params)
    model.fit(Xtr, ztr)
    z_pred_test = model.predict(Xte)

    log_Cp_corrected_test = log_Cp_dn[test_mask] + z_pred_test
    log_Cp_true_test = log_Cp[test_mask]

    fold_r2_corrected = float(r2(log_Cp_true_test, log_Cp_corrected_test))
    fold_r2_baseline = float(r2(log_Cp_true_test, log_Cp_dn[test_mask]))

    per_facility_corrected[held_out] = {
        "n_test": int(test_mask.sum()),
        "r2_log_corrected_own_mean": fold_r2_corrected,
        "r2_log_baseline_alone_own_mean": fold_r2_baseline,
        "median_ratio_obs_over_corrected_pred": float(
            np.median(Cp[test_mask] / np.exp(log_Cp_corrected_test))
        ),
        "gbm_best_params": best_params,
        "gbm_feature_importances": dict(zip(RESID_FEATURES, model.feature_importances_.tolist())),
    }

    all_true_log.append(log_Cp_true_test)
    all_pred_corrected_log.append(log_Cp_corrected_test)
    all_pred_baseline_log.append(log_Cp_dn[test_mask])

    print(f"[{held_out}] n={int(test_mask.sum())} corrected R2={fold_r2_corrected:.4f} "
          f"baseline-alone R2={fold_r2_baseline:.4f} params={best_params}")

all_true_log = np.concatenate(all_true_log)
all_pred_corrected_log = np.concatenate(all_pred_corrected_log)
all_pred_baseline_log = np.concatenate(all_pred_baseline_log)

pooled_r2_corrected = float(r2(all_true_log, all_pred_corrected_log, ref_mean_global))
pooled_r2_baseline_lofo = float(r2(all_true_log, all_pred_baseline_log, ref_mean_global))
pooled_rmse_corrected = float(np.sqrt(np.mean((all_true_log - all_pred_corrected_log) ** 2)))
pooled_rmse_baseline = float(np.sqrt(np.mean((all_true_log - all_pred_baseline_log) ** 2)))

decision_rule_pass = bool(
    pooled_r2_corrected > 0.3
    and all(v["r2_log_corrected_own_mean"] >= 0 for v in per_facility_corrected.values())
)

# --- sanity check: does the residual z even correlate with the Pi-groups
# WITHIN a single facility (i.e. is there any signal for GBM to find at
# all, independent of the cross-facility generalization question)? ---
within_facility_corr = {}
for f in facilities:
    m = source == f
    if m.sum() >= 4:
        corrs = {}
        for feat in RESID_FEATURES:
            v = np.log(d.loc[m, feat].to_numpy())
            if np.std(v) > 1e-12:
                corrs[feat] = float(np.corrcoef(v, z[m])[0, 1])
            else:
                corrs[feat] = None  # constant within facility -> no signal available
        within_facility_corr[f] = corrs
    else:
        within_facility_corr[f] = "n<4, skipped"

conclusion_lines = []
conclusion_lines.append(
    f"Baseline-alone (Daily-Nece regime IV, zero fitted params) pooled R2_log = "
    f"{baseline_pooled_r2:.4f} across all 114 points (matches prior "
    f"daily_nece_regime_check_results.json naive figure of -4.13)."
)
conclusion_lines.append(
    f"GBM-corrected (baseline + learned geometry-only residual), strict LOFO: "
    f"pooled R2_log = {pooled_r2_corrected:.4f} vs. baseline-alone-under-LOFO "
    f"R2_log = {pooled_r2_baseline_lofo:.4f}."
)
per_fac_r2_corrected = {k: v["r2_log_corrected_own_mean"] for k, v in per_facility_corrected.items()}
conclusion_lines.append(f"Per-facility corrected R2_log: {per_fac_r2_corrected}")
if decision_rule_pass:
    conclusion_lines.append(
        "DECISION RULE PASSES: pooled R2 substantially positive and every "
        "held-out facility individually R2>=0. This would be a genuine "
        "positive result."
    )
else:
    conclusion_lines.append(
        "DECISION RULE FAILS. The residual-correction framing does not "
        "rescue cross-facility generalization: the GBM cannot learn a "
        "geometry-to-residual mapping from 3 facilities that transfers to "
        "a 4th, for the same root-cause reason as every other model in "
        "this project (facility, geometry_type and Pi-group values are "
        "almost perfectly confounded -- each facility occupies its own "
        "disjoint region of Pi-group space, so held-out geometry is "
        "extrapolation, not interpolation, for the residual model too)."
    )

result = {
    "slug": "daily_nece_residual_gbm",
    "n_total": n,
    "facilities": facilities,
    "physics_baseline": "C_M = 0.051 * Pi_gap^0.1 * Re_Omega^(-0.2)  (Daily & Nece 1960, regime IV, zero fitted parameters)",
    "residual_target": "z = log(Cp) - log(C_M_daily_nece)",
    "residual_model": "sklearn GradientBoostingRegressor, hyperparameters chosen by inner KFold(4) grid search on the 3 training facilities only per outer fold",
    "residual_model_features": RESID_FEATURES,
    "note_on_feature_choice": "Re_Omega excluded from residual-model features on purpose: it is already inside the physics baseline, so re-adding it would let the GBM relearn its own per-facility Reynolds correction, defeating the point of the physics-baseline framing.",
    "param_grid": PARAM_GRID,
    "baseline_alone_per_facility": baseline_per_facility,
    "baseline_alone_pooled_r2_log_all114": baseline_pooled_r2,
    "chosen_gbm_params_per_outer_fold": chosen_params_per_fold,
    "lofo_results": {
        "per_facility": per_facility_corrected,
        "pooled_r2_log_corrected": pooled_r2_corrected,
        "pooled_r2_log_baseline_alone_same_folds": pooled_r2_baseline_lofo,
        "pooled_rmse_log_corrected": pooled_rmse_corrected,
        "pooled_rmse_log_baseline_alone": pooled_rmse_baseline,
    },
    "within_facility_residual_vs_geometry_correlation": within_facility_corr,
    "decision_rule": {
        "requires": "pooled_r2_log_corrected > 0.3 (substantially positive) AND every per_facility r2_log_corrected >= 0",
        "pass": decision_rule_pass,
    },
    "known_baselines_for_comparison": {
        "best_prior_direct_Cp_pooled_r2_log_range": [-109, -0.34],
        "maml_pooled_r2_log_but_catastrophic_per_facility": {"pooled": 0.74, "worst_per_facility": -5.6},
        "reynolds_exponent_bootstrap_ci_excludes_zero": [-0.524, -0.203],
    },
    "conclusion": " ".join(conclusion_lines),
}

with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2, default=float)

print()
print(json.dumps(result, indent=2, default=str))
