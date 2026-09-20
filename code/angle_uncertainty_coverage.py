"""
Angle: honest predictive UNCERTAINTY under LOFO, instead of point accuracy.

Question: even though point predictions (R2) are known to be bad/unstable
cross-facility (pooled LOFO R2_log from -0.34 best to -109 worst across 16
prior methods), can a model at least produce prediction INTERVALS whose
nominal coverage (e.g. 90%) is honestly achieved on the truly unseen,
held-out facility? If yes -> genuinely useful engineering result ("the model
doesn't know the right answer but knows how uncertain it is"). If no ->
important honest negative: uncertainty is ALSO miscalibrated cross-facility,
not just the point estimate.

Two independent, principled interval-construction methods, both under
strict leave-one-facility-out (LOFO):

  METHOD A: Split Conformal Prediction on top of an OLS linear baseline
    (log Cp ~ features). Classic split-conformal (Lei et al. 2018 /
    Vovk et al.): the 3 training facilities are split into a proper-train
    set (fit OLS) and a calibration set (compute |residual| conformity
    scores, take the finite-sample-corrected quantile). The held-out
    facility NEVER touches fitting or calibration -- only final scoring.
    This gives, by construction, EXACT marginal (1-alpha) coverage on
    exchangeable data. LOFO breaks exchangeability by design (held-out
    facility is a different data-generating regime), so this is precisely
    the assumption under test.

  METHOD B: Conformalized Quantile Regression (CQR, Romano et al. 2019)
    using GradientBoostingRegressor(loss="quantile") for the lower/upper
    quantile functions, again fit on proper-train only, corrected with a
    calibration-set conformity score, and evaluated purely out-of-sample
    on the held-out facility. CQR is locally adaptive (interval width can
    vary with the input), unlike method A's constant-width interval, so
    it is a meaningfully different way the "genuinely positive" story
    could play out.

For both methods: nominal levels 90% and 80%; feature sets RE_ONLY
(log Re_Omega alone -- the one predictor with real transferable signal
found so far) and FULL (log Re_Omega + Pi_confinement + Pi_gap +
Pi_aspect_axial -- the set that overfits to facility in point-prediction).

Repeated random proper-train/calibration splits (fixed seeds, not
np.random unseeded) to average out calibration-split noise. All scoring
on the held-out facility is out-of-sample in every repeat: calibration
data is always drawn only from the 3 training facilities.

Decision rule (this angle's honest bar, consistent with the paper's
existing bar): a genuinely positive result requires the pooled empirical
coverage to be close to nominal (say within ~10 percentage points) AND
no individual held-out facility's coverage catastrophically below nominal
(say not more than ~20-25 points under). Anything else is a negative /
partial result and must be reported as such.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

RNG_BASE_SEED = 20260818
N_REPEATS = 40
CALIB_FRACTION = 0.3
ALPHAS = [0.10, 0.20]  # nominal miscoverage -> 90% and 80% intervals
FEATURE_SETS = {
    "RE_ONLY": ["logRe"],
    "FULL": ["logRe", "Pi_confinement", "Pi_gap", "Pi_aspect_axial"],
}

DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT_PATH = _ROOT + "/results/angle_uncertainty_coverage_results.json"


def load_data():
    df = pd.read_csv(DATA_PATH)
    d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_confinement",
                           "Pi_aspect_axial", "source"]).copy()
    d = d[d["Cp"] > 0].reset_index(drop=True)
    d["logCp"] = np.log(d["Cp"])
    d["logRe"] = np.log(d["Re_Omega"])
    return d


def split_conformal_once(train_df, test_df, feat_cols, alpha, seed):
    """Method A: OLS + split conformal. Returns (covered_bool_array, widths_array, interval_halfwidth)."""
    rng = np.random.RandomState(seed)
    n_train = len(train_df)
    idx = rng.permutation(n_train)
    n_calib = max(3, int(round(CALIB_FRACTION * n_train)))
    calib_idx = idx[:n_calib]
    proper_idx = idx[n_calib:]
    if len(proper_idx) < 2:
        return None  # not enough data to fit

    Xtr_raw = train_df.iloc[proper_idx][feat_cols].values
    ytr = train_df.iloc[proper_idx]["logCp"].values
    Xcal_raw = train_df.iloc[calib_idx][feat_cols].values
    ycal = train_df.iloc[calib_idx]["logCp"].values
    Xtest_raw = test_df[feat_cols].values
    ytest = test_df["logCp"].values

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)
    Xcal = scaler.transform(Xcal_raw)
    Xtest = scaler.transform(Xtest_raw)

    model = LinearRegression().fit(Xtr, ytr)
    resid_cal = np.abs(ycal - model.predict(Xcal))

    n = len(resid_cal)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = np.quantile(resid_cal, q_level, method="higher")

    pred_test = model.predict(Xtest)
    lower = pred_test - q_hat
    upper = pred_test + q_hat
    covered = (ytest >= lower) & (ytest <= upper)
    width = upper - lower
    return covered, width


def cqr_once(train_df, test_df, feat_cols, alpha, seed):
    """Method B: GBM quantile regression + CQR calibration correction."""
    rng = np.random.RandomState(seed)
    n_train = len(train_df)
    idx = rng.permutation(n_train)
    n_calib = max(3, int(round(CALIB_FRACTION * n_train)))
    calib_idx = idx[:n_calib]
    proper_idx = idx[n_calib:]
    if len(proper_idx) < 5:
        return None

    Xtr_raw = train_df.iloc[proper_idx][feat_cols].values
    ytr = train_df.iloc[proper_idx]["logCp"].values
    Xcal_raw = train_df.iloc[calib_idx][feat_cols].values
    ycal = train_df.iloc[calib_idx]["logCp"].values
    Xtest_raw = test_df[feat_cols].values
    ytest = test_df["logCp"].values

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)
    Xcal = scaler.transform(Xcal_raw)
    Xtest = scaler.transform(Xtest_raw)

    lo_q = alpha / 2
    hi_q = 1 - alpha / 2

    gbm_kwargs = dict(n_estimators=80, max_depth=2, learning_rate=0.08,
                       min_samples_leaf=max(2, len(proper_idx) // 15),
                       random_state=seed)
    model_lo = GradientBoostingRegressor(loss="quantile", alpha=lo_q, **gbm_kwargs)
    model_hi = GradientBoostingRegressor(loss="quantile", alpha=hi_q, **gbm_kwargs)
    model_lo.fit(Xtr, ytr)
    model_hi.fit(Xtr, ytr)

    cal_lo = model_lo.predict(Xcal)
    cal_hi = model_hi.predict(Xcal)
    # CQR conformity score (Romano et al. 2019)
    scores = np.maximum(cal_lo - ycal, ycal - cal_hi)
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = np.quantile(scores, q_level, method="higher")

    test_lo = model_lo.predict(Xtest) - q_hat
    test_hi = model_hi.predict(Xtest) + q_hat
    covered = (ytest >= test_lo) & (ytest <= test_hi)
    width = test_hi - test_lo
    return covered, width


def run_method(method_fn, d, sources, feat_cols, alpha):
    per_facility = {}
    pooled_covered = []
    pooled_width = []
    for held_out in sources:
        train_df = d[d["source"] != held_out].reset_index(drop=True)
        test_df = d[d["source"] == held_out].reset_index(drop=True)
        n_test = len(test_df)
        covered_accum = np.zeros(n_test)
        width_accum = np.zeros(n_test)
        n_valid_reps = 0
        for r in range(N_REPEATS):
            seed = RNG_BASE_SEED + 1000 * r + hash(held_out) % 997
            out = method_fn(train_df, test_df, feat_cols, alpha, seed)
            if out is None:
                continue
            covered, width = out
            covered_accum += covered.astype(float)
            width_accum += width
            n_valid_reps += 1
        mean_covered_per_point = covered_accum / n_valid_reps
        mean_width_per_point = width_accum / n_valid_reps
        facility_coverage = float(np.mean(mean_covered_per_point))
        facility_width = float(np.mean(mean_width_per_point))
        per_facility[held_out] = {
            "n_test": n_test,
            "n_valid_reps": n_valid_reps,
            "coverage": facility_coverage,
            "mean_interval_width_logspace": facility_width,
        }
        pooled_covered.extend(mean_covered_per_point.tolist())
        pooled_width.extend(mean_width_per_point.tolist())
    pooled_coverage = float(np.mean(pooled_covered))
    pooled_width_mean = float(np.mean(pooled_width))
    return per_facility, pooled_coverage, pooled_width_mean


def main():
    d = load_data()
    sources = sorted(d["source"].unique().tolist())
    print(f"n={len(d)} usable rows, facilities={sources}")

    results = {
        "meta": {
            "n_repeats": N_REPEATS,
            "calib_fraction": CALIB_FRACTION,
            "alphas_nominal_miscoverage": ALPHAS,
            "nominal_coverage_pct": [round((1 - a) * 100) for a in ALPHAS],
            "feature_sets": FEATURE_SETS,
            "facility_sizes": {s: int((d["source"] == s).sum()) for s in sources},
            "decision_rule": ("pooled coverage within ~10pp of nominal AND no held-out "
                               "facility more than ~20-25pp below nominal counts as "
                               "genuinely positive; otherwise negative/partial."),
        },
        "methods": {},
    }

    for method_name, method_fn in [
        ("split_conformal_OLS", split_conformal_once),
        ("CQR_gbm_quantile", cqr_once),
    ]:
        results["methods"][method_name] = {}
        for fset_name, feat_cols in FEATURE_SETS.items():
            results["methods"][method_name][fset_name] = {}
            for alpha in ALPHAS:
                nominal_pct = round((1 - alpha) * 100)
                per_fac, pooled_cov, pooled_w = run_method(method_fn, d, sources, feat_cols, alpha)
                results["methods"][method_name][fset_name][f"nominal_{nominal_pct}pct"] = {
                    "per_facility": per_fac,
                    "pooled_coverage": pooled_cov,
                    "pooled_mean_interval_width_logspace": pooled_w,
                    "pooled_coverage_gap_pp": round((pooled_cov - (1 - alpha)) * 100, 2),
                }
                print(f"[{method_name} | {fset_name} | nominal {nominal_pct}%] "
                      f"pooled_coverage={pooled_cov:.3f} "
                      f"(gap={pooled_cov - (1 - alpha):+.3f}) "
                      f"pooled_width_log={pooled_w:.3f}")
                for fac, info in per_fac.items():
                    print(f"    {fac} (n={info['n_test']}): coverage={info['coverage']:.3f}, "
                          f"width_log={info['mean_interval_width_logspace']:.3f}")

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
