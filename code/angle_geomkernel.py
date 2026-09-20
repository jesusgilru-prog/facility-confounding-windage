"""
Angle: EMPIRICAL - kernel/locally-weighted regression using geometric similarity
(distance in Pi-group space, and/or geometry_type exact match) to reweight
training points when predicting a held-out facility, under strict LOFO.

Rules respected:
- Held-out facility's Cp values are NEVER used for fitting (strict LOFO).
- Held-out facility's geometry (Pi_confinement, Pi_gap, Pi_aspect_axial,
  geometry_type) IS used at test time to build weights - this is realistic
  (you know a new facility's geometry before you have measured its windage).
- Bandwidth for the Gaussian kernel is selected by NESTED LOFO *inside the
  training pool only* (never touching the held-out facility's target),
  to avoid leaking test information into a tuning choice.
- No numbers are invented; everything below is computed from the real CSV.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT_JSON = _ROOT + "/results/angle_geomkernel_results.json"

RNG = 0
FEATURES_GEOM = ["Pi_confinement", "Pi_gap", "Pi_aspect_axial"]  # distance metric space
REGR_COLS = ["log_Re", "Pi_confinement", "Pi_gap", "Pi_aspect_axial"]  # regression predictors

df = pd.read_csv(DATA_PATH)
df["log_Cp"] = np.log10(df["Cp"])
df["log_Re"] = np.log10(df["Re_Omega"])

FACILITIES = df["source"].unique().tolist()


def standardize(train_vals, apply_vals, cols):
    """Fit mean/std on train_vals[cols], apply to apply_vals[cols]. Returns np arrays."""
    mu = train_vals[cols].mean()
    sd = train_vals[cols].std(ddof=0).replace(0, 1.0)
    return (apply_vals[cols] - mu) / sd, mu, sd


def gaussian_weights(test_row_std, train_std_arr, bandwidth):
    d2 = np.sum((train_std_arr - test_row_std.values) ** 2, axis=1)
    if np.isinf(bandwidth):
        return np.ones_like(d2)
    w = np.exp(-d2 / (2.0 * bandwidth ** 2))
    return w


def geomtype_weights(test_geomtype, train_geomtypes, mismatch_weight):
    """Binary/soft weight: 1.0 for exact geometry_type match, mismatch_weight otherwise."""
    return np.where(train_geomtypes.values == test_geomtype, 1.0, mismatch_weight)


def fit_predict_weighted(train_df, test_df, weights_per_test_point):
    """
    weights_per_test_point: list of arrays, one per row of test_df, each of length len(train_df)
    Fits a separate weighted linear regression per test point (true locally-weighted regression).
    Returns predictions (log_Cp).
    """
    X_train = train_df[REGR_COLS].values
    y_train = train_df["log_Cp"].values
    preds = np.zeros(len(test_df))
    X_test = test_df[REGR_COLS].values
    for i in range(len(test_df)):
        w = weights_per_test_point[i]
        if w.sum() <= 1e-12:
            w = np.ones_like(w)  # degenerate fallback: uniform
        model = LinearRegression()
        model.fit(X_train, y_train, sample_weight=w)
        preds[i] = model.predict(X_test[i : i + 1])[0]
    return preds


def run_variant(bandwidth_geom, mismatch_weight, nested_select=False, candidate_bandwidths=None):
    """
    bandwidth_geom: float or np.inf -> Gaussian kernel bandwidth in standardized Pi-space
                    (np.inf reduces to unweighted pooling, our baseline check)
    mismatch_weight: float in [0,1] -> extra multiplicative weight for geometry_type mismatch
                    (1.0 = no geometry_type effect, 0.0 = hard-exclude mismatched geometry_type)
    nested_select: if True, ignore bandwidth_geom and select best bandwidth per outer fold via
                   nested LOFO among training facilities only (no test-facility leakage).
    Returns dict with pooled and per-facility R2 (in log_Cp space).
    """
    all_true = []
    all_pred = []
    per_facility = {}
    chosen_bandwidths = {}

    for held_out in FACILITIES:
        train_df = df[df["source"] != held_out].reset_index(drop=True)
        test_df = df[df["source"] == held_out].reset_index(drop=True)

        bw = bandwidth_geom
        if nested_select:
            # nested LOFO strictly inside train_df (held_out never touched)
            inner_facilities = train_df["source"].unique().tolist()
            best_bw, best_score = None, -np.inf
            for cand in candidate_bandwidths:
                inner_true, inner_pred = [], []
                for inner_held in inner_facilities:
                    inner_train = train_df[train_df["source"] != inner_held].reset_index(drop=True)
                    inner_test = train_df[train_df["source"] == inner_held].reset_index(drop=True)
                    if len(inner_train) < 5 or len(inner_test) == 0:
                        continue
                    std_train, mu, sd = standardize(inner_train, inner_train, FEATURES_GEOM)
                    std_test, _, _ = standardize(inner_train, inner_test, FEATURES_GEOM)
                    w_list = []
                    for j in range(len(inner_test)):
                        w = gaussian_weights(std_test.iloc[j], std_train.values, cand)
                        if mismatch_weight < 1.0:
                            w = w * geomtype_weights(
                                inner_test["geometry_type"].iloc[j],
                                inner_train["geometry_type"],
                                mismatch_weight,
                            )
                        w_list.append(w)
                    preds = fit_predict_weighted(inner_train, inner_test, w_list)
                    inner_true.extend(inner_test["log_Cp"].tolist())
                    inner_pred.extend(preds.tolist())
                if len(inner_true) >= 3:
                    score = r2_score(inner_true, inner_pred)
                    if score > best_score:
                        best_score, best_bw = score, cand
            bw = best_bw if best_bw is not None else np.inf
        chosen_bandwidths[held_out] = bw

        std_train, mu, sd = standardize(train_df, train_df, FEATURES_GEOM)
        std_test, _, _ = standardize(train_df, test_df, FEATURES_GEOM)
        w_list = []
        for j in range(len(test_df)):
            w = gaussian_weights(std_test.iloc[j], std_train.values, bw)
            if mismatch_weight < 1.0:
                w = w * geomtype_weights(
                    test_df["geometry_type"].iloc[j], train_df["geometry_type"], mismatch_weight
                )
            w_list.append(w)
        preds = fit_predict_weighted(train_df, test_df, w_list)

        r2_fac = r2_score(test_df["log_Cp"].values, preds)
        per_facility[held_out] = {
            "r2_log": float(r2_fac),
            "n": int(len(test_df)),
            "bandwidth_used": None if bw is None else (float(bw) if np.isfinite(bw) else "inf"),
        }
        all_true.extend(test_df["log_Cp"].tolist())
        all_pred.extend(preds.tolist())

    pooled_r2 = float(r2_score(all_true, all_pred))
    return {
        "pooled_r2_log": pooled_r2,
        "per_facility": per_facility,
        "chosen_bandwidths": {k: (float(v) if np.isfinite(v) else "inf") for k, v in chosen_bandwidths.items()},
    }


results = {}

# 1. Baseline: unweighted pooled OLS (bandwidth = inf => uniform weights), sanity check
# against the already-established "classical linear regression" LOFO result.
results["baseline_unweighted_ols"] = run_variant(bandwidth_geom=np.inf, mismatch_weight=1.0)

# 2. Fixed-bandwidth Gaussian kernel in Pi-space, several fixed bandwidths (no nested tuning,
#    to see the raw sensitivity / best-case-if-we-cheated-on-bandwidth ceiling).
for bw in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]:
    results[f"fixed_bandwidth_{bw}"] = run_variant(bandwidth_geom=bw, mismatch_weight=1.0)

# 3. Honest nested-CV bandwidth selection (bandwidth chosen without ever touching the held-out
#    facility's Cp values) -- this is the methodologically defensible version.
candidate_bws = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]
results["nested_selected_bandwidth"] = run_variant(
    bandwidth_geom=None, mismatch_weight=1.0, nested_select=True, candidate_bandwidths=candidate_bws
)

# 4. geometry_type hard match (only use points with the SAME geometry_type as the held-out
#    facility; if none exist in the training pool, falls back to uniform weights via the
#    w.sum()<=0 fallback in fit_predict_weighted). No Pi-space kernel (bandwidth=inf) combined
#    with hard geometry_type exclusion of mismatches (mismatch_weight=0).
results["geomtype_hard_match_only"] = run_variant(bandwidth_geom=np.inf, mismatch_weight=0.0)

# 5. Combination: nested-selected Pi-space bandwidth AND geometry_type soft downweighting
#    (mismatch gets weight 0.3 instead of being excluded, since geometry_type overlap across
#    facilities is very sparse -- see notes).
results["nested_bandwidth_plus_geomtype_soft"] = run_variant(
    bandwidth_geom=None,
    mismatch_weight=0.3,
    nested_select=True,
    candidate_bandwidths=candidate_bws,
)

# Coverage check: which held-out facilities actually HAVE a geometry_type match available
# in their training pool (needed to interpret variant 4 honestly).
geomtype_overlap = {}
for held_out in FACILITIES:
    train_df = df[df["source"] != held_out]
    test_types = set(df[df["source"] == held_out]["geometry_type"].unique())
    train_types = set(train_df["geometry_type"].unique())
    geomtype_overlap[held_out] = {
        "test_geometry_types": sorted(test_types),
        "train_geometry_types_available": sorted(train_types),
        "has_exact_match": bool(test_types & train_types),
        "n_train_rows_with_match": int(train_df["geometry_type"].isin(test_types).sum()),
    }
results["geomtype_overlap_diagnostic"] = geomtype_overlap

# 6. Diagnostic: how isolated is each held-out facility in standardized Pi-space?
#    (nearest-neighbor distance to the training pool -- explains WHY kernel weighting
#    can/cannot help: if the nearest training neighbor is already many std-devs away,
#    no kernel bandwidth choice can produce a meaningful local fit.)
nn_diag = {}
for held_out in FACILITIES:
    train_df = df[df["source"] != held_out]
    test_df = df[df["source"] == held_out]
    std_train, mu, sd = standardize(train_df, train_df, FEATURES_GEOM)
    std_test, _, _ = standardize(train_df, test_df, FEATURES_GEOM)
    dists = []
    for j in range(len(std_test)):
        d = np.sqrt(((std_train.values - std_test.iloc[j].values) ** 2).sum(axis=1))
        dists.append(d.min())
    dists = np.array(dists)
    nn_diag[held_out] = {
        "mean_nn_dist_std_units": float(dists.mean()),
        "median_nn_dist_std_units": float(np.median(dists)),
        "min_nn_dist_std_units": float(dists.min()),
        "max_nn_dist_std_units": float(dists.max()),
    }
results["nearest_neighbor_distance_diagnostic_std_units"] = nn_diag

# 7. Oracle (LEAKAGE, NOT a valid protocol) bandwidth sweep per facility: selects, for each
#    held-out facility, the bandwidth that maximizes THAT facility's own R2 -- i.e. it cheats
#    by looking at the held-out facility's true Cp values to pick the bandwidth. This is
#    reported ONLY to establish an upper-bound ceiling ("even if we cheated, how good could
#    kernel weighting possibly get?"), never as a legitimate result.
oracle_bandwidths = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, np.inf]
oracle = {}
for held_out in FACILITIES:
    train_df = df[df["source"] != held_out].reset_index(drop=True)
    test_df = df[df["source"] == held_out].reset_index(drop=True)
    best_r2, best_bw = -np.inf, None
    for bw in oracle_bandwidths:
        std_train, _, _ = standardize(train_df, train_df, FEATURES_GEOM)
        std_test, _, _ = standardize(train_df, test_df, FEATURES_GEOM)
        w_list = [gaussian_weights(std_test.iloc[j], std_train.values, bw) for j in range(len(test_df))]
        preds = fit_predict_weighted(train_df, test_df, w_list)
        r2 = r2_score(test_df["log_Cp"].values, preds)
        if r2 > best_r2:
            best_r2, best_bw = r2, bw
    oracle[held_out] = {
        "oracle_best_r2_log_LEAKAGE_NOT_VALID": float(best_r2),
        "oracle_bandwidth_LEAKAGE_NOT_VALID": float(best_bw) if np.isfinite(best_bw) else "inf",
        "caveat": "bandwidth chosen using this facility's own true Cp values - illustrates ceiling only, not a defensible LOFO result",
    }
results["oracle_leakage_ceiling_DO_NOT_USE_AS_RESULT"] = oracle

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
