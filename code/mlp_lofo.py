"""Small-MLP baseline under real leave-one-facility-out (LOFO) validation,
for the 'Windage Power' paper (2026-08-18).

Question: given that the pooled log-log LINEAR model with the 4 winning
predictors (Re, Pi_gap, Pi_confinement, Pi_aspect_axial) catastrophically
fails LOFO (pooled R2=-0.885 in log-space, individual facilities down to
-553.8 -- see scaling_law_search_results.json), and that even the
Re-only linear model (the best-performing baseline found so far across
5 prior lines of investigation) still fails on 3 of 4 facilities despite
a positive pooled R2 (+0.4526), does swapping the LINEAR functional form
for a small MLP (nonlinear, but still trained on the exact same log-
transformed inputs) rescue generalization?

Prior: with n=114 total and as few as 8-45 points per training fold once
one facility is held out, a neural net is expected to overfit AT LEAST
as badly as the linear model, likely worse (more free parameters, same
information-poverty problem: the Pi-groups are near-constant within each
facility, so they still act as disguised facility dummies regardless of
the model class used on top of them). This script tests that prediction
honestly instead of assuming it.

Library check per task instructions: sklearn.neural_network.MLPRegressor
is used (import checked first; torch is NOT installed in this project's
venv and was not pip-installed, since sklearn's MLP is explicitly listed
as an acceptable choice by the task and is available -- no substitute
needed).

Validation standard (mandatory, no exceptions): true LOFO. For every
outer fold, the held-out facility contributes ZERO points to (a) model
training, (b) feature scaling (StandardScaler fit on train only), and
(c) hyperparameter selection (nested: hyperparameters are chosen via an
INNER LOFO loop over the 3 training facilities only, never touching the
held-out outer facility). Pooled R2 is computed in log-space exactly as
in scaling_law_search.py: concatenate held-out residuals across the 4
outer folds, divide by the TSS of the full-dataset log(Cp) around its
global mean.

Two feature sets are tested, per task instructions:
  - FULL: log(Re_Omega), log(Pi_gap), log(Pi_confinement), log(Pi_aspect_axial)
    (same 4 predictors as the linear M6 winner)
  - RE_ONLY: log(Re_Omega) alone (same single predictor as linear M1)

MLP init is stochastic, so every outer-fold evaluation is repeated over
N_SEEDS random seeds and reported as mean +/- std, not a single lucky
draw.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import warnings

import numpy as np
import pandas as pd

try:
    import sklearn
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_OK = True
    SKLEARN_VERSION = sklearn.__version__
except ImportError:
    SKLEARN_OK = False
    SKLEARN_VERSION = None

try:
    import torch  # noqa: F401
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

RNG_SEED = 12345
N_SEEDS = 10  # repeats of the stochastic MLP fit per outer fold, for mean+/-std
INNER_SEED_FOR_SELECTION = 0  # single seed used during hyperparameter search (kept cheap)

warnings.filterwarnings("ignore", category=UserWarning)  # sklearn convergence warnings on tiny folds

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
d = df.dropna(subset=["Cp", "Re_Omega", "g_level", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "M_tip", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)

y = np.log(d["Cp"].values)
lRe = np.log(d["Re_Omega"].values)
lgap = np.log(d["Pi_gap"].values)
lconf = np.log(d["Pi_confinement"].values)
lasp = np.log(d["Pi_aspect_axial"].values)
sources = d["source"].values
facilities = sorted(set(sources))

FEATURE_SETS = {
    "FULL_Re_gap_conf_asp": np.column_stack([lRe, lgap, lconf, lasp]),
    "RE_ONLY": lRe.reshape(-1, 1),
}

# Small architecture / strong-L2 grid, per task instructions (1-2 hidden
# layers, 4-8 neurons, strong L2, early stopping). Kept deliberately small
# (not an exhaustive search) because with as few as ~20-30 points per
# inner-training-fold, an aggressive hyperparameter search would itself
# be a form of overfitting to noise.
HIDDEN_LAYER_GRID = [(4,), (8,), (4, 4), (8, 4)]
ALPHA_GRID = [1.0, 10.0, 30.0]


def make_pipeline(hidden_layer_sizes, alpha, seed):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            activation="tanh",
            solver="adam",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            max_iter=3000,
            random_state=seed,
        )),
    ])


def pooled_r2_log(resid_concat, y_full):
    tss = np.sum((y_full - y_full.mean()) ** 2)
    rss = np.sum(resid_concat ** 2)
    return float(1 - rss / tss)


def fit_eval_once(X, ytrain_idx, ytest_idx, hidden_layer_sizes, alpha, seed):
    Xtr, Xte = X[ytrain_idx], X[ytest_idx]
    ytr, yte = y[ytrain_idx], y[ytest_idx]
    pipe = make_pipeline(hidden_layer_sizes, alpha, seed)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    return yte - pred


def inner_select_hyperparams(X, train_facility_mask):
    """Nested LOFO over the facilities INSIDE the outer-training set only.
    Selects (hidden_layer_sizes, alpha) by mean pooled R2 across the
    inner folds, using a single fixed seed to keep the search cheap and
    reproducible. Never touches the outer held-out facility."""
    inner_facilities = sorted(set(sources[train_facility_mask]))
    if len(inner_facilities) < 2:
        # Can't do inner LOFO with <2 facilities (e.g. degenerate case);
        # fall back to a fixed, conservative default.
        return (4,), 10.0, {"note": "inner LOFO skipped, <2 facilities in train fold; used default hyperparams"}

    best_r2, best_combo = -np.inf, None
    search_log = []
    train_idx_global = np.where(train_facility_mask)[0]
    for hls in HIDDEN_LAYER_GRID:
        for alpha in ALPHA_GRID:
            resids = []
            for f in inner_facilities:
                inner_test_mask = sources[train_idx_global] == f
                inner_test_idx = train_idx_global[inner_test_mask]
                inner_train_idx = train_idx_global[~inner_test_mask]
                if len(inner_train_idx) < 3 or len(inner_test_idx) < 1:
                    continue
                r = fit_eval_once(X, inner_train_idx, inner_test_idx, hls, alpha, INNER_SEED_FOR_SELECTION)
                resids.append(r)
            if not resids:
                continue
            resids = np.concatenate(resids)
            y_train_full = y[train_idx_global]
            # pooled R2 restricted to the training-fold subset (its own mean/TSS)
            tss = np.sum((y_train_full - y_train_full.mean()) ** 2)
            r2 = float(1 - np.sum(resids ** 2) / tss) if tss > 0 else -np.inf
            search_log.append({"hidden_layer_sizes": hls, "alpha": alpha, "inner_lofo_pooled_r2": r2})
            if r2 > best_r2:
                best_r2, best_combo = r2, (hls, alpha)
    if best_combo is None:
        return (4,), 10.0, {"note": "inner search produced no valid combo, used default", "search_log": search_log}
    return best_combo[0], best_combo[1], {"inner_lofo_pooled_r2_at_selection": best_r2, "search_log": search_log}


def run_feature_set(fs_name, X):
    outer_results = {}
    all_resid_mean_over_seeds = []  # one residual vector per outer fold, averaged over seeds (for pooled R2)
    selection_log = {}
    for f in facilities:
        test_mask = sources == f
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        hls, alpha, sel_info = inner_select_hyperparams(X, train_mask)
        selection_log[f] = {"selected_hidden_layer_sizes": hls, "selected_alpha": alpha, **sel_info}

        seed_resids = []
        seed_r2s = []
        for s in range(N_SEEDS):
            seed = RNG_SEED + s
            resid = fit_eval_once(X, train_idx, test_idx, hls, alpha, seed)
            seed_resids.append(resid)
            y_test = y[test_idx]
            tss_f = np.sum((y_test - y_test.mean()) ** 2)
            r2_f = float(1 - np.sum(resid ** 2) / tss_f) if tss_f > 0 else None
            seed_r2s.append(r2_f)
        seed_resids = np.array(seed_resids)  # (N_SEEDS, n_test)
        mean_resid = seed_resids.mean(axis=0)
        all_resid_mean_over_seeds.append(mean_resid)

        outer_results[f] = {
            "n_test": int(test_mask.sum()),
            "selected_hidden_layer_sizes": hls,
            "selected_alpha": alpha,
            "held_out_r2_log_per_seed": seed_r2s,
            "held_out_r2_log_mean": float(np.mean(seed_r2s)) if all(v is not None for v in seed_r2s) else None,
            "held_out_r2_log_std": float(np.std(seed_r2s)) if all(v is not None for v in seed_r2s) else None,
            "held_out_rmse_log_mean_of_seed_means": float(np.sqrt(np.mean(mean_resid ** 2))),
        }

    resid_concat = np.concatenate(all_resid_mean_over_seeds)
    pooled_r2 = pooled_r2_log(resid_concat, y)
    pooled_rmse = float(np.sqrt(np.mean(resid_concat ** 2)))

    return {
        "feature_set": fs_name,
        "n_features": X.shape[1],
        "hyperparam_search_grid": {"hidden_layer_sizes": HIDDEN_LAYER_GRID, "alpha": ALPHA_GRID},
        "hyperparam_selection_per_outer_fold": selection_log,
        "per_facility": outer_results,
        "pooled_lofo_r2_log_mean_over_seeds": pooled_r2,
        "pooled_lofo_rmse_log_mean_over_seeds": pooled_rmse,
    }


if not SKLEARN_OK:
    out = {
        "error": "sklearn not available in this environment, MLP experiment could not run",
        "sklearn_available": False,
        "torch_available": TORCH_OK,
    }
    with open(_ROOT + "/results/mlp_lofo_results.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    raise SystemExit(0)

results_by_feature_set = {}
for fs_name, X in FEATURE_SETS.items():
    print(f"\n=== Running feature set: {fs_name} (n_features={X.shape[1]}) ===")
    res = run_feature_set(fs_name, X)
    results_by_feature_set[fs_name] = res
    print(f"Pooled LOFO R2 (log-space, mean over {N_SEEDS} seeds): {res['pooled_lofo_r2_log_mean_over_seeds']:.4f}")
    for f, r in res["per_facility"].items():
        print(f"  {f:15s} n={r['n_test']:3d}  hls={r['selected_hidden_layer_sizes']}  alpha={r['selected_alpha']:5.1f}  "
              f"R2_mean={r['held_out_r2_log_mean']:.4f}  R2_std={r['held_out_r2_log_std']:.4f}")

# Reference numbers from the established linear baselines (scaling_law_search_results.json),
# reproduced here verbatim from the task context for direct comparison in the output JSON
# (NOT recomputed -- these are cited from the already-verified prior result file).
linear_baseline_reference = {
    "note": "Reproduced from results/scaling_law_search_results.json and prior task context, "
            "NOT recomputed in this script. Shown here only as a comparison anchor.",
    "linear_full_Re_gap_conf_asp_pooled_r2_log": -0.885,
    "linear_full_worst_facility_r2_log": -553.8,
    "linear_re_only_pooled_r2_log": 0.4526,
    "linear_re_only_per_facility_r2_log": {
        "Liu2024": -61.7, "Guo2024_ex_Xia2024": -2.36, "Zheng2024": -9.59, "Vrancik1968": 0.468,
    },
}

out = {
    "n_points": int(n),
    "sources": {k: int(v) for k, v in pd.Series(sources).value_counts().to_dict().items()},
    "library_used": "sklearn.neural_network.MLPRegressor",
    "sklearn_version": SKLEARN_VERSION,
    "torch_available": TORCH_OK,
    "torch_used": False,
    "n_seeds_per_outer_fold": N_SEEDS,
    "rng_seed_base": RNG_SEED,
    "validation_protocol": "True leave-one-facility-out. Hyperparameters (hidden_layer_sizes, alpha) "
                            "selected per outer fold via a NESTED inner-LOFO loop over the 3 training "
                            "facilities only (never touching the held-out outer facility). Feature "
                            "scaling (StandardScaler) fit on the training fold only.",
    "results_by_feature_set": results_by_feature_set,
    "linear_baseline_reference_from_prior_work": linear_baseline_reference,
}

with open(_ROOT + "/results/mlp_lofo_results.json", "w") as fh:
    json.dump(out, fh, indent=2, default=float)

print("\n=== SUMMARY ===")
for fs_name, res in results_by_feature_set.items():
    print(f"{fs_name}: pooled R2 = {res['pooled_lofo_r2_log_mean_over_seeds']:.4f}")
print("\nLinear baselines (reference, not recomputed here):")
print(json.dumps(linear_baseline_reference, indent=2))
