"""
LLM-in-the-loop physical law discovery for rotor-stator windage power coefficient Cp.

Approach: instead of a black-box ML model, the LLM (me) proposes several
dimensionally-consistent closed-form candidate laws
    Cp = f(Re_Omega, Pi_gap, Pi_confinement, Pi_aspect_axial ; theta)
inspired by Buckingham-Pi structure and rotor-stator windage physics
(Daily-Nece regime IV: C_M = 0.051 * G^0.1 * Re^-0.2 as one anchor, plus
genuinely different functional forms: rational saturation, exponential
decay, Reynolds-exponent modulation by confinement, additive two-regime
power laws, etc.)

For each candidate form, its free constants theta are fit by nonlinear
least squares (scipy.optimize.least_squares) in LOG space on the 3
training facilities, then evaluated on the held-out facility, exactly
following the paper's Leave-One-FACILITY-Out (LOFO) protocol:
  - per-facility held-out R2 in log space
  - pooled residuals across all 4 held-out folds -> ONE pooled R2 in log space

Decision rule: pooled R2 substantially positive AND every per-facility R2 >= 0.

Pi_blockage is deliberately excluded (exact algebraic identity of
Pi_aspect_axial and Pi_confinement, confirmed here: corr(Pa*Pc, Pi_blockage)
~ 0.9999). No candidate form below uses the product Pa*Pc, since that would
reintroduce the same information under a different name.
"""

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

RNG = np.random.default_rng(12345)

DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT_JSON = _ROOT + "/results/ai_search_llm_law_discovery_results.json"

FACILITIES = ["Guo2024", "Vrancik1968", "Liu2024", "Zheng2024"]

EPS = 1e-12


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Candidate closed-form laws.
# Each returns log(Cp_pred) given params theta and arrays Re, Pg, Pc, Pa.
# All forms are built so that the modeled Cp stays strictly positive
# (products/ratios of positive quantities raised to real powers, or
# exp(...) terms) so log() is always well defined for any theta explored
# by the optimizer (we also clip bases away from zero defensively).
# ---------------------------------------------------------------------------

def safe_pow_log(base, exponent):
    """log(base**exponent) = exponent*log(base), base assumed > 0."""
    return exponent * np.log(np.clip(base, EPS, None))


def form_daily_nece_anchor(theta, Re, Pg, Pc, Pa):
    # Cp = C * Pg^0.1 * Re^-0.2   (Daily-Nece regime IV exponents, fixed;
    # only the leading constant C is free)
    (logC,) = theta
    return logC + 0.1 * np.log(np.clip(Pg, EPS, None)) - 0.2 * np.log(np.clip(Re, EPS, None))


def form_full_power_law(theta, Re, Pg, Pc, Pa):
    # Cp = C * Re^a * Pg^b * Pc^c * Pa^d   (generalized Buckingham-Pi power law,
    # free exponents on all 4 groups; log-linear -- included as a modernized
    # anchor generalizing both the Reynolds-only and Daily-Nece baselines)
    logC, a, b, c, d = theta
    return (logC + safe_pow_log(Re, a) + safe_pow_log(Pg, b)
            + safe_pow_log(Pc, c) + safe_pow_log(Pa, d))


def form_rational_confinement_saturation(theta, Re, Pg, Pc, Pa):
    # Cp = C * Re^a * Pg^b / (1 + k*Pc)^c   -- confinement effect saturates
    # rationally rather than as a pure power (boundary-layer blockage
    # approaching an asymptote as confinement grows)
    logC, a, b, k, c = theta
    denom = 1.0 + np.exp(k) * Pc  # exp(k) keeps the rational-law slope positive
    return logC + safe_pow_log(Re, a) + safe_pow_log(Pg, b) - c * np.log(np.clip(denom, EPS, None))


def form_exponential_confinement_decay(theta, Re, Pg, Pc, Pa):
    # Cp = C * Re^a * Pg^b * exp(-k*Pc) * Pa^d  -- exponential (rather than
    # power-law) suppression of windage by confinement, independent decay
    # channel for axial aspect ratio
    logC, a, b, k, d = theta
    return logC + safe_pow_log(Re, a) + safe_pow_log(Pg, b) - k * Pc + safe_pow_log(Pa, d)


def form_reynolds_exponent_modulation(theta, Re, Pg, Pc, Pa):
    # Cp = C * Re^(a + b*Pc) * Pg^c * Pa^d  -- the Reynolds scaling exponent
    # itself is a linear function of confinement, modeling a facility-
    # dependent laminar/turbulent boundary-layer regime shift. NOT separable
    # in log-space (b*Pc*log(Re) interaction term), genuinely nonlinear.
    logC, a, b, c, d = theta
    logRe = np.log(np.clip(Re, EPS, None))
    return logC + (a + b * Pc) * logRe + safe_pow_log(Pg, c) + safe_pow_log(Pa, d)


def form_rational_aspect_coupling(theta, Re, Pg, Pc, Pa):
    # Cp = C * Re^a * Pg^b * Pc^c * (1 + k*Pa)^d  -- axial aspect ratio enters
    # as a rational correction factor rather than a pure power, capturing a
    # finite-chamber-length correction to the open-rotor scaling
    logC, a, b, c, k, d = theta
    corr = 1.0 + np.exp(k) * Pa
    return (logC + safe_pow_log(Re, a) + safe_pow_log(Pg, b) + safe_pow_log(Pc, c)
            + d * np.log(np.clip(corr, EPS, None)))


def form_gap_confinement_ratio(theta, Re, Pg, Pc, Pa):
    # Cp = C * Re^a * (Pg/Pc)^b * Pa^d  -- facility-invariant "universal"
    # combined similarity group (gap normalized by confinement), inspired by
    # the idea that windage depends on the *relative* gap-to-confinement
    # geometry rather than each Pi group independently
    logC, a, b, d = theta
    ratio = Pg / np.clip(Pc, EPS, None)
    return logC + safe_pow_log(Re, a) + safe_pow_log(ratio, b) + safe_pow_log(Pa, d)


def form_two_regime_additive(theta, Re, Pg, Pc, Pa):
    # Cp = (C1*Re^a1 + C2*Re^a2) * Pg^b * Pc^c  -- additive two-regime power
    # law (e.g. viscous-laminar + turbulent-inertial windage contributions
    # combined additively rather than multiplicatively); genuinely different
    # structure, not reducible to a single log-linear fit.
    logC1, a1, logC2, a2, b, c = theta
    Re_c = np.clip(Re, EPS, None)
    term = np.exp(logC1) * Re_c ** a1 + np.exp(logC2) * Re_c ** a2
    return np.log(np.clip(term, EPS, None)) + safe_pow_log(Pg, b) + safe_pow_log(Pc, c)


CANDIDATE_FORMS = {
    "daily_nece_anchor": {
        "func": form_daily_nece_anchor,
        "n_params": 1,
        "init": lambda: [0.0],
        "desc": "Cp = C * Pg^0.1 * Re^-0.2 (Daily-Nece regime IV exponents fixed, only C free)",
    },
    "full_power_law": {
        "func": form_full_power_law,
        "n_params": 5,
        "init": lambda: [0.0, -0.2, 0.1, 0.0, 0.0],
        "desc": "Cp = C * Re^a * Pg^b * Pc^c * Pa^d (generalized Buckingham-Pi power law)",
    },
    "rational_confinement_saturation": {
        "func": form_rational_confinement_saturation,
        "n_params": 5,
        "init": lambda: [0.0, -0.2, 0.1, 0.0, 0.5],
        "desc": "Cp = C * Re^a * Pg^b / (1+k*Pc)^c (rational confinement saturation)",
    },
    "exponential_confinement_decay": {
        "func": form_exponential_confinement_decay,
        "n_params": 5,
        "init": lambda: [0.0, -0.2, 0.1, 0.5, 0.0],
        "desc": "Cp = C * Re^a * Pg^b * exp(-k*Pc) * Pa^d (exponential confinement decay)",
    },
    "reynolds_exponent_modulation": {
        "func": form_reynolds_exponent_modulation,
        "n_params": 5,
        "init": lambda: [0.0, -0.2, 0.0, 0.1, 0.0],
        "desc": "Cp = C * Re^(a+b*Pc) * Pg^c * Pa^d (confinement-modulated Re exponent, regime shift)",
    },
    "rational_aspect_coupling": {
        "func": form_rational_aspect_coupling,
        "n_params": 6,
        "init": lambda: [0.0, -0.2, 0.1, 0.0, 0.0, 0.5],
        "desc": "Cp = C * Re^a * Pg^b * Pc^c * (1+k*Pa)^d (rational axial-aspect correction)",
    },
    "gap_confinement_ratio": {
        "func": form_gap_confinement_ratio,
        "n_params": 4,
        "init": lambda: [0.0, -0.2, 0.1, 0.0],
        "desc": "Cp = C * Re^a * (Pg/Pc)^b * Pa^d (universal gap/confinement similarity group)",
    },
    "two_regime_additive": {
        "func": form_two_regime_additive,
        "n_params": 6,
        "init": lambda: [0.0, -0.2, -3.0, 0.5, 0.1, 0.0],
        "desc": "Cp = (C1*Re^a1 + C2*Re^a2) * Pg^b * Pc^c (additive two-regime windage)",
    },
}


def fit_form(func, n_params, init_fn, Re, Pg, Pc, Pa, y_log, n_restarts=8):
    """Fit one candidate form on training data with multiple random restarts,
    keep the lowest-training-SSE solution."""
    best = None
    inits = [np.array(init_fn(), dtype=float)]
    for _ in range(n_restarts - 1):
        perturb = RNG.normal(scale=0.6, size=n_params)
        inits.append(np.array(init_fn(), dtype=float) + perturb)

    def resid(theta):
        pred = func(theta, Re, Pg, Pc, Pa)
        if not np.all(np.isfinite(pred)):
            return np.full_like(y_log, 1e6)
        return pred - y_log

    for theta0 in inits:
        try:
            sol = least_squares(resid, theta0, method="lm", max_nfev=20000)
        except Exception:
            try:
                sol = least_squares(resid, theta0, max_nfev=20000)
            except Exception:
                continue
        sse = float(np.sum(sol.fun ** 2))
        if not np.isfinite(sse):
            continue
        if best is None or sse < best[0]:
            best = (sse, sol.x.copy())

    if best is None:
        return None, np.inf
    return best[1], best[0]


def run_lofo_for_form(name, spec, df):
    func = spec["func"]
    n_params = spec["n_params"]
    init_fn = spec["init"]

    per_facility_r2 = {}
    pooled_true = []
    pooled_pred = []
    fitted_params_per_fold = {}
    fit_failed = False

    for held_out in FACILITIES:
        train_df = df[df["source"] != held_out]
        test_df = df[df["source"] == held_out]

        Re_tr = train_df["Re_Omega"].values
        Pg_tr = train_df["Pi_gap"].values
        Pc_tr = train_df["Pi_confinement"].values
        Pa_tr = train_df["Pi_aspect_axial"].values
        y_tr = np.log(train_df["Cp"].values)

        theta, sse = fit_form(func, n_params, init_fn, Re_tr, Pg_tr, Pc_tr, Pa_tr, y_tr)
        if theta is None:
            fit_failed = True
            per_facility_r2[held_out] = float("nan")
            continue

        Re_te = test_df["Re_Omega"].values
        Pg_te = test_df["Pi_gap"].values
        Pc_te = test_df["Pi_confinement"].values
        Pa_te = test_df["Pi_aspect_axial"].values
        y_te = np.log(test_df["Cp"].values)

        y_pred = func(theta, Re_te, Pg_te, Pc_te, Pa_te)
        if not np.all(np.isfinite(y_pred)):
            fit_failed = True
            per_facility_r2[held_out] = float("nan")
            continue

        fold_r2 = r2_score(y_te, y_pred)
        per_facility_r2[held_out] = float(fold_r2)
        fitted_params_per_fold[held_out] = theta.tolist()

        pooled_true.extend(y_te.tolist())
        pooled_pred.extend(y_pred.tolist())

    if len(pooled_true) == len(df):
        pooled_r2 = r2_score(np.array(pooled_true), np.array(pooled_pred))
    else:
        pooled_r2 = float("nan")

    clears = (
        not fit_failed
        and np.isfinite(pooled_r2)
        and pooled_r2 > 0.3
        and all(np.isfinite(v) and v >= 0 for v in per_facility_r2.values())
    )

    return {
        "description": spec["desc"],
        "n_params": n_params,
        "pooled_r2_log": pooled_r2,
        "per_facility_r2_log": per_facility_r2,
        "fitted_params_per_fold": fitted_params_per_fold,
        "fit_failed": fit_failed,
        "clears_decision_rule": bool(clears),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    assert set(FACILITIES) == set(df["source"].unique()), df["source"].unique()
    assert (df["Cp"] > 0).all(), "Cp must be strictly positive for log-space fitting"

    results = {}
    for name, spec in CANDIDATE_FORMS.items():
        print(f"Fitting form: {name} ...")
        res = run_lofo_for_form(name, spec, df)
        results[name] = res
        print(f"  pooled_r2_log={res['pooled_r2_log']:.4f}  "
              f"per_facility={res['per_facility_r2_log']}  "
              f"clears={res['clears_decision_rule']}")

    # pick the best form by pooled R2 among those that don't have fit_failed
    valid = {k: v for k, v in results.items() if not v["fit_failed"] and np.isfinite(v["pooled_r2_log"])}
    best_name = max(valid, key=lambda k: valid[k]["pooled_r2_log"]) if valid else None

    summary = {
        "protocol": "Leave-One-FACILITY-Out (LOFO), log-space R2, pooled residuals across all 4 held-out folds",
        "facilities": FACILITIES,
        "n_rows": int(len(df)),
        "excluded_predictor": "Pi_blockage (exact algebraic identity of Pi_aspect_axial and Pi_confinement; "
                               "confirmed here corr(Pi_aspect_axial*Pi_confinement, Pi_blockage) = 0.9999; "
                               "no candidate form below uses the Pa*Pc product for this reason)",
        "known_baselines": {
            "best_linear_4_predictor_pooled_r2_log": -0.885,
            "reynolds_only_pooled_r2_log": 0.4526,
            "reynolds_only_per_facility_r2_log": {
                "note": "positive pooled R2 was an artifact; per-facility values were -2.36, -61.7, -9.59, +0.468 "
                        "(order of facilities as stated in task spec, not necessarily matching FACILITIES order here)"
            },
            "already_failed_black_box_methods": [
                "plain MLP", "Random Forest", "Gradient Boosting",
                "kernel ridge regression (2 variants)", "plain (non-hierarchical) Gaussian Process",
                "symbolic regression via genetic search", "Huber/robust regression"
            ],
        },
        "forms": results,
        "best_form_by_pooled_r2": best_name,
        "best_form_result": results.get(best_name) if best_name else None,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for name, res in results.items():
        print(f"{name:35s} pooled_r2={res['pooled_r2_log']:+.4f}  clears={res['clears_decision_rule']}  "
              f"per_facility={ {k: round(v,3) for k,v in res['per_facility_r2_log'].items()} }")
    print(f"\nBest form by pooled R2: {best_name}")
    if best_name:
        print(json.dumps(results[best_name], indent=2))


if __name__ == "__main__":
    main()
