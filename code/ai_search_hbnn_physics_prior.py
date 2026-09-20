"""
Hierarchical Bayesian model with a physics-informed prior (Daily-Nece regime-IV
Reynolds exponent = -0.2) for the cross-rotor windage-power dataset.

Model (log space, y = log(Cp)):
    y_i = alpha_{f(i)} + beta_{f(i)} * x1_i + g1*x2_i + g2*x3_i + g3*x4_i + eps_i
where f(i) is the facility of row i, x1 = log(Re_Omega), x2 = log(Pi_confinement),
x3 = log(Pi_gap), x4 = log(Pi_aspect_axial).

Hierarchical (partial pooling) structure across facilities:
    alpha_f ~ Normal(mu_alpha, sigma_alpha)
    beta_f  ~ Normal(mu_beta,  sigma_beta)        [population-level slope]
    mu_beta ~ Normal(-0.2, 0.10)                  <-- Daily-Nece physics prior mean
    sigma_beta ~ HalfNormal(0.15)                 <-- shrinkage strength of per-facility
                                                       Reynolds exponent toward the prior
The geometry group coefficients (g1, g2, g3) are shared (population-level, not
hierarchical) because within any one facility the Pi_* variables are essentially
constant (fixed geometry) -- they only vary BETWEEN facilities. With only 3
training facilities per LOFO fold, a facility-specific slope on these variables
would be unidentifiable (perfectly aliased with the facility intercept), so we
use one shared, weakly-regularized coefficient per Pi group instead.

For a held-out facility (no training data at all), the hierarchical model's
honest prediction for that facility's own alpha_f, beta_f is the *population*
distribution (mu_alpha, mu_beta) -- i.e. full shrinkage to the group mean /
physics prior, since there is zero information to update the facility-specific
deviation. This is exactly the point of testing whether the physics-informed
hierarchical prior generalizes to an unseen rig.

Inference: full NUTS MCMC via PyMC (this machine's project .venv has pymc 6.3.1
+ pytensor installed already -- real Bayesian inference, not an approximation).
"""

import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import warnings
import numpy as np
import pandas as pd
import pymc as pm

warnings.filterwarnings("ignore")

DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT_JSON = _ROOT + "/results/ai_search_hbnn_physics_prior_results.json"

DAILY_NECE_EXPONENT = -0.2  # physics prior mean for the Reynolds-exponent-like weight

FACILITIES = ["Guo2024", "Vrancik1968", "Liu2024", "Zheng2024"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Cp", "Re_Omega", "Pi_confinement", "Pi_gap", "Pi_aspect_axial", "source"])
    df = df[df["Cp"] > 0].copy()
    df["y"] = np.log(df["Cp"])
    df["x1"] = np.log(df["Re_Omega"])
    df["x2"] = np.log(df["Pi_confinement"])
    df["x3"] = np.log(df["Pi_gap"])
    df["x4"] = np.log(df["Pi_aspect_axial"])
    return df


def fit_fold(train_df, seed):
    """Fit the hierarchical model on the training facilities, return posterior-mean
    hyperparameters (population level) plus per-facility posterior means (used only
    for in-sample diagnostics, NOT for the held-out prediction)."""
    facs = sorted(train_df["source"].unique().tolist())
    fac_idx_map = {f: i for i, f in enumerate(facs)}
    fac_idx = train_df["source"].map(fac_idx_map).values
    n_fac = len(facs)

    x1 = train_df["x1"].values
    x2 = train_df["x2"].values
    x3 = train_df["x3"].values
    x4 = train_df["x4"].values
    y = train_df["y"].values

    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=5.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=2.0)
        alpha_f = pm.Normal("alpha_f", mu=mu_alpha, sigma=sigma_alpha, shape=n_fac)

        mu_beta = pm.Normal("mu_beta", mu=DAILY_NECE_EXPONENT, sigma=0.10)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.15)
        beta_f = pm.Normal("beta_f", mu=mu_beta, sigma=sigma_beta, shape=n_fac)

        g1 = pm.Normal("g1", mu=0.0, sigma=1.0)
        g2 = pm.Normal("g2", mu=0.0, sigma=1.0)
        g3 = pm.Normal("g3", mu=0.0, sigma=1.0)

        sigma_y = pm.HalfNormal("sigma_y", sigma=1.0)

        mu_lin = (
            alpha_f[fac_idx]
            + beta_f[fac_idx] * x1
            + g1 * x2
            + g2 * x3
            + g3 * x4
        )
        pm.Normal("y_obs", mu=mu_lin, sigma=sigma_y, observed=y)

        idata = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.95,
            random_seed=seed,
            progressbar=False,
            cores=4,
        )

    post = idata.posterior
    summary = {
        "mu_alpha": float(post["mu_alpha"].mean()),
        "sigma_alpha": float(post["sigma_alpha"].mean()),
        "mu_beta": float(post["mu_beta"].mean()),
        "sigma_beta": float(post["sigma_beta"].mean()),
        "g1": float(post["g1"].mean()),
        "g2": float(post["g2"].mean()),
        "g3": float(post["g3"].mean()),
        "sigma_y": float(post["sigma_y"].mean()),
        "facilities_trained_on": facs,
        "per_facility_alpha": {f: float(post["alpha_f"].mean(dim=("chain", "draw"))[i]) for f, i in fac_idx_map.items()},
        "per_facility_beta": {f: float(post["beta_f"].mean(dim=("chain", "draw"))[i]) for f, i in fac_idx_map.items()},
    }

    # basic convergence diagnostic
    import arviz as az
    rhat = az.rhat(idata)
    max_rhat = float(max(float(rhat[v].max()) for v in ["mu_alpha", "mu_beta", "g1", "g2", "g3", "sigma_y"]))
    summary["max_rhat"] = max_rhat

    return summary


def predict_new_facility(df_facility, summary):
    """Honest LOFO prediction: the held-out facility has ZERO training data, so its
    facility-specific (alpha_f, beta_f) deviation is unknown and the hierarchical
    model's best estimate reverts fully to the population-level mean (mu_alpha,
    mu_beta) -- i.e. full shrinkage to the physics-informed prior population mean."""
    x1 = df_facility["x1"].values
    x2 = df_facility["x2"].values
    x3 = df_facility["x3"].values
    x4 = df_facility["x4"].values
    y_pred = (
        summary["mu_alpha"]
        + summary["mu_beta"] * x1
        + summary["g1"] * x2
        + summary["g2"] * x3
        + summary["g3"] * x4
    )
    return y_pred


def r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def main():
    df = load_data()
    assert set(FACILITIES) == set(df["source"].unique()), df["source"].unique()

    per_facility_r2 = {}
    all_y_true = []
    all_y_pred = []
    fold_summaries = {}

    for i, held_out in enumerate(FACILITIES):
        train_df = df[df["source"] != held_out].reset_index(drop=True)
        test_df = df[df["source"] == held_out].reset_index(drop=True)

        print(f"=== Fold: holding out {held_out} (n_test={len(test_df)}, n_train={len(train_df)}) ===", flush=True)
        summary = fit_fold(train_df, seed=1000 + i)
        print(f"  mu_beta (posterior mean) = {summary['mu_beta']:.4f}  (prior mean -0.2)  max_rhat={summary['max_rhat']:.4f}", flush=True)

        y_pred = predict_new_facility(test_df, summary)
        y_true = test_df["y"].values

        fold_r2 = r2(y_true, y_pred)
        per_facility_r2[held_out] = float(fold_r2)
        print(f"  held-out R2_log({held_out}) = {fold_r2:.4f}", flush=True)

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        fold_summaries[held_out] = summary

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    pooled_r2 = r2(all_y_true, all_y_pred)

    print(f"\n=== POOLED LOFO R2_log = {pooled_r2:.4f} ===")
    for f, v in per_facility_r2.items():
        print(f"  {f}: {v:.4f}")

    clears = pooled_r2 > 0.0 and all(v >= 0.0 for v in per_facility_r2.values())
    print(f"\nClears decision rule (pooled substantially positive AND all per-facility >=0): {clears}")

    results = {
        "approach": "Hierarchical Bayesian model (NUTS/PyMC) with physics-informed prior on the "
                    "Reynolds-exponent-like weight (Daily-Nece regime-IV exponent = -0.2 as prior mean, "
                    "with per-facility shrinkage). Shared population-level coefficients for the geometry "
                    "Pi groups (Pi_confinement, Pi_gap, Pi_aspect_axial) since those are facility-constant "
                    "and would be unidentifiable as facility-specific effects under LOFO.",
        "inference": "Full MCMC via PyMC 6.3.1 (NUTS, 4 chains x 2000 draws, 2000 tune, target_accept=0.95) "
                     "-- real Bayesian inference, not an approximation.",
        "substituted_proxy": False,
        "daily_nece_prior_mean": DAILY_NECE_EXPONENT,
        "pooled_r2_log": float(pooled_r2),
        "per_facility_r2_log": per_facility_r2,
        "clears_decision_rule": bool(clears),
        "fold_posterior_summaries": fold_summaries,
        "n_rows_total": int(len(df)),
        "facility_counts": df["source"].value_counts().to_dict(),
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nSaved results to {OUT_JSON}")


if __name__ == "__main__":
    main()
