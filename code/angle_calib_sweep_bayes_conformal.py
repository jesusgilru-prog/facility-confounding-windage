"""Angle: EMPIRICAL few-shot calibration sweep (Bayesian hierarchical updating +
split conformal prediction), n_cal = 1..20, real data, strict LOFO.

Question: with N real calibration points drawn from a NEW (held-out) facility,
does a principled update of a hierarchical prior (all 4 log-log slopes +
intercept, not just the intercept as in hierarchical_calibration_lofo.py)
produce genuinely useful predictions, and/or valid conformal coverage, for the
REMAINING points of that facility? Sweep n_cal and report real numbers.

Explicitly NOT the existing James-Stein intercept-only recalibration
(code/hierarchical_calibration_lofo.py) -- that already showed shrinkage
barely moves the intercept (lambda ~1e-3) because between-facility intercept
variance (tau2) is tiny compared to within-facility noise for some facilities,
and R2 stays deeply negative (Guo2024: -8.8 to -9.1 for n_cal in 0..10) because
the SLOPES also differ by facility and were never touched.

Method here (both requested variants implemented):

1. Base/global model (M6 form used throughout this project):
     log(Cp) = b0 + q*log(Re_Omega) + p*log(Pi_gap)
               + r*log(Pi_confinement) + t*log(Pi_aspect_axial)
   Fit ONLY on the 3 training facilities (target held out entirely, strict
   LOFO), using per-facility weights = 1/n_facility so the largest facility
   (Guo2024, n=45) cannot dominate a pooled fit against Zheng2024 (n=8).

2. Hierarchical prior on the FULL 5-vector beta = (b0,q,p,r,t):
   - beta_hat_global = weighted OLS of (1) above.
   - For each of the 3 training facilities, fit its OWN local OLS (same
     5 params) -- gives 3 independent estimates of beta.
   - tau_j^2 (j=1..5) = sample variance (ddof=1, n=3) of each parameter
     across those 3 local fits. Declared limitation: df=2, this is a very
     noisy estimate of between-facility variance (same caveat as the
     existing intercept-only script, now extended to 5 dims). We keep the
     prior DIAGONAL (no cross-covariance) because n=3 cannot support a
     5x5 covariance estimate at all.
   - sigma2 = pooled within-facility residual variance (residuals of each
     training facility around ITS OWN local fit, i.e. noise net of the
     facility offset/slope), pooled by residual degrees of freedom across
     the 3 training facilities.

3. Bayesian conjugate update using n_cal real points (X_cal, y_cal) from the
   held-out facility (standard Gaussian-linear posterior, closed form):
     Sigma0    = diag(tau_j^2)                      [prior covariance]
     Sigma_post = (Sigma0^-1 + X_cal'X_cal/sigma2)^-1
     beta_post = Sigma_post (Sigma0^-1 beta_hat + X_cal'y_cal/sigma2)
   This is well-defined even for n_cal=1 (prior regularizes the otherwise
   unidentified 5-parameter fit) and smoothly interpolates towards the
   facility's own local OLS as n_cal grows.

4. Split conformal prediction on top of the SAME posterior-mean predictor:
   nonconformity score = |y - yhat_post(x)| on the n_cal calibration points
   (points NOT reused for testing); interval half-width = the standard
   finite-sample conformal quantile
     q_hat = the ceil((n_cal+1)*(1-alpha))-th order statistic of the scores
   (only defined/reported when ceil((n_cal+1)*(1-alpha)) <= n_cal, else
   marked infeasible -- no invented numbers). Evaluate EMPIRICAL coverage
   and mean interval width on the held-out test points at alpha=0.20 (80%)
   and alpha=0.10 (90%).

5. Evaluate on the facility's OWN remaining points (never used for
   calibration): R2_log (only when >=2 test points remain, needs a real
   variance denominator), MAE_log, RMSE_log; report both PER-FACILITY and
   POOLED-across-facilities-within-repetition numbers, averaged over many
   random calibration-subset draws per n_cal (with std across reps -- not
   hidden).

Decision rule (same honest bar as the rest of the paper): "genuinely useful"
requires pooled R2_log substantially positive (say >=0.5 as a real
improvement bar, not just >0) AND every individual held-out facility's
R2_log >= 0 at that n_cal. We check this explicitly and report the verdict,
we do not spin it either way.

No fabricated numbers: every number below is computed from
data/cross_rotor_dataset_v3.csv by this script.
"""
import json
import numpy as np
import pandas as pd

RNG_SEED = 20260818
N_REPS = 300  # random calibration-subset draws per (facility, n_cal)
N_CAL_GRID = list(range(1, 21))
ALPHAS = [0.20, 0.10]  # 80% / 90% nominal conformal coverage

FEATS = ["Re_Omega", "Pi_gap", "Pi_confinement", "Pi_aspect_axial"]
PARAM_NAMES = ["intercept", "log_Re_Omega", "log_Pi_gap", "log_Pi_confinement",
               "log_Pi_aspect_axial"]


def build_design(df):
    X = np.column_stack([
        np.ones(len(df)),
        np.log(df["Re_Omega"].values),
        np.log(df["Pi_gap"].values),
        np.log(df["Pi_confinement"].values),
        np.log(df["Pi_aspect_axial"].values),
    ])
    y = np.log(df["Cp"].values)
    return X, y


def weighted_ols(X, y, w):
    """Weighted least squares, closed form."""
    W = np.diag(w)
    XtW = X.T @ W
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    return beta


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def r2_log(y_true, y_pred):
    if len(y_true) < 2:
        return None
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return None
    return 1.0 - ss_res / ss_tot


def conformal_quantile(scores, alpha):
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return None  # infeasible at this n_cal, do not invent a number
    return float(np.sort(scores)[k - 1])


def main():
    df = pd.read_csv("data/cross_rotor_dataset_v3.csv")
    facilities = df["source"].unique().tolist()
    rng = np.random.default_rng(RNG_SEED)

    # ---- Step 1-2: per-facility local fits + global weighted fit + priors,
    # done ONCE per held-out target (strict LOFO), reused across all reps/n_cal.
    global_setup = {}
    for target in facilities:
        train_df = df[df["source"] != target]
        train_facilities = train_df["source"].unique().tolist()

        # weighted pooled OLS across training facilities (equal facility weight)
        Xtr, ytr = build_design(train_df)
        w = np.array([1.0 / (train_df["source"] == f).sum() for f in train_df["source"]])
        beta_hat = weighted_ols(Xtr, ytr, w)

        # per-training-facility local OLS fits -> between-facility variance
        local_betas = []
        resid_list = []
        dof_total = 0
        for f in train_facilities:
            sub = train_df[train_df["source"] == f]
            Xf, yf = build_design(sub)
            if len(sub) > Xf.shape[1]:
                bf = ols(Xf, yf)
                resid = yf - Xf @ bf
                dof = len(sub) - Xf.shape[1]
            else:
                # not enough points for a full local fit; fall back to global
                # beta for this facility's contribution to tau2 (declared),
                # residuals computed against global beta instead.
                bf = beta_hat.copy()
                resid = yf - Xf @ bf
                dof = max(len(sub) - 1, 1)
            local_betas.append(bf)
            resid_list.append(resid)
            dof_total += dof
        local_betas = np.array(local_betas)  # (n_train_facilities, 5)
        tau2 = local_betas.var(axis=0, ddof=1) if len(local_betas) > 1 else np.full(5, np.nan)
        tau2 = np.maximum(tau2, 1e-8)  # numerical floor, never used to inflate a result

        all_resid = np.concatenate(resid_list)
        sigma2 = float(np.sum(all_resid ** 2) / max(dof_total, 1))

        global_setup[target] = dict(
            beta_hat=beta_hat, tau2=tau2, sigma2=sigma2,
            n_train_facilities=len(train_facilities),
        )

    # ---- Step 3-5: sweep n_cal, random calibration subsets, Bayesian update
    # + conformal calibration, evaluate on remaining points.
    per_facility_curve = {f: {} for f in facilities}
    for target in facilities:
        sub = df[df["source"] == target].reset_index(drop=True)
        n_total = len(sub)
        Xf, yf = build_design(sub)
        setup = global_setup[target]
        beta_hat = setup["beta_hat"]
        Sigma0_inv = np.diag(1.0 / setup["tau2"])
        sigma2 = setup["sigma2"]

        max_n_cal = min(N_CAL_GRID[-1], n_total - 1)  # need >=1 test point
        for n_cal in [n for n in N_CAL_GRID if n <= max_n_cal]:
            n_test = n_total - n_cal
            reps_r2, reps_mae, reps_rmse = [], [], []
            reps_cov = {a: [] for a in ALPHAS}
            reps_width = {a: [] for a in ALPHAS}
            n_reps_this = N_REPS if n_total > 25 else min(N_REPS, 4000)
            # exact enumeration when combinatorially small, else random draws
            from math import comb
            total_combos = comb(n_total, n_cal)
            exact = total_combos <= 60
            if exact:
                from itertools import combinations
                idx_sets = list(combinations(range(n_total), n_cal))
            else:
                idx_sets = [
                    tuple(sorted(rng.choice(n_total, size=n_cal, replace=False)))
                    for _ in range(n_reps_this)
                ]

            for cal_idx in idx_sets:
                cal_idx = np.array(cal_idx)
                test_idx = np.array([i for i in range(n_total) if i not in set(cal_idx)])
                if len(test_idx) < 1:
                    continue
                Xc, yc = Xf[cal_idx], yf[cal_idx]
                Xt, yt = Xf[test_idx], yf[test_idx]

                # Bayesian conjugate update (diagonal prior)
                XtX = Xc.T @ Xc / sigma2
                Sigma_post_inv = Sigma0_inv + XtX
                rhs = Sigma0_inv @ beta_hat + Xc.T @ yc / sigma2
                beta_post = np.linalg.solve(Sigma_post_inv, rhs)

                yhat_test = Xt @ beta_post
                resid_test = yt - yhat_test
                reps_mae.append(np.mean(np.abs(resid_test)))
                reps_rmse.append(np.sqrt(np.mean(resid_test ** 2)))
                r2 = r2_log(yt, yhat_test)
                if r2 is not None:
                    reps_r2.append(r2)

                # split conformal using calibration residuals under beta_post
                yhat_cal = Xc @ beta_post
                scores = np.abs(yc - yhat_cal)
                for a in ALPHAS:
                    q = conformal_quantile(scores, a)
                    if q is None:
                        continue
                    covered = np.mean(np.abs(resid_test) <= q)
                    reps_cov[a].append(covered)
                    reps_width[a].append(2 * q)

            entry = dict(
                n_cal=n_cal, n_test=n_test,
                n_reps=len(idx_sets), exact_enumeration=exact,
                r2_mean=float(np.mean(reps_r2)) if reps_r2 else None,
                r2_median=float(np.median(reps_r2)) if reps_r2 else None,
                r2_p25=float(np.percentile(reps_r2, 25)) if reps_r2 else None,
                r2_p75=float(np.percentile(reps_r2, 75)) if reps_r2 else None,
                r2_std=float(np.std(reps_r2)) if reps_r2 else None,
                r2_n=len(reps_r2),
                mae_mean=float(np.mean(reps_mae)),
                mae_median=float(np.median(reps_mae)),
                rmse_mean=float(np.mean(reps_rmse)),
                small_test_set=bool(n_test < 3),
            )
            for a in ALPHAS:
                key = f"cov_{int((1-a)*100)}"
                entry[key + "_mean"] = float(np.mean(reps_cov[a])) if reps_cov[a] else None
                entry[key + "_n_feasible_reps"] = len(reps_cov[a])
                entry[f"width_{int((1-a)*100)}_mean"] = float(np.mean(reps_width[a])) if reps_width[a] else None
            per_facility_curve[target][n_cal] = entry

    # ---- pooled-across-facilities curve (only n_cal values available for ALL
    # facilities, i.e. up to Zheng2024's cap of 7) ----
    common_n_cal = sorted(set.intersection(*[
        set(per_facility_curve[f].keys()) for f in facilities
    ]))
    pooled_curve = {}
    for n_cal in common_n_cal:
        # pool by re-deriving reps? We only stored aggregated means per facility;
        # for a defensible "pooled" number, weight each facility's mean R2 by
        # its number of test points at this n_cal (standard pooled-R2 style
        # used elsewhere in this project is residual-concatenation; here we
        # approximate with n_test-weighted mean of per-rep residual arrays is
        # not stored, so we report n_test-weighted MEAN R2 across facilities,
        # explicitly labeled as such, not a true residual-pooled R2).
        r2s, ws, r2_medians = [], [], []
        maes, rmses = [], []
        for f in facilities:
            e = per_facility_curve[f][n_cal]
            if e["r2_mean"] is not None:
                r2s.append(e["r2_mean"])
                ws.append(e["n_test"])
            if e["r2_median"] is not None:
                r2_medians.append(e["r2_median"])
            maes.append(e["mae_mean"])
            rmses.append(e["rmse_mean"])
        pooled_curve[n_cal] = dict(
            weighted_mean_r2=float(np.average(r2s, weights=ws[: len(r2s)])) if r2s else None,
            unweighted_mean_r2=float(np.mean(r2s)) if r2s else None,
            unweighted_median_of_medians_r2=float(np.mean(r2_medians)) if r2_medians else None,
            min_facility_r2=float(min(r2s)) if r2s else None,
            mean_mae=float(np.mean(maes)),
            mean_rmse=float(np.mean(rmses)),
            worst_facility=(facilities[int(np.argmin([per_facility_curve[f][n_cal]["r2_mean"]
                             if per_facility_curve[f][n_cal]["r2_mean"] is not None else np.inf
                             for f in facilities]))] if r2s else None),
        )

    # ---- decision rule check ----
    decision = {}
    for n_cal in common_n_cal:
        pc = pooled_curve[n_cal]
        per_facility_r2 = {f: per_facility_curve[f][n_cal]["r2_mean"] for f in facilities}
        all_nonneg = all(v is not None and v >= 0 for v in per_facility_r2.values())
        pooled_substantial = pc["weighted_mean_r2"] is not None and pc["weighted_mean_r2"] >= 0.5
        decision[n_cal] = dict(
            all_facilities_r2_nonneg=all_nonneg,
            pooled_r2_substantial=pooled_substantial,
            passes_honest_bar=bool(all_nonneg and pooled_substantial),
            per_facility_r2=per_facility_r2,
        )

    out = dict(
        method="Bayesian hierarchical (diagonal prior on 5-param log-log OLS) "
               "+ split conformal, strict LOFO, n_cal sweep",
        n_reps_target=N_REPS,
        seed=RNG_SEED,
        global_setup={f: dict(
            beta_hat=global_setup[f]["beta_hat"].tolist(),
            tau2=global_setup[f]["tau2"].tolist(),
            sigma2=global_setup[f]["sigma2"],
            n_train_facilities=global_setup[f]["n_train_facilities"],
        ) for f in facilities},
        param_names=PARAM_NAMES,
        per_facility_curve=per_facility_curve,
        pooled_curve_common_n_cal=pooled_curve,
        decision_rule_check=decision,
        facility_sizes={f: int((df["source"] == f).sum()) for f in facilities},
        notes=[
            "Zheng2024 n=8 caps the *common* n_cal grid at 7 (needs >=1 test point); "
            "per-facility curves for Guo2024/Vrancik1968/Liu2024 go further (up to 20) "
            "and are reported separately in per_facility_curve.",
            "tau2 estimated from only 3 training facilities per LOFO fold (df=2 per "
            "parameter) -- inherently noisy, same caveat as the pre-existing "
            "intercept-only James-Stein script, now extended to all 5 params.",
            "Conformal coverage/width fields are None when infeasible at that n_cal "
            "(ceil((n_cal+1)(1-alpha)) > n_cal) -- never filled with an invented value.",
            "pooled_curve here is an n_test-weighted MEAN of per-facility mean R2 "
            "(not a residual-concatenation pooled R2 as used in some other scripts "
            "in this repo) -- labeled explicitly to avoid confusion between the two "
            "pooling conventions used across the paper's 16+ prior methods.",
        ],
    )

    with open("results/angle_calib_sweep_bayes_conformal_results.json", "w") as fh:
        json.dump(out, fh, indent=2)

    # console summary
    print("=== Global setup (beta_hat, tau2, sigma2) per LOFO target ===")
    for f in facilities:
        gs = global_setup[f]
        print(f"{f}: sigma2={gs['sigma2']:.4f}  tau2={np.round(gs['tau2'],5)}")
    print()
    print("=== Pooled (n_test-weighted mean) R2_log by n_cal (common grid, capped by Zheng n=8) ===")
    for n_cal in common_n_cal:
        pc = pooled_curve[n_cal]
        d = decision[n_cal]
        mm = pc['unweighted_median_of_medians_r2']
        mms = f"{mm:+.3f}" if mm is not None else "n/a"
        print(f"n_cal={n_cal:2d}  pooled_mean_R2={pc['weighted_mean_r2']:+.3f}  "
              f"median_of_medians_R2={mms}  "
              f"min_facility_R2={pc['min_facility_r2']:+.3f} ({pc['worst_facility']})  "
              f"passes_honest_bar={d['passes_honest_bar']}")
    print()
    print("=== Per-facility R2_log curves (full range up to min(20, n_facility-1)) ===")
    for f in facilities:
        print(f"-- {f} (n={len(df[df.source==f])}) --")
        for n_cal, e in per_facility_curve[f].items():
            r2s = f"{e['r2_mean']:+.3f}" if e['r2_mean'] is not None else "  n/a"
            r2med = f"{e['r2_median']:+.3f}" if e['r2_median'] is not None else "  n/a"
            cov80 = e.get("cov_80_mean")
            cov90 = e.get("cov_90_mean")
            cov80s = f"{cov80:.2f}" if cov80 is not None else "n/a"
            cov90s = f"{cov90:.2f}" if cov90 is not None else "n/a"
            print(f"  n_cal={n_cal:2d}  R2_mean={r2s}  R2_median={r2med}  MAE={e['mae_mean']:.3f}  "
                  f"cov80={cov80s}  cov90={cov90s}  n_test={e['n_test']}  n_reps={e['n_reps']}"
                  f"{' (exact)' if e['exact_enumeration'] else ''}")


if __name__ == "__main__":
    main()
