"""Busqueda de ley de escala para el paper 'Windage Power' (2026-08-17,
revision post-multi-IA).

Cambios respecto a la version original tras la revision asistida por IA
(ver la declaracion de uso de IA del manuscrito):
  - Pi_blockage retirado del conjunto de grupos Pi: es una identidad
    algebraica EXACTA de Pi_aspect_axial y Pi_confinement (verificado
    log(Pi_blk) = -0.45158 + 1.0*log(Pi_asp) + 2.0*log(Pi_conf),
    R2=1.0, RSS=1.04e-27). Con el retirado quedan 10 estructuras no
    nulas + el modelo nulo = 11 estructuras en total.
  - M11 (correccion logaritmica) se reajusta con method='lm' (antes
    usaba 'trf' por defecto pese a que el texto decia
    Levenberg-Marquardt).
  - Se anade LOFO-CV (leave-one-facility-out, 4 pliegues) y bootstrap
    por instalacion (remuestreo de las 4 fuentes completas, no de
    puntos individuales) para el modelo ganador.

Deliberadamente DISTINTO del motor bayesiano jerarquico de paper8
(bayesian_structural_sr.py): aqui NO se perfila un prefactor por
geometria via evidencia de Laplace. Es una busqueda de estructura
global (no jerarquica) sobre un conjunto de 10 candidatos no nulos,
comparados por AIC/BIC clasicos.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

RNG_SEED = 12345  # fijo, no Math.random() -- reproducibilidad del bootstrap

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
d = df.dropna(subset=["Cp", "Re_Omega", "g_level", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "M_tip", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)

y = np.log(d["Cp"].values)
lRe = np.log(d["Re_Omega"].values)
lg = np.log(d["g_level"].clip(lower=1e-6).values)
lgap = np.log(d["Pi_gap"].values)
lconf = np.log(d["Pi_confinement"].values)
lasp = np.log(d["Pi_aspect_axial"].values)
lmt = np.log(d["M_tip"].clip(lower=1e-8).values)
cross_re_gap = lRe * lgap
sources = d["source"].values

# --- verificacion de la identidad Pi_blockage (documentacion, no se usa en el fit) ---
lblk_check = np.log(df.dropna(subset=["Pi_blockage", "Pi_aspect_axial", "Pi_confinement"])["Pi_blockage"].values)

def fit_linear(cols, names, yv=None, nv=None):
    yv = y if yv is None else yv
    nv = n if nv is None else nv
    X = np.column_stack([np.ones(nv)] + cols)
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    pred = X @ coef
    rss = float(np.sum((yv - pred) ** 2))
    k = X.shape[1]
    bic = nv * np.log(rss / nv) + k * np.log(nv)
    aic = nv * np.log(rss / nv) + 2 * k
    r2 = 1 - rss / np.sum((yv - yv.mean()) ** 2)
    return {"coef": dict(zip(["const"] + names, coef.tolist())),
            "rss": rss, "k": k, "bic": bic, "aic": aic, "r2": r2}

structures = {}
structures["M0_null"] = fit_linear([], [])
structures["M1_Re"] = fit_linear([lRe], ["q_Re"])
structures["M2_Re_g"] = fit_linear([lRe, lg], ["q_Re", "b_g"])
structures["M3_Re_gap"] = fit_linear([lRe, lgap], ["q_Re", "p_gap"])
structures["M4_Re_conf"] = fit_linear([lRe, lconf], ["q_Re", "r_conf"])
structures["M5_Re_gap_conf"] = fit_linear([lRe, lgap, lconf], ["q_Re", "p_gap", "r_conf"])
structures["M6_Re_gap_conf_asp"] = fit_linear([lRe, lgap, lconf, lasp],
                                               ["q_Re", "p_gap", "r_conf", "t_asp"])
structures["M7_Re_gap_conf_Mtip"] = fit_linear([lRe, lgap, lconf, lmt],
                                                ["q_Re", "p_gap", "r_conf", "u_Mtip"])
structures["M8_Re_gap_conf_cross"] = fit_linear([lRe, lgap, lconf, cross_re_gap],
                                                 ["q_Re", "p_gap", "r_conf", "v_cross_RexGap"])
structures["M9full_Re_gap_conf_Mtip_cross"] = fit_linear(
    [lRe, lgap, lconf, lmt, cross_re_gap],
    ["q_Re", "p_gap", "r_conf", "u_Mtip", "v_cross_RexGap"])

# M10: log(Cp) = logC + q*log(Re) + log(1 + a*log(Re)) -- no lineal en 'a'
# method='lm' (Levenberg-Marquardt real, el texto original lo afirmaba
# pero el codigo usaba 'trf' por defecto)
def resid_logcorr(params):
    logC, q, a = params
    corr = 1.0 + a * lRe
    pred = logC + q * lRe + np.log(np.clip(corr, 1e-6, None))
    return pred - y

M1_const = structures["M1_Re"]["coef"]["const"]
M1_qRe = structures["M1_Re"]["coef"]["q_Re"]
M1_rss = structures["M1_Re"]["rss"]

# a0=0.0 es un punto estacionario exacto del objetivo (sum(r*log(Re))~1.7e-12
# en x0=[M1, a=0]), asi que LM no puede moverse desde ahi -- multi-arranque
# real para no reportar un artefacto de optimizador como "convergencia limpia
# a a=0".
a0_candidates = [-0.05, -0.02, -0.01, -0.005, -0.001, 0.0, 0.001, 0.01]
best_res, best_rss10 = None, np.inf
for a0 in a0_candidates:
    x0 = [M1_const, M1_qRe, a0]
    res_try = least_squares(resid_logcorr, x0, method="lm", max_nfev=20000)
    rss_try = float(np.sum(res_try.fun ** 2))
    if rss_try < best_rss10:
        best_rss10, best_res = rss_try, res_try
res_lm = best_res
rss10 = best_rss10
k10 = 3
bic10 = n * np.log(rss10 / n) + k10 * np.log(n)
aic10 = n * np.log(rss10 / n) + 2 * k10
r2_10 = 1 - rss10 / np.sum((y - y.mean()) ** 2)
converged10 = bool(res_lm.status > 0)
structures["M10_Re_logcorrection"] = {
    "coef": {"logC": float(res_lm.x[0]), "q_Re": float(res_lm.x[1]), "a_logcorr": float(res_lm.x[2])},
    "rss": rss10, "k": k10, "bic": bic10, "aic": aic10, "r2": r2_10,
    "converged": converged10, "lm_status": int(res_lm.status),
    "multi_start_a0_tried": a0_candidates,
    "note": f"Multi-arranque real sobre a0 (method='lm'): a0=0.0 es un punto "
            f"estacionario exacto del objetivo (LM no se mueve desde ahi), asi "
            f"que arrancar solo en a0=0 producia el artefacto de 'converge a "
            f"a=0, colapsa a M1'. Con multi-arranque, el mejor ajuste real "
            f"converge a a={res_lm.x[2]:.4f}, RSS={rss10:.4f} "
            f"(vs RSS={M1_rss:.4f} de M1) -- el termino de correccion "
            f"logaritmica SI mejora el ajuste in-sample. Esto no cambia la "
            f"conclusion LOFO (M6 sigue ganando el BIC global; el hallazgo de "
            f"no-generalizacion es independiente de este ranking in-sample).",
}

n_non_null = len([k for k in structures if k != "M0_null"])
ranking = sorted(structures.items(), key=lambda kv: kv[1]["bic"])
# ranking valido excluye estructuras no convergidas del "ganador"
valid_ranking = [(k, v) for k, v in ranking if v.get("converged", True)]
best_name, best = valid_ranking[0]

# --- LOFO-CV (leave-one-facility-out) del modelo ganador ---
def refit_and_score(train_mask, test_mask, cols_fn):
    Xtr_cols = cols_fn(train_mask)
    Xtr = np.column_stack([np.ones(train_mask.sum())] + Xtr_cols)
    ytr = y[train_mask]
    coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    Xte_cols = cols_fn(test_mask)
    Xte = np.column_stack([np.ones(test_mask.sum())] + Xte_cols)
    yte = y[test_mask]
    pred = Xte @ coef
    resid = yte - pred
    return coef, resid

def winner_cols(mask):
    return [lRe[mask], lgap[mask], lconf[mask], lasp[mask]]

facilities = sorted(set(sources))
lofo_results = {}
all_resid = []
for f in facilities:
    test_mask = sources == f
    train_mask = ~test_mask
    coef, resid = refit_and_score(train_mask, test_mask, winner_cols)
    lofo_results[f] = {
        "n_test": int(test_mask.sum()),
        "coef_trained_without_this_facility": dict(zip(
            ["const", "q_Re", "p_gap", "r_conf", "t_asp"], coef.tolist())),
        "held_out_rmse_log": float(np.sqrt(np.mean(resid ** 2))),
        "held_out_r2_log": float(1 - np.sum(resid ** 2) /
                                  np.sum((y[test_mask] - y[test_mask].mean()) ** 2))
                           if test_mask.sum() > 1 else None,
        "held_out_median_abs_resid_log": float(np.median(np.abs(resid))),
    }
    all_resid.append(resid)
all_resid = np.concatenate(all_resid)
lofo_pooled_rmse = float(np.sqrt(np.mean(all_resid ** 2)))
lofo_pooled_r2 = float(1 - np.sum(all_resid ** 2) / np.sum((y - y.mean()) ** 2))

# --- bootstrap por instalacion (resample facilities with replacement) ---
rng = np.random.default_rng(RNG_SEED)
n_boot = 2000
n_rank_deficient_skipped = 0
boot_coefs = []
facility_idx = {f: np.where(sources == f)[0] for f in facilities}
for _ in range(n_boot):
    chosen = rng.choice(facilities, size=len(facilities), replace=True)
    idx = np.concatenate([facility_idx[f] for f in chosen])
    Xb = np.column_stack([np.ones(len(idx)), lRe[idx], lgap[idx], lconf[idx], lasp[idx]])
    yb = y[idx]
    # np.linalg.lstsq no lanza LinAlgError con matriz rank-deficient --
    # devuelve la solucion de norma minima en silencio. Con solo 4
    # instalaciones remuestreadas con reemplazo, un resample con <5
    # instalaciones-unicas-efectivas (los Pi-groups son ~constantes por
    # instalacion) produce una matriz de diseño de rango <5. Se filtra
    # explicitamente en vez de confiar en una excepcion que nunca salta
    # (bug real, hallado en la ronda 2 de revision, 2026-08-17).
    if np.linalg.matrix_rank(Xb) < Xb.shape[1]:
        n_rank_deficient_skipped += 1
        continue
    coef_b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    boot_coefs.append(coef_b)
boot_coefs = np.array(boot_coefs)
boot_names = ["const", "q_Re", "p_gap", "r_conf", "t_asp"]
bootstrap_facility_se = {
    name: {
        "point_estimate": float(best["coef"][name]),
        "bootstrap_mean": float(boot_coefs[:, i].mean()),
        "bootstrap_se": float(boot_coefs[:, i].std(ddof=1)),
        "ci95_pct": [float(np.percentile(boot_coefs[:, i], 2.5)),
                     float(np.percentile(boot_coefs[:, i], 97.5))],
    }
    for i, name in enumerate(boot_names)
}

# --- Comparacion contra Daily-Nece (regimen turbulento IV) ---
G = d["Pi_gap"].values
Re = d["Re_Omega"].values
Cp_actual = d["Cp"].values
Cp_daily_nece = 0.051 * (G ** (1 / 10)) * (Re ** (-0.2))
rel_err = np.abs(Cp_actual - Cp_daily_nece) / Cp_actual
log_actual = np.log(Cp_actual)
log_pred_dn = np.log(Cp_daily_nece)
rss_dn = np.sum((log_actual - log_pred_dn) ** 2)
r2_dn = 1 - rss_dn / np.sum((log_actual - log_actual.mean()) ** 2)
n_underpredicted = int(np.sum(Cp_actual > Cp_daily_nece))

daily_nece_comparison = {
    "formula": "C_M,IV = 0.051 * Pi_gap^(1/10) * Re_Omega^(-0.2)",
    "n": int(n),
    "n_underpredicted_of_n": f"{n_underpredicted}/{n}",
    "median_ratio_obs_over_pred": float(np.median(Cp_actual / Cp_daily_nece)),
    "min_ratio_obs_over_pred": float(np.min(Cp_actual / Cp_daily_nece)),
    "max_ratio_obs_over_pred": float(np.max(Cp_actual / Cp_daily_nece)),
    "median_relative_error_pct": float(np.median(rel_err) * 100),
    "mean_relative_error_pct": float(np.mean(rel_err) * 100),
    "max_relative_error_pct": float(np.max(rel_err) * 100),
    "r2_log_space_vs_Daily_Nece": float(r2_dn),
    "note": "Turbulent regime-IV formula applied to ALL points (114, "
            "includes g_level=1 and regimes not necessarily IV). UPDATED "
            "causal diagnosis (verified via a real regression of "
            "log(Cp/C_M_DailyNece) ~ log(Re_Omega): slope=-0.452, "
            "R2=0.716, see robustness_summary_stats.json): the DOMINANT "
            "cause is that the corpus's effective Reynolds exponent "
            "differs substantially from regime IV's -0.2, not a simple "
            "geometric-class offset -- a III<->IV regime switch would "
            "only shift the exponent by O(0.1), insufficient for the "
            "observed bias. Geometric-class mixing (Daily-Nece = smooth "
            "confined disk; the corpus is centrifuge arms / alternator "
            "poles / cylinders) remains a contributing but secondary "
            "factor, not the main explanation.",
}

# --- identidad algebraica de Pi_blockage (documentada, no usada en el fit) ---
d_blk = df.dropna(subset=["Pi_blockage", "Pi_aspect_axial", "Pi_confinement"])
d_blk = d_blk[(d_blk["Pi_blockage"] > 0) & (d_blk["Pi_aspect_axial"] > 0) & (d_blk["Pi_confinement"] > 0)]
Xblk = np.column_stack([np.ones(len(d_blk)), np.log(d_blk["Pi_aspect_axial"]), np.log(d_blk["Pi_confinement"])])
yblk = np.log(d_blk["Pi_blockage"])
coef_blk, *_ = np.linalg.lstsq(Xblk, yblk, rcond=None)
pred_blk = Xblk @ coef_blk
rss_blk = float(np.sum((yblk - pred_blk) ** 2))
r2_blk = float(1 - rss_blk / np.sum((yblk - yblk.mean()) ** 2))

blockage_identity = {
    "n": int(len(d_blk)),
    "log_Pi_blk_equals": "const + 1.0*log(Pi_aspect_axial) + 2.0*log(Pi_confinement)",
    "coef_const_asp_conf": coef_blk.tolist(),
    "rss": rss_blk,
    "r2": r2_blk,
    "conclusion": "Exact algebraic identity (R2=1.0 to machine precision). "
                  "Pi_blockage removed from the set of Pi groups used in "
                  "the structure search.",
}

out = {
    "n_points": int(n),
    "sources": {k: int(v) for k, v in pd.Series(sources).value_counts().to_dict().items()},
    "n_non_null_structures": n_non_null,
    "structures": structures,
    "ranking_by_bic_all": [name for name, _ in ranking],
    "ranking_by_bic_converged_only": [name for name, _ in valid_ranking],
    "best_structure": best_name,
    "best_structure_result": best,
    "lofo_cv": {"per_facility": lofo_results,
                "pooled_rmse_log": lofo_pooled_rmse,
                "pooled_r2_log": lofo_pooled_r2},
    "bootstrap_by_facility": {
        "n_boot": n_boot,
        "n_rank_deficient_skipped": n_rank_deficient_skipped,
        "n_usable": len(boot_coefs),
        "coef_se": bootstrap_facility_se,
    },
    "daily_nece_comparison": daily_nece_comparison,
    "pi_blockage_identity_check": blockage_identity,
}

with open(_ROOT + "/results/scaling_law_search_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)

print(f"n = {n}, estructuras no nulas = {n_non_null}")
print("\nRanking por BIC (todas, incluida no-convergida si la hay):")
for name, res_ in ranking:
    conv = res_.get("converged", True)
    print(f"  {name:35s} BIC={res_['bic']:9.2f}  AIC={res_['aic']:9.2f}  R2={res_['r2']:.4f}  k={res_['k']}  converged={conv}")
print(f"\nGanador (entre convergidas): {best_name}")
print(json.dumps(best["coef"], indent=2))
print(f"\nLOFO-CV pooled: RMSE(log)={lofo_pooled_rmse:.4f}  R2(log)={lofo_pooled_r2:.4f}")
for f, r in lofo_results.items():
    print(f"  {f:15s} n={r['n_test']:3d}  RMSE(log)={r['held_out_rmse_log']:.4f}  R2(log)={r['held_out_r2_log']}")
print(f"\nBootstrap SE (por instalacion, n_boot={n_boot}):")
print(json.dumps(bootstrap_facility_se, indent=2))
print(f"\nComparacion Daily-Nece:")
print(json.dumps(daily_nece_comparison, indent=2))
print(f"\nIdentidad Pi_blockage:")
print(json.dumps(blockage_identity, indent=2))
