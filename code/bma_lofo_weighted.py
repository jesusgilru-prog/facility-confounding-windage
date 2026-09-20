"""Bayesian Model Averaging ponderado por LOFO-R2 (no por BIC), sobre el
conjunto de estructuras ya conocidas del proyecto (M0-M9full lineales +
M10 no lineal de scaling_law_search.py, mas las 17 estructuras N1-N17 de
new_pi_groups_lofo_test.py). 2026-08-18.

Motivacion: BIC premia el ajuste in-sample, que en este dataset esta
dominado por sobreajuste de instalacion (los grupos Pi son casi
constantes dentro de cada fuente). El ganador BIC (M6) es catastrofico
en LOFO real (pooled R2 = -0.885). En cambio, M1 (solo Reynolds, sin
ningun grupo geometrico) es el mejor individual en LOFO (pooled R2 =
+0.4526) pese a NO ganar nunca el BIC. La idea de este script es no
elegir un unico ganador sino promediar todas las estructuras conocidas
con un peso proporcional a como de bien generalizan (LOFO-R2), no a como
de bien ajustan in-sample (BIC).

Diseno anti-fuga (CRITICO, leido dos veces antes de escribir el codigo):
Para predecir la instalacion held-out f, el ENSEMBLE necesita dos cosas
por cada estructura S:
  (a) el modelo S reentrenado en las 3 instalaciones de entrenamiento
      (todas menos f) y aplicado a f -> esto da la prediccion punto a
      punto que se promedia (igual que en un LOFO normal de una sola
      estructura).
  (b) el PESO de S para el fold f. Si este peso se calculara con el
      LOFO-R2 "global" de S (que incluye el fold f), estariamos usando
      informacion de f para decidir cuanto pesa S al predecir f -> fuga.
      En vez de eso, el peso de S para el fold f se calcula con un LOFO
      ANIDADO restringido a las OTRAS 3 instalaciones (entrena con 2,
      predice la 3a, para las 3 combinaciones dentro de las instalaciones
      de entrenamiento) y jamas toca los puntos de f. Es un
      leave-one-facility-out anidado (nested LOFO / double cross-validation).

Estandar LOFO obligatorio del proyecto: se reporta el R2 pooled en
log-espacio Y los 4 R2 individuales por separado, nunca solo el promedio.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

RNG_SEED = 12345

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
l_slender = lasp - lgap
l_Re_x_g = lRe + lg
l_Mtip_x_gap = lmt + lgap
l_Mtip_x_conf = lmt + lconf
l_g_x_conf = lg + lconf
l_g_x_gap = lg + lgap
sources = d["source"].values
facilities = sorted(set(sources))

# ---------------------------------------------------------------
# Conjunto de candidatos: M0-M9full (lineales, scaling_law_search.py)
# + M10 (no lineal) + N1-N17 (new_pi_groups_lofo_test.py). 28 estructuras.
# Cada builder recibe una mascara booleana y devuelve la lista de
# columnas (SIN intercepto, que se anade siempre).
# ---------------------------------------------------------------
LINEAR_BUILDERS = {
    "M0_null":                     lambda m: [],
    "M1_Re":                       lambda m: [lRe[m]],
    "M2_Re_g":                     lambda m: [lRe[m], lg[m]],
    "M3_Re_gap":                   lambda m: [lRe[m], lgap[m]],
    "M4_Re_conf":                  lambda m: [lRe[m], lconf[m]],
    "M5_Re_gap_conf":              lambda m: [lRe[m], lgap[m], lconf[m]],
    "M6_Re_gap_conf_asp":          lambda m: [lRe[m], lgap[m], lconf[m], lasp[m]],
    "M7_Re_gap_conf_Mtip":         lambda m: [lRe[m], lgap[m], lconf[m], lmt[m]],
    "M8_Re_gap_conf_cross":        lambda m: [lRe[m], lgap[m], lconf[m], cross_re_gap[m]],
    "M9full_Re_gap_conf_Mtip_cross": lambda m: [lRe[m], lgap[m], lconf[m], lmt[m], cross_re_gap[m]],
    "N1_Re_Mtip":                  lambda m: [lRe[m], lmt[m]],
    "N2_Mtip_only":                lambda m: [lmt[m]],
    "N3_g_only":                   lambda m: [lg[m]],
    "N4_M6_plus_g":                lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], lg[m]],
    "N5_M6_plus_Mtip":             lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], lmt[m]],
    "N6_Re_Mtip_g":                lambda m: [lRe[m], lmt[m], lg[m]],
    "N7_Re_conf_Mtip":             lambda m: [lRe[m], lconf[m], lmt[m]],
    "N8_Re_conf_slender":          lambda m: [lRe[m], lconf[m], l_slender[m]],
    "N9_Re_x_g_combined":          lambda m: [l_Re_x_g[m]],
    "N10_Re_x_g_plus_conf_gap":    lambda m: [l_Re_x_g[m], lgap[m], lconf[m]],
    "N11_M6_plus_Mtip_x_gap":      lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], l_Mtip_x_gap[m]],
    "N12_M6_plus_Mtip_x_conf":     lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], l_Mtip_x_conf[m]],
    "N13_M6_plus_g_x_conf":        lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], l_g_x_conf[m]],
    "N14_M6_plus_g_x_gap":         lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], l_g_x_gap[m]],
    "N15_Mtip_x_gap_only":         lambda m: [l_Mtip_x_gap[m]],
    "N16_g_x_conf_only":           lambda m: [l_g_x_conf[m]],
    "N17_M6_full_plus_g_plus_Mtip": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], lg[m], lmt[m]],
}
STRUCT_NAMES = list(LINEAR_BUILDERS.keys()) + ["M10_Re_logcorrection"]

def fit_predict_linear(cols_tr, y_tr, cols_te, n_te):
    """OLS en log-espacio (min-norm si rank-deficient, igual que el resto
    del proyecto: np.linalg.lstsq no lanza excepcion, documentado)."""
    n_tr = len(y_tr)
    Xtr = np.column_stack([np.ones(n_tr)] + cols_tr) if cols_tr else np.ones((n_tr, 1))
    coef, *_ = np.linalg.lstsq(Xtr, y_tr, rcond=None)
    Xte = np.column_stack([np.ones(n_te)] + cols_te) if cols_te else np.ones((n_te, 1))
    return Xte @ coef, coef

def fit_predict_M10(mask_tr, mask_te):
    """M10: log(Cp) = logC + q*log(Re) + log(1 + a*log(Re)), no lineal en 'a'.
    Multi-arranque real (igual criterio que scaling_law_search.py: a0=0 es
    un punto estacionario del objetivo, hay que probar varios a0)."""
    y_tr = y[mask_tr]
    lRe_tr = lRe[mask_tr]
    lRe_te = lRe[mask_te]

    def resid(params):
        logC, q, a = params
        corr = 1.0 + a * lRe_tr
        pred = logC + q * lRe_tr + np.log(np.clip(corr, 1e-6, None))
        return pred - y_tr

    # punto de partida M1 (OLS Re-only) sobre el mismo train set
    X1 = np.column_stack([np.ones(mask_tr.sum()), lRe_tr])
    c1, *_ = np.linalg.lstsq(X1, y_tr, rcond=None)
    a0_candidates = [-0.05, -0.02, -0.01, -0.005, -0.001, 0.0, 0.001, 0.01]
    best_res, best_rss = None, np.inf
    for a0 in a0_candidates:
        x0 = [c1[0], c1[1], a0]
        try:
            res_try = least_squares(resid, x0, method="lm", max_nfev=20000)
        except Exception:
            continue
        rss_try = float(np.sum(res_try.fun ** 2))
        if rss_try < best_rss:
            best_rss, best_res = rss_try, res_try
    if best_res is None:
        # fallback: colapsa a M1 si el no lineal falla por completo
        pred_te = c1[0] + c1[1] * lRe_te
        return pred_te, np.array([c1[0], c1[1], 0.0]), False
    logC, q, a = best_res.x
    corr_te = 1.0 + a * lRe_te
    pred_te = logC + q * lRe_te + np.log(np.clip(corr_te, 1e-6, None))
    return pred_te, best_res.x, bool(best_res.status > 0)

def predict_structure(name, mask_tr, mask_te):
    if name == "M10_Re_logcorrection":
        pred_te, coef, converged = fit_predict_M10(mask_tr, mask_te)
        return pred_te
    cols_tr = LINEAR_BUILDERS[name](mask_tr)
    cols_te = LINEAR_BUILDERS[name](mask_te)
    pred_te, coef = fit_predict_linear(cols_tr, y[mask_tr], cols_te, int(mask_te.sum()))
    return pred_te

def r2_of(resid, y_true):
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return None
    return float(1 - np.sum(resid ** 2) / ss_tot)

# =================================================================
# PASO 1: LOFO "exterior" (estandar) de cada estructura -> esto da (a)
# la prediccion final por fold f de cada estructura (modelo entrenado
# en las 3 facilities != f, aplicado a f), y de paso el LOFO-R2
# individual "global" de cada estructura, solo para comparacion/reporte
# (NO se usa para pesos, por diseno anti-fuga).
# =================================================================
outer_pred = {name: {} for name in STRUCT_NAMES}   # outer_pred[name][f] = array de predicciones log(Cp) en f
outer_resid_all = {name: [] for name in STRUCT_NAMES}
individual_lofo = {}
for name in STRUCT_NAMES:
    per_facility = {}
    all_resid = []
    for f in facilities:
        test_mask = sources == f
        train_mask = ~test_mask
        pred_te = predict_structure(name, train_mask, test_mask)
        outer_pred[name][f] = pred_te
        resid = y[test_mask] - pred_te
        all_resid.append(resid)
        per_facility[f] = {
            "n_test": int(test_mask.sum()),
            "held_out_r2_log": r2_of(resid, y[test_mask]) if test_mask.sum() > 1 else None,
            "held_out_rmse_log": float(np.sqrt(np.mean(resid ** 2))),
        }
    all_resid_cat = np.concatenate(all_resid)
    pooled_r2 = r2_of(all_resid_cat, y)
    individual_lofo[name] = {
        "per_facility": per_facility,
        "pooled_r2_log": pooled_r2,
        "pooled_rmse_log": float(np.sqrt(np.mean(all_resid_cat ** 2))),
    }

# =================================================================
# PASO 2: LOFO "interior" ANIDADO -> peso de cada estructura para CADA
# fold exterior f, calculado SOLO con las 3 facilities de entrenamiento
# de ese fold (entrena con 2, predice la 3a, para las 3 combinaciones
# posibles dentro de esas 3 facilities). Nunca toca los puntos de f.
# =================================================================
inner_lofo_r2 = {name: {} for name in STRUCT_NAMES}  # inner_lofo_r2[name][f] = R2 pooled de S usando solo las otras 3 facilities
for f_outer in facilities:
    train_facilities = [g for g in facilities if g != f_outer]  # 3 facilities
    for name in STRUCT_NAMES:
        all_resid_inner = []
        for g in train_facilities:
            inner_test_mask = sources == g
            inner_train_mask = (sources != g) & (sources != f_outer)  # las otras 2 (nunca f_outer)
            pred_te = predict_structure(name, inner_train_mask, inner_test_mask)
            all_resid_inner.append(y[inner_test_mask] - pred_te)
        all_resid_inner = np.concatenate(all_resid_inner)
        y_true_inner = np.concatenate([y[sources == g] for g in train_facilities])
        inner_lofo_r2[name][f_outer] = r2_of(all_resid_inner, y_true_inner)

# =================================================================
# PASO 3: pesos = softmax(alpha * R2_normalizado[0,1] via min-max DENTRO
# del conjunto de candidatos, por fold f) -- se prueban varios alpha
# (temperatura) para reportar sensibilidad, con alpha_primary como el
# valor principal reportado.
# =================================================================
def softmax_weights(r2_dict_for_fold, alpha):
    names = list(r2_dict_for_fold.keys())
    vals = np.array([r2_dict_for_fold[nm] if r2_dict_for_fold[nm] is not None else -1e9 for nm in names])
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-12:
        norm = np.zeros_like(vals)
    else:
        norm = (vals - vmin) / (vmax - vmin)
    z = alpha * norm
    z = z - z.max()  # estabilidad numerica del softmax
    w = np.exp(z)
    w = w / w.sum()
    return dict(zip(names, w.tolist()))

ALPHAS = [1.0, 3.0, 5.0, 10.0, 20.0]
ALPHA_PRIMARY = 5.0

ensemble_results_by_alpha = {}
weights_by_fold_alpha_primary = {}
for alpha in ALPHAS:
    all_resid = []
    per_facility = {}
    weights_per_fold = {}
    for f in facilities:
        test_mask = sources == f
        w_f = softmax_weights(inner_lofo_r2_for_fold := {nm: inner_lofo_r2[nm][f] for nm in STRUCT_NAMES}, alpha)
        weights_per_fold[f] = w_f
        pred_ens = np.zeros(test_mask.sum())
        for nm in STRUCT_NAMES:
            pred_ens += w_f[nm] * outer_pred[nm][f]
        resid = y[test_mask] - pred_ens
        all_resid.append(resid)
        per_facility[f] = {
            "n_test": int(test_mask.sum()),
            "held_out_r2_log": r2_of(resid, y[test_mask]) if test_mask.sum() > 1 else None,
            "held_out_rmse_log": float(np.sqrt(np.mean(resid ** 2))),
            "top3_weights": sorted(w_f.items(), key=lambda kv: -kv[1])[:3],
        }
    all_resid_cat = np.concatenate(all_resid)
    ensemble_results_by_alpha[alpha] = {
        "per_facility": per_facility,
        "pooled_r2_log": r2_of(all_resid_cat, y),
        "pooled_rmse_log": float(np.sqrt(np.mean(all_resid_cat ** 2))),
    }
    if alpha == ALPHA_PRIMARY:
        weights_by_fold_alpha_primary = weights_per_fold

# =================================================================
# BASELINE DE CONTROL: ensemble NO ponderado (media aritmetica simple de
# las 28 estructuras) -- para separar "promediar ayuda per se" de
# "ponderar por LOFO ayuda mas que promediar sin mas".
# =================================================================
uniform_all_resid = []
uniform_per_facility = {}
for f in facilities:
    test_mask = sources == f
    pred_ens = np.mean([outer_pred[nm][f] for nm in STRUCT_NAMES], axis=0)
    resid = y[test_mask] - pred_ens
    uniform_all_resid.append(resid)
    uniform_per_facility[f] = {
        "n_test": int(test_mask.sum()),
        "held_out_r2_log": r2_of(resid, y[test_mask]) if test_mask.sum() > 1 else None,
    }
uniform_all_resid_cat = np.concatenate(uniform_all_resid)
uniform_ensemble = {
    "per_facility": uniform_per_facility,
    "pooled_r2_log": r2_of(uniform_all_resid_cat, y),
    "pooled_rmse_log": float(np.sqrt(np.mean(uniform_all_resid_cat ** 2))),
}

# =================================================================
# Ranking de mejores individuales para contexto
# =================================================================
individual_ranking = sorted(individual_lofo.items(), key=lambda kv: (kv[1]["pooled_r2_log"] if kv[1]["pooled_r2_log"] is not None else -1e18), reverse=True)
best_individual_name, best_individual = individual_ranking[0]

primary_ensemble = ensemble_results_by_alpha[ALPHA_PRIMARY]

out = {
    "method": "Bayesian Model Averaging con pesos = softmax(alpha * LOFO_R2_normalizado), "
              "calculado con nested-LOFO (leakage-free) por fold exterior",
    "n_points": int(n),
    "n_structures": len(STRUCT_NAMES),
    "structure_names": STRUCT_NAMES,
    "alpha_primary": ALPHA_PRIMARY,
    "alphas_tested": ALPHAS,
    "individual_structures_lofo_ranking_pooled_r2": [
        (nm, res["pooled_r2_log"]) for nm, res in individual_ranking
    ],
    "individual_lofo_detail": individual_lofo,
    "best_individual_structure": {
        "name": best_individual_name,
        "pooled_r2_log": best_individual["pooled_r2_log"],
        "per_facility": best_individual["per_facility"],
    },
    "nested_inner_lofo_r2_used_for_weights": inner_lofo_r2,
    "ensemble_bma_lofo_weighted_by_alpha": ensemble_results_by_alpha,
    "ensemble_bma_primary_alpha": ALPHA_PRIMARY,
    "ensemble_bma_primary_result": primary_ensemble,
    "weights_by_fold_alpha_primary_full": weights_by_fold_alpha_primary,
    "control_uniform_average_ensemble_no_weighting": uniform_ensemble,
    "comparison_summary": {
        "best_individual_structure": best_individual_name,
        "best_individual_pooled_r2_log": best_individual["pooled_r2_log"],
        "bma_lofo_weighted_pooled_r2_log_at_alpha_primary": primary_ensemble["pooled_r2_log"],
        "uniform_unweighted_average_pooled_r2_log": uniform_ensemble["pooled_r2_log"],
        "bma_beats_best_individual": (
            primary_ensemble["pooled_r2_log"] is not None and best_individual["pooled_r2_log"] is not None
            and primary_ensemble["pooled_r2_log"] > best_individual["pooled_r2_log"]
        ),
        "bma_beats_uniform_average": (
            primary_ensemble["pooled_r2_log"] is not None and uniform_ensemble["pooled_r2_log"] is not None
            and primary_ensemble["pooled_r2_log"] > uniform_ensemble["pooled_r2_log"]
        ),
    },
    "note_anti_leakage": (
        "Los pesos de cada estructura para el fold donde se excluye la "
        "instalacion f se calculan con un LOFO ANIDADO restringido a las "
        "otras 3 instalaciones (entrena con 2, predice la 3a, para las 3 "
        "combinaciones dentro de esas 3), sin tocar nunca los puntos de f. "
        "La prediccion final del ensemble para f usa el modelo de cada "
        "estructura reentrenado en las 3 instalaciones de entrenamiento "
        "completas (no las 2 del interior) aplicado a f, ponderado por esos "
        "pesos anidados."
    ),
}

with open(_ROOT + "/results/bma_lofo_weighted_results.json", "w") as fp:
    json.dump(out, fp, indent=2, default=float)

print(f"n = {n}, n_structures = {len(STRUCT_NAMES)}")
print("\n--- Ranking LOFO individual (pooled R2, estandar de una sola estructura) ---")
for nm, r2 in out["individual_structures_lofo_ranking_pooled_r2"][:8]:
    print(f"  {nm:35s} pooled_R2={r2}")
print("...")
for nm, r2 in out["individual_structures_lofo_ranking_pooled_r2"][-5:]:
    print(f"  {nm:35s} pooled_R2={r2}")

print(f"\nMejor individual: {best_individual_name}  pooled_R2={best_individual['pooled_r2_log']:.4f}")
for f, r in best_individual["per_facility"].items():
    print(f"    {f:15s} R2={r['held_out_r2_log']}")

print(f"\n--- Ensemble BMA ponderado por LOFO-R2 anidado, por alpha ---")
for alpha in ALPHAS:
    res = ensemble_results_by_alpha[alpha]
    print(f"  alpha={alpha:5.1f}  pooled_R2={res['pooled_r2_log']:.4f}")
    for f in facilities:
        print(f"      {f:15s} R2={res['per_facility'][f]['held_out_r2_log']}")

print(f"\n--- Control: promedio NO ponderado (uniforme) de las {len(STRUCT_NAMES)} estructuras ---")
print(f"  pooled_R2={uniform_ensemble['pooled_r2_log']:.4f}")
for f, r in uniform_ensemble["per_facility"].items():
    print(f"    {f:15s} R2={r['held_out_r2_log']}")

print(f"\n=== RESUMEN ===")
print(json.dumps(out["comparison_summary"], indent=2))

print(f"\nPesos por fold (alpha_primary={ALPHA_PRIMARY}), top-3 estructuras por fold:")
for f in facilities:
    top3 = sorted(weights_by_fold_alpha_primary[f].items(), key=lambda kv: -kv[1])[:3]
    print(f"  held-out={f:15s} -> {top3}")
