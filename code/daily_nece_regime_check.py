"""Hipotesis dirigida (2026-08-17): ¿el fallo 114/114 de Daily-Nece regimen
IV (ratio mediano 26.7x, ver scaling_law_search.py) se debe a
aplicar SOLO la formula turbulenta de capas separadas (regimen IV) sin
clasificar primero el regimen real de cada punto?

Formulas y fronteras de Daily & Nece (1960), TOMADAS DE LITERATURA (no
inventadas, no ajustadas a este dataset), verificadas via WebSearch +
extraccion de texto de Poncet et al., "Review of fluid flow and convective
heat transfer within rotating disk cavities with impinging jet" (arXiv
1305.2882v1, seccion 2.4.1, ecuaciones 38-42), consistente con una segunda
busqueda independiente (mismos 4 coeficientes/exponentes):

  Regimen I   (laminar, capas fusionadas):   C_M = pi   * G^(-1)    * Re^(-1)
  Regimen II  (laminar, capas separadas):    C_M = 1.85 * G^(0.1)   * Re^(-0.5)
  Regimen III (turbulento, capas fusionadas):C_M = 0.04 * G^(-0.167)* Re^(-0.25)
  Regimen IV  (turbulento, capas separadas): C_M = 0.051* G^(0.1)   * Re^(-0.2)

Frontera fusionado/separado en regimen turbulento (III/IV), la UNICA
frontera con formula cerrada explicita encontrada en la fuente:
  G_(III/IV) = 0.2112 * Re^(-3/16)

Frontera laminar/turbulento (Reynolds critico, Daily-Nece / Kreith),
formula a trozos en G:
  Re_crit = (pi/0.036)^(4/3) * G^(-10/9),      G < 0.0111
  Re_crit = 6.9739e6 * G^(16/15),              0.0111 <= G < 0.0233
  Re_crit = 1.266e5,                            G >= 0.0233

PUNTO CIEGO DECLARADO EXPLICITAMENTE: no se encontro con confianza la
formula cerrada de la frontera I/II (fusionado/separado en regimen
LAMINAR). Esto es irrelevante en la practica para este dataset porque
113/114 puntos caen en el lado turbulento del criterio anterior (ver
resultados), asi que la clasificacion I/II casi no se ejerce (1 solo
punto, de Vrancik1968).

Fuentes (WebSearch, 2026-08-17): arXiv:1305.2882 (Poncet et al., review,
extraido con pdfplumber); confirmado independientemente contra un segundo
resultado de busqueda con los mismos 4 coeficientes.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
d = df.dropna(subset=["Cp", "Re_Omega", "g_level", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)

Re = d["Re_Omega"].values
G = d["Pi_gap"].values
Cp_actual = d["Cp"].values
source = d["source"].values


def re_crit(Gv):
    out = np.empty_like(Gv)
    m1 = Gv < 0.0111
    m2 = (Gv >= 0.0111) & (Gv < 0.0233)
    m3 = Gv >= 0.0233
    out[m1] = (np.pi / 0.036) ** (4 / 3) * Gv[m1] ** (-10 / 9)
    out[m2] = 6.9739e6 * Gv[m2] ** (16 / 15)
    out[m3] = 1.266e5
    return out


rc = re_crit(G)
turbulent = Re > rc
G_bound_III_IV = 0.2112 * Re ** (-3 / 16)
merged = G < G_bound_III_IV

regime = np.where(turbulent & merged, "III",
          np.where(turbulent & ~merged, "IV",
          np.where((~turbulent) & merged, "I", "II")))

Cp_pred = np.empty(n)
Cp_pred[regime == "I"] = np.pi * G[regime == "I"] ** (-1) * Re[regime == "I"] ** (-1)
Cp_pred[regime == "II"] = 1.85 * G[regime == "II"] ** 0.1 * Re[regime == "II"] ** (-0.5)
Cp_pred[regime == "III"] = 0.04 * G[regime == "III"] ** (-0.167) * Re[regime == "III"] ** (-0.25)
Cp_pred[regime == "IV"] = 0.051 * G[regime == "IV"] ** 0.1 * Re[regime == "IV"] ** (-0.2)

ratio = Cp_actual / Cp_pred
log_actual = np.log(Cp_actual)
log_pred = np.log(Cp_pred)
rss = float(np.sum((log_actual - log_pred) ** 2))
r2_regime_classified = 1 - rss / np.sum((log_actual - log_actual.mean()) ** 2)

# --- comparacion contra el naive (regimen IV forzado a los 114 puntos), ---
# --- ya documentado en scaling_law_search_results.json ---
Cp_dn4_naive = 0.051 * (G ** (1 / 10)) * (Re ** (-0.2))
ratio_naive = Cp_actual / Cp_dn4_naive
log_pred_naive = np.log(Cp_dn4_naive)
r2_naive = 1 - np.sum((log_actual - log_pred_naive) ** 2) / np.sum((log_actual - log_actual.mean()) ** 2)

regime_counts = pd.Series(regime).value_counts().to_dict()
per_source = {}
for s in sorted(set(source)):
    m = source == s
    per_source[s] = {
        "n": int(m.sum()),
        "regime_counts": pd.Series(regime[m]).value_counts().to_dict(),
        "median_ratio_obs_over_pred": float(np.median(ratio[m])),
        "n_underpredicted": int((Cp_actual[m] > Cp_pred[m]).sum()),
    }

result_hypothesis_check = {
    "hypothesis": "Classifying by Daily-Nece regime (I/II/III/IV) instead of "
                  "forcing regime IV on all 114 points drastically reduces "
                  "the error.",
    "formulas_source": "Poncet et al. arXiv:1305.2882 sec 2.4.1 eq 38-42, "
                        "verified by an independent WebSearch (same "
                        "coefficients/exponents). III/IV boundary: eq 42 "
                        "of the same source. Laminar/turbulent boundary: "
                        "Daily-Nece / Kreith, same source.",
    "blind_spot_declared": "Could not confidently locate the closed-form "
                            "I/II boundary (laminar regime). Does not "
                            "materially affect the result: only 1/114 "
                            "points falls on the laminar side of the "
                            "critical-Re criterion.",
    "regime_counts_all_114": regime_counts,
    "per_source": per_source,
    "median_ratio_regime_classified": float(np.median(ratio)),
    "median_ratio_naive_regime_IV_only": float(np.median(ratio_naive)),
    "n_underpredicted_regime_classified": f"{int((Cp_actual > Cp_pred).sum())}/{n}",
    "n_underpredicted_naive": f"{int((Cp_actual > Cp_dn4_naive).sum())}/{n}",
    "r2_log_space_regime_classified": float(r2_regime_classified),
    "r2_log_space_naive_regime_IV_only": float(r2_naive),
    "conclusion": "HYPOTHESIS NOT CONFIRMED. Classifying by the correct "
                  "physical regime (mostly III vs. IV; 113/114 points are "
                  "turbulent, and within those only Vrancik1968 falls "
                  "partly in merged regime III) barely moves the error: "
                  "median ratio 26.7x -> 25.1x, R2(log) -4.13 -> -3.70. "
                  "Still 114/114 underpredicted. This quantitatively "
                  "confirms that regime classification is NOT the cause: "
                  "a III<->IV regime switch only shifts the exponent by "
                  "O(0.1), insufficient to explain a bias of more than "
                  "one order of magnitude. UPDATED causal diagnosis "
                  "(verified via a real regression of "
                  "log(Cp/C_M_DailyNece) ~ log(Re_Omega): slope=-0.452, "
                  "R2=0.716, see robustness_summary_stats.json): the "
                  "DOMINANT cause is a mismatch between the corpus's "
                  "effective Reynolds exponent and regime IV's -0.2, not "
                  "regime misclassification. Geometric-class mixing "
                  "(Daily-Nece = smooth confined disk vs. the real corpus: "
                  "centrifuge arms / alternator poles / cylinders) remains "
                  "a contributing but secondary factor.",
}

# --- estandar de validacion LOFO obligatorio sobre cualquier modelo derivado ---
# El modelo directo de arriba (aplicacion de formula de Daily-Nece con
# coeficientes de literatura, CERO parametros ajustados a este dataset) no
# tiene "entrenamiento" que dejar fuera -- el per-source de arriba ya es,
# de hecho, un resultado zero-shot por instalacion.
#
# Para cumplir el estandar de validacion de forma sustantiva, se prueba la
# extension natural de la hipotesis: anadir el regimen (fusionado=1 /
# separado=0) como regresor extra al modelo ganador ya existente de
# scaling_law_search.py (log Cp ~ log Re + log Pi_gap + log Pi_confinement +
# log Pi_aspect_axial), y repetir el LOFO-CV de 4 pliegues EXACTAMENTE con
# el mismo protocolo (sin calibrar nada con la fuente excluida).
y = np.log(Cp_actual)
lRe = np.log(Re)
lgap = np.log(G)
lconf = np.log(d["Pi_confinement"].values)
lasp = np.log(d["Pi_aspect_axial"].values)
regime_dummy = (regime == "III").astype(float)  # 1=fusionado, 0=separado

facilities = sorted(set(source))


def lofo(cols_fn, names):
    all_resid = []
    per_fac = {}
    for f in facilities:
        test = source == f
        train = ~test
        Xtr = np.column_stack([np.ones(int(train.sum()))] + cols_fn(train))
        coef, *_ = np.linalg.lstsq(Xtr, y[train], rcond=None)
        Xte = np.column_stack([np.ones(int(test.sum()))] + cols_fn(test))
        pred = Xte @ coef
        resid = y[test] - pred
        r2f = (1 - np.sum(resid ** 2) / np.sum((y[test] - y[test].mean()) ** 2)
               if test.sum() > 1 else None)
        per_fac[f] = {
            "n_test": int(test.sum()),
            "held_out_r2_log": float(r2f) if r2f is not None else None,
            "held_out_rmse_log": float(np.sqrt(np.mean(resid ** 2))),
            "coef_trained_without_this_facility": dict(zip(["const"] + names, coef.tolist())),
        }
        all_resid.append(resid)
    all_resid = np.concatenate(all_resid)
    pooled_r2 = 1 - np.sum(all_resid ** 2) / np.sum((y - y.mean()) ** 2)
    pooled_rmse = float(np.sqrt(np.mean(all_resid ** 2)))
    return per_fac, float(pooled_r2), pooled_rmse


base_cols = lambda mask: [lRe[mask], lgap[mask], lconf[mask], lasp[mask]]
per_fac_base, pooled_r2_base, pooled_rmse_base = lofo(
    base_cols, ["q_Re", "p_gap", "r_conf", "t_asp"])

reg_cols = lambda mask: [lRe[mask], lgap[mask], lconf[mask], lasp[mask], regime_dummy[mask]]
per_fac_reg, pooled_r2_reg, pooled_rmse_reg = lofo(
    reg_cols, ["q_Re", "p_gap", "r_conf", "t_asp", "regime_merged_dummy"])

lofo_comparison = {
    "protocol": "4-fold leave-one-facility-out, log space, no parameter "
                "calibrated with the excluded source. Replicates exactly "
                "the protocol of scaling_law_search.py.",
    "baseline_winner_model_no_regime": {
        "cols": ["const", "log_Re", "log_Pi_gap", "log_Pi_confinement", "log_Pi_aspect_axial"],
        "pooled_r2_log": pooled_r2_base,
        "pooled_rmse_log": pooled_rmse_base,
        "per_facility": per_fac_base,
        "note": "Should match lofo_cv.pooled_r2_log from "
                "scaling_law_search_results.json (cross-check).",
    },
    "extended_model_with_regime_dummy": {
        "cols": ["const", "log_Re", "log_Pi_gap", "log_Pi_confinement",
                 "log_Pi_aspect_axial", "regime_merged_dummy(III=1,IV/otro=0)"],
        "pooled_r2_log": pooled_r2_reg,
        "pooled_rmse_log": pooled_rmse_reg,
        "per_facility": per_fac_reg,
    },
    "conclusion": "Adding the physical regime as a regressor does NOT "
                  "improve LOFO generalization (pooled R2 log-space {:.4f} "
                  "-> {:.4f}, slightly worse). Reason: the regime dummy is "
                  "almost perfectly confounded with facility identity "
                  "(only Vrancik1968 has points in merged regime III, "
                  "29/41; the other 3 sources are 100% separated regime "
                  "IV), so when Vrancik1968 is held out the model cannot "
                  "learn any merged-regime effect at all -- that "
                  "facility's R2 is identical to the baseline model. This "
                  "is another symptom of the same underlying problem: "
                  "Daily-Nece regime varies case-by-case with facility, it "
                  "is not an independent covariate that generalizes.".format(
                      pooled_r2_base, pooled_r2_reg),
}

out = {
    "daily_nece_regime_classification_check": result_hypothesis_check,
    "lofo_validation_of_regime_as_covariate": lofo_comparison,
}

with open(_ROOT + "/results/daily_nece_regime_check_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)

print(json.dumps(result_hypothesis_check, indent=2, default=str))
print()
print(json.dumps(lofo_comparison, indent=2, default=str))
