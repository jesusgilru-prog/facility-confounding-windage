"""Hipotesis (2026-08-17, investigacion adicional): grupos adimensionales
NUEVOS construibles a partir de columnas crudas no usadas en el modelo
original (R_chamber_m, h_rotor_m, H_pole_m, H_chamber_m, gap_axial_m,
M_tip, g_level), evaluados con el MISMO estandar LOFO obligatorio del
resto del proyecto (entrena con 3 fuentes, predice sobre la 4a nunca
vista, ni siquiera para calibrar un intercepto).

Este script es DELIBERADAMENTE separado de scaling_law_search.py (no lo
modifica) para no tocar los resultados ya congelados del paper. Extiende
el mismo dataset y la misma metodologia (BIC en log-log, LOFO de 4
pliegues) con un conjunto ampliado de estructuras.

Paso 0 (obligatorio antes de proponer nada): mapear que columnas crudas
estan realmente disponibles en las 4 fuentes, porque un grupo que solo
existe en una fuente es indistinguible de un dummy de fuente y NO es
testable en un LOFO honesto de 4 vias.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")

# ---------------------------------------------------------------
# PASO 0: disponibilidad por fuente de las columnas crudas candidatas
# ---------------------------------------------------------------
availability = {}
for col in ["R_chamber_m", "gap_radial_m", "h_rotor_m", "H_pole_m",
            "H_chamber_m", "gap_axial_m", "M_tip", "g_level"]:
    availability[col] = df.groupby("source")[col].apply(lambda s: int(s.notna().sum())).to_dict()

# ---------------------------------------------------------------
# Subconjunto usable (mismas 114 filas que el modelo original: Cp,
# Re_Omega, g_level, Pi_gap, Pi_confinement, Pi_aspect_axial, M_tip,
# source todas no-nulas)
# ---------------------------------------------------------------
d = df.dropna(subset=["Cp", "Re_Omega", "g_level", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "M_tip", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)
sources = d["source"].values
facilities = sorted(set(sources))

y = np.log(d["Cp"].values)
lRe = np.log(d["Re_Omega"].values)
lg = np.log(d["g_level"].clip(lower=1e-6).values)
lgap = np.log(d["Pi_gap"].values)
lconf = np.log(d["Pi_confinement"].values)
lasp = np.log(d["Pi_aspect_axial"].values)
lmt = np.log(d["M_tip"].clip(lower=1e-8).values)

# ---------------------------------------------------------------
# GRUPOS NUEVOS PROPUESTOS (justificacion fisica de cada uno)
# ---------------------------------------------------------------
# Los tres candidatos "obvios" a partir de columnas crudas no usadas
# (H_pole_m, H_chamber_m, gap_axial_m) resultan ser EXCLUSIVOS de una
# sola fuente cada uno (ver `availability` mas abajo) -> no se pueden
# calcular para las otras 3 fuentes, luego no son testables en un LOFO
# real de 4 vias sin imputacion inventada. Se documentan como
# estructuralmente no testables, no se fuerzan al modelo pooled.
#
# Los grupos que SI se pueden calcular en las 114 filas (las 4
# fuentes) combinan Re_Omega, Pi_gap, Pi_confinement, Pi_aspect_axial,
# M_tip y g_level -- las unicas columnas 100% pobladas en las 4
# fuentes ademas de las ya usadas en el modelo original.
#
# 1. Pi_slenderness = h_rotor_m / gap_radial_m = Pi_aspect_axial/Pi_gap
#    (razon altura del rotor / holgura radial). Fisicamente: controla
#    si el windage esta dominado por efectos de borde axial (rotor
#    "delgado" en un hueco ancho) o por flujo tipo Couette confinado
#    (rotor "alto" en un hueco estrecho). Es una recombinacion de
#    columnas ya usadas, pero como UN SOLO termino (1 parametro en vez
#    de 2) reduce grados de libertad -- interesante para LOFO aunque
#    algebraicamente student de lasp-lgap.
l_slender = lasp - lgap

# 2. Pi_Re_g = interaccion Re x g_level en log-espacio, forzada a
#    coeficiente comun (1 parametro): un "numero de Reynolds efectivo"
#    que se degrada con la gravedad artificial -- justificacion fisica:
#    en hipergravedad la capa limite se adelgaza por el aumento del
#    empuje centrifugo, efecto que podria modular el termino viscoso
#    en vez de sumarse a el de forma independiente.
l_Re_x_g = lRe + lg

# 3. Pi_Mtip_gap = M_tip * Pi_gap (interaccion compresibilidad x
#    holgura radial). Justificacion: los efectos de compresibilidad en
#    la punta del rotor (M_tip) deberian manifestarse sobre todo en la
#    region de holgura estrecha, no de forma aditiva independiente del
#    tamano del hueco.
l_Mtip_x_gap = lmt + lgap

# 4. Pi_Mtip_conf = M_tip * Pi_confinement (compresibilidad x
#    confinamiento). Justificacion: el efecto de compresibilidad de la
#    punta se transmite/refleja en la camara segun cuan confinado este
#    el rotor.
l_Mtip_x_conf = lmt + lconf

# 5. Pi_g_conf = g_level * Pi_confinement (fuerza centrifuga relativa
#    x confinamiento). Justificacion: el flujo secundario de
#    recirculacion inducido por la gravedad artificial depende tanto
#    de cuanto "pesa" el fluido relativamente (g_level) como de cuanto
#    espacio tiene para recircular (confinamiento).
l_g_x_conf = lg + lconf

# 6. Pi_g_gap = g_level * Pi_gap (fuerza centrifuga relativa x holgura
#    radial). Justificacion analoga a la anterior pero para el hueco
#    radial en vez del confinamiento global.
l_g_x_gap = lg + lgap

# ---------------------------------------------------------------
# Motor de ajuste identico al de scaling_law_search.py (OLS en
# log-espacio, BIC/AIC/R2)
# ---------------------------------------------------------------
def fit_linear(cols, names):
    X = np.column_stack([np.ones(n)] + cols)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    rss = float(np.sum((y - pred) ** 2))
    k = X.shape[1]
    bic = n * np.log(rss / n) + k * np.log(n)
    aic = n * np.log(rss / n) + 2 * k
    r2 = 1 - rss / np.sum((y - y.mean()) ** 2)
    return {"coef": dict(zip(["const"] + names, coef.tolist())),
            "rss": rss, "k": k, "bic": bic, "aic": aic, "r2": r2,
            "cols_key": names}

structures = {}
# --- referencia (ya conocidas, recalculadas aqui identicas para verificar) ---
structures["M0_null"] = fit_linear([], [])
structures["M1_Re"] = fit_linear([lRe], ["q_Re"])
structures["M2_Re_g"] = fit_linear([lRe, lg], ["q_Re", "b_g"])
structures["M6_Re_gap_conf_asp_WINNER_ORIG"] = fit_linear(
    [lRe, lgap, lconf, lasp], ["q_Re", "p_gap", "r_conf", "t_asp"])
structures["M7_Re_gap_conf_Mtip"] = fit_linear(
    [lRe, lgap, lconf, lmt], ["q_Re", "p_gap", "r_conf", "u_Mtip"])

# --- NUEVAS estructuras con los 6 grupos propuestos ---
structures["N1_Re_Mtip"] = fit_linear([lRe, lmt], ["q_Re", "u_Mtip"])
structures["N2_Mtip_only"] = fit_linear([lmt], ["u_Mtip"])
structures["N3_g_only"] = fit_linear([lg], ["b_g"])
structures["N4_M6_plus_g"] = fit_linear(
    [lRe, lgap, lconf, lasp, lg], ["q_Re", "p_gap", "r_conf", "t_asp", "b_g"])
structures["N5_M6_plus_Mtip"] = fit_linear(
    [lRe, lgap, lconf, lasp, lmt], ["q_Re", "p_gap", "r_conf", "t_asp", "u_Mtip"])
structures["N6_Re_Mtip_g"] = fit_linear([lRe, lmt, lg], ["q_Re", "u_Mtip", "b_g"])
structures["N7_Re_conf_Mtip"] = fit_linear([lRe, lconf, lmt], ["q_Re", "r_conf", "u_Mtip"])
structures["N8_Re_conf_slender"] = fit_linear(
    [lRe, lconf, l_slender], ["q_Re", "r_conf", "s_slender"])
structures["N9_Re_x_g_combined"] = fit_linear([l_Re_x_g], ["w_Rexg"])
structures["N10_Re_x_g_plus_conf_gap"] = fit_linear(
    [l_Re_x_g, lgap, lconf], ["w_Rexg", "p_gap", "r_conf"])
structures["N11_M6_plus_Mtip_x_gap"] = fit_linear(
    [lRe, lgap, lconf, lasp, l_Mtip_x_gap], ["q_Re", "p_gap", "r_conf", "t_asp", "z_MtipXgap"])
structures["N12_M6_plus_Mtip_x_conf"] = fit_linear(
    [lRe, lgap, lconf, lasp, l_Mtip_x_conf], ["q_Re", "p_gap", "r_conf", "t_asp", "z_MtipXconf"])
structures["N13_M6_plus_g_x_conf"] = fit_linear(
    [lRe, lgap, lconf, lasp, l_g_x_conf], ["q_Re", "p_gap", "r_conf", "t_asp", "z_gXconf"])
structures["N14_M6_plus_g_x_gap"] = fit_linear(
    [lRe, lgap, lconf, lasp, l_g_x_gap], ["q_Re", "p_gap", "r_conf", "t_asp", "z_gXgap"])
structures["N15_Mtip_x_gap_only"] = fit_linear([l_Mtip_x_gap], ["z_MtipXgap"])
structures["N16_g_x_conf_only"] = fit_linear([l_g_x_conf], ["z_gXconf"])
structures["N17_M6_full_plus_g_plus_Mtip"] = fit_linear(
    [lRe, lgap, lconf, lasp, lg, lmt],
    ["q_Re", "p_gap", "r_conf", "t_asp", "b_g", "u_Mtip"])

ranking = sorted(structures.items(), key=lambda kv: kv[1]["bic"])

# ---------------------------------------------------------------
# LOFO-CV (protocolo obligatorio) para TODAS las estructuras nuevas
# (no solo la ganadora por BIC: el objetivo de esta investigacion es
# ver si ALGUNA de ellas, aunque no gane BIC in-sample, reduce el
# fallo de generalizacion cruzada)
# ---------------------------------------------------------------
col_builders = {
    "M1_Re_only": lambda m: [lRe[m]],
    "M2_Re_g": lambda m: [lRe[m], lg[m]],
    "M6_Re_gap_conf_asp_WINNER_ORIG": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m]],
    "M7_Re_gap_conf_Mtip": lambda m: [lRe[m], lgap[m], lconf[m], lmt[m]],
    "N1_Re_Mtip": lambda m: [lRe[m], lmt[m]],
    "N2_Mtip_only": lambda m: [lmt[m]],
    "N3_g_only": lambda m: [lg[m]],
    "N4_M6_plus_g": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], lg[m]],
    "N5_M6_plus_Mtip": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], lmt[m]],
    "N6_Re_Mtip_g": lambda m: [lRe[m], lmt[m], lg[m]],
    "N7_Re_conf_Mtip": lambda m: [lRe[m], lconf[m], lmt[m]],
    "N8_Re_conf_slender": lambda m: [lRe[m], lconf[m], l_slender[m]],
    "N9_Re_x_g_combined": lambda m: [l_Re_x_g[m]],
    "N10_Re_x_g_plus_conf_gap": lambda m: [l_Re_x_g[m], lgap[m], lconf[m]],
    "N11_M6_plus_Mtip_x_gap": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], l_Mtip_x_gap[m]],
    "N12_M6_plus_Mtip_x_conf": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], l_Mtip_x_conf[m]],
    "N13_M6_plus_g_x_conf": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], l_g_x_conf[m]],
    "N14_M6_plus_g_x_gap": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], l_g_x_gap[m]],
    "N15_Mtip_x_gap_only": lambda m: [l_Mtip_x_gap[m]],
    "N16_g_x_conf_only": lambda m: [l_g_x_conf[m]],
    "N17_M6_full_plus_g_plus_Mtip": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], lg[m], lmt[m]],
}

def lofo_eval(cols_fn):
    all_resid = []
    per_facility = {}
    for f in facilities:
        test_mask = sources == f
        train_mask = ~test_mask
        Xtr = np.column_stack([np.ones(train_mask.sum())] + cols_fn(train_mask))
        ytr = y[train_mask]
        coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        Xte = np.column_stack([np.ones(test_mask.sum())] + cols_fn(test_mask))
        yte = y[test_mask]
        resid = yte - Xte @ coef
        all_resid.append(resid)
        ss_tot = np.sum((yte - yte.mean()) ** 2)
        per_facility[f] = {
            "n_test": int(test_mask.sum()),
            "held_out_r2_log": float(1 - np.sum(resid ** 2) / ss_tot) if test_mask.sum() > 1 else None,
            "held_out_rmse_log": float(np.sqrt(np.mean(resid ** 2))),
        }
    all_resid = np.concatenate(all_resid)
    pooled_r2 = float(1 - np.sum(all_resid ** 2) / np.sum((y - y.mean()) ** 2))
    pooled_rmse = float(np.sqrt(np.mean(all_resid ** 2)))
    return {"per_facility": per_facility, "pooled_r2_log": pooled_r2, "pooled_rmse_log": pooled_rmse}

lofo_all = {}
for name, cols_fn in col_builders.items():
    lofo_all[name] = lofo_eval(cols_fn)

lofo_ranking = sorted(lofo_all.items(), key=lambda kv: kv[1]["pooled_r2_log"], reverse=True)

# ---------------------------------------------------------------
# Salida
# ---------------------------------------------------------------
out = {
    "n_points": int(n),
    "raw_column_availability_by_source": availability,
    "note_facility_exclusive_columns": (
        "H_pole_m has data only for Vrancik1968 (41/41, 0 in the other 3); "
        "gap_axial_m only for Liu2024 (20/20, 0 in the other 3); "
        "H_chamber_m is entirely missing for Vrancik1968 (0/41) although "
        "present in the other 3. Any Pi group built from these 3 columns is "
        "indistinguishable from a facility dummy for at least one facility -- "
        "it is NOT testable in an honest 4-fold LOFO without imputing data "
        "that does not exist. Excluded from the pooled candidate set for "
        "this structural reason, not for lack of physical interest."
    ),
    "bic_ranking": [(name, s["bic"], s["r2"], s["k"]) for name, s in ranking],
    "bic_winner_overall": ranking[0][0],
    "bic_winner_is_original_M6": ranking[0][0] == "M6_Re_gap_conf_asp_WINNER_ORIG",
    "structures_detail": structures,
    "lofo_pooled_r2_by_structure_sorted_best_first": [(name, r["pooled_r2_log"]) for name, r in lofo_ranking],
    "lofo_detail": lofo_all,
    "baseline_lofo_pooled_r2_M6_from_original_search": -0.8848004392997473,
    "any_new_structure_beats_baseline_pooled": any(
        r["pooled_r2_log"] > -0.8848004392997473 for name, r in lofo_all.items() if name != "M6_Re_gap_conf_asp_WINNER_ORIG"
    ),
    "best_new_structure_lofo": lofo_ranking[0],
}

with open(_ROOT + "/results/new_pi_groups_lofo_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)

print(f"n = {n}\n")
print("Disponibilidad de columnas crudas candidatas por fuente:")
print(json.dumps(availability, indent=2))
print("\nRanking por BIC (todas las estructuras, original + nuevas):")
for name, s in ranking:
    print(f"  {name:38s} BIC={s['bic']:9.2f}  R2={s['r2']:.4f}  k={s['k']}")
print(f"\nGanador BIC global: {ranking[0][0]}  (es el M6 original: {ranking[0][0]=='M6_Re_gap_conf_asp_WINNER_ORIG'})")

print("\n" + "=" * 70)
print("LOFO-CV pooled R2(log) por estructura, ordenado de mejor a peor:")
print("=" * 70)
for name, r in lofo_ranking:
    print(f"  {name:38s} pooled_R2={r['pooled_r2_log']:10.4f}  pooled_RMSE={r['pooled_rmse_log']:.4f}")
    for f in facilities:
        fr = r["per_facility"][f]
        print(f"       {f:15s} n={fr['n_test']:3d}  R2={fr['held_out_r2_log']}")

print(f"\nBaseline conocido (M6 original, ya verificado antes de este script): pooled R2 = -0.8848")
print(f"Alguna estructura nueva SUPERA el baseline: {out['any_new_structure_beats_baseline_pooled']}")
print(f"Mejor estructura nueva en LOFO pooled: {out['best_new_structure_lofo']}")
