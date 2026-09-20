"""Linea de investigacion (2026-08-18): Random Forest + Gradient Boosting
sobre log(Cp), evaluados con el MISMO estandar LOFO obligatorio del resto
del proyecto (entrena con 3 fuentes, predice sobre la 4a nunca vista, sin
calibrar nada con datos de la fuente excluida). Se espera a priori que
generalicen mal (los arboles no extrapolan fuera del rango de valores de
las features vistas en entrenamiento, y con solo 4 instalaciones los
grupos Pi geometricos son casi constantes dentro de cada una -> actuan
como dummies de fuente para un arbol igual que para un modelo lineal).

Lo mas valioso de esta linea es el analisis SHAP DIAGNOSTICO (no
predictivo) sobre el modelo entrenado con las 114 filas completas: que
feature domina la prediccion de Cp, y si el modelo esta usando alguna Pi
geometrica como proxy oculto de instalacion (senal: SHAP de esa feature
casi constante dentro de cada fuente pero muy distinto entre fuentes,
igual que la propia feature).

Librerias: sklearn y shap instaladas en un venv nuevo del proyecto
(paper_windage_power/.venv) porque el sistema no las tenia y es un
entorno "externally-managed" (PEP 668); se creo el venv y se instalaron
con pip normal, sin --break-system-packages y sin tocar el Python de
sistema.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

import shap

RNG_SEED = 42

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")

FEATURES = [
    "Re_Omega", "Pi_gap", "Pi_confinement", "Pi_aspect_axial",
    "Pi_blockage", "M_tip", "g_level",
]
GEOM_FEATURES = ["Pi_gap", "Pi_confinement", "Pi_aspect_axial", "Pi_blockage"]

d = df.dropna(subset=["Cp", "source"] + FEATURES).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)
sources = d["source"].values
facilities = sorted(set(sources))

X = d[FEATURES].values
y = np.log(d["Cp"].values)  # log(Cp), consistente con el resto del proyecto

# -----------------------------------------------------------------
# Hiperparametros: fijos, elegidos a mano para un dataset muy pequeno
# (n=114, folds LOFO de entrenamiento tan chicos como 66-106 filas y de
# test tan chicos como 8 filas). NO se hizo busqueda de hiperparametros
# con CV anidada (anadiria complejidad no pedida); son valores
# conservadores tipicos para evitar arboles profundos sobre pocos datos.
# -----------------------------------------------------------------
def make_rf():
    return RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=3,
        max_features="sqrt", random_state=RNG_SEED,
    )

def make_gbm():
    return GradientBoostingRegressor(
        n_estimators=150, max_depth=2, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=3, random_state=RNG_SEED,
    )

MODEL_FACTORIES = {"RandomForest": make_rf, "GradientBoosting": make_gbm}

# -----------------------------------------------------------------
# LOFO-CV obligatorio: entrena con 3 fuentes, predice la 4a nunca vista.
# Ningun hiperparametro ni escalado se calibra con la fuente excluida.
# -----------------------------------------------------------------
def lofo_eval(model_factory):
    all_resid = []
    per_facility = {}
    for f in facilities:
        test_mask = sources == f
        train_mask = ~test_mask
        model = model_factory()
        model.fit(X[train_mask], y[train_mask])
        pred = model.predict(X[test_mask])
        yte = y[test_mask]
        resid = yte - pred
        all_resid.append(resid)
        ss_tot = np.sum((yte - yte.mean()) ** 2)
        per_facility[f] = {
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
            "held_out_r2_log": float(1 - np.sum(resid ** 2) / ss_tot) if test_mask.sum() > 1 else None,
            "held_out_rmse_log": float(np.sqrt(np.mean(resid ** 2))),
        }
    all_resid = np.concatenate(all_resid)
    pooled_r2 = float(1 - np.sum(all_resid ** 2) / np.sum((y - y.mean()) ** 2))
    pooled_rmse = float(np.sqrt(np.mean(all_resid ** 2)))
    return {"per_facility": per_facility, "pooled_r2_log": pooled_r2, "pooled_rmse_log": pooled_rmse}

lofo_results = {}
for name, factory in MODEL_FACTORIES.items():
    lofo_results[name] = lofo_eval(factory)

# Referencia ya establecida en el proyecto (misma metrica, mismo protocolo)
BASELINE_LOG_LOG_POOLED = -0.8848004392997473  # modelo log-log pooled (Re,gap,conf,asp)
BASELINE_RE_ONLY_POOLED = 0.4526  # modelo solo-Reynolds, mejor de lo probado hasta ahora

# -----------------------------------------------------------------
# SHAP DIAGNOSTICO (no predictivo): modelo entrenado con las 114 filas
# completas. Sirve para explicar que usa el modelo, no para predecir
# fuera de muestra -- se declara explicitamente como tal.
# -----------------------------------------------------------------
full_models = {name: factory() for name, factory in MODEL_FACTORIES.items()}
shap_diag = {}
for name, model in full_models.items():
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)  # (n, n_features)

    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_ranking = sorted(
        zip(FEATURES, mean_abs_shap.tolist()), key=lambda kv: kv[1], reverse=True
    )

    # Test de "proxy de instalacion": para cada feature geometrica Pi,
    # que fraccion de la varianza total de su SHAP value se explica por
    # la identidad de la fuente (eta^2 tipo ANOVA de un factor). Un
    # eta^2 cercano a 1 significa que el SHAP de esa feature es
    # practicamente un dummy de fuente disfrazado (igual que ya se
    # diagnostico para los coeficientes del modelo lineal).
    proxy_diag = {}
    grand_mean_by_feat = sv.mean(axis=0)
    for j, feat in enumerate(FEATURES):
        col = sv[:, j]
        ss_total = np.sum((col - grand_mean_by_feat[j]) ** 2)
        ss_between = 0.0
        per_source_mean = {}
        per_source_std = {}
        for f in facilities:
            m = sources == f
            fmean = col[m].mean()
            per_source_mean[f] = float(fmean)
            per_source_std[f] = float(col[m].std())
            ss_between += m.sum() * (fmean - grand_mean_by_feat[j]) ** 2
        eta2 = float(ss_between / ss_total) if ss_total > 0 else None
        proxy_diag[feat] = {
            "eta2_shap_explained_by_source": eta2,
            "shap_mean_by_source": per_source_mean,
            "shap_std_within_source": per_source_std,
            "is_geometric_pi": feat in GEOM_FEATURES,
        }

    shap_diag[name] = {
        "mean_abs_shap_ranking": importance_ranking,
        "dominant_feature": importance_ranking[0][0],
        "proxy_diagnostic_by_feature": proxy_diag,
        "sklearn_feature_importances_ranking": sorted(
            zip(FEATURES, model.feature_importances_.tolist()),
            key=lambda kv: kv[1], reverse=True,
        ),
    }

# Ranking de eta^2 para las features geometricas Pi especificamente,
# para responder directamente la pregunta "hay senal de proxy oculto".
geom_proxy_summary = {}
for name in shap_diag:
    geom_proxy_summary[name] = sorted(
        [(feat, shap_diag[name]["proxy_diagnostic_by_feature"][feat]["eta2_shap_explained_by_source"])
         for feat in GEOM_FEATURES],
        key=lambda kv: kv[1], reverse=True,
    )
    # Para contraste: eta2 de Re_Omega y M_tip/g_level (features "fisicas",
    # no geometricas-de-instalacion), que deberian variar mas libremente
    # dentro de cada fuente (Re_Omega SI varia bastante dentro de fuente).
    geom_proxy_summary[name + "_non_geom_contrast"] = sorted(
        [(feat, shap_diag[name]["proxy_diagnostic_by_feature"][feat]["eta2_shap_explained_by_source"])
         for feat in FEATURES if feat not in GEOM_FEATURES],
        key=lambda kv: kv[1], reverse=True,
    )

# -----------------------------------------------------------------
# Salida
# -----------------------------------------------------------------
out = {
    "n_points": int(n),
    "features_used": FEATURES,
    "geometric_pi_features": GEOM_FEATURES,
    "libraries": {"sklearn_available": True, "shap_available": True,
                  "note": "instaladas en venv nuevo del proyecto (paper_windage_power/.venv), "
                          "el Python de sistema no las tenia (entorno externally-managed)."},
    "lofo_cv_results": lofo_results,
    "baseline_reference_log_log_pooled_M6": BASELINE_LOG_LOG_POOLED,
    "baseline_reference_re_only_pooled": BASELINE_RE_ONLY_POOLED,
    "any_tree_model_beats_re_only_baseline_pooled": {
        name: r["pooled_r2_log"] > BASELINE_RE_ONLY_POOLED for name, r in lofo_results.items()
    },
    "shap_diagnostic_full_data_NOT_predictive": shap_diag,
    "geometric_pi_proxy_eta2_summary_sorted_desc": geom_proxy_summary,
}

with open(_ROOT + "/results/rf_gbm_shap_diagnostics.json", "w") as f:
    json.dump(out, f, indent=2, default=float)

print(f"n = {n}, features = {FEATURES}\n")

print("=" * 78)
print("LOFO-CV (obligatorio) -- pooled R2(log-Cp) y por instalacion")
print("=" * 78)
for name, r in lofo_results.items():
    print(f"\n{name}: pooled R2 = {r['pooled_r2_log']:.4f}  pooled RMSE(log) = {r['pooled_rmse_log']:.4f}")
    for f in facilities:
        fr = r["per_facility"][f]
        print(f"    {f:15s} n_train={fr['n_train']:3d} n_test={fr['n_test']:3d}  R2={fr['held_out_r2_log']}")

print(f"\nBaseline ya establecido (log-log pooled OLS, M6): pooled R2 = {BASELINE_LOG_LOG_POOLED:.4f}")
print(f"Baseline ya establecido (solo-Reynolds, mejor probado hasta ahora): pooled R2 = {BASELINE_RE_ONLY_POOLED:.4f}")
for name in lofo_results:
    print(f"  {name} supera al baseline solo-Reynolds: {out['any_tree_model_beats_re_only_baseline_pooled'][name]}")

print("\n" + "=" * 78)
print("SHAP DIAGNOSTICO (modelo entrenado con TODOS los datos -- no predictivo)")
print("=" * 78)
for name, sd in shap_diag.items():
    print(f"\n--- {name} ---")
    print("Ranking |SHAP| medio (dominancia de features):")
    for feat, val in sd["mean_abs_shap_ranking"]:
        tag = " [GEOM]" if feat in GEOM_FEATURES else ""
        print(f"    {feat:20s} {val:.4f}{tag}")
    print(f"  Feature dominante: {sd['dominant_feature']}")
    print("  eta^2 (SHAP explicado por fuente) -- Pi geometricas, ordenado desc:")
    for feat, eta2 in geom_proxy_summary[name]:
        print(f"    {feat:20s} eta2={eta2:.4f}")
    print("  eta^2 -- features NO geometricas (contraste):")
    for feat, eta2 in geom_proxy_summary[name + "_non_geom_contrast"]:
        print(f"    {feat:20s} eta2={eta2:.4f}")

print("\nResultados completos guardados en results/rf_gbm_shap_diagnostics.json")
