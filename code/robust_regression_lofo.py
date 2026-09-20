"""Diagnostico (2026-08-18): regresion robusta (Huber, RANSAC) vs OLS,
evaluada con LOFO real (leave-one-facility-out, 4 pliegues), sobre las
MISMAS features log-transformadas que el modelo M6 ya establecido
(log Re_Omega, log Pi_gap, log Pi_confinement, log Pi_aspect_axial) y,
como referencia adicional, sobre el modelo M1 solo-Reynolds (mejor de
todo lo probado hasta ahora, LOFO pooled R2=+0.4526).

Pregunta que responde este script: el fallo catastrofico LOFO de M6,
es un problema de OUTLIERS puntuales dentro de cada instalacion (en
cuyo caso Huber/RANSAC, que dan menos peso a residuos grandes durante
el AJUSTE, deberian mejorar el R2 LOFO), o es un DESPLAZAMIENTO
SISTEMATICO de toda una instalacion completa (en cuyo caso la
robustez a outliers puntuales no puede ayudar, porque no hay "ruido
puntual" que downweightear -- el problema es que la instalacion
completa vive en una region distinta del espacio de log-features/log-Cp
que las otras tres).

Protocolo LOFO: para cada fold, se entrena SOLO con las 3 fuentes de
entrenamiento (ajuste de coeficientes, incluido cualquier hyperparametro
interno de Huber/RANSAC que dependa de los datos de entrenamiento, p.ej.
el residual_threshold automatico de RANSAC via MAD) y se predice sobre
la 4a fuente nunca vista. No se usa ni un solo punto de la fuente
excluida ni para entrenar el modelo global ni para fijar ningun
hyperparametro.

Requiere scikit-learn (instalado en esta sesion via
`pip3 install --user --break-system-packages scikit-learn==1.9.0`, no
estaba disponible en el sistema; numpy/pandas/scipy si estaban
preinstalados a nivel de sistema).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT_PATH = _ROOT + "/results/robust_regression_lofo_results.json"

df = pd.read_csv(DATA_PATH)

# Mismo subconjunto que el modelo M6 original (114 filas, 4 fuentes)
d = df.dropna(subset=["Cp", "Re_Omega", "g_level", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "M_tip", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)
sources = d["source"].values
facilities = sorted(set(sources))

y = np.log(d["Cp"].values)
lRe = np.log(d["Re_Omega"].values)
lgap = np.log(d["Pi_gap"].values)
lconf = np.log(d["Pi_confinement"].values)
lasp = np.log(d["Pi_aspect_axial"].values)

FEATURE_SETS = {
    "M6_Re_gap_conf_asp": {
        "cols": [lRe, lgap, lconf, lasp],
        "names": ["log_Re", "log_Pi_gap", "log_Pi_confinement", "log_Pi_aspect_axial"],
    },
    "M1_Re_only": {
        "cols": [lRe],
        "names": ["log_Re"],
    },
}


def make_estimator(kind, n_train, n_features):
    if kind == "OLS":
        return LinearRegression()
    if kind == "Huber":
        # epsilon y alpha por defecto de sklearn (1.35 / 1e-4): no se
        # tunean con el fold de test, se fijan a priori para todos los
        # folds por igual (evita fuga de informacion del facility
        # excluido).
        return HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500)
    if kind == "RANSAC":
        # min_samples explicito porque con muy pocas features (M1: 1
        # feature) el valor por defecto de sklearn puede ser demasiado
        # pequeno/grande segun version; se fija a un valor razonable
        # (mitad del set de entrenamiento, con piso en n_features+2)
        # a partir SOLO del tamano del set de entrenamiento, nunca del
        # de test.
        min_samples = max(n_features + 2, int(0.5 * n_train))
        min_samples = min(min_samples, n_train - 1)
        return RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=min_samples,
            residual_threshold=None,  # MAD-based, calculado del propio train
            random_state=0,
            max_trials=1000,
        )
    raise ValueError(kind)


def lofo_eval(cols, kind):
    X_full = np.column_stack(cols)
    all_resid = []
    per_facility = {}
    outlier_fracs = {}
    for f in facilities:
        test_mask = sources == f
        train_mask = ~test_mask
        Xtr, ytr = X_full[train_mask], y[train_mask]
        Xte, yte = X_full[test_mask], y[test_mask]

        est = make_estimator(kind, n_train=Xtr.shape[0], n_features=Xtr.shape[1])
        est.fit(Xtr, ytr)
        pred = est.predict(Xte)
        resid = yte - pred
        all_resid.append(resid)

        ss_tot = np.sum((yte - yte.mean()) ** 2)
        per_facility[f] = {
            "n_test": int(test_mask.sum()),
            "held_out_r2_log": (float(1 - np.sum(resid ** 2) / ss_tot)
                                 if test_mask.sum() > 1 else None),
            "held_out_rmse_log": float(np.sqrt(np.mean(resid ** 2))),
            "held_out_mean_resid_log": float(np.mean(resid)),
            "held_out_std_resid_log": float(np.std(resid)),
        }

        if kind == "RANSAC":
            inlier_mask = est.inlier_mask_
            outlier_fracs[f] = float(1 - inlier_mask.mean())
        elif kind == "Huber":
            # sklearn expone los pesos internos solo indirectamente:
            # aproximamos "downweighted" como |residuo estandarizado
            # del ajuste final en TRAIN| > epsilon (mismo criterio que
            # usa Huber para arrancar la ponderacion IRLS)
            train_pred = est.predict(Xtr)
            train_resid = ytr - train_pred
            scale = est.scale_ if hasattr(est, "scale_") and est.scale_ > 0 else np.std(train_resid)
            standardized = np.abs(train_resid) / max(scale, 1e-12)
            outlier_fracs[f] = float(np.mean(standardized > 1.35))

    all_resid = np.concatenate(all_resid)
    pooled_r2 = float(1 - np.sum(all_resid ** 2) / np.sum((y - y.mean()) ** 2))
    pooled_rmse = float(np.sqrt(np.mean(all_resid ** 2)))
    return {
        "per_facility": per_facility,
        "pooled_r2_log": pooled_r2,
        "pooled_rmse_log": pooled_rmse,
        "train_outlier_fraction_by_held_out_fold": outlier_fracs,
    }


results = {"n_points": int(n), "facilities": facilities, "feature_sets": {}}

for fs_name, fs in FEATURE_SETS.items():
    results["feature_sets"][fs_name] = {"feature_names": fs["names"], "models": {}}
    for kind in ["OLS", "Huber", "RANSAC"]:
        res = lofo_eval(fs["cols"], kind)
        results["feature_sets"][fs_name]["models"][kind] = res

with open(OUT_PATH, "w") as fh:
    json.dump(results, fh, indent=2, default=float)

# ---------------------------------------------------------------
# Reporte por consola
# ---------------------------------------------------------------
print(f"n = {n} puntos, {len(facilities)} instalaciones: {facilities}\n")
for fs_name, fs_res in results["feature_sets"].items():
    print("=" * 78)
    print(f"FEATURE SET: {fs_name}  ({fs_res['feature_names']})")
    print("=" * 78)
    for kind, res in fs_res["models"].items():
        print(f"\n-- {kind} --  LOFO pooled R2(log) = {res['pooled_r2_log']:.4f}   "
              f"RMSE(log) = {res['pooled_rmse_log']:.4f}")
        for f in facilities:
            pf = res["per_facility"][f]
            r2s = f"{pf['held_out_r2_log']:.4f}" if pf["held_out_r2_log"] is not None else "NA"
            of = res["train_outlier_fraction_by_held_out_fold"].get(f)
            of_s = f"  train_outlier_frac(other 3 folds)={of:.3f}" if of is not None else ""
            print(f"   {f:15s} n={pf['n_test']:3d}  R2={r2s:>10s}  "
                  f"RMSE={pf['held_out_rmse_log']:.4f}  "
                  f"mean_resid={pf['held_out_mean_resid_log']:+.4f}  "
                  f"std_resid={pf['held_out_std_resid_log']:.4f}{of_s}")
    print()

print(f"\nResultados guardados en: {OUT_PATH}")
