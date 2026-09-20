"""Angle: OOD/trust detector (2026-08-18).

Pregunta: en vez de intentar predecir Cp entre instalaciones (lo que ya
ha fallado con 16 metodos bajo LOFO estricto), se puede al menos
DETECTAR, mirando solo las Pi-groups y Re de un punto de consulta (sin
ver su Cp), si ese punto esta "dentro de distribucion" respecto a las
instalaciones de entrenamiento o si es "fuera de distribucion" -- es
decir, predecir cuales son los puntos que el modelo baseline va a
predecir mal, sin necesidad de acertar el valor de Cp en si.

Protocolo (LOFO estricto, igual que el resto del proyecto):
Para cada instalacion excluida f (Guo2024, Vrancik1968, Liu2024,
Zheng2024):
  1. Se entrena el baseline ya establecido como mejor referencia,
     "M1 solo-Reynolds" (log Cp ~ a + b*log(Re_Omega), OLS), SOLO con
     las 3 instalaciones restantes. Se predice log Cp en la instalacion
     excluida y se calcula el residuo absoluto |resid| = |y_true - y_pred|
     por punto. Esto reproduce exactamente robust_regression_lofo.py
     (mismo pooled R2 = 0.4526 ya documentado), pero aqui se guardan
     los residuos POR PUNTO en vez de solo el resumen agregado.
  2. Se entrena un detector de anomalias/densidad SOLO sobre las
     features Pi (log Re_Omega, log Pi_confinement, log Pi_gap,
     log Pi_aspect_axial) de las 3 instalaciones de entrenamiento (sin
     ver Cp en absoluto, ni la etiqueta de instalacion, ni
     geom_confidence -- eso estaria confundido 1:1 con la instalacion
     y haria trampa). Se estandariza con media/std del train.
     Tres detectores independientes:
       a) Distancia de Mahalanobis al centroide del train (covarianza
          del train, con regularizacion diagonal minima si hiciera
          falta).
       b) Isolation Forest (score de anomalia).
       c) Distancia media a los k-vecinos mas cercanos del train
          (k = min(5, n_train-1)).
  3. Se guarda, por cada punto excluido: instalacion, |resid| del
     baseline M1, y las 3 puntuaciones OOD.

Analisis honesto de si el detector "funciona":
  - Correlacion de Spearman POOLED (los 114 puntos held-out juntos,
    cada uno con el detector entrenado SIN verlo) entre cada score OOD
    y |resid|.
  - Correlacion de Spearman POR INSTALACION (dentro de cada una de las
    4 instalaciones por separado) -- esto es la prueba real de si el
    detector aporta algo MAS ALLA de simplemente reconocer de que
    instalacion viene el punto, porque dentro de una misma instalacion
    ya no hay variacion de "instalacion" que explote trivialmente.
  - Correlacion "facility-demeaned": se resta a cada score y a cada
    |resid| la MEDIANA de su propia instalacion antes de correlacionar
    globalmente. Esto aisla la señal intra-instalacion pooled (mas
    puntos que una instalacion sola) del efecto trivial de que
    instalaciones distintas tengan a la vez peor Cp-baseline Y mayor
    "rareza" media en el espacio de features.
  - AUC (score OOD prediciendo "|resid| > mediana global de |resid|")
    tanto pooled como con el split de mediana calculado POR
    INSTALACION (asi el detector tiene que acertar quien es el peor
    DENTRO de cada instalacion, no solo "adivinar la instalacion mala").

Nota critica: NO se usa geom_confidence ni el nombre de la instalacion
como feature de entrada al detector -- eso reproduciria exactamente la
trampa de confusion perfecta facility<->geom_confidence descrita en el
enunciado del proyecto. Los detectores solo ven Re_Omega, Pi_confinement,
Pi_gap, Pi_aspect_axial (en log).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT_PATH = _ROOT + "/results/angle_ood_detector_results.json"

RNG_SEED = 0

df = pd.read_csv(DATA_PATH)

d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n_total = len(d)
sources = d["source"].values
facilities = sorted(set(sources))

y = np.log(d["Cp"].values)
lRe = np.log(d["Re_Omega"].values)
lgap = np.log(d["Pi_gap"].values)
lconf = np.log(d["Pi_confinement"].values)
lasp = np.log(d["Pi_aspect_axial"].values)

# Feature matrix for OOD detectors (Pi-groups + Re only, NO facility id,
# NO geom_confidence).
X_full = np.column_stack([lRe, lconf, lgap, lasp])
feature_names = ["log_Re", "log_Pi_confinement", "log_Pi_gap", "log_Pi_aspect_axial"]

assert n_total == 114, f"expected 114 rows after cleaning, got {n_total}"


def mahalanobis_scores(X_train, X_test):
    mu = X_train.mean(axis=0)
    cov = np.cov(X_train, rowvar=False)
    # regularizacion minima solo si la covarianza es casi singular
    # (evita fallo numerico con pocas muestras / features colineales)
    eps = 1e-8 * np.trace(cov) / cov.shape[0]
    cov_reg = cov + eps * np.eye(cov.shape[0])
    cov_inv = np.linalg.pinv(cov_reg)
    diffs = X_test - mu
    d2 = np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs)
    return np.sqrt(np.clip(d2, 0, None))


def isoforest_scores(X_train, X_test):
    clf = IsolationForest(n_estimators=300, random_state=RNG_SEED, contamination="auto")
    clf.fit(X_train)
    # score_samples: mayor = mas normal. Invertimos para que mayor =
    # mas anomalo, consistente con los otros dos detectores.
    return -clf.score_samples(X_test)


def knn_dist_scores(X_train, X_test, k=5):
    k_eff = min(k, X_train.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(X_train)
    dist, _ = nn.kneighbors(X_test)
    return dist.mean(axis=1)


records = []
per_facility_pooled_r2 = {}

for f in facilities:
    train_mask = sources != f
    test_mask = sources == f

    X_train_raw = X_full[train_mask]
    X_test_raw = X_full[test_mask]

    # Estandarizacion con media/std del TRAIN unicamente (sin fuga del
    # facility excluido).
    mu_s = X_train_raw.mean(axis=0)
    sd_s = X_train_raw.std(axis=0, ddof=1)
    sd_s[sd_s == 0] = 1.0
    X_train = (X_train_raw - mu_s) / sd_s
    X_test = (X_test_raw - mu_s) / sd_s

    # --- Baseline M1 solo-Reynolds (identico a robust_regression_lofo.py) ---
    lRe_train = lRe[train_mask].reshape(-1, 1)
    lRe_test = lRe[test_mask].reshape(-1, 1)
    y_train = y[train_mask]
    y_test = y[test_mask]

    m1 = LinearRegression().fit(lRe_train, y_train)
    y_pred = m1.predict(lRe_test)
    resid = y_test - y_pred
    abs_resid = np.abs(resid)

    r2_fold = 1 - np.sum(resid**2) / np.sum((y_test - y_test.mean())**2) if len(y_test) > 1 else np.nan
    per_facility_pooled_r2[f] = float(r2_fold)

    # --- Detectores OOD (SOLO ven X, nunca y/Cp, nunca el label de facility) ---
    maha = mahalanobis_scores(X_train, X_test)
    isof = isoforest_scores(X_train, X_test)
    knnd = knn_dist_scores(X_train, X_test, k=5)

    # Deteccion de extrapolacion MAS DIRECTA: distancia (en log_Re
    # estandarizado con media/std del train) al punto de train mas
    # cercano, SOLO en la variable que el propio baseline M1 usa. Mas
    # mecanisticamente ligado al fallo del baseline que la novedad en
    # el espacio Pi completo.
    lRe_train_std = (lRe[train_mask] - lRe[train_mask].mean()) / lRe[train_mask].std(ddof=1)
    lRe_test_std = (lRe[test_mask] - lRe[train_mask].mean()) / lRe[train_mask].std(ddof=1)
    re_extrap = knn_dist_scores(lRe_train_std.reshape(-1, 1), lRe_test_std.reshape(-1, 1), k=1)

    for i in range(test_mask.sum()):
        records.append({
            "source": f,
            "abs_resid_M1": float(abs_resid[i]),
            "resid_M1": float(resid[i]),
            "maha": float(maha[i]),
            "isoforest": float(isof[i]),
            "knn_dist": float(knnd[i]),
            "re_extrap_1nn": float(re_extrap[i]),
        })

res_df = pd.DataFrame(records)
assert len(res_df) == n_total

detector_cols = ["maha", "isoforest", "knn_dist", "re_extrap_1nn"]

# ---------------------------------------------------------------
# 1) Correlacion pooled (todas las 114 filas held-out juntas)
# ---------------------------------------------------------------
pooled_spearman = {}
for col in detector_cols:
    rho, p = spearmanr(res_df[col], res_df["abs_resid_M1"])
    pooled_spearman[col] = {"rho": float(rho), "p_value": float(p)}

# ---------------------------------------------------------------
# 2) Correlacion por instalacion (dentro de cada facility por separado)
# ---------------------------------------------------------------
per_facility_spearman = {}
for f in facilities:
    sub = res_df[res_df["source"] == f]
    per_facility_spearman[f] = {"n": int(len(sub))}
    for col in detector_cols:
        if len(sub) >= 4 and sub[col].nunique() > 1 and sub["abs_resid_M1"].nunique() > 1:
            rho, p = spearmanr(sub[col], sub["abs_resid_M1"])
        else:
            rho, p = (np.nan, np.nan)
        per_facility_spearman[f][col] = {"rho": float(rho) if not np.isnan(rho) else None,
                                          "p_value": float(p) if not np.isnan(p) else None}

# ---------------------------------------------------------------
# 3) Facility-demeaned (resta la mediana de la propia instalacion a
#    score y a |resid| antes de correlacionar globalmente) -- aisla
#    señal intra-instalacion pooled de las 4 facilidades juntas.
# ---------------------------------------------------------------
res_df["abs_resid_demeaned"] = res_df.groupby("source")["abs_resid_M1"].transform(lambda s: s - s.median())
demeaned_spearman = {}
for col in detector_cols:
    res_df[col + "_demeaned"] = res_df.groupby("source")[col].transform(lambda s: s - s.median())
    rho, p = spearmanr(res_df[col + "_demeaned"], res_df["abs_resid_demeaned"])
    demeaned_spearman[col] = {"rho": float(rho), "p_value": float(p)}

# ---------------------------------------------------------------
# 4) AUC: score OOD prediciendo "mal predicho" (|resid| por encima de
#    mediana). Version A: mediana GLOBAL (pooled, deja que el detector
#    gane solo por identificar la instalacion mala). Version B:
#    mediana POR INSTALACION (obliga a acertar quien es el peor DENTRO
#    de cada instalacion, elimina el atajo trivial de "adivina la
#    instalacion").
# ---------------------------------------------------------------
global_median = res_df["abs_resid_M1"].median()
res_df["bad_global"] = (res_df["abs_resid_M1"] > global_median).astype(int)
res_df["bad_within_facility"] = res_df.groupby("source")["abs_resid_M1"].transform(
    lambda s: (s > s.median()).astype(int)
)

auc_results = {"global_median_split": {}, "within_facility_median_split": {}}
for col in detector_cols:
    y_bad_g = res_df["bad_global"].values
    if len(np.unique(y_bad_g)) > 1:
        auc_g = roc_auc_score(y_bad_g, res_df[col].values)
    else:
        auc_g = None
    auc_results["global_median_split"][col] = auc_g

    y_bad_w = res_df["bad_within_facility"].values
    if len(np.unique(y_bad_w)) > 1:
        auc_w = roc_auc_score(y_bad_w, res_df[col].values)
    else:
        auc_w = None
    auc_results["within_facility_median_split"][col] = auc_w

# ---------------------------------------------------------------
# 5) Sanity check: cuanto de la señal pooled es solo "reconocer la
#    instalacion" -- media de |resid| y de cada score OOD por
#    instalacion, y correlacion Spearman ENTRE ESAS 4 MEDIAS (n=4,
#    evidencia debil per se, se reporta solo como diagnostico, no como
#    prueba).
# ---------------------------------------------------------------
facility_means = res_df.groupby("source")[["abs_resid_M1"] + detector_cols].mean()
facility_level_corr = {}
for col in detector_cols:
    if facility_means[col].nunique() > 1:
        rho, p = spearmanr(facility_means[col], facility_means["abs_resid_M1"])
    else:
        rho, p = (np.nan, np.nan)
    facility_level_corr[col] = {"rho": float(rho) if not np.isnan(rho) else None,
                                 "p_value": float(p) if not np.isnan(p) else None,
                                 "n_facilities": int(len(facility_means))}

output = {
    "protocol": "LOFO estricto (4 folds, 1 instalacion excluida por fold). "
                "Detector OOD entrenado SOLO con X (log Re, log Pi_confinement, "
                "log Pi_gap, log Pi_aspect_axial) de las 3 instalaciones de "
                "entrenamiento, sin ver Cp ni la etiqueta de instalacion. "
                "Baseline de referencia = M1 solo-Reynolds OLS (identico a "
                "robust_regression_lofo.py).",
    "n_points": n_total,
    "facilities": facilities,
    "feature_names_ood_detector": feature_names,
    "per_facility_M1_pooled_r2_log_this_run": per_facility_pooled_r2,
    "pooled_spearman_score_vs_abs_resid": pooled_spearman,
    "per_facility_spearman_score_vs_abs_resid": per_facility_spearman,
    "facility_demeaned_pooled_spearman": demeaned_spearman,
    "auc_score_predicts_bad_point": auc_results,
    "facility_level_mean_correlation_diagnostic_n4_weak": facility_level_corr,
    "facility_mean_table": facility_means.to_dict(orient="index"),
}

with open(OUT_PATH, "w") as fh:
    json.dump(output, fh, indent=2)

print(json.dumps(output, indent=2))
