"""Gaussian Process Regression LOFO probe (2026-08-18).

Pregunta que responde: todos los modelos parametricos (log-log OLS
pooled, regimenes Daily-Nece, hueco axial vs radial, transferencia
intra-clase, 17 grupos Pi nuevos, calibracion jerarquica James-Stein)
fallan en LOFO real porque los grupos Pi geometricos son casi
constantes dentro de cada instalacion (actuan como dummies de fuente
disfrazados) y solo hay 4 instalaciones (n_eff=4 clusters). Un GP no
asume una FORMA funcional fija -- interpola con un kernel RBF y, mas
importante, declara su propia incertidumbre (varianza predictiva) via
verosimilitud marginal. Este script pregunta dos cosas:

  1. ¿Generaliza mejor en LOFO real que el mejor modelo parametrico
     hasta ahora (solo-Reynolds, LOFO pooled R2=+0.4526)?
  2. Si NO generaliza (que es lo esperable dado que el problema es de
     identificabilidad/escasez de datos, no de capacidad de modelo),
     ¿al menos SABE que esta extrapolando? Es decir, ¿la varianza
     predictiva del GP es mayor sobre la instalacion excluida que
     sobre el propio training, y esa varianza esta correlacionada con
     el error real cometido?

Metodo (declarado explicitamente):
  - Features: log(Re_Omega), log(Pi_gap), log(Pi_confinement),
    log(Pi_aspect_axial) -- las mismas 4 variables adimensionales que
    en scaling_law_search.py (modelo M6 ganador) y en el resto de la
    linea, para que el resultado sea comparable termino a termino.
  - Target: log(Cp).
  - Kernel: ConstantKernel (amplitud) * RBF anisotropica (un
    length_scale por feature, permite que el GP pese cada Pi-group de
    forma distinta) + WhiteKernel (ruido/nugget). Hiperparametros
    optimizados por maxima verosimilitud marginal (LML) sobre el
    training de cada fold, con normalize_y=True y 10 reinicios del
    optimizador L-BFGS-B para evitar minimos locales.
  - Estandarizacion: las 4 features se estandarizan (media/std) usando
    SOLO estadisticas del training de cada fold (nunca de la
    instalacion excluida) antes de entrar al kernel, para que los
    length_scales optimizados sean comparables entre features de
    escalas muy distintas (Re_Omega abarca ~3 ordenes de magnitud mas
    que los Pi-groups).
  - LOFO real: para cada instalacion f, se entrena el GP con las otras
    3 (nunca se usa f ni para el modelo global ni para ningun ajuste),
    se predice sobre f completa, y se calcula R2 en log-espacio (media
    global de y usada para la varianza total, igual que en
    scaling_law_search.py, para que el numero sea comparable).
  - Incertidumbre: se reporta return_std=True del GP -- la desviacion
    estandar predictiva en log-espacio, SIN la varianza de ruido
    (WhiteKernel) sumada aparte (return_std de sklearn ya incluye el
    ruido si el WhiteKernel es parte del kernel -- se declara asi, no
    se resta a mano). Se compara la std media en training (in-sample,
    CV interno leave-one-out no se hace aqui; se usa la std sobre los
    propios puntos de training tras el fit, que es un piso optimista)
    contra la std media sobre la instalacion excluida, y se calcula la
    correlacion (Pearson) entre |residuo| y std predictiva dentro de
    la instalacion excluida.

Limitaciones declaradas:
  - Con n_train ~= 73-106 puntos y 4 features, el GP tiene margen de
    sobreajustar el kernel (length_scales largos = casi lineal,
    length_scales cortos = memorizacion local); no se restringen los
    bounds mas alla de lo razonable (declarado en KERNEL_BOUNDS) para
    no imponer a mano una solucion sesgada hacia "generaliza bien".
  - No se hace nested-CV para elegir bounds/arquitectura del kernel
    por fold -- eso violaria el estandar LOFO (usar datos de test para
    elegir hiperparametros de estructura). Los bounds son fijos y
    razonables a priori, iguales en los 4 folds.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

RNG_SEED = 12345
N_RESTARTS = 10
KERNEL_BOUNDS = {
    "constant": (1e-3, 1e3),
    "length_scale": (1e-2, 1e2),
    "noise": (1e-10, 1e1),
}

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)

y = np.log(d["Cp"].values)
lRe = np.log(d["Re_Omega"].values)
lgap = np.log(d["Pi_gap"].values)
lconf = np.log(d["Pi_confinement"].values)
lasp = np.log(d["Pi_aspect_axial"].values)
sources = d["source"].values
facilities = sorted(set(sources))

X_raw = np.column_stack([lRe, lgap, lconf, lasp])  # (n,4)
feature_names = ["log_Re_Omega", "log_Pi_gap", "log_Pi_confinement", "log_Pi_aspect_axial"]

y_global_mean = y.mean()  # para R2 pooled, consistente con scaling_law_search.py


def r2_log(y_true, y_pred, ref_mean):
    if len(y_true) < 2:
        return None
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - ref_mean) ** 2)
    if ss_tot == 0:
        return None
    return float(1 - ss_res / ss_tot)


def make_kernel(n_features):
    return (
        ConstantKernel(1.0, constant_value_bounds=KERNEL_BOUNDS["constant"])
        * RBF(length_scale=np.ones(n_features),
              length_scale_bounds=KERNEL_BOUNDS["length_scale"])
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=KERNEL_BOUNDS["noise"])
    )


results = {"per_facility": {}, "meta": {
    "n_total": int(n), "features": feature_names,
    "kernel_bounds": KERNEL_BOUNDS, "n_restarts_optimizer": N_RESTARTS,
}}

all_resid = []
all_std = []
all_y_true = []
all_y_pred = []

for f in facilities:
    test_mask = sources == f
    train_mask = ~test_mask
    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())

    Xtr_raw = X_raw[train_mask]
    ytr = y[train_mask]
    Xte_raw = X_raw[test_mask]
    yte = y[test_mask]

    # estandarizacion SOLO con estadisticas de training (nunca de f)
    mu = Xtr_raw.mean(axis=0)
    sd = Xtr_raw.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0  # guard, no aplica aqui pero por robustez
    Xtr = (Xtr_raw - mu) / sd
    Xte = (Xte_raw - mu) / sd

    kernel = make_kernel(Xtr.shape[1])
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=N_RESTARTS,
        random_state=RNG_SEED, alpha=0.0,
    )
    gp.fit(Xtr, ytr)

    pred_te, std_te = gp.predict(Xte, return_std=True)
    pred_tr, std_tr = gp.predict(Xtr, return_std=True)

    resid_te = yte - pred_te
    resid_tr = ytr - pred_tr

    r2_f = r2_log(yte, pred_te, y_global_mean)

    # correlacion entre |residuo| y std predictiva dentro de la instalacion excluida
    if n_test >= 3 and np.std(std_te) > 0 and np.std(np.abs(resid_te)) > 0:
        corr_abs_resid_std = float(np.corrcoef(np.abs(resid_te), std_te)[0, 1])
    else:
        corr_abs_resid_std = None

    results["per_facility"][f] = {
        "n_train": n_train, "n_test": n_test,
        "held_out_r2_log": r2_f,
        "held_out_rmse_log": float(np.sqrt(np.mean(resid_te ** 2))),
        "held_out_median_abs_resid_log": float(np.median(np.abs(resid_te))),
        "mean_pred_std_log_on_held_out_facility": float(np.mean(std_te)),
        "mean_pred_std_log_on_own_training": float(np.mean(std_tr)),
        "std_ratio_heldout_over_train": float(np.mean(std_te) / np.mean(std_tr))
                                          if np.mean(std_tr) > 0 else None,
        "corr_abs_resid_vs_pred_std_within_heldout": corr_abs_resid_std,
        "fitted_kernel": str(gp.kernel_),
        "log_marginal_likelihood_train": float(gp.log_marginal_likelihood_value_),
    }

    all_resid.append(resid_te)
    all_std.append(std_te)
    all_y_true.append(yte)
    all_y_pred.append(pred_te)

all_resid = np.concatenate(all_resid)
all_std = np.concatenate(all_std)
all_y_true = np.concatenate(all_y_true)
all_y_pred = np.concatenate(all_y_pred)

pooled_r2 = r2_log(all_y_true, all_y_pred, y_global_mean)
pooled_rmse = float(np.sqrt(np.mean(all_resid ** 2)))
if np.std(all_std) > 0 and np.std(np.abs(all_resid)) > 0:
    pooled_corr_abs_resid_std = float(np.corrcoef(np.abs(all_resid), all_std)[0, 1])
else:
    pooled_corr_abs_resid_std = None

results["lofo_cv"] = {
    "pooled_r2_log": pooled_r2,
    "pooled_rmse_log": pooled_rmse,
    "pooled_corr_abs_resid_vs_pred_std": pooled_corr_abs_resid_std,
}

# comparacion explicita con la mejor linea previa (solo-Reynolds)
results["comparison_to_prior_best"] = {
    "solo_reynolds_lofo_pooled_r2": 0.4526,
    "solo_reynolds_per_facility_r2_approx": {
        "Vrancik1968": 0.468, "Guo2024_aka_Xia2024": -2.36,
        "Liu2024": -61.7, "Zheng2024": -9.59,
    },
    "gp_beats_solo_reynolds_pooled": (pooled_r2 is not None and pooled_r2 > 0.4526),
    "note": "cifras solo-Reynolds citadas del contexto de la tarea (linea previa "
            "ya cerrada), no recalculadas aqui.",
}

out_path = _ROOT + "/results/gp_lofo_test_results.json"
with open(out_path, "w") as fh:
    json.dump(results, fh, indent=2)

print("=== GP (RBF anisotropica + WhiteKernel) LOFO real ===")
print(f"n_total={n}  features={feature_names}")
print()
for f in facilities:
    r = results["per_facility"][f]
    print(f"{f:14s} n_test={r['n_test']:3d}  R2_log={r['held_out_r2_log']:+8.4f}  "
          f"RMSE_log={r['held_out_rmse_log']:.4f}  "
          f"std_heldout={r['mean_pred_std_log_on_held_out_facility']:.4f}  "
          f"std_train={r['mean_pred_std_log_on_own_training']:.4f}  "
          f"ratio={r['std_ratio_heldout_over_train']:.2f}  "
          f"corr(|resid|,std)={r['corr_abs_resid_vs_pred_std_within_heldout']}")
print()
print(f"POOLED LOFO R2(log) = {pooled_r2:.4f}   RMSE(log) = {pooled_rmse:.4f}")
print(f"POOLED corr(|resid|, pred_std) = {pooled_corr_abs_resid_std}")
print()
print(f"Comparacion: solo-Reynolds LOFO pooled R2 = 0.4526  |  GP = {pooled_r2:.4f}  "
      f"|  GP mejora = {results['comparison_to_prior_best']['gp_beats_solo_reynolds_pooled']}")
print(f"\nGuardado en {out_path}")
