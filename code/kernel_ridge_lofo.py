"""Kernel Ridge Regression (RBF) sobre las features log-transformadas del
modelo pooled (Re_Omega, Pi_gap, Pi_confinement, Pi_aspect_axial), evaluado
con el estandar LOFO obligatorio del proyecto (leave-one-facility-out real:
entrena con 3 fuentes, predice sobre la 4a nunca vista).

Dos variantes:
  A) KRR "puro": solo las 4 features log-Pi/Re, sin ninguna senal de
     instalacion. Hiperparametros (alpha, gamma) elegidos por
     GridSearchCV con GroupKFold DENTRO de las 3 instalaciones de
     entrenamiento (leave-one-facility-out anidado, nunca toca la
     instalacion excluida del fold externo).
  B) KRR + one-hot de instalacion (SOLO en entrenamiento, con
     handle_unknown='ignore' de sklearn -- la instalacion excluida del
     fold LOFO externo produce un vector one-hot de ceros, es decir,
     "categoria nueva/desconocida"). Mismo esquema de hiperparametros
     anidado. El objetivo de esta variante es cuantificar cuanto se
     apoya el modelo en la dummy de instalacion: si B es mucho mejor
     in-sample / en CV interna pero colapsa igual o peor que A en el
     fold LOFO externo, confirma que la dummy es un atajo de
     identificacion, no una feature generalizable.

Librerias: sklearn.kernel_ridge.KernelRidge + sklearn.preprocessing +
sklearn.model_selection (todas presentes en el venv del proyecto,
scikit-learn==1.9.0, a local virtual environment).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, GroupKFold

RNG_SEED = 12345

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)

y = np.log(d["Cp"].values)
X4 = np.column_stack([
    np.log(d["Re_Omega"].values),
    np.log(d["Pi_gap"].values),
    np.log(d["Pi_confinement"].values),
    np.log(d["Pi_aspect_axial"].values),
])
feat_names = ["log_Re", "log_Pi_gap", "log_Pi_confinement", "log_Pi_aspect_axial"]
sources = d["source"].values
facilities = sorted(set(sources))

alpha_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4]
gamma_grid = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
param_grid = {"krr__alpha": alpha_grid, "krr__gamma": gamma_grid}


def inner_cv_splits(groups_train):
    """LOFO anidado: GroupKFold con tantos folds como instalaciones
    distintas queden en el set de entrenamiento (normalmente 3), nunca
    usa la instalacion del fold externo."""
    n_groups = len(set(groups_train))
    return GroupKFold(n_splits=n_groups)


def fit_predict_variant_A(Xtr, ytr, groups_tr, Xte):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("krr", KernelRidge(kernel="rbf")),
    ])
    gkf = inner_cv_splits(groups_tr)
    gs = GridSearchCV(pipe, param_grid, cv=gkf, scoring="neg_mean_squared_error",
                       n_jobs=2)
    gs.fit(Xtr, ytr, groups=groups_tr)
    pred = gs.best_estimator_.predict(Xte)
    return pred, gs.best_params_, gs.best_score_


def fit_predict_variant_B(Xtr4, ytr, groups_tr, Xte4, groups_te):
    """Igual que A pero anadiendo one-hot de instalacion SOLO calculado
    sobre las instalaciones de entrenamiento (handle_unknown='ignore' ->
    la instalacion held-out del fold externo llega como fila de ceros,
    'categoria nueva')."""
    ohe = OneHotEncoder(handle_unknown="ignore")
    ohe.fit(groups_tr.reshape(-1, 1))
    Otr = ohe.transform(groups_tr.reshape(-1, 1)).toarray()
    Ote = ohe.transform(groups_te.reshape(-1, 1)).toarray()

    scaler = StandardScaler()
    Xtr4_s = scaler.fit_transform(Xtr4)
    Xte4_s = scaler.transform(Xte4)

    Xtr_full = np.column_stack([Xtr4_s, Otr])
    Xte_full = np.column_stack([Xte4_s, Ote])

    pipe = Pipeline([("krr", KernelRidge(kernel="rbf"))])
    gkf = inner_cv_splits(groups_tr)
    gs = GridSearchCV(pipe, param_grid, cv=gkf, scoring="neg_mean_squared_error",
                       n_jobs=2)
    gs.fit(Xtr_full, ytr, groups=groups_tr)
    pred = gs.best_estimator_.predict(Xte_full)
    n_unknown_rows = int((Ote.sum(axis=1) == 0).sum())
    return pred, gs.best_params_, gs.best_score_, n_unknown_rows


def r2_log(y_true, resid):
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else None


results_A, results_B = {}, {}
all_resid_A, all_resid_B = [], []

for f in facilities:
    test_mask = sources == f
    train_mask = ~test_mask

    Xtr, ytr = X4[train_mask], y[train_mask]
    Xte, yte = X4[test_mask], y[test_mask]
    groups_tr = sources[train_mask]
    groups_te = sources[test_mask]

    # --- Variant A: KRR puro, sin dummy de instalacion ---
    predA, best_params_A, best_cv_score_A = fit_predict_variant_A(Xtr, ytr, groups_tr, Xte)
    residA = yte - predA
    results_A[f] = {
        "n_test": int(test_mask.sum()),
        "n_train": int(train_mask.sum()),
        "best_params": best_params_A,
        "inner_cv_neg_mse": float(best_cv_score_A),
        "held_out_rmse_log": float(np.sqrt(np.mean(residA ** 2))),
        "held_out_r2_log": r2_log(yte, residA) if test_mask.sum() > 1 else None,
        "held_out_median_abs_resid_log": float(np.median(np.abs(residA))),
    }
    all_resid_A.append(residA)

    # --- Variant B: KRR + one-hot instalacion (solo train), unknown en test ---
    predB, best_params_B, best_cv_score_B, n_unknown = fit_predict_variant_B(
        Xtr, ytr, groups_tr, Xte, groups_te)
    residB = yte - predB
    results_B[f] = {
        "n_test": int(test_mask.sum()),
        "n_train": int(train_mask.sum()),
        "n_test_rows_with_unknown_onehot": n_unknown,
        "best_params": best_params_B,
        "inner_cv_neg_mse": float(best_cv_score_B),
        "held_out_rmse_log": float(np.sqrt(np.mean(residB ** 2))),
        "held_out_r2_log": r2_log(yte, residB) if test_mask.sum() > 1 else None,
        "held_out_median_abs_resid_log": float(np.median(np.abs(residB))),
    }
    all_resid_B.append(residB)

all_resid_A = np.concatenate(all_resid_A)
all_resid_B = np.concatenate(all_resid_B)
pooled_r2_A = r2_log(y, all_resid_A)
pooled_r2_B = r2_log(y, all_resid_B)
pooled_rmse_A = float(np.sqrt(np.mean(all_resid_A ** 2)))
pooled_rmse_B = float(np.sqrt(np.mean(all_resid_B ** 2)))

# --- referencia: in-sample fit (todo el dataset, sin LOFO) para contraste ---
pipe_full = Pipeline([("scaler", StandardScaler()), ("krr", KernelRidge(kernel="rbf"))])
gkf_full = GroupKFold(n_splits=4)
gs_full = GridSearchCV(pipe_full, param_grid, cv=gkf_full, scoring="neg_mean_squared_error", n_jobs=2)
gs_full.fit(X4, y, groups=sources)
pred_full_insample = gs_full.best_estimator_.predict(X4)
resid_full_insample = y - pred_full_insample
r2_insample = r2_log(y, resid_full_insample)

out = {
    "n_points": int(n),
    "sources": {k: int(v) for k, v in pd.Series(sources).value_counts().to_dict().items()},
    "features": feat_names,
    "method": "KernelRidge(kernel='rbf') sobre features log-transformadas; "
              "alpha,gamma elegidos por GridSearchCV con GroupKFold "
              "(leave-one-facility-out ANIDADO, solo dentro de las 3 "
              "instalaciones de entrenamiento del fold externo). Estandar "
              "LOFO real: la instalacion held-out del fold externo nunca "
              "se usa para elegir hiperparametros ni para entrenar.",
    "param_grid": {"alpha": alpha_grid, "gamma": gamma_grid},
    "variant_A_pure_krr_no_facility_dummy": {
        "per_facility": results_A,
        "pooled_rmse_log": pooled_rmse_A,
        "pooled_r2_log": pooled_r2_A,
    },
    "variant_B_krr_plus_onehot_facility_unknown_at_test": {
        "per_facility": results_B,
        "pooled_rmse_log": pooled_rmse_B,
        "pooled_r2_log": pooled_r2_B,
        "note": "El one-hot de instalacion se ajusta SOLO sobre las 3 "
                "instalaciones de entrenamiento (OneHotEncoder("
                "handle_unknown='ignore')). La instalacion excluida del "
                "fold LOFO externo llega como fila de ceros (categoria "
                "nueva/desconocida) -- el modelo NO ha visto nunca esa "
                "columna activada durante el entrenamiento de ese fold.",
    },
    "reference_in_sample_groupkfold4_no_lofo_holdout": {
        "note": "SOLO para contraste (no es una metrica LOFO valida): "
                "ajuste sobre TODO el dataset con hiperparametros elegidos "
                "por GroupKFold(4) pero evaluado sobre los mismos puntos "
                "de entrenamiento (in-sample). Muestra cuanto se infla el "
                "R2 cuando no hay holdout real de instalacion.",
        "best_params": gs_full.best_params_,
        "r2_log_in_sample": r2_insample,
    },
    "comparison_to_prior_best_result": {
        "solo_reynolds_lofo_pooled_r2": 0.4526,
        "solo_reynolds_per_facility": {
            "Vrancik1968": 0.468, "Guo2024_ex_Xia2024": -2.36,
            "Liu2024": -61.7, "Zheng2024": -9.59,
        },
        "pooled_loglinear_4pi_lofo_pooled_r2": -0.885,
    },
}

with open(_ROOT + "/results/kernel_ridge_lofo_results.json", "w") as fh:
    json.dump(out, fh, indent=2, default=float)

print(f"n = {n}")
print(f"Facilities: {facilities}")
print("\n=== Variant A: KRR puro (sin dummy de instalacion) ===")
for f in facilities:
    r = results_A[f]
    print(f"  {f:15s} n_test={r['n_test']:3d}  best_params={r['best_params']}  "
          f"RMSE(log)={r['held_out_rmse_log']:.4f}  R2(log)={r['held_out_r2_log']}")
print(f"  POOLED  RMSE(log)={pooled_rmse_A:.4f}  R2(log)={pooled_r2_A:.4f}")

print("\n=== Variant B: KRR + one-hot instalacion (unknown en test) ===")
for f in facilities:
    r = results_B[f]
    print(f"  {f:15s} n_test={r['n_test']:3d}  n_unknown_onehot={r['n_test_rows_with_unknown_onehot']}  "
          f"best_params={r['best_params']}  RMSE(log)={r['held_out_rmse_log']:.4f}  "
          f"R2(log)={r['held_out_r2_log']}")
print(f"  POOLED  RMSE(log)={pooled_rmse_B:.4f}  R2(log)={pooled_r2_B:.4f}")

print(f"\n=== Referencia in-sample (GroupKFold4, sin holdout real) ===")
print(f"  best_params={gs_full.best_params_}  R2(log) in-sample={r2_insample:.4f}")

print(f"\n=== Contraste con mejor resultado previo del proyecto ===")
print(f"  solo-Reynolds LOFO pooled R2 = 0.4526 (mejor hasta ahora)")
print(f"  pooled log-lineal 4-Pi LOFO pooled R2 = -0.885")
