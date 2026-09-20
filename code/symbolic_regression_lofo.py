"""Busqueda de regresion simbolica AUTOMATIZADA (gplearn SymbolicRegressor)
para log(Cp) en funcion de log(Re_Omega), log(Pi_gap), log(Pi_confinement),
log(Pi_aspect_axial), con evaluacion leave-one-facility-out (LOFO) real.

Linea 6 de investigacion sobre el dataset cross_rotor_dataset_v3.csv
(paper_windage_power). A diferencia de las 5 lineas previas (todas con
formulas escritas a mano), aqui la busqueda de estructura funcional es
automatica: gplearn evoluciona una poblacion de expresiones simbolicas
por programacion genetica (cruce, mutacion, seleccion por torneo)
minimizando el error cuadratico medio en log(Cp).

Protocolo LOFO OBLIGATORIO: para cada una de las 4 instalaciones,
- se entrena el SymbolicRegressor SOLO con las 3 instalaciones restantes
  (ni la formula ni ningun parametro ve la instalacion excluida durante
  el entrenamiento -- no hay ajuste few-shot en este metodo),
- se evalua la MEJOR expresión de esa corrida sobre la instalacion
  excluida (nunca vista),
- se reporta el R2 en espacio log tanto pooled (concatenando los 4
  residuales held-out) como por instalacion individual.

Variables de entrada (todas en log, todas estrictamente positivas en
el dataset -- verificado antes de tomar log):
  X0 = log(Re_Omega)
  X1 = log(Pi_gap)
  X2 = log(Pi_confinement)
  X3 = log(Pi_aspect_axial)
Objetivo: y = log(Cp)

Operadores permitidos: +, -, *, /, log, sqrt (todos con las versiones
protegidas de gplearn: div protegido evita division por cero devolviendo
1.0, log protegido usa log(|x|) con floor en |x|<0.001, sqrt protegido
usa sqrt(|x|) -- necesario porque durante la evolucion las subexpresiones
intermedias pueden volverse negativas aunque las variables de entrada no
lo sean).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import warnings

import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor

warnings.filterwarnings("ignore")

RNG_SEED = 12345  # fijo, no Math.random() -- reproducibilidad

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)
assert (d["Re_Omega"] > 0).all() and (d["Pi_gap"] > 0).all()
assert (d["Pi_confinement"] > 0).all() and (d["Pi_aspect_axial"] > 0).all()

y_all = np.log(d["Cp"].values)
X_all = np.column_stack([
    np.log(d["Re_Omega"].values),
    np.log(d["Pi_gap"].values),
    np.log(d["Pi_confinement"].values),
    np.log(d["Pi_aspect_axial"].values),
])
feature_names = ["log_Re", "log_Pigap", "log_Piconf", "log_Piasp"]
sources = d["source"].values
facilities = sorted(set(sources))

GP_PARAMS = dict(
    population_size=800,
    generations=40,
    tournament_size=20,
    stopping_criteria=0.0,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=1.0,
    verbose=0,
    parsimony_coefficient=0.001,
    function_set=("add", "sub", "mul", "div", "log", "sqrt"),
    feature_names=feature_names,
    const_range=(-2.0, 2.0),
    init_depth=(2, 6),
    metric="mean absolute error",
    n_jobs=1,
)


def r2_log(y_true, y_pred):
    rss = float(np.sum((y_true - y_pred) ** 2))
    tss = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - rss / tss if tss > 0 else float("nan")


fold_results = {}
pooled_resid = []
pooled_y = []

for i, held_out in enumerate(facilities):
    test_mask = sources == held_out
    train_mask = ~test_mask
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    gp = SymbolicRegressor(random_state=RNG_SEED + i, **GP_PARAMS)
    gp.fit(X_train, y_train)

    pred_train = gp.predict(X_train)
    pred_test = gp.predict(X_test)
    pred_test = np.clip(pred_test, -50, 50)  # proteger contra explosiones numericas de la formula evolucionada

    r2_train = r2_log(y_train, pred_train)
    r2_test = r2_log(y_test, pred_test) if test_mask.sum() > 1 else None
    rmse_test = float(np.sqrt(np.mean((y_test - pred_test) ** 2)))

    resid = y_test - pred_test
    pooled_resid.append(resid)
    pooled_y.append(y_test)

    fold_results[held_out] = {
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "best_formula": str(gp._program),
        "best_formula_raw_fitness": float(gp._program.raw_fitness_),
        "r2_train_in_sample_log": r2_train,
        "r2_test_held_out_log": r2_test,
        "rmse_test_held_out_log": rmse_test,
        "median_abs_resid_log": float(np.median(np.abs(resid))),
    }
    print(f"[{held_out}] n_train={train_mask.sum()} n_test={test_mask.sum()}")
    print(f"  formula: {gp._program}")
    print(f"  R2_train(in-sample, log)={r2_train:.4f}  R2_test(held-out, log)={r2_test}")

pooled_resid = np.concatenate(pooled_resid)
pooled_y = np.concatenate(pooled_y)
pooled_r2 = r2_log(pooled_y, pooled_y - pooled_resid)
pooled_rmse = float(np.sqrt(np.mean(pooled_resid ** 2)))

# --- baseline de referencia: mismo protocolo LOFO con el modelo lineal
#     log-log completo (Re+3 grupos Pi) y con el modelo solo-Reynolds,
#     para comparar en igualdad de condiciones con el mismo pipeline ---
def lofo_linear(cols_idx):
    resids, r2s = [], {}
    all_r = []
    for held_out in facilities:
        test_mask = sources == held_out
        train_mask = ~test_mask
        Xtr = np.column_stack([np.ones(train_mask.sum())] + [X_all[train_mask, j] for j in cols_idx])
        Xte = np.column_stack([np.ones(test_mask.sum())] + [X_all[test_mask, j] for j in cols_idx])
        coef, *_ = np.linalg.lstsq(Xtr, y_all[train_mask], rcond=None)
        pred = Xte @ coef
        r = y_all[test_mask] - pred
        all_r.append(r)
        r2s[held_out] = r2_log(y_all[test_mask], pred) if test_mask.sum() > 1 else None
    all_r = np.concatenate(all_r)
    pooled = r2_log(np.concatenate([y_all[sources == f] for f in facilities]), np.concatenate([y_all[sources == f] for f in facilities]) - all_r)
    return {"per_facility_r2": r2s, "pooled_r2": pooled}

baseline_full = lofo_linear([0, 1, 2, 3])
baseline_re_only = lofo_linear([0])

out = {
    "method": "gplearn SymbolicRegressor, LOFO por instalacion (4 folds)",
    "gp_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in GP_PARAMS.items()},
    "n_points_total": int(n),
    "n_per_facility": {f: int((sources == f).sum()) for f in facilities},
    "fold_results": fold_results,
    "lofo_pooled": {
        "r2_log": pooled_r2,
        "rmse_log": pooled_rmse,
        "n_pooled": int(len(pooled_y)),
    },
    "baseline_comparison_same_pipeline": {
        "linear_loglog_Re_gap_conf_asp": baseline_full,
        "linear_loglog_Re_only": baseline_re_only,
    },
}

with open(_ROOT + "/results/symbolic_regression_lofo_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)

print("\n=== RESUMEN LOFO (regresion simbolica gplearn) ===")
for f in facilities:
    r = fold_results[f]
    print(f"  held-out={f:15s} n_test={r['n_test']:3d}  R2(log)={r['r2_test_held_out_log']}")
print(f"\nPooled R2(log) = {pooled_r2:.4f}   RMSE(log) = {pooled_rmse:.4f}")
print(f"\nBaseline lineal completo (mismo pipeline LOFO): pooled R2 = {baseline_full['pooled_r2']:.4f}")
print(f"Baseline lineal solo-Re (mismo pipeline LOFO): pooled R2 = {baseline_re_only['pooled_r2']:.4f}")
print("\nGuardado en results/symbolic_regression_lofo_results.json")
