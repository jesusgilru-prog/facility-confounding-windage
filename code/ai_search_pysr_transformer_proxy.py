"""Busqueda de estructura funcional via PySR (SymbolicRegression.jl) para
log(Cp), como aproximacion honesta a "transformer-guided symbolic
regression" cuando NINGUN modelo transformer pre-entrenado de regresion
simbolica end-to-end resulto disponible/cargable en Hugging Face Hub en
tiempo razonable (ver seccion "BUSQUEDA DE MODELO PREENTRENADO" abajo).

Linea 7 de investigacion sobre cross_rotor_dataset_v3.csv (paper_windage_power).

===================================================================
BUSQUEDA DE MODELO PREENTRENADO (documentada, resultado: NEGATIVO)
===================================================================
Se consulto el Hugging Face Hub (via herramientas MCP, usuario
autenticado) con las queries:
  - "symbolic regression"            -> 1 resultado (modelo con 0
    descargas, sin card/pesos utilizables directamente via
    transformers.AutoModel; repo personal sin config compatible)
  - "equation discovery transformer" -> 0 resultados
  - "symbolicregression"             -> 0 modelos, 1 Space (Streamlit,
    no es un modelo cargable localmente)
  - "end-to-end symbolic regression" -> 0 resultados
  - "NSRTS neural symbolic regression that scales" -> 0 resultados
Los modelos conocidos en la literatura para regresion simbolica
end-to-end basada en transformers (Kamienny et al. 2022 "End-to-end
symbolic regression with transformers"/E2E, Biggio et al. 2021 NSRTS,
Landajuela et al. uDSR) se distribuyen como repos de GitHub con
checkpoints en formato propio (no HF `config.json`/`AutoModel`
compatibles), y no aparecieron indexados en el Hub con pesos
descargables. Cargar cualquiera de ellos exigiria clonar el repo del
paper, instalar su entorno especifico y adaptar su tokenizador de
expresiones -- fuera de un tiempo razonable para esta prueba puntual.
CONCLUSION: no se encontro un transformer preentrenado de regresion
simbolica utilizable. Se sustituye por el proxy descrito abajo, dejando
constancia explicita de la sustitucion.

===================================================================
PROXY UTILIZADO: PySR con gramatica RESTRINGIDA Y DISTINTA a la ya
probada (symbolic_regression_lofo.py, gplearn, resultado pooled
R2=-2.67, FALLIDO)
===================================================================
PySR (SymbolicRegression.jl) es tambien evolutivo, tal como se advierte
en el encargo. Para que no sea "lo mismo con otra libreria", la
gramatica de busqueda aqui es DELIBERADAMENTE MAS RESTRICTIVA y
DISTINTA de la de gplearn:

  gplearn (script previo)      -> operadores {+,-,*,/,log,sqrt} SIN
                                    restriccion de anidamiento, arboles
                                    de profundidad libre (2 a 6),
                                    poblacion 800, 40 generaciones:
                                    equivale a una busqueda de forma
                                    funcional practicamente libre sobre
                                    las variables ya logaritmizadas.

  PySR (este script)            -> operadores binarios SOLO {+, *, /},
                                    operador unario SOLO {square}
                                    (nada de log, nada de sqrt, nada de
                                    resta), Y con restriccion explicita
                                    de anidamiento "no division dentro
                                    de division" (maximo UNA division
                                    en todo el arbol) via
                                    nested_constraints, maxsize=20.
                                    Esto fuerza expresiones del tipo
                                    "superficie de respuesta acotada":
                                    sumas de terminos polinomicos de
                                    grado <=2 en las variables
                                    logaritmizadas, con como mucho UNA
                                    correccion racional de baja orden
                                    (p.ej. a0 + a1*X + a2*X^2 + X3/(c+X0)).
                                    Es la forma funcional tipica de
                                    correlaciones de friccion en disco
                                    tipo Daily-Nece con correccion de
                                    holgura (Cp ~ Re^n * (1 + f(Pi))),
                                    que en espacio log es exactamente
                                    "polinomio de bajo orden + una
                                    correccion racional", NO un arbol
                                    libre con log/sqrt anidados.

Variables de entrada (logaritmizadas, igual convencion que el resto
del corpus -- todas estrictamente positivas, verificado):
  X0 = log(Re_Omega)
  X1 = log(Pi_gap)
  X2 = log(Pi_confinement)
  X3 = log(Pi_aspect_axial)
Objetivo: y = log(Cp)  (fitting en espacio log, estandar en este corpus)

Protocolo LOFO OBLIGATORIO (identico al resto de la linea): para cada
una de las 4 instalaciones, se entrena PySR SOLO con las 3 restantes,
se evalua la mejor expresion (por score de PySR, penalizando
complejidad) sobre la instalacion excluida (nunca vista durante el
ajuste), se reporta R2 en log tanto pooled como por instalacion.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RNG_SEED = 20260818  # fijo, reproducible

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


def r2_log(y_true, y_pred):
    rss = float(np.sum((y_true - y_pred) ** 2))
    tss = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - rss / tss if tss > 0 else float("nan")


def make_regressor(seed, out_dir):
    from pysr import PySRRegressor
    return PySRRegressor(
        niterations=80,
        populations=24,
        population_size=40,
        binary_operators=["+", "*", "/"],
        unary_operators=["square"],
        # gramatica restringida: nada de log/sqrt/resta, y como mucho
        # UNA division en todo el arbol (evita arboles racionales
        # anidados tipo a/(b/(c+d)) que gplearn ya explora libremente)
        nested_constraints={"/": {"/": 0}},
        maxsize=20,
        parsimony=0.01,
        model_selection="best",
        random_state=seed,
        deterministic=True,
        parallelism="serial",
        procs=0,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
        tempdir=out_dir,
        warm_start=False,
    )


fold_results = {}
pooled_resid = []
pooled_y = []
pooled_facility = []

for i, held_out in enumerate(facilities):
    test_mask = sources == held_out
    train_mask = ~test_mask
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    out_dir = f_ROOT + "/code/pysr_runs/fold_{held_out}"
    import os
    os.makedirs(out_dir, exist_ok=True)

    model = make_regressor(RNG_SEED + i, out_dir)
    model.fit(X_train, y_train, variable_names=feature_names)

    best_eq = str(model.sympy())
    pred_train = np.asarray(model.predict(X_train), dtype=float)
    pred_test = np.asarray(model.predict(X_test), dtype=float)
    pred_test = np.clip(pred_test, -50, 50)

    r2_train = r2_log(y_train, pred_train)
    r2_test = r2_log(y_test, pred_test) if test_mask.sum() > 1 else None
    rmse_test = float(np.sqrt(np.mean((y_test - pred_test) ** 2)))

    resid = y_test - pred_test
    pooled_resid.append(resid)
    pooled_y.append(y_test)
    pooled_facility.extend([held_out] * len(resid))

    fold_results[held_out] = {
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "best_formula_sympy": best_eq,
        "r2_train_in_sample_log": r2_train,
        "r2_test_held_out_log": r2_test,
        "rmse_test_held_out_log": rmse_test,
        "median_abs_resid_log": float(np.median(np.abs(resid))),
    }
    print(f"[{held_out}] n_train={train_mask.sum()} n_test={test_mask.sum()}")
    print(f"  formula: {best_eq}")
    print(f"  R2_train(in-sample, log)={r2_train:.4f}  R2_test(held-out, log)={r2_test}")

pooled_resid = np.concatenate(pooled_resid)
pooled_y = np.concatenate(pooled_y)
pooled_r2 = r2_log(pooled_y, pooled_y - pooled_resid)
pooled_rmse = float(np.sqrt(np.mean(pooled_resid ** 2)))

out = {
    "method": "PySR (SymbolicRegression.jl), gramatica restringida {+,*,/,square} "
              "con maximo 1 division anidada, como proxy de busqueda simbolica "
              "guiada tras NO encontrar transformer preentrenado utilizable en HF Hub",
    "pretrained_transformer_search": {
        "attempted": True,
        "found_usable_model": False,
        "queries_tried": [
            "symbolic regression",
            "equation discovery transformer",
            "symbolicregression",
            "end-to-end symbolic regression",
            "NSRTS neural symbolic regression that scales",
        ],
        "conclusion": "Ningun modelo con pesos HF-compatibles (AutoModel) "
                      "encontrado; unico hit relevante (0 descargas) sin "
                      "checkpoint utilizable. Sustituido por proxy PySR.",
    },
    "grammar": {
        "binary_operators": ["+", "*", "/"],
        "unary_operators": ["square"],
        "nested_constraints": {"/": {"/": 0}},
        "maxsize": 20,
        "difference_from_prior_gplearn_attempt": (
            "gplearn previo: {+,-,*,/,log,sqrt} sin restriccion de "
            "anidamiento, arboles libres profundidad 2-6. Este PySR: "
            "sin log/sqrt/resta, maximo 1 division anidada en todo el "
            "arbol -> fuerza polinomios de bajo orden + correccion "
            "racional simple en log-espacio, forma tipica Daily-Nece "
            "con correccion de holgura."
        ),
    },
    "n_points_total": int(n),
    "n_per_facility": {f: int((sources == f).sum()) for f in facilities},
    "fold_results": fold_results,
    "lofo_pooled": {
        "r2_log": pooled_r2,
        "rmse_log": pooled_rmse,
        "n_pooled": int(len(pooled_y)),
    },
    "decision_rule": {
        "pooled_r2_substantially_positive": bool(pooled_r2 > 0.3),
        "all_facilities_r2_nonneg": bool(all(
            (fold_results[f]["r2_test_held_out_log"] is not None and
             fold_results[f]["r2_test_held_out_log"] >= 0)
            for f in facilities
        )),
        "clears_decision_rule": bool(
            pooled_r2 > 0.3 and all(
                (fold_results[f]["r2_test_held_out_log"] is not None and
                 fold_results[f]["r2_test_held_out_log"] >= 0)
                for f in facilities
            )
        ),
    },
    "known_baselines_for_reference": {
        "best_linear_4_predictor_pooled_r2_log": -0.885,
        "reynolds_only_linear_pooled_r2_log": 0.4526,
        "reynolds_only_per_facility_r2_log": {
            "note": "orden no confirmado aqui, tomado del encargo: "
                    "-2.36, -61.7, -9.59, +0.468 (Vrancik1968)"
        },
        "already_failed_methods_pooled_r2_range": [-2.67, -0.55],
    },
}

with open(_ROOT + "/results/ai_search_pysr_transformer_proxy_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)

print("\n=== RESUMEN LOFO (PySR, gramatica restringida) ===")
for f in facilities:
    r = fold_results[f]
    print(f"  held-out={f:15s} n_test={r['n_test']:3d}  R2(log)={r['r2_test_held_out_log']}")
print(f"\nPooled R2(log) = {pooled_r2:.4f}   RMSE(log) = {pooled_rmse:.4f}")
print(f"Decision rule cleared: {out['decision_rule']['clears_decision_rule']}")
print("\nGuardado en results/ai_search_pysr_transformer_proxy_results.json")
