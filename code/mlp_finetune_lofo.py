"""MLP fine-tuning few-shot LOFO probe (2026-08-18).

Extiende hierarchical_calibration_lofo.py (James-Stein / shrinkage lineal
de UN solo parametro, el intercepto) a una red neuronal pequena real
con fine-tuning por GRADIENTE de TODOS sus pesos con los mismos puntos
de calibracion, para responder: ¿el fine-tuning con gradiente aprovecha
mejor los pocos puntos de calibracion que el shrinkage lineal simple,
o es igual/peor?

sklearn y pytorch NO estan instalados en este entorno (verificado con
import, pip install bloqueado por "externally-managed-environment" sin
--break-system-packages, que no se ha forzado). Sustituto declarado:
MLP pequeno (4 -> 8 tanh -> 1 lineal, 49 parametros) implementado a
mano en numpy puro, con forward/backward manual y descenso de
gradiente (Adam para el pre-entrenamiento, SGD plano para el
fine-tuning). Esto es un sustituto razonable y honesto de un MLP de
PyTorch para una red de este tamano, no una simplificacion que cambie
la pregunta que se responde.

Metodo (declarado explicitamente):
  Mismas 4 features que el modelo ganador M6 de scaling_law_search.py:
  log(Re_Omega), log(Pi_gap), log(Pi_confinement), log(Pi_aspect_axial)
  -> log(Cp). Estandarizadas (z-score) con media/std calculadas SOLO
  con las 3 instalaciones de entrenamiento (nunca con la excluida f).

  Para cada instalacion excluida f (LOFO real, f nunca en pre-entreno):
    1. Pre-entrenamiento: MLP entrenado con Adam (lr=0.01, 3000 epocas,
       weight decay L2=1e-4 en pesos no en biases) sobre las 3
       instalaciones de entrenamiento pooled, full-batch, semilla fija
       determinista por fold (SEED_BASE + hash(f) truncado).
    2. n_cal=0: evaluacion zero-shot del modelo pre-entrenado sobre TODA
       f (chequeo de consistencia / equivalente al lambda=0 del script
       James-Stein, pero para la red).
    3. Para n_cal en {1,3,5,10}, 20 repeticiones (mismo esquema de RNG
       que hierarchical_calibration_lofo.py: rng.choice sin reemplazo,
       semilla global fija, avance deterministico):
         a. Se muestrean n_cal puntos REALES de f (calibracion).
         b. Fine-tuning: partiendo de los pesos PRE-ENTRENADOS (nunca
            desde cero), se dan POCOS pasos de descenso de gradiente
            plano (FT_STEPS=10, FT_LR=0.02, sin Adam, sin momentum) con
            SOLO esos n_cal puntos como batch completo. Todos los pesos
            (W1,b1,W2,b2) son entrenables, sin regularizacion extra
            (los pocos pasos + lr bajo son la unica proteccion contra
            sobreajuste inmediato, tal como se pidio).
         c. Evaluacion SOLO sobre los puntos de f que NO se usaron para
            calibrar (test_idx = f menos cal_idx). R2 en log(Cp) con la
            varianza propia de ese subconjunto de test (misma definicion
            r2_log que en hierarchical_calibration_lofo.py).
         d. Si f tiene tan pocos puntos que no quedan >=2 para test tras
            apartar n_cal (Zheng2024, n=8, n_cal=10), se marca no
            computable, NO se rellena con numero inventado.
    4. R2 pooled por n_cal: concatenando residuos de test de las 4
       instalaciones DENTRO de cada repeticion, promediado sobre las 20.

Los hiperparametros (H=8, epocas, lr, pasos de fine-tuning, lr de
fine-tuning) se fijaron a priori con criterio estandar de ML (no se
ajustaron mirando el R2 de la instalacion excluida en ningun fold --
eso violaria LOFO). Se comprueba por separado que el pre-entrenamiento
converge razonablemente (R2 in-sample sobre las 3 instalaciones de
entrenamiento) antes de fiarse del resultado LOFO.

Comparacion directa pedida: curva James-Stein ya obtenida (pooled log-R2)
n_cal = 0,1,3,5,10 -> -0.885, -0.859, -0.818, -0.743, -0.683
(results/hierarchical_calibration_lofo_results.json).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd

RNG_SEED = 12345
N_REPEATS = 20
N_CAL_GRID = [1, 3, 5, 10]
MIN_TEST_POINTS = 2

H = 8              # hidden units
PRETRAIN_EPOCHS = 3000
PRETRAIN_LR = 0.01
WEIGHT_DECAY = 1e-4
FT_STEPS = 10
FT_LR = 0.02

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
d = df.dropna(subset=["Cp", "Re_Omega", "g_level", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "M_tip", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)

y_raw = np.log(d["Cp"].values)
lRe = np.log(d["Re_Omega"].values)
lgap = np.log(d["Pi_gap"].values)
lconf = np.log(d["Pi_confinement"].values)
lasp = np.log(d["Pi_aspect_axial"].values)
sources = d["source"].values
facilities = sorted(set(sources))

X_raw = np.column_stack([lRe, lgap, lconf, lasp])  # (n,4)


def r2_log(y_true, y_pred):
    if len(y_true) < 2:
        return None
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return None
    return float(1 - ss_res / ss_tot)


# ---------------- MLP: 4 -> H (tanh) -> 1 (linear), numpy puro ----------------

def init_weights(rng, n_in=4, n_hidden=H):
    # Xavier-ish init
    W1 = rng.normal(0, np.sqrt(1.0 / n_in), size=(n_in, n_hidden))
    b1 = np.zeros(n_hidden)
    W2 = rng.normal(0, np.sqrt(1.0 / n_hidden), size=(n_hidden, 1))
    b2 = np.zeros(1)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    h = np.tanh(z1)
    yhat = (h @ params["W2"] + params["b2"]).ravel()
    cache = (X, z1, h)
    return yhat, cache


def backward(params, cache, y_true, yhat, weight_decay=0.0):
    X, z1, h = cache
    n = X.shape[0]
    dyhat = (2.0 / n) * (yhat - y_true)  # dMSE/dyhat, shape (n,)
    dW2 = h.T @ dyhat.reshape(-1, 1) + weight_decay * params["W2"]
    db2 = np.array([dyhat.sum()])
    dh = dyhat.reshape(-1, 1) @ params["W2"].T
    dz1 = dh * (1 - h ** 2)
    dW1 = X.T @ dz1 + weight_decay * params["W1"]
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def adam_pretrain(params, X, y_true, epochs, lr, weight_decay, seed):
    rng_local = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(vv) for k, vv in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        yhat, cache = forward(params, X)
        grads = backward(params, cache, y_true, yhat, weight_decay=weight_decay)
        for k in params:
            m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
            v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)
            mhat = m[k] / (1 - beta1 ** t)
            vhat = v[k] / (1 - beta2 ** t)
            params[k] = params[k] - lr * mhat / (np.sqrt(vhat) + eps)
    return params


def sgd_finetune(params, X, y_true, steps, lr):
    p = {k: v.copy() for k, v in params.items()}
    for _ in range(steps):
        yhat, cache = forward(p, X)
        grads = backward(p, cache, y_true, yhat, weight_decay=0.0)
        for k in p:
            p[k] = p[k] - lr * grads[k]
    return p


# ---------------- estandarizacion por fold (solo con train) ----------------

def standardize_fit(X, y):
    mu_X, sd_X = X.mean(axis=0), X.std(axis=0)
    sd_X[sd_X == 0] = 1.0
    mu_y, sd_y = y.mean(), y.std()
    if sd_y == 0:
        sd_y = 1.0
    return mu_X, sd_X, mu_y, sd_y


def standardize_apply(X, y, mu_X, sd_X, mu_y, sd_y):
    return (X - mu_X) / sd_X, (y - mu_y) / sd_y


def unstandardize_y(y_std, mu_y, sd_y):
    return y_std * sd_y + mu_y


results = {"per_facility": {}, "pooled_by_n_cal": {}, "pretrain_diagnostics": {},
           "hyperparameters": {"H": H, "pretrain_epochs": PRETRAIN_EPOCHS,
                                "pretrain_lr": PRETRAIN_LR, "weight_decay": WEIGHT_DECAY,
                                "ft_steps": FT_STEPS, "ft_lr": FT_LR,
                                "n_repeats": N_REPEATS, "n_cal_grid": N_CAL_GRID}}

curve_grid = [0] + N_CAL_GRID
pretrained_cache = {}  # facility -> (params, mu_X, sd_X, mu_y, sd_y)

for f in facilities:
    test_mask = sources == f
    train_mask = ~test_mask
    f_idx_all = np.where(test_mask)[0]
    n_f = len(f_idx_all)

    Xtr, ytr = X_raw[train_mask], y_raw[train_mask]
    mu_X, sd_X, mu_y, sd_y = standardize_fit(Xtr, ytr)
    Xtr_s, ytr_s = standardize_apply(Xtr, ytr, mu_X, sd_X, mu_y, sd_y)

    fold_seed = RNG_SEED + sum(ord(c) for c in f) * 97  # deterministic, no built-in hash() (PYTHONHASHSEED-dependent)
    params0 = init_weights(np.random.default_rng(fold_seed))
    params_pre = adam_pretrain(params0, Xtr_s, ytr_s, PRETRAIN_EPOCHS, PRETRAIN_LR,
                                WEIGHT_DECAY, seed=fold_seed)

    # diagnostico: R2 in-sample sobre las 3 instalaciones de entreno
    yhat_tr_s, _ = forward(params_pre, Xtr_s)
    yhat_tr = unstandardize_y(yhat_tr_s, mu_y, sd_y)
    r2_train_insample = r2_log(ytr, yhat_tr)
    results["pretrain_diagnostics"][f] = {
        "r2_train_insample_pooled_other3": r2_train_insample,
        "note": "R2 in-sample del MLP sobre las 3 instalaciones de entreno "
                "(pooled), NO es LOFO -- solo verifica que el pre-entreno convergio."}

    pretrained_cache[f] = (params_pre, mu_X, sd_X, mu_y, sd_y)

    # --- n_cal = 0: zero-shot sobre TODA f ---
    Xf_s, _ = standardize_apply(X_raw[f_idx_all], y_raw[f_idx_all], mu_X, sd_X, mu_y, sd_y)
    yhat0_s, _ = forward(params_pre, Xf_s)
    yhat0 = unstandardize_y(yhat0_s, mu_y, sd_y)
    r2_0 = r2_log(y_raw[f_idx_all], yhat0)

    results["per_facility"][f] = {"n_total_facility": int(n_f), "by_n_cal": {}}
    results["per_facility"][f]["by_n_cal"]["0"] = {
        "feasible": True, "mean_r2": r2_0, "std_r2": 0.0, "n_reps_used": 1,
        "note": "zero-shot MLP pre-entrenado, sin fine-tuning, evaluado sobre TODA f "
                "(analogo a lambda=0 en el script James-Stein)."}

    rng = np.random.default_rng(RNG_SEED)
    for n_cal in N_CAL_GRID:
        n_test_after = n_f - n_cal
        if n_test_after < MIN_TEST_POINTS:
            results["per_facility"][f]["by_n_cal"][str(n_cal)] = {
                "feasible": False,
                "reason": f"facility {f} has only {n_f} points; holding out "
                          f"{n_cal} for calibration leaves {n_test_after} < "
                          f"{MIN_TEST_POINTS} for testing",
                "mean_r2": None, "std_r2": None, "n_reps_used": 0}
            continue

        rep_r2 = []
        for rep in range(N_REPEATS):
            cal_idx = rng.choice(f_idx_all, size=n_cal, replace=False)
            test_idx = np.setdiff1d(f_idx_all, cal_idx, assume_unique=False)

            Xcal_s, ycal_s = standardize_apply(X_raw[cal_idx], y_raw[cal_idx],
                                                mu_X, sd_X, mu_y, sd_y)
            params_ft = sgd_finetune(params_pre, Xcal_s, ycal_s, FT_STEPS, FT_LR)

            Xtest_s, _ = standardize_apply(X_raw[test_idx], y_raw[test_idx],
                                            mu_X, sd_X, mu_y, sd_y)
            yhat_test_s, _ = forward(params_ft, Xtest_s)
            yhat_test = unstandardize_y(yhat_test_s, mu_y, sd_y)
            r2 = r2_log(y_raw[test_idx], yhat_test)
            if r2 is not None:
                rep_r2.append(r2)

        results["per_facility"][f]["by_n_cal"][str(n_cal)] = {
            "feasible": True,
            "mean_r2": float(np.mean(rep_r2)) if rep_r2 else None,
            "std_r2": float(np.std(rep_r2, ddof=1)) if len(rep_r2) > 1 else 0.0,
            "n_reps_used": len(rep_r2)}

# ---------------- R2 pooled por n_cal (acoplado entre instalaciones dentro de cada rep) ----------------
# n_cal=0 pooled: concatenar zero-shot de las 4 (una sola "repeticion", determinista)
all_y0, all_pred0 = [], []
for f in facilities:
    f_idx_all = np.where(sources == f)[0]
    params_pre, mu_X, sd_X, mu_y, sd_y = pretrained_cache[f]
    Xf_s, _ = standardize_apply(X_raw[f_idx_all], y_raw[f_idx_all], mu_X, sd_X, mu_y, sd_y)
    yhat0_s, _ = forward(params_pre, Xf_s)
    yhat0 = unstandardize_y(yhat0_s, mu_y, sd_y)
    all_y0.append(y_raw[f_idx_all]); all_pred0.append(yhat0)
all_y0 = np.concatenate(all_y0); all_pred0 = np.concatenate(all_pred0)
results["pooled_by_n_cal"]["0"] = {
    "mean_r2_pooled": r2_log(all_y0, all_pred0), "std_r2_pooled": 0.0,
    "n_reps_used": 1, "note": "zero-shot MLP, todas las instalaciones, sin fine-tuning."}

# Se repite el mismo bucle de muestreo (rng con la MISMA semilla RNG_SEED, mismo orden
# facility -> n_cal -> rep que el bucle per-facility de arriba, por lo que consume la
# secuencia aleatoria idéntica y produce las MISMAS muestras de calibración) y esta vez
# se agrupan los resultados por rep across facilities en vez de por facility, para poder
# calcular el R2 pooled de cada repetición.
rep_pooled_by_ncal = {nc: [] for nc in N_CAL_GRID}
for f in facilities:
    f_idx_all = np.where(sources == f)[0]
    n_f = len(f_idx_all)
    params_pre, mu_X, sd_X, mu_y, sd_y = pretrained_cache[f]
    rng = np.random.default_rng(RNG_SEED)  # identico al bucle de arriba para esta facility
    for n_cal in N_CAL_GRID:
        n_test_after = n_f - n_cal
        if n_test_after < MIN_TEST_POINTS:
            for rep in range(N_REPEATS):
                pass  # no consume rng (coincide con el bucle de arriba: se salta sin muestrear)
            continue
        for rep in range(N_REPEATS):
            cal_idx = rng.choice(f_idx_all, size=n_cal, replace=False)
            test_idx = np.setdiff1d(f_idx_all, cal_idx, assume_unique=False)
            Xcal_s, ycal_s = standardize_apply(X_raw[cal_idx], y_raw[cal_idx], mu_X, sd_X, mu_y, sd_y)
            params_ft = sgd_finetune(params_pre, Xcal_s, ycal_s, FT_STEPS, FT_LR)
            Xtest_s, _ = standardize_apply(X_raw[test_idx], y_raw[test_idx], mu_X, sd_X, mu_y, sd_y)
            yhat_test_s, _ = forward(params_ft, Xtest_s)
            yhat_test = unstandardize_y(yhat_test_s, mu_y, sd_y)
            rep_pooled_by_ncal[n_cal].append((rep, f, y_raw[test_idx], yhat_test))

for n_cal in N_CAL_GRID:
    per_rep_r2 = {}
    for rep, f, yt, yp in rep_pooled_by_ncal[n_cal]:
        per_rep_r2.setdefault(rep, {"y": [], "p": []})
        per_rep_r2[rep]["y"].append(yt)
        per_rep_r2[rep]["p"].append(yp)
    rep_r2_vals = []
    zheng_skipped = False
    for rep, dd in per_rep_r2.items():
        yy = np.concatenate(dd["y"]); pp = np.concatenate(dd["p"])
        r2 = r2_log(yy, pp)
        if r2 is not None:
            rep_r2_vals.append(r2)
    n_facilities_contributing = len(set(f for _, f, _, _ in rep_pooled_by_ncal[n_cal]))
    zheng_skipped = n_facilities_contributing < len(facilities)
    results["pooled_by_n_cal"][str(n_cal)] = {
        "mean_r2_pooled": float(np.mean(rep_r2_vals)) if rep_r2_vals else None,
        "std_r2_pooled": float(np.std(rep_r2_vals, ddof=1)) if len(rep_r2_vals) > 1 else 0.0,
        "n_reps_used": len(rep_r2_vals),
        "any_rep_skipped_zheng2024": bool(zheng_skipped),
        "note": "all 4 facilities contribute" if not zheng_skipped else
                "Zheng2024 (n=8) cannot supply test points for this pooled round when "
                "n_f - n_cal < 2; excluded ONLY from this pooled round."}

# ---------------- comparacion directa con James-Stein ----------------
js_baseline = {"0": -0.8848004392997484, "1": -0.8594553795566364,
               "3": -0.8179778180758814, "5": -0.7428989686137879,
               "10": -0.683270527499946}
comparison = {}
for k in ["0", "1", "3", "5", "10"]:
    mlp_r2 = results["pooled_by_n_cal"][k]["mean_r2_pooled"]
    comparison[k] = {"james_stein_pooled_r2": js_baseline[k],
                      "mlp_finetune_pooled_r2": mlp_r2,
                      "mlp_minus_js": (mlp_r2 - js_baseline[k]) if mlp_r2 is not None else None}
results["comparison_vs_james_stein"] = comparison

with open(_ROOT + "/results/mlp_finetune_lofo_results.json", "w") as fh:
    json.dump(results, fh, indent=2)

print(json.dumps(results["pretrain_diagnostics"], indent=2))
print(json.dumps(results["pooled_by_n_cal"], indent=2))
print(json.dumps(comparison, indent=2))
