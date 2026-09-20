"""Few-shot / partial-pooling LOFO probe (2026-08-17).

Pregunta que responde: la ley pooled ingenua falla catastroficamente en
LOFO puro (R2 pooled en log-espacio = -0.885, ver
results/scaling_law_search_results.json, generado por
scaling_law_search.py). Eso es "zero-shot": cero puntos de la
instalacion excluida. Pero un investigador real que aplique esto a una
instalacion NUEVA normalmente SI tiene un puñado de mediciones de
calibracion de esa instalacion. Este script pregunta: ¿cuantos puntos
de calibracion hacen falta, con un encogimiento (shrinkage) simple del
intercepto hacia la media global, para que el LOFO deje de ser
catastrofico?

Metodo (declarado explicitamente, NO es el motor bayesiano jerarquico
de paper8 -- es un James-Stein / empirical-Bayes de libro de texto
sobre UN solo parametro, el intercepto):

  1. Se fija la MISMA forma funcional que el modelo ganador M6 de
     scaling_law_search.py: log(Cp) = b0 + q*log(Re) + p*log(Pi_gap)
     + r*log(Pi_confinement) + t*log(Pi_aspect_axial).
  2. Para cada instalacion excluida f (LOFO real: f NUNCA se usa para
     entrenar el modelo global ni las pendientes):
       a. Se ajustan por minimos cuadrados ordinarios el intercepto
          GLOBAL b0_hat y las 4 pendientes (q,p,r,t) SOLO con las
          otras 3 instalaciones agrupadas (pooled), igual que en el
          LOFO original.
       b. Con esas mismas 3 instalaciones de entrenamiento se estima
          la varianza ENTRE instalaciones del intercepto (tau^2): para
          cada una de las 3 instalaciones de entrenamiento se calcula
          su propio "offset" de intercepto (media de los residuos
          usando las pendientes globales fijas, menos b0_hat). tau^2
          es la varianza (ddof=1, n=3, MUY poca muestra, declarado
          como limitacion) de esos 3 offsets.
       c. Se estima sigma^2 = varianza residual DENTRO de instalacion
          (residuos tras restar el offset propio de cada instalacion
          de entrenamiento), pooled sobre las 3.
       d. Para n_cal en {1,3,5,10}: se muestrean sin reemplazo n_cal
          puntos de la instalacion f (semilla fija, 20 repeticiones
          independientes). Con esos puntos SOLAMENTE (las pendientes
          siguen fijas, NUNCA se reajustan con datos de f) se calcula
          el intercepto local ingenuo:
              b0_naive_f = mean(y_cal - pendientes_globales . X_cal)
          y se encoge hacia el global con el estimador James-Stein /
          empirical-Bayes de libro:
              lambda = tau^2 / (tau^2 + sigma^2 / n_cal)
              b0_shrunk = lambda * b0_naive_f + (1-lambda) * b0_hat
          (n_cal=0 => lambda=0 => b0_shrunk = b0_hat, es decir,
          colapsa EXACTAMENTE al LOFO puro original -- se usa como
          chequeo de consistencia, no como punto de la curva pedida).
       e. Se evalua SOLO sobre los puntos de f que NO se usaron como
          calibracion (para no medir el ajuste sobre los mismos puntos
          que se usaron para calibrar). R2 en log-espacio con la
          varianza propia de ese subconjunto de test.
       f. Si f tiene tan pocos puntos que no quedan >=2 para test tras
          apartar n_cal (caso Zheng2024, n=8: n_cal=10 imposible,
          incluso n_cal=... se limita), se marca ese (facility, n_cal)
          como no computable y se declara la razon, NO se rellena con
          un numero inventado.
  3. Se repite 20 veces por (facility, n_cal) con muestreo aleatorio
     distinto (semilla fija global, RNG unico avanzando en orden
     determinista) y se promedia -- tambien se reporta la desviacion
     estandar entre repeticiones para no esconder la varianza.
  4. R2 pooled por n_cal: se concatenan los residuos de test de las 4
     instalaciones DENTRO DE CADA repeticion y se calcula R2 pooled de
     esa repeticion; luego se promedia sobre las 20 repeticiones.

Limitaciones declaradas en el propio script (no solo en el reporte):
  - tau^2 se estima con n=3 instalaciones de entrenamiento -- muestra
    minima, la estimacion puede ser inestable de una ronda LOFO a otra.
  - Las PENDIENTES siguen siendo las globales pooled, que ya se sabe
    (bootstrap por instalacion, scaling_law_search.py) que NO son
    estables entre instalaciones (varian 12x en pendiente de Re sola).
    Este experimento SOLO adapta el intercepto; si el fallo real
    estuviera tambien en la pendiente, encoger solo el intercepto
    puede no bastar -- eso es precisamente lo que este script mide,
    no lo que asume.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd

RNG_SEED = 12345
N_REPEATS = 20
N_CAL_GRID = [1, 3, 5, 10]
MIN_TEST_POINTS = 2  # minimo de puntos de test tras apartar calibracion para que R2 tenga sentido

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
d = df.dropna(subset=["Cp", "Re_Omega", "g_level", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "M_tip", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)

y = np.log(d["Cp"].values)
lRe = np.log(d["Re_Omega"].values)
lgap = np.log(d["Pi_gap"].values)
lconf = np.log(d["Pi_confinement"].values)
lasp = np.log(d["Pi_aspect_axial"].values)
sources = d["source"].values
facilities = sorted(set(sources))

# Diseno X sin columna de intercepto (el intercepto se maneja aparte,
# es el unico parametro que se encoge)
X_slopes = np.column_stack([lRe, lgap, lconf, lasp])  # (n,4): q,p,r,t


def fit_pooled(mask):
    """OLS pooled (intercepto + 4 pendientes) sobre mask."""
    Xd = np.column_stack([np.ones(mask.sum()), X_slopes[mask]])
    coef, *_ = np.linalg.lstsq(Xd, y[mask], rcond=None)
    return coef[0], coef[1:]  # b0, slopes(4,)


def naive_local_intercept(idx, slopes):
    """Intercepto local ingenuo a partir de puntos de calibracion (idx),
    usando las pendientes GLOBALES fijas (nunca reajustadas con f)."""
    resid = y[idx] - X_slopes[idx] @ slopes
    return float(np.mean(resid))


def r2_log(y_true, y_pred):
    if len(y_true) < 2:
        return None
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return None
    return float(1 - ss_res / ss_tot)


rng = np.random.default_rng(RNG_SEED)

results = {"per_facility": {}, "pooled_by_n_cal": {}}
# curva completa: incluye n_cal=0 (chequeo de consistencia con LOFO original)
curve_grid = [0] + N_CAL_GRID

# repeticion-a-repeticion, pooled across facilities (para el R2 pooled por n_cal)
pooled_r2_per_ncal_per_rep = {nc: [] for nc in curve_grid}

sanity_check_vs_original_lofo = {}

for f in facilities:
    test_facility_mask = sources == f
    train_mask = ~test_facility_mask
    f_idx_all = np.where(test_facility_mask)[0]
    n_f = len(f_idx_all)

    # 1) modelo global pooled entrenado SOLO con las otras 3 instalaciones
    b0_hat, slopes_hat = fit_pooled(train_mask)

    # 2) tau^2 (varianza entre instalaciones de entrenamiento) y sigma^2
    #    (varianza residual dentro de instalacion), estimadas SOLO con
    #    las 3 instalaciones de entrenamiento (nunca con f)
    train_facilities = [g for g in facilities if g != f]
    offsets = []
    within_resid_all = []
    for g in train_facilities:
        g_idx = np.where(sources == g)[0]
        resid_g = y[g_idx] - X_slopes[g_idx] @ slopes_hat
        mean_resid_g = resid_g.mean()
        offsets.append(mean_resid_g - b0_hat)
        within_resid_all.append(resid_g - mean_resid_g)
    offsets = np.array(offsets)
    tau2 = float(np.var(offsets, ddof=1)) if len(offsets) > 1 else 0.0
    within_resid_all = np.concatenate(within_resid_all)
    # dof: n_train - n_train_facilities (se resta una media por instalacion)
    dof_sigma = max(len(within_resid_all) - len(train_facilities), 1)
    sigma2 = float(np.sum(within_resid_all ** 2) / dof_sigma)

    # --- n_cal = 0 : chequeo de consistencia con el LOFO original (lambda=0) ---
    pred0 = b0_hat + X_slopes[f_idx_all] @ slopes_hat
    r2_0 = r2_log(y[f_idx_all], pred0)
    sanity_check_vs_original_lofo[f] = {
        "n_test": int(n_f), "r2_log_ncal0": r2_0,
        "note": "should match lofo_cv.per_facility.<f>.held_out_r2_log from "
                "scaling_law_search_results.json (same model, same test "
                "set = whole facility, lambda=0)",
    }

    results["per_facility"][f] = {"n_total_facility": int(n_f), "tau2": tau2,
                                   "sigma2": sigma2, "b0_hat_global": float(b0_hat),
                                   "by_n_cal": {}}

    for n_cal in curve_grid:
        if n_cal == 0:
            # lambda=0 exacto, no hay muestreo aleatorio que hacer
            results["per_facility"][f]["by_n_cal"]["0"] = {
                "feasible": True, "mean_r2": r2_0, "std_r2": 0.0,
                "n_reps_used": 1, "note": "lambda=0 (no calibration), identical "
                                           "to the original pure LOFO",
            }
            pooled_r2_per_ncal_per_rep[0] = [None]  # se llena mas abajo con logica pooled real
            continue

        n_test_after = n_f - n_cal
        if n_test_after < MIN_TEST_POINTS:
            results["per_facility"][f]["by_n_cal"][str(n_cal)] = {
                "feasible": False,
                "reason": f"facility {f} has only {n_f} points; "
                          f"holding out {n_cal} for calibration leaves "
                          f"{n_test_after} < {MIN_TEST_POINTS} for testing",
                "mean_r2": None, "std_r2": None, "n_reps_used": 0,
            }
            continue

        rep_r2 = []
        for rep in range(N_REPEATS):
            cal_idx = rng.choice(f_idx_all, size=n_cal, replace=False)
            test_idx = np.setdiff1d(f_idx_all, cal_idx, assume_unique=False)

            b0_naive = naive_local_intercept(cal_idx, slopes_hat)
            lam = tau2 / (tau2 + sigma2 / n_cal) if (tau2 + sigma2 / n_cal) > 0 else 0.0
            b0_shrunk = lam * b0_naive + (1 - lam) * b0_hat

            pred_test = b0_shrunk + X_slopes[test_idx] @ slopes_hat
            r2 = r2_log(y[test_idx], pred_test)
            if r2 is not None:
                rep_r2.append(r2)

        results["per_facility"][f]["by_n_cal"][str(n_cal)] = {
            "feasible": True,
            "mean_r2": float(np.mean(rep_r2)) if rep_r2 else None,
            "std_r2": float(np.std(rep_r2, ddof=1)) if len(rep_r2) > 1 else 0.0,
            "n_reps_used": len(rep_r2),
            "lambda_last_rep": float(lam),
        }

# --- R2 pooled por n_cal: repetir el muestreo de forma acoplada entre
#     las 4 instalaciones dentro de cada repeticion (misma logica, RNG
#     re-arrancado con la MISMA semilla para que la comparacion n_cal=0
#     vs n_cal>0 sea limpia y reproducible por separado) ---
rng2 = np.random.default_rng(RNG_SEED + 1)
pooled_curve = {}
for n_cal in N_CAL_GRID:
    rep_pooled_r2 = []
    for rep in range(N_REPEATS):
        all_y_test, all_pred_test = [], []
        skipped_any = False
        for f in facilities:
            f_idx_all = np.where(sources == f)[0]
            n_f = len(f_idx_all)
            train_mask = sources != f
            b0_hat, slopes_hat = fit_pooled(train_mask)
            train_facilities = [g for g in facilities if g != f]
            offsets, within_resid_all = [], []
            for g in train_facilities:
                g_idx = np.where(sources == g)[0]
                resid_g = y[g_idx] - X_slopes[g_idx] @ slopes_hat
                mean_resid_g = resid_g.mean()
                offsets.append(mean_resid_g - b0_hat)
                within_resid_all.append(resid_g - mean_resid_g)
            tau2 = float(np.var(np.array(offsets), ddof=1))
            within_resid_all = np.concatenate(within_resid_all)
            dof_sigma = max(len(within_resid_all) - len(train_facilities), 1)
            sigma2 = float(np.sum(within_resid_all ** 2) / dof_sigma)

            n_test_after = n_f - n_cal
            if n_test_after < MIN_TEST_POINTS:
                skipped_any = True
                continue  # esta instalacion no aporta a esta ronda pooled (Zheng2024/n_cal=10)

            cal_idx = rng2.choice(f_idx_all, size=n_cal, replace=False)
            test_idx = np.setdiff1d(f_idx_all, cal_idx, assume_unique=False)
            b0_naive = naive_local_intercept(cal_idx, slopes_hat)
            lam = tau2 / (tau2 + sigma2 / n_cal) if (tau2 + sigma2 / n_cal) > 0 else 0.0
            b0_shrunk = lam * b0_naive + (1 - lam) * b0_hat
            pred_test = b0_shrunk + X_slopes[test_idx] @ slopes_hat
            all_y_test.append(y[test_idx])
            all_pred_test.append(pred_test)

        y_te = np.concatenate(all_y_test)
        pred_te = np.concatenate(all_pred_test)
        r2p = r2_log(y_te, pred_te)
        rep_pooled_r2.append({"r2": r2p, "skipped_a_facility": skipped_any,
                               "n_pooled_test": int(len(y_te))})
    r2_vals = [r["r2"] for r in rep_pooled_r2 if r["r2"] is not None]
    pooled_curve[str(n_cal)] = {
        "mean_r2_pooled": float(np.mean(r2_vals)) if r2_vals else None,
        "std_r2_pooled": float(np.std(r2_vals, ddof=1)) if len(r2_vals) > 1 else None,
        "n_reps_used": len(r2_vals),
        "any_rep_skipped_zheng2024": any(r["skipped_a_facility"] for r in rep_pooled_r2),
        "note": ("Zheng2024 (n=8) cannot supply test points for the pooled "
                 "round when n_cal=10 (8-10<2); it is excluded ONLY from "
                 "that pooled round, the other 3 facilities still "
                 "contribute."
                 if n_cal == 10 else
                 "all 4 facilities contribute to this pooled round."),
    }

# n_cal=0 pooled: identico al LOFO original (lambda=0 en todas las instalaciones,
# sin aleatoriedad -- una sola pasada, no 20 repeticiones porque no hay muestreo)
all_y0, all_pred0 = [], []
for f in facilities:
    f_idx_all = np.where(sources == f)[0]
    train_mask = sources != f
    b0_hat, slopes_hat = fit_pooled(train_mask)
    pred0 = b0_hat + X_slopes[f_idx_all] @ slopes_hat
    all_y0.append(y[f_idx_all])
    all_pred0.append(pred0)
r2_pooled_ncal0 = r2_log(np.concatenate(all_y0), np.concatenate(all_pred0))
pooled_curve["0"] = {"mean_r2_pooled": r2_pooled_ncal0, "std_r2_pooled": 0.0,
                      "n_reps_used": 1, "any_rep_skipped_zheng2024": False,
                      "note": "lambda=0 across all 4 facilities = original pure LOFO "
                              "(consistency check, should be -0.885)"}

results["pooled_by_n_cal"] = pooled_curve
results["sanity_check_vs_original_lofo"] = sanity_check_vs_original_lofo
results["config"] = {"rng_seed": RNG_SEED, "n_repeats": N_REPEATS,
                      "n_cal_grid": N_CAL_GRID, "min_test_points": MIN_TEST_POINTS,
                      "model_form": "log(Cp) = b0 + q*log(Re) + p*log(Pi_gap) + "
                                     "r*log(Pi_confinement) + t*log(Pi_aspect_axial), "
                                     "identical to the winning model M6 from "
                                     "scaling_law_search.py",
                      "shrinkage": "single-parameter James-Stein/empirical-Bayes "
                                   "(the intercept only): lambda = tau2/(tau2+sigma2/n_cal), "
                                   "tau2 and sigma2 estimated from the 3 training "
                                   "facilities in each LOFO round. Slopes are "
                                   "NEVER refit with data from the excluded "
                                   "facility."}

with open(_ROOT + "/results/hierarchical_calibration_lofo_results.json", "w") as fh:
    json.dump(results, fh, indent=2, default=float)

print("=== Chequeo de consistencia (n_cal=0 debe igualar LOFO original) ===")
for f, v in sanity_check_vs_original_lofo.items():
    print(f"  {f:15s} n={v['n_test']:3d}  R2(log, n_cal=0) = {v['r2_log_ncal0']}")

print("\n=== Curva por instalacion: R2(log) vs n_cal (media +- std sobre 20 reps) ===")
for f in facilities:
    print(f"\n{f}  (n_total={results['per_facility'][f]['n_total_facility']}, "
          f"tau2={results['per_facility'][f]['tau2']:.4f}, "
          f"sigma2={results['per_facility'][f]['sigma2']:.4f})")
    for nc in curve_grid:
        v = results["per_facility"][f]["by_n_cal"][str(nc)]
        if v.get("feasible", True) and v["mean_r2"] is not None:
            print(f"    n_cal={nc:3d}  R2_mean={v['mean_r2']:9.4f}  "
                  f"R2_std={v['std_r2']:.4f}  (n_reps={v['n_reps_used']})")
        else:
            print(f"    n_cal={nc:3d}  NO COMPUTABLE: {v.get('reason')}")

print("\n=== Curva pooled (4 instalaciones combinadas): R2(log) vs n_cal ===")
for nc in curve_grid:
    v = pooled_curve[str(nc)]
    print(f"  n_cal={nc:3d}  R2_pooled_mean={v['mean_r2_pooled']}  "
          f"R2_pooled_std={v['std_r2_pooled']}  n_reps={v['n_reps_used']}  "
          f"nota={v['note']}")
