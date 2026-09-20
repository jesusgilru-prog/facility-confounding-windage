"""Ensemble de MLPs pequenos + dispersion como estimador de incertidumbre,
bajo LOFO real (leave-one-facility-out).

Pregunta central (mas importante que el R2 predictivo): cuando el ensemble
predice sobre la instalacion excluida (fuera de distribucion, OOD), la
dispersion entre miembros del ensemble -- proxy de incertidumbre epistemica
-- es sistematicamente mas alta que sobre datos de instalaciones vistas
pero no usados para entrenar cada miembro (validacion interna, ID)?

Metodo:
  - Para cada uno de los 4 folds LOFO (entrena en 3 fuentes, evalua en la
    4a nunca vista, ni para el modelo global ni para el ensemble):
      1. El conjunto de entrenamiento (3 fuentes) se divide una vez en
         train interno (70%) / validacion interna ID (30%), particion fija
         para todos los miembros del ensemble de ese fold.
      2. N_MEMBERS=10 MLPRegressor pequenos (16,8) se entrenan cada uno
         sobre un remuestreo bootstrap del train interno, con semilla
         distinta (bagging + init aleatoria distinta -> diversidad real).
      3. Prediccion del ensemble = media de los N miembros.
      4. Incertidumbre del ensemble en cada punto = desviacion estandar
         entre los N miembros (en espacio log(Cp)).
      5. Se registra esa dispersion en los puntos de validacion interna
         (ID, mismas fuentes que entrenamiento) y en los puntos de la
         fuente excluida (OOD).
  - Se repite todo con dos conjuntos de predictores:
      (A) Re-only: [log(Re_Omega)]  -- el mejor generalizador conocido
          hasta ahora (LOFO pooled R2=+0.4526 con regresion lineal).
      (B) Full: [log(Re_Omega), Pi_gap, Pi_confinement, Pi_aspect_axial]
          -- el modelo que sobreajusta a instalacion (LOFO pooled
          R2=-0.885 con regresion lineal).
  - Comparacion ID vs OOD: media, mediana, y test de Mann-Whitney U
    (una cola: OOD > ID) sobre la dispersion agrupada de los 4 folds.

Nada se calibra con datos de la fuente excluida: el escalado (StandardScaler)
se ajusta solo con el train interno de cada fold; el hold-out OOD solo se usa
para prediccion final, nunca para fit de escalado, pesos, ni bootstrap.

Librerias: sklearn.neural_network.MLPRegressor (torch no esta instalado en
este entorno; se declara explicitamente y se usa este sustituto razonable,
ya disponible en el venv del proyecto).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats

RNG_SEED = 20260818  # fijo, no Math.random()
N_MEMBERS = 10
HIDDEN = (16, 8)

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
d["logCp"] = np.log(d["Cp"])
d["logRe"] = np.log(d["Re_Omega"])
sources = sorted(d["source"].unique().tolist())
print(f"n={len(d)} filas usables, fuentes={sources}")


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def run_feature_set(feature_cols, label):
    print("=" * 70)
    print(f"Conjunto de predictores: {label} = {feature_cols}")
    print("=" * 70)

    master_rng = np.random.default_rng(RNG_SEED)

    fold_results = {}
    pooled_true, pooled_pred_mean = [], []
    id_std_all, ood_std_all = [], []

    for test_src in sources:
        train_df = d[d["source"] != test_src].reset_index(drop=True)
        test_df = d[d["source"] == test_src].reset_index(drop=True)

        # Particion interna fija train/val (ID) -- 70/30, semilla derivada
        # deterministicamente del nombre de fuente excluida para reproducibilidad.
        fold_seed = RNG_SEED + abs(hash(test_src)) % 10000
        rng_split = np.random.default_rng(fold_seed)
        n_train_total = len(train_df)
        idx = rng_split.permutation(n_train_total)
        n_val = max(3, int(round(0.30 * n_train_total)))
        val_idx = idx[:n_val]
        int_train_idx = idx[n_val:]

        Xtr_full = train_df[feature_cols].values
        ytr_full = train_df["logCp"].values

        X_int_train = Xtr_full[int_train_idx]
        y_int_train = ytr_full[int_train_idx]
        X_id_val = Xtr_full[val_idx]
        y_id_val = ytr_full[val_idx]

        X_ood = test_df[feature_cols].values
        y_ood = test_df["logCp"].values

        # Escalado ajustado SOLO con el train interno de este fold.
        scaler = StandardScaler().fit(X_int_train)
        X_int_train_s = scaler.transform(X_int_train)
        X_id_val_s = scaler.transform(X_id_val)
        X_ood_s = scaler.transform(X_ood)

        y_mean_tr = y_int_train.mean()
        y_std_tr = y_int_train.std() if y_int_train.std() > 1e-8 else 1.0

        preds_id = np.zeros((N_MEMBERS, len(X_id_val)))
        preds_ood = np.zeros((N_MEMBERS, len(X_ood)))

        n_int_train = len(X_int_train)
        for m in range(N_MEMBERS):
            member_seed = int(master_rng.integers(0, 1_000_000))
            boot_rng = np.random.default_rng(member_seed)
            boot_idx = boot_rng.integers(0, n_int_train, size=n_int_train)
            Xb = X_int_train_s[boot_idx]
            yb = (y_int_train[boot_idx] - y_mean_tr) / y_std_tr

            mlp = MLPRegressor(
                hidden_layer_sizes=HIDDEN,
                activation="tanh",
                solver="lbfgs",
                alpha=1e-2,
                max_iter=4000,
                random_state=member_seed,
            )
            mlp.fit(Xb, yb)

            preds_id[m] = mlp.predict(X_id_val_s) * y_std_tr + y_mean_tr
            preds_ood[m] = mlp.predict(X_ood_s) * y_std_tr + y_mean_tr

        mean_id = preds_id.mean(axis=0)
        std_id = preds_id.std(axis=0, ddof=1)
        mean_ood = preds_ood.mean(axis=0)
        std_ood = preds_ood.std(axis=0, ddof=1)

        fold_r2 = r2_score(y_ood, mean_ood)
        print(f"-- Fold test={test_src}: n_train_int={n_int_train}, n_id_val={len(X_id_val)}, "
              f"n_ood={len(X_ood)}")
        print(f"   held-out R2(log), ensemble mean pred: {fold_r2:.4f}")
        print(f"   dispersion ID  (val interna, misma fuente que train): "
              f"mean={std_id.mean():.4f}, median={np.median(std_id):.4f}")
        print(f"   dispersion OOD (fuente excluida, nunca vista): "
              f"mean={std_ood.mean():.4f}, median={np.median(std_ood):.4f}")
        ratio = std_ood.mean() / std_id.mean() if std_id.mean() > 1e-12 else float("nan")
        print(f"   ratio OOD/ID (medias): {ratio:.3f}")

        fold_results[test_src] = {
            "n_train_internal": int(n_int_train),
            "n_id_val": int(len(X_id_val)),
            "n_ood": int(len(X_ood)),
            "r2_log_ood": float(fold_r2),
            "id_dispersion_mean": float(std_id.mean()),
            "id_dispersion_median": float(np.median(std_id)),
            "ood_dispersion_mean": float(std_ood.mean()),
            "ood_dispersion_median": float(np.median(std_ood)),
            "ratio_ood_over_id_mean": float(ratio),
        }

        pooled_true.extend(y_ood.tolist())
        pooled_pred_mean.extend(mean_ood.tolist())
        id_std_all.extend(std_id.tolist())
        ood_std_all.extend(std_ood.tolist())

    pooled_r2 = r2_score(pooled_true, pooled_pred_mean)
    id_std_all = np.array(id_std_all)
    ood_std_all = np.array(ood_std_all)

    # Mann-Whitney U, una cola: H1 = dispersion OOD > dispersion ID
    u_stat, p_one_sided = stats.mannwhitneyu(ood_std_all, id_std_all, alternative="greater")

    print()
    print(f"POOLED (4 folds) held-out R2(log), ensemble mean, {label}: {pooled_r2:.4f}")
    print(f"Dispersion ID  pooled: mean={id_std_all.mean():.4f}, median={np.median(id_std_all):.4f}, "
          f"n={len(id_std_all)}")
    print(f"Dispersion OOD pooled: mean={ood_std_all.mean():.4f}, median={np.median(ood_std_all):.4f}, "
          f"n={len(ood_std_all)}")
    ratio_pooled = ood_std_all.mean() / id_std_all.mean() if id_std_all.mean() > 1e-12 else float("nan")
    print(f"Ratio pooled OOD/ID (medias): {ratio_pooled:.3f}")
    print(f"Mann-Whitney U (OOD > ID, una cola): U={u_stat:.1f}, p={p_one_sided:.4g}")
    print()

    return {
        "feature_cols": feature_cols,
        "per_facility": fold_results,
        "pooled_r2_log_space": float(pooled_r2),
        "pooled_r2_per_facility": {k: v["r2_log_ood"] for k, v in fold_results.items()},
        "id_dispersion_pooled_mean": float(id_std_all.mean()),
        "id_dispersion_pooled_median": float(np.median(id_std_all)),
        "ood_dispersion_pooled_mean": float(ood_std_all.mean()),
        "ood_dispersion_pooled_median": float(np.median(ood_std_all)),
        "ratio_ood_over_id_pooled_mean": float(ratio_pooled),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p_one_sided_ood_greater": float(p_one_sided),
        "n_id_points_pooled": int(len(id_std_all)),
        "n_ood_points_pooled": int(len(ood_std_all)),
    }


results = {
    "method": "ensemble_of_MLPs_dispersion_as_uncertainty",
    "library_used": "sklearn.neural_network.MLPRegressor (torch not installed in this "
                     "environment; declared explicitly, sklearn MLP used as reasonable "
                     "substitute already available in project .venv)",
    "n_members": N_MEMBERS,
    "hidden_layer_sizes": list(HIDDEN),
    "rng_seed": RNG_SEED,
    "protocol": (
        "LOFO real: 4 folds, train on 3 facilities / test on 1 unseen facility, "
        "never used for global fit or per-facility calibration. Internal 70/30 "
        "split of the 3 training facilities gives an in-distribution (ID) held-out "
        "validation set. Each of N_MEMBERS MLPs is trained on a bootstrap resample "
        "of the internal 70% train split with a distinct random seed (bagging + "
        "distinct init = ensemble diversity). Ensemble dispersion (std across "
        "members) compared between ID validation points and OOD (excluded "
        "facility) points."
    ),
    "feature_set_A_re_only": run_feature_set(["logRe"], "A: Re-only"),
    "feature_set_B_full": run_feature_set(
        ["logRe", "Pi_gap", "Pi_confinement", "Pi_aspect_axial"], "B: Full (Re+3 Pi groups)"
    ),
}

out_path = _ROOT + "/results/ensemble_uncertainty_lofo_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Escrito: {out_path}")
