"""Angle: active-learning / experimental-design recommendation for a
hypothetical 5th facility (2026-08-18).

Pregunta que responde: dado que NINGUN modelo predictivo generaliza
entre instalaciones (16 metodos probados, LOFO pooled R2 en
[-109, -0.34], y el unico caso con R2>0 pooled -MAML- esconde fallos
catastroficos por instalacion), ?podemos al menos dar una
recomendacion accionable y honesta a un futuro experimentalista sobre
DONDE construir una 5a instalacion / que rango medir para reducir la
incertidumbre epistemica del "mapa" Cp(Re, Pi_gap, Pi_confinement,
Pi_aspect_axial) tal y como lo conocemos hoy? Esto NO es una promesa de
que esa medida arreglaria la generalizacion (eso ya se probo que falla
por razones estructurales, no de cobertura), sino una afirmacion mas
modesta y verificable: "esta region del espacio de entrada es la peor
cubierta hoy segun un GP ajustado a los 114 puntos reales, y anadir
puntos ahi es lo que mas reduce la varianza predictiva integrada".

Metodo:
  1. Mismas 4 features que en el resto de la linea (log Re_Omega,
     log Pi_gap, log Pi_confinement, log Pi_aspect_axial), target
     log(Cp), estandarizadas con media/std de TODO el dataset (aqui no
     hay LOFO: se modela "el conocimiento actual del campo" con las 4
     instalaciones juntas, que es exactamente la pregunta -- que
     mediria alguien que parte de todo lo publicado hasta hoy).
  2. Se ajusta un GP (Constant*RBF anisotropica ARD + WhiteKernel,
     mismos bounds y misma receta de reinicios que gp_lofo_test.py)
     sobre los 114 puntos completos.
  3. Se calcula el CAMPO DE VARIANZA POSTERIOR LATENTE (sin ruido,
     i.e. incertidumbre epistemica de la funcion Cp(x), no de una
     observacion futura con ruido) en una malla de referencia que
     cubre el casco convexo de los datos mas un margen de extrapolacion
     moderado (30% del rango por dimension).
  4. Para un conjunto de disenos candidatos de "5a instalacion" (una
     geometria fija Pi_gap/Pi_confinement/Pi_aspect_axial mas un
     barrido de Re representativo), se simula anadir esos puntos al GP
     (SIN usar ninguna etiqueta y -- la varianza posterior de un GP no
     depende de las observaciones, solo de las posiciones de entrada y
     el kernel ya ajustado, propiedad estandar de regresion gaussiana,
     Rasmussen & Williams 2006 cap. 2) y se mide la reduccion de la
     varianza integrada (promedio sobre la malla de referencia) que
     resultaria: criterio de diseno experimental tipo IMSPE
     (integrated mean squared prediction error) / active learning por
     reduccion de varianza (MacKay 1992; Cohn et al. 1996).
  5. Se rankean los candidatos. Controles de honestidad:
       a) candidatos que repiten la geometria de una instalacion ya
          existente deben dar una reduccion ~0 (validacion de que el
          metodo no es perverso).
       b) se compara explicitamente "explorar una geometria nueva" vs
          "extender el rango de Re en una geometria ya conocida", para
          ver si la recomendacion es trivial (solo "sube el Re") o
          realmente senala un hueco geometrico.
       c) se inspecciona si las longitudes de escala ARD ajustadas
          saturan el limite superior del bound en alguna dimension
          (senal de que el kernel esencialmente IGNORA esa dimension,
          lo que haria vacua cualquier recomendacion centrada en ella).
       d) se recalcula el ranking con un kernel isotropico (una sola
          longitud de escala) y con un criterio totalmente
          independiente del GP -- distancia de Mahalanobis al vecino
          mas cercano en el dataset real -- y se reporta la correlacion
          de Spearman entre rankings. Si no correlacionan, el resultado
          depende fragilmente de la eleccion de kernel y se declara asi.
       e) se recuerda explicitamente el hallazgo ya calculado en
          gp_lofo_test.py: la std predictiva del GP NO correlaciona de
          forma fiable con el error real cross-facility (corr pooled
          =-0.018, con signos opuestos por instalacion). Por tanto esta
          recomendacion es una afirmacion sobre COBERTURA/INFORMACION
          geometrica del dataset actual, NUNCA una garantia de que
          medir ahi arreglaria la generalizacion.

Limitaciones declaradas:
  - La malla de referencia y el barrido de Re por candidato son
    elecciones de diseno (declaradas explicitamente abajo), no
    verdades fisicas; se reportan resultados de sensibilidad para que
    el lector juzgue cuanto dependen de esas elecciones.
  - Tratar Pi_confinement/Pi_gap/Pi_aspect_axial como continuos ignora
    que geometry_type es categorico (un facility real ocupa un punto,
    no una region); se interpreta el "hueco" encontrado como una
    region de geometria factible, no como una maquina concreta.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

RNG_SEED = 12345
N_RESTARTS = 10
KERNEL_BOUNDS = {
    "constant": (1e-3, 1e3),
    "length_scale": (1e-2, 1e2),
    "noise": (1e-10, 1e1),
}
JITTER = 1e-10

# ---------------------------------------------------------------- data
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
feature_names = ["log_Re_Omega", "log_Pi_gap", "log_Pi_confinement", "log_Pi_aspect_axial"]

X_raw = np.column_stack([lRe, lgap, lconf, lasp])  # (n,4)
mu_full = X_raw.mean(axis=0)
sd_full = X_raw.std(axis=0, ddof=0)
X_train = (X_raw - mu_full) / sd_full


def make_kernel(n_features, ard=True):
    ls0 = np.ones(n_features) if ard else 1.0
    return (
        ConstantKernel(1.0, constant_value_bounds=KERNEL_BOUNDS["constant"])
        * RBF(length_scale=ls0, length_scale_bounds=KERNEL_BOUNDS["length_scale"])
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=KERNEL_BOUNDS["noise"])
    )


# ---------------------------------------------------------- fit full GP
gp = GaussianProcessRegressor(
    kernel=make_kernel(4, ard=True), normalize_y=True,
    n_restarts_optimizer=N_RESTARTS, random_state=RNG_SEED, alpha=0.0,
)
gp.fit(X_train, y)
kernel_fitted = gp.kernel_
signal_kernel = kernel_fitted.k1   # ConstantKernel * RBF
noise_kernel = kernel_fitted.k2    # WhiteKernel
fitted_length_scales = signal_kernel.k2.length_scale
fitted_amplitude = float(np.sqrt(signal_kernel.k1.constant_value))
fitted_noise = float(noise_kernel.noise_level)

length_scale_bound_hi = KERNEL_BOUNDS["length_scale"][1]
saturated_dims = [feature_names[i] for i, ls in enumerate(fitted_length_scales)
                  if ls >= 0.95 * length_scale_bound_hi]

print("=== GP ajustado a los 114 puntos completos ===")
print("kernel:", kernel_fitted)
print("length_scales (ARD, orden features):", dict(zip(feature_names, fitted_length_scales.tolist())))
print("dimensiones SATURADAS (kernel esencialmente ciego a esa dimension):", saturated_dims)
print()

# also fit isotropic-kernel version for robustness check
gp_iso = GaussianProcessRegressor(
    kernel=make_kernel(4, ard=False), normalize_y=True,
    n_restarts_optimizer=N_RESTARTS, random_state=RNG_SEED, alpha=0.0,
)
gp_iso.fit(X_train, y)
kernel_iso = gp_iso.kernel_
print("kernel isotropico ajustado:", kernel_iso)
print()


# ----------------------------------------------- GP algebra utilities
def build_predictor(kernel, X_ref):
    """Devuelve funcion latent_var(Xg) usando SOLO el kernel (ya ajustado)
    y las posiciones X_ref (nunca las y) -- la varianza posterior de un GP
    no depende de las observaciones."""
    sk = kernel.k1  # signal part (Constant*RBF)
    K_full = kernel(X_ref)          # incluye ruido en la diagonal (Y=None)
    K_full = K_full + JITTER * np.eye(len(X_ref))
    c_and_lower = cho_factor(K_full, lower=True)

    def latent_var(Xg):
        prior_var = sk.diag(Xg)                      # sin ruido
        k_cross = sk(X_ref, Xg)                       # (n, M) sin ruido
        alpha = cho_solve(c_and_lower, k_cross)        # K_full^-1 k_cross
        reduction = np.einsum("ij,ij->j", k_cross, alpha)
        var = prior_var - reduction
        return np.maximum(var, 0.0)

    return latent_var


latent_var_full = build_predictor(kernel_fitted, X_train)
latent_var_iso = build_predictor(kernel_iso, X_train)

# ---------------------------------------------------- reference grid
# PAD=0.0 (analisis PRINCIPAL): malla de referencia y malla de candidatos
# estrictamente DENTRO del casco convexo observado -- esto es un hueco de
# INTERPOLACION real, no una recompensa por extrapolar. Se repite todo el
# analisis con PAD=0.30 mas abajo como control de sensibilidad.
PAD = 0.0
lo = X_train.min(axis=0)
hi = X_train.max(axis=0)
rng = hi - lo
grid_lo = lo - PAD * rng
grid_hi = hi + PAD * rng

rng_mc = np.random.default_rng(RNG_SEED)
M_REF = 4000
Xref = rng_mc.uniform(grid_lo, grid_hi, size=(M_REF, 4))

baseline_var_full = latent_var_full(Xref)
baseline_mean_var_full = float(baseline_var_full.mean())
baseline_var_iso = latent_var_iso(Xref)
baseline_mean_var_iso = float(baseline_var_iso.mean())

print(f"Varianza latente media (malla de referencia, M={M_REF}), kernel ARD: {baseline_mean_var_full:.5f}")
print(f"Varianza latente media (malla de referencia), kernel isotropico:     {baseline_mean_var_iso:.5f}")
print()


# ------------------------------------------------------ candidate designs
def facility_center(fac):
    mask = sources == fac
    return X_train[mask].mean(axis=0), X_train[mask, 0].min(), X_train[mask, 0].max()


def make_candidate(geom_std, re_lo_std, re_hi_std, n_sweep=6):
    """geom_std = (Pi_gap, Pi_confinement, Pi_aspect_axial) estandarizados.
    Barrido de Re log-espaciado en unidades estandarizadas entre re_lo_std y
    re_hi_std."""
    re_vals = np.linspace(re_lo_std, re_hi_std, n_sweep)
    pts = np.column_stack([re_vals,
                            np.full(n_sweep, geom_std[0]),
                            np.full(n_sweep, geom_std[1]),
                            np.full(n_sweep, geom_std[2])])
    return pts


def variance_reduction(kernel, X_ref_base, Xref_grid, X_new, baseline_mean):
    """Anade X_new (SIN etiquetas) al conjunto de entrenamiento y recalcula
    la varianza latente media en Xref_grid. No usa y en ningun momento."""
    X_aug = np.vstack([X_ref_base, X_new])
    pred = build_predictor(kernel, X_aug)
    new_mean = float(pred(Xref_grid).mean())
    return baseline_mean - new_mean, new_mean


# overall Re range (standardized) across the whole dataset, used as the
# "comprehensive sweep" any new facility is assumed able to attempt
re_overall_lo, re_overall_hi = X_train[:, 0].min(), X_train[:, 0].max()

candidates = {}

# (a) controls: repeat each existing facility's own geometry + own Re range
for fac in facilities:
    center, re_lo, re_hi = facility_center(fac)
    pts = make_candidate(center[1:], re_lo, re_hi, n_sweep=6)
    candidates[f"CONTROL_repeat_{fac}"] = pts

# (b) grid over geometry space (Pi_gap, Pi_confinement, Pi_aspect_axial),
# standardized, spanning [lo-0.3rng, hi+0.3rng] in each of the 3 geometry dims,
# combined with the OVERALL Re sweep (comprehensive new facility assumption)
n_grid_per_dim = 6
gap_grid = np.linspace(grid_lo[1], grid_hi[1], n_grid_per_dim)
conf_grid = np.linspace(grid_lo[2], grid_hi[2], n_grid_per_dim)
asp_grid = np.linspace(grid_lo[3], grid_hi[3], n_grid_per_dim)

geom_candidates = []
for gp_ in gap_grid:
    for cf in conf_grid:
        for ap in asp_grid:
            geom_candidates.append((gp_, cf, ap))

for i, geom in enumerate(geom_candidates):
    pts = make_candidate(geom, re_overall_lo, re_overall_hi, n_sweep=6)
    candidates[f"GRID_{i:03d}"] = pts

# (c) "trivial" comparison candidates: same geometry as each facility but
# EXTENDING Re well beyond that facility's own observed range (up to the
# overall dataset max/min) -- tests whether "just push Re further" rivals
# exploring new geometry
for fac in facilities:
    center, re_lo, re_hi = facility_center(fac)
    pts = make_candidate(center[1:], re_overall_lo, re_overall_hi, n_sweep=6)
    candidates[f"EXTEND_RE_{fac}"] = pts

print(f"Total candidatos evaluados: {len(candidates)} "
      f"({len(facilities)} control + {len(geom_candidates)} grid + {len(facilities)} extend-Re)")
print()

results_per_candidate = {}
for name, pts in candidates.items():
    dv_ard, newmean_ard = variance_reduction(kernel_fitted, X_train, Xref, pts, baseline_mean_var_full)
    dv_iso, newmean_iso = variance_reduction(kernel_iso, X_train, Xref, pts, baseline_mean_var_iso)
    # un-standardize geometry point (use first non-Re row's geom cols) for reporting
    geom_std = pts[0, 1:]
    geom_phys = np.exp(geom_std * sd_full[1:] + mu_full[1:])  # back to Pi_gap, Pi_conf, Pi_aspect
    re_lo_phys = float(np.exp(pts[:, 0].min() * sd_full[0] + mu_full[0]))
    re_hi_phys = float(np.exp(pts[:, 0].max() * sd_full[0] + mu_full[0]))
    results_per_candidate[name] = {
        "delta_var_ard": float(dv_ard),
        "delta_var_iso": float(dv_iso),
        "pct_reduction_ard": float(dv_ard / baseline_mean_var_full * 100),
        "Pi_gap": float(geom_phys[0]),
        "Pi_confinement": float(geom_phys[1]),
        "Pi_aspect_axial": float(geom_phys[2]),
        "Re_Omega_sweep_lo": re_lo_phys,
        "Re_Omega_sweep_hi": re_hi_phys,
    }

# rank GRID candidates only (the genuinely "new geometry" search)
grid_names = [k for k in candidates if k.startswith("GRID_")]
grid_ranked = sorted(grid_names, key=lambda k: -results_per_candidate[k]["delta_var_ard"])

control_names = [k for k in candidates if k.startswith("CONTROL_")]
extend_names = [k for k in candidates if k.startswith("EXTEND_RE_")]

print("=== Controles (repetir geometria+Re de una instalacion ya existente) ===")
for k in control_names:
    r = results_per_candidate[k]
    print(f"{k:28s} delta_var={r['delta_var_ard']:+.6f} ({r['pct_reduction_ard']:+.3f}% de la varianza base)")
print()

print("=== Extender SOLO el rango de Re en geometria ya conocida ===")
for k in extend_names:
    r = results_per_candidate[k]
    print(f"{k:28s} delta_var={r['delta_var_ard']:+.6f} ({r['pct_reduction_ard']:+.3f}%)  "
          f"Re=[{r['Re_Omega_sweep_lo']:.3g},{r['Re_Omega_sweep_hi']:.3g}]")
print()

print("=== TOP 10 candidatos de NUEVA geometria (malla b) por reduccion de varianza ===")
for k in grid_ranked[:10]:
    r = results_per_candidate[k]
    print(f"{k:10s} delta_var={r['delta_var_ard']:+.6f} ({r['pct_reduction_ard']:+.3f}%)  "
          f"Pi_gap={r['Pi_gap']:.4f} Pi_conf={r['Pi_confinement']:.4f} "
          f"Pi_asp={r['Pi_aspect_axial']:.4f}  Re=[{r['Re_Omega_sweep_lo']:.3g},{r['Re_Omega_sweep_hi']:.3g}]")
print()
print("=== BOTTOM 5 candidatos de nueva geometria (menor reduccion) ===")
for k in grid_ranked[-5:]:
    r = results_per_candidate[k]
    print(f"{k:10s} delta_var={r['delta_var_ard']:+.6f} ({r['pct_reduction_ard']:+.3f}%)  "
          f"Pi_gap={r['Pi_gap']:.4f} Pi_conf={r['Pi_confinement']:.4f} Pi_asp={r['Pi_aspect_axial']:.4f}")
print()

# ------------------------------------------- robustness: ARD vs isotropic
grid_ranked_iso = sorted(grid_names, key=lambda k: -results_per_candidate[k]["delta_var_iso"])
rank_ard = {k: i for i, k in enumerate(grid_ranked)}
rank_iso = {k: i for i, k in enumerate(grid_ranked_iso)}
common = grid_names
rho, pval = spearmanr([rank_ard[k] for k in common], [rank_iso[k] for k in common])
print(f"Spearman rho entre ranking ARD vs isotropico (mismos {len(common)} candidatos): "
      f"rho={rho:.4f}, p={pval:.4g}")
print()


# ------------------------------------------- independent check: NN-distance
def mahalanobis_gap_score(Xg, X_ref, k_neighbors=3):
    """Distancia media a los k vecinos mas cercanos reales (dataset), en el
    espacio estandarizado -- criterio de cobertura totalmente independiente
    del GP (no usa kernel ni longitudes de escala ajustadas)."""
    d2 = ((Xg[:, None, :] - X_ref[None, :, :]) ** 2).sum(axis=2)
    d2_sorted = np.sort(d2, axis=1)
    return np.sqrt(d2_sorted[:, :k_neighbors]).mean(axis=1)


geom_pts_for_nn = np.array([[re_overall_lo, *geom_candidates[i]] for i in range(len(geom_candidates))])
# use midpoint Re (geometric mean of overall sweep) for a single representative NN distance per geometry
re_mid = 0.5 * (re_overall_lo + re_overall_hi)
geom_pts_for_nn[:, 0] = re_mid
nn_scores = mahalanobis_gap_score(geom_pts_for_nn, X_train, k_neighbors=3)
nn_rank_order = np.argsort(-nn_scores)  # descending: biggest gap first
grid_names_arr = np.array(grid_names)
grid_ranked_nn = list(grid_names_arr[nn_rank_order])
rank_nn = {k: i for i, k in enumerate(grid_ranked_nn)}
rho_nn, pval_nn = spearmanr([rank_ard[k] for k in common], [rank_nn[k] for k in common])
print(f"Spearman rho entre ranking GP-ARD vs distancia-a-vecino-mas-cercano (independiente del kernel): "
      f"rho={rho_nn:.4f}, p={pval_nn:.4g}")
print()

# top candidate detail + nearest real facility distance
top_name = grid_ranked[0]
top = results_per_candidate[top_name]
top_std = np.array([re_mid, *[k for k in geom_candidates[int(top_name.split('_')[1])]]])
dists_to_each_facility = {}
for fac in facilities:
    mask = sources == fac
    fac_pts = X_train[mask]
    d2 = ((fac_pts - top_std[None, :]) ** 2).sum(axis=1)
    dists_to_each_facility[fac] = float(np.sqrt(d2.min()))

print(f"Candidato #1 ({top_name}): distancia (espacio estandarizado 4D) al punto real mas cercano de cada instalacion:")
for fac, dd in dists_to_each_facility.items():
    print(f"   {fac:14s} {dd:.3f}")
print(f"   (para referencia, distancia media punto-a-vecino-mas-cercano DENTRO del dataset real: "
      f"{np.sqrt(((X_train[:,None,:]-X_train[None,:,:])**2).sum(2)).__class__}")
print()

# ---------------------------------- pointwise diagnostic: local uncertainty
# AT the existing facility centers (should be low = well covered) vs AT the
# winning candidate (should be much higher = genuine gap), independent of
# the integrated-over-grid criterion above.
pointwise_var_at_facility_centers = {}
for fac in facilities:
    center, _, _ = facility_center(fac)
    v = float(latent_var_full(center[None, :])[0])
    pointwise_var_at_facility_centers[fac] = v

top_idx = int(grid_ranked[0].split("_")[1])
top_geom = geom_candidates[top_idx]
top_pt_std = np.array([re_mid, *top_geom])
pointwise_var_at_top_candidate = float(latent_var_full(top_pt_std[None, :])[0])
prior_var_no_data = fitted_amplitude ** 2

print("=== Diagnostico puntual (independiente del criterio integrado) ===")
print("Varianza latente PUNTUAL en el centro de cada instalacion (cobertura actual):")
for fac, v in pointwise_var_at_facility_centers.items():
    print(f"   {fac:14s} var={v:.5f}")
print(f"Varianza latente PUNTUAL en el candidato ganador (Re=mediana): {pointwise_var_at_top_candidate:.5f}")
print(f"Varianza a priori (sin ningun dato): {prior_var_no_data:.5f}")
print()

# --------------------------------- sensitivity: repeat with PAD=0.30
PAD_SENS = 0.30
grid_lo_s = lo - PAD_SENS * rng
grid_hi_s = hi + PAD_SENS * rng
Xref_s = rng_mc.uniform(grid_lo_s, grid_hi_s, size=(M_REF, 4))
baseline_mean_s = float(latent_var_full(Xref_s).mean())
gap_grid_s = np.linspace(grid_lo_s[1], grid_hi_s[1], n_grid_per_dim)
conf_grid_s = np.linspace(grid_lo_s[2], grid_hi_s[2], n_grid_per_dim)
asp_grid_s = np.linspace(grid_lo_s[3], grid_hi_s[3], n_grid_per_dim)
geom_candidates_s = [(g, c, a) for g in gap_grid_s for c in conf_grid_s for a in asp_grid_s]
dv_s = []
for geom in geom_candidates_s:
    pts = make_candidate(geom, re_overall_lo, re_overall_hi, n_sweep=6)
    X_aug = np.vstack([X_train, pts])
    pred = build_predictor(kernel_fitted, X_aug)
    new_mean = float(pred(Xref_s).mean())
    dv_s.append(baseline_mean_s - new_mean)
order_s = np.argsort(dv_s)[::-1]
print(f"=== Control de sensibilidad: malla de referencia CON extrapolacion (PAD={PAD_SENS}) ===")
print(f"baseline_mean_var(PAD={PAD_SENS})={baseline_mean_s:.5f}  (vs PAD=0: {baseline_mean_var_full:.5f})")
print("Top5 bajo PAD=0.30:")
sens_top5 = []
for i in order_s[:5]:
    geom = geom_candidates_s[i]
    phys = np.exp(np.array(geom) * sd_full[1:] + mu_full[1:])
    row = {"delta_var": float(dv_s[i]), "pct": float(dv_s[i] / baseline_mean_s * 100),
           "Pi_gap": float(phys[0]), "Pi_confinement": float(phys[1]), "Pi_aspect_axial": float(phys[2])}
    sens_top5.append(row)
    print(f"   dv={row['delta_var']:.5f} ({row['pct']:.2f}%)  Pi_gap={row['Pi_gap']:.4f} "
          f"Pi_conf={row['Pi_confinement']:.4f} Pi_asp={row['Pi_aspect_axial']:.4f}")
print()

out = {
    "meta": {
        "n_total": int(n), "features": feature_names,
        "kernel_bounds": KERNEL_BOUNDS, "n_restarts_optimizer": N_RESTARTS,
        "reference_grid_M": M_REF, "reference_grid_padding_frac": PAD,
        "n_geometry_grid_points": len(geom_candidates),
        "re_sweep_points_per_candidate": 6,
    },
    "fitted_kernel_full_ard": str(kernel_fitted),
    "fitted_length_scales_ard": dict(zip(feature_names, fitted_length_scales.tolist())),
    "fitted_amplitude": fitted_amplitude,
    "fitted_noise_level": fitted_noise,
    "saturated_dimensions_ge_95pct_of_upper_bound": saturated_dims,
    "fitted_kernel_isotropic": str(kernel_iso),
    "baseline_mean_latent_variance_ARD": baseline_mean_var_full,
    "baseline_mean_latent_variance_isotropic": baseline_mean_var_iso,
    "controls_repeat_existing_facility": {k: results_per_candidate[k] for k in control_names},
    "extend_re_only_existing_geometry": {k: results_per_candidate[k] for k in extend_names},
    "top10_new_geometry_candidates": [
        {"name": k, **results_per_candidate[k]} for k in grid_ranked[:10]
    ],
    "bottom5_new_geometry_candidates": [
        {"name": k, **results_per_candidate[k]} for k in grid_ranked[-5:]
    ],
    "spearman_rank_correlation_ARD_vs_isotropic_kernel": {"rho": float(rho), "p": float(pval)},
    "spearman_rank_correlation_ARD_vs_nearest_neighbor_distance": {"rho": float(rho_nn), "p": float(pval_nn)},
    "top_candidate_distance_to_nearest_real_point_per_facility_std_space": dists_to_each_facility,
    "pointwise_diagnostic": {
        "latent_var_at_facility_centers": pointwise_var_at_facility_centers,
        "latent_var_at_top_candidate": pointwise_var_at_top_candidate,
        "prior_var_no_data_amplitude_squared": prior_var_no_data,
    },
    "sensitivity_check_PAD_0.30_reference_grid": {
        "baseline_mean_var": baseline_mean_s,
        "top5_candidates": sens_top5,
        "note": "recomputado con la malla de referencia y de candidatos extendida "
                "30% mas alla del casco convexo observado (permite extrapolacion). "
                "Comparar con el analisis principal (PAD=0, estrictamente interior) "
                "para ver si la recomendacion es un hueco real de interpolacion o "
                "un artefacto de recompensar la extrapolacion.",
    },
    "context_from_prior_script_gp_lofo_test": {
        "pooled_corr_abs_resid_vs_pred_std_LOFO": -0.0179044343357672,
        "interpretation": "la std predictiva del GP en LOFO real NO correlaciona de forma fiable "
                           "con el error verdadero cross-facility (signo positivo en 2 instalaciones, "
                           "negativo en las otras 2). Por tanto esta recomendacion es una afirmacion "
                           "sobre cobertura/informacion del dataset actual (donde el GP AUN NO ha visto "
                           "datos), no una garantia de que medir ahi arreglaria la generalizacion.",
    },
}

out_path = _ROOT + "/results/angle_activelearning_gapmap_results.json"
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)
print(f"Guardado en {out_path}")
