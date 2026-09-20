"""Genera las 4 figuras del paper 'Windage Power' a partir de datos reales
(CSV + resultados de scaling_law_search.py). Sin datos inventados."""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_confinement", "Pi_aspect_axial", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)

res = json.load(open(_ROOT + "/results/scaling_law_search_results.json"))
coef = res["best_structure_result"]["coef"]

colors = {"Guo2024": "#1f77b4", "Vrancik1968": "#ff7f0e", "Liu2024": "#2ca02c", "Zheng2024": "#d62728"}
markers = {"Guo2024": "o", "Vrancik1968": "s", "Liu2024": "^", "Zheng2024": "D"}
# La etiqueta "Xia2024" (interna, heredada del dataset compartido con
# paper8_hipergravedad) fue renombrada a "Guo2024" directamente en el CSV
# de ESTE paper el 2026-08-18 (bloqueante #1, ronda 3) -- el autor real de
# esa fuente es Guo et al. 2024 (verificado por Crossref, DOI
# 10.3390/app14177613). Esta es la copia local de paper_windage_power,
# distinta de la copia de paper8 (que conserva "Xia2024" sin tocar).
display_label = {"Guo2024": "Guo2024", "Vrancik1968": "Vrancik1968",
                  "Liu2024": "Liu2024", "Zheng2024": "Zheng2024"}

# --- Fig 1: Cp vs Re_Omega por fuente ---
fig, ax = plt.subplots(figsize=(5.5, 4))
for src in sorted(d["source"].unique()):
    sub = d[d["source"] == src]
    ax.scatter(sub["Re_Omega"], sub["Cp"], label=display_label[src], color=colors[src], marker=markers[src], alpha=0.75, edgecolors="k", linewidths=0.3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\mathrm{Re}_\Omega$"); ax.set_ylabel(r"$C_p$")
ax.legend(fontsize=8, frameon=False)
ax.set_title("Power coefficient vs. rotational Reynolds number")
fig.tight_layout()
fig.savefig(_ROOT + "/figures/fig1_cp_vs_re.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 2: observado vs predicho del modelo ganador (M6, aspect) ---
lRe = np.log(d["Re_Omega"]); lgap = np.log(d["Pi_gap"]); lconf = np.log(d["Pi_confinement"]); lasp = np.log(d["Pi_aspect_axial"])
log_pred = (coef["const"] + coef["q_Re"] * lRe + coef["p_gap"] * lgap
            + coef["r_conf"] * lconf + coef["t_asp"] * lasp)
Cp_pred = np.exp(log_pred)

fig, ax = plt.subplots(figsize=(4.5, 4.5))
for src in sorted(d["source"].unique()):
    mask = d["source"] == src
    ax.scatter(d.loc[mask, "Cp"], Cp_pred[mask], label=display_label[src], color=colors[src], marker=markers[src], alpha=0.75, edgecolors="k", linewidths=0.3)
lims = [min(d["Cp"].min(), Cp_pred.min()) * 0.5, max(d["Cp"].max(), Cp_pred.max()) * 2]
ax.plot(lims, lims, "k--", linewidth=1, label="1:1")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel(r"Observed $C_p$"); ax.set_ylabel(r"Predicted $C_p$ (structure M6)")
ax.legend(fontsize=8, frameon=False)
ax.set_title(r"Observed vs. predicted, in-sample fit ($R^2=0.870$)")
fig.tight_layout()
fig.savefig(_ROOT + "/figures/fig2_obs_vs_pred.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 3: residuos (log) por fuente ---
resid = np.log(d["Cp"].values) - log_pred.values
fig, ax = plt.subplots(figsize=(5.5, 4))
srcs = sorted(d["source"].unique())
data_by_src = [resid[d["source"] == s] for s in srcs]
bp = ax.boxplot(data_by_src, tick_labels=[display_label[s] for s in srcs], patch_artist=True)
for patch, s in zip(bp["boxes"], srcs):
    patch.set_facecolor(colors[s]); patch.set_alpha(0.5)
ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
ax.set_ylabel(r"$\log C_p^{\mathrm{obs}} - \log C_p^{\mathrm{pred}}$ (in-sample)")
ax.set_title("In-sample residuals by facility")
fig.tight_layout()
fig.savefig(_ROOT + "/figures/fig3_residuals_by_source.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 4: fallo de Daily-Nece (ratio obs/pred vs Re) ---
G = d["Pi_gap"].values; Re = d["Re_Omega"].values
Cp_dn = 0.051 * (G ** (1 / 10)) * (Re ** (-0.2))
ratio = d["Cp"].values / Cp_dn
fig, ax = plt.subplots(figsize=(5.5, 4))
for src in sorted(d["source"].unique()):
    mask = d["source"] == src
    ax.scatter(d.loc[mask, "Re_Omega"], ratio[mask], label=display_label[src], color=colors[src], marker=markers[src], alpha=0.75, edgecolors="k", linewidths=0.3)
ax.axhline(1, color="k", linewidth=1, linestyle="--", label="Daily–Nece regime IV (ratio=1)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\mathrm{Re}_\Omega$"); ax.set_ylabel(r"$C_p^{\mathrm{obs}} / C_{M,\mathrm{IV}}^{\mathrm{Daily-Nece}}$")
ax.legend(fontsize=8, frameon=False)
ax.set_title("Underprediction by the unfiltered Daily–Nece regime-IV formula")
fig.tight_layout()
fig.savefig(_ROOT + "/figures/fig5_daily_nece_ratio.png", bbox_inches="tight")
plt.close(fig)

# --- Fig 5: LOFO-CV, observado vs predicho fuera de muestra ---
lofo = res["lofo_cv"]["per_facility"]
fig, ax = plt.subplots(figsize=(4.5, 4.5))
all_pred_oos = []
for src in sorted(d["source"].unique()):
    mask = (d["source"] == src).values
    c = lofo[src]["coef_trained_without_this_facility"]
    lp = (c["const"] + c["q_Re"] * lRe[mask].values + c["p_gap"] * lgap[mask].values
          + c["r_conf"] * lconf[mask].values + c["t_asp"] * lasp[mask].values)
    pred_oos = np.exp(lp)
    all_pred_oos.append(pred_oos)
    ax.scatter(d.loc[mask, "Cp"], pred_oos, label=display_label[src], color=colors[src], marker=markers[src], alpha=0.75, edgecolors="k", linewidths=0.3)
all_pred_oos = np.concatenate(all_pred_oos)
# limites dinamicos a partir de los datos reales (antes fijos en [1e-4,1e8],
# dejaban ~90% de la caja en blanco -- bloqueante de figuras, ronda 2)
lo = min(d["Cp"].min(), all_pred_oos.min()) * 0.5
hi = max(d["Cp"].max(), all_pred_oos.max()) * 2
lims2 = [lo, hi]
ax.plot(lims2, lims2, "k--", linewidth=1, label="1:1")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lims2); ax.set_ylim(lims2)
ax.set_xlabel(r"Observed $C_p$"); ax.set_ylabel(r"Predicted $C_p$ (LOFO, facility held out)")
ax.legend(fontsize=7, frameon=False)
ax.set_title(r"Leave-one-facility-out, held-out predictions ($R^2_{\mathrm{log}}=-0.885$ pooled)")
fig.tight_layout()
fig.savefig(_ROOT + "/figures/fig4_lofo_obs_vs_pred.png", bbox_inches="tight")
plt.close(fig)

print("Figuras generadas: fig1..fig5 en figures/")
