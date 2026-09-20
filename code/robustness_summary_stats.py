"""Produce y guarda las cifras titulares de la seccion Robustness checks
del manuscrito: regresion Daily-Nece vs Re, los cuatro
eta^2 de identidad de instalacion, y las cuatro pendientes de Reynolds
por instalacion por separado. Reejecutado sobre el CSV con la etiqueta
Guo2024 ya unificada.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
df = df[df["Cp"] > 0]

# --- (a) regresion log(Cp/C_M_DailyNece) vs log(Re_Omega) ---
G = df["Pi_gap"]
Re = df["Re_Omega"]
C_M_DN = 0.051 * G ** (1 / 10) * Re ** (-0.2)
ratio = df["Cp"] / C_M_DN
X = np.column_stack([np.ones(len(df)), np.log(Re)])
y = np.log(ratio)
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
r2_ratio = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

daily_nece_ratio_regression = {
    "formula": "log(Cp / C_M_DailyNece_regimeIV) ~ log(Re_Omega)",
    "intercept": float(coef[0]),
    "slope": float(coef[1]),
    "r2": float(r2_ratio),
    "n": int(len(df)),
}

# --- (b) eta^2 de identidad de instalacion (ANOVA de un factor) ---
def eta_squared(values, groups):
    values = np.log(values.values)
    groups = groups.values
    grand_mean = values.mean()
    ss_total = np.sum((values - grand_mean) ** 2)
    ss_between = 0.0
    for g in np.unique(groups):
        vg = values[groups == g]
        ss_between += len(vg) * (vg.mean() - grand_mean) ** 2
    return float(ss_between / ss_total)

eta2 = {
    "log_Re_Omega": eta_squared(df["Re_Omega"], df["source"]),
    "log_Pi_gap": eta_squared(df["Pi_gap"], df["source"]),
    "log_Pi_confinement": eta_squared(df["Pi_confinement"], df["source"]),
    "log_Pi_aspect_axial": eta_squared(df["Pi_aspect_axial"], df["source"]),
}

# --- (c) pendiente de Reynolds por instalacion, ajuste separado ---
slopes_per_source = {}
for src, g in df.groupby("source"):
    if len(g) < 3:
        continue
    Xs = np.column_stack([np.ones(len(g)), np.log(g["Re_Omega"])])
    ys = np.log(g["Cp"])
    c, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    slopes_per_source[src] = float(c[1])

out = {
    "daily_nece_ratio_regression": daily_nece_ratio_regression,
    "eta_squared_facility_identity": eta2,
    "reynolds_slope_per_facility": slopes_per_source,
    "reynolds_slope_spread_ratio": float(
        max(abs(v) for v in slopes_per_source.values())
        / min(abs(v) for v in slopes_per_source.values())
    ),
}

with open(_ROOT + "/results/robustness_summary_stats.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
