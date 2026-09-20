"""Chequeo de viabilidad (2026-08-17): ajuste global de ley de potencia
Cp ~ Re_Omega^a * g_level^b * Pi_gap^c * Pi_confinement^d sobre el
dataset corregido de paper8_hipergravedad (114 puntos, 4 fuentes:
Xia2024, Vrancik1968, Liu2024, Zheng2024). Sin corrección de
domain-confounding (eso ya lo cubre paper8) -- este es el punto de
partida del modelo predictivo directo para "Windage Power".
"""
import pandas as pd
import numpy as np

df = pd.read_csv("../data/cross_rotor_dataset_v3.csv")
d = df.dropna(subset=["Cp", "Re_Omega", "g_level", "Pi_gap", "Pi_confinement"]).copy()
d = d[d["Cp"] > 0]

X = np.column_stack([
    np.ones(len(d)),
    np.log(d["Re_Omega"]),
    np.log(d["g_level"].clip(lower=1e-6)),
    np.log(d["Pi_gap"]),
    np.log(d["Pi_confinement"]),
])
y = np.log(d["Cp"])
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

print(f"n = {len(d)}")
print(f"coef (const, Re, g, Pi_gap, Pi_conf) = {coef}")
print(f"R2 global (sin corregir confounding) = {r2:.4f}")
print(d.groupby("source")["error_pct"].agg(["count", "mean", "max"]))
