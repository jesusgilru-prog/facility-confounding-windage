"""Verifica que Cp del corpus == P_w / (1/2 * rho * omega^3 * R^5),
la normalizacion estandar del coeficiente de par C_M usada por
Daily & Nece (1960), no rho*omega^3*R^5 (que da un factor exacto de 2
de diferencia). Ejecutado 2026-08-17 en respuesta a la ronda 2 de
revision (bloqueante #3): la version anterior del manuscrito citaba
esta verificacion sin script ni normalizacion correcta.
"""
import pandas as pd
import numpy as np
from pathlib import Path

d = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "cross_rotor_dataset_v3.csv")
Cp = d["Cp"]
Pw = d["P_w_W"]
denom = 0.5 * d["rho_kgm3"] * d["omega_rad_s"] ** 3 * d["R_m"] ** 5
ratio = Cp / (Pw / denom)

print(f"n = {len(d)}")
print(f"ratio Cp / (Pw / (0.5*rho*omega^3*R^5)): min={ratio.min():.10f} max={ratio.max():.10f}")
assert np.allclose(ratio, 1.0, atol=1e-9), "normalization mismatch"
print("Verificado: Cp == P_w / (1/2 * rho * omega^3 * R^5) en las 114 filas, ratio=1.000000000")
