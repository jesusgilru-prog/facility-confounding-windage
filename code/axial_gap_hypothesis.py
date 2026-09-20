"""
Hipotesis: Pi_gap (=gap_radial_m/R_m) es un proxy RADIAL, pero la G
clasica de Daily-Nece es un ratio de holgura AXIAL (gap_axial_m /
H_chamber_m-h_rotor_m). Se investiga si sustituir Pi_gap por un
Pi_gap_axial construido con datos axiales mejora la comparacion contra
Daily-Nece regimen IV y si un modelo ajustado con ese proxy generaliza
en LOFO.

Ejecutado 2026-08-17 sobre data/cross_rotor_dataset_v3.csv
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import pandas as pd

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
n_total = len(df)

print("=" * 70)
print("(a) Poblacion de columnas de holgura por fuente (no-nulos)")
print("=" * 70)
cols = ["gap_radial_m", "gap_axial_m", "H_pole_m", "H_chamber_m", "h_rotor_m", "R_chamber_m", "R_m"]
pop = df.groupby("source")[cols].apply(lambda x: x.notna().sum())
pop["n_rows"] = df.groupby("source").size()
print(pop.to_string())

print()
print("Nulos (NaN count) por fuente para gap_axial_m y H_chamber_m:")
null_tab = df.groupby("source")[["gap_axial_m", "H_chamber_m", "H_pole_m"]].apply(lambda x: x.isna().sum())
print(null_tab.to_string())

# -------------------------------------------------------------------
# (b) Verificar formula de gap axial en Liu2024 (unica fuente con
# gap_axial_m poblado directamente) y extenderla donde H_chamber_m y
# h_rotor_m estan ambos disponibles (Guo2024, Zheng2024), asumiendo
# rotor centrado axialmente: gap_axial = (H_chamber - h_rotor) / 2.
# -------------------------------------------------------------------
print()
print("=" * 70)
print("(b) Verificacion formula gap_axial = (H_chamber_m - h_rotor_m)/2")
print("=" * 70)
liu = df[df.source == "Liu2024"]
derived_liu = (liu["H_chamber_m"] - liu["h_rotor_m"]) / 2
match = np.allclose(derived_liu.values, liu["gap_axial_m"].values)
print(f"Liu2024: gap_axial_m real vs (H_chamber-h_rotor)/2 derivado -> coincide exacto: {match}")
print(f"  valores: real={liu['gap_axial_m'].unique()}, derivado={derived_liu.unique()}")

# Vrancik1968 NO tiene H_chamber_m en absoluto (solo H_pole_m, que es
# la altura axial del polo saliente, no la camara) -> gap axial
# INDEFINIBLE para esta fuente con las columnas disponibles.
vr_h_chamber_missing = df[df.source == "Vrancik1968"]["H_chamber_m"].isna().all()
print(f"\nVrancik1968: H_chamber_m 100% nulo (sin dato de camara axial) -> {vr_h_chamber_missing}")
print("  Vrancik1968 solo registra H_pole_m (altura del polo/saliente), NO la")
print("  altura de camara; no existe forma no-inventada de derivar un gap axial")
print("  para esta fuente con las columnas del CSV.")

# Construir Pi_gap_axial donde sea posible SIN inventar datos:
df["gap_axial_derived_m"] = np.where(
    df["source"] == "Liu2024", df["gap_axial_m"],
    np.where(df["H_chamber_m"].notna() & df["h_rotor_m"].notna(),
             (df["H_chamber_m"] - df["h_rotor_m"]) / 2, np.nan))
df["Pi_gap_axial"] = df["gap_axial_derived_m"] / df["R_m"]

cov = df.groupby("source")["Pi_gap_axial"].apply(lambda x: x.notna().sum())
cov_tab = pd.DataFrame({"n_rows": df.groupby("source").size(), "n_with_Pi_gap_axial": cov})
print("\nCobertura final de Pi_gap_axial por fuente:")
print(cov_tab.to_string())
print(f"\nTotal filas con Pi_gap_axial definible: {df['Pi_gap_axial'].notna().sum()} / {n_total}"
      f" ({100*df['Pi_gap_axial'].notna().sum()/n_total:.1f}%)")
print(f"Fuentes cubiertas: {sorted(df.loc[df['Pi_gap_axial'].notna(),'source'].unique())}")
print(f"Fuente SIN cobertura: Vrancik1968 (41/114 = {100*41/114:.1f}% del corpus queda fuera)")

# Dentro de cada fuente cubierta, ¿cuanta varianza real tiene el proxy?
print("\nValores unicos de Pi_gap_axial por fuente cubierta (evalua si es")
print("realmente variable dentro de instalacion o es una constante por fuente):")
for src in ["Liu2024", "Guo2024", "Zheng2024"]:
    vals = df.loc[df.source == src, "Pi_gap_axial"].dropna().unique()
    print(f"  {src}: {len(vals)} valor(es) unico(s) -> {np.round(vals, 5)}")

# -------------------------------------------------------------------
# (b-cont) Repetir comparacion Daily-Nece regimen IV con Pi_gap_axial
# en vez de Pi_gap (radial), SOLO sobre las 73 filas donde el proxy
# axial existe (Liu+Xia+Zheng). Se reporta tambien el mismo calculo
# con Pi_gap radial restringido a las MISMAS 73 filas, para que la
# comparacion sea limpia (mismo subconjunto de puntos, solo cambia el
# proxy de G).
# -------------------------------------------------------------------
print()
print("=" * 70)
print("(b-cont) Comparacion Daily-Nece regimen IV: Pi_gap (radial) vs")
print("         Pi_gap_axial, sobre el mismo subconjunto de 73 filas")
print("         (Liu2024+Guo2024+Zheng2024; Vrancik1968 excluido por falta")
print("         de dato axial)")
print("=" * 70)

sub = df[df["Pi_gap_axial"].notna()].copy()
sub = sub.dropna(subset=["Cp", "Re_Omega", "Pi_gap"])
sub = sub[sub["Cp"] > 0]
n_sub = len(sub)
print(f"n subconjunto = {n_sub} (fuentes: {sorted(sub['source'].unique())})")

def daily_nece_eval(G, Re, Cp_actual, label):
    Cp_pred = 0.051 * (G ** (1 / 10)) * (Re ** (-0.2))
    rel_err = np.abs(Cp_actual - Cp_pred) / Cp_actual
    log_actual = np.log(Cp_actual)
    log_pred = np.log(Cp_pred)
    rss = np.sum((log_actual - log_pred) ** 2)
    r2 = 1 - rss / np.sum((log_actual - log_actual.mean()) ** 2)
    n_under = int(np.sum(Cp_actual > Cp_pred))
    print(f"\n-- {label} --")
    print(f"  n = {len(Cp_actual)}")
    print(f"  n_underpredicted/n = {n_under}/{len(Cp_actual)}")
    print(f"  median ratio obs/pred = {np.median(Cp_actual/Cp_pred):.3f}")
    print(f"  min/max ratio = {np.min(Cp_actual/Cp_pred):.3f} / {np.max(Cp_actual/Cp_pred):.3f}")
    print(f"  median rel err % = {np.median(rel_err)*100:.1f}")
    print(f"  R2 (log space) = {r2:.4f}")
    return r2

r2_radial_sub = daily_nece_eval(sub["Pi_gap"].values, sub["Re_Omega"].values, sub["Cp"].values,
                                 "Pi_gap RADIAL (mismo subconjunto de 73 filas)")
r2_axial_sub = daily_nece_eval(sub["Pi_gap_axial"].values, sub["Re_Omega"].values, sub["Cp"].values,
                                "Pi_gap_axial (proxy axial, hipotesis)")

# Y tambien el numero de referencia original (114 filas, radial, ya
# verificado por el autor) para contexto:
full = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap"])
full = full[full["Cp"] > 0]
r2_radial_full = daily_nece_eval(full["Pi_gap"].values, full["Re_Omega"].values, full["Cp"].values,
                                  "Pi_gap RADIAL (114 filas completas, referencia)")

print(f"\nResumen R2(log): radial-114={r2_radial_full:.4f}  radial-73={r2_radial_sub:.4f}  "
      f"axial-73={r2_axial_sub:.4f}")

# -------------------------------------------------------------------
# (c) Estandar de validacion LOFO obligatorio para cualquier modelo
# AJUSTADO (no solo aplicado) con el nuevo proxy. Se ajusta un
# analogo de M3 (log Cp = const + q*log(Re) + p*log(Pi_gap_axial))
# sobre el subconjunto de 3 fuentes con dato axial, y se compara con
# el mismo M3 pero usando Pi_gap radial sobre el MISMO subconjunto de
# 3 fuentes (para que la comparacion sea justa).
#
# DESVIACION DECLARADA DEL ESTANDAR: el estandar obligatorio pide LOFO
# de 4 pliegues (una fuente excluida totalmente por ronda) con R2 por
# fuente Y pooled de las 4. Con Pi_gap_axial eso es IMPOSIBLE de
# cumplir tal cual: Vrancik1968 no tiene dato axial en absoluto, así
# que no se le puede ni evaluar (falta el predictor) ni usar como
# entrenamiento completo del modelo axial. Por tanto aqui se reporta
# un LOFO de 3 pliegues (Liu/Xia/Zheng) explicitamente rebajado del
# estandar de 4, y se declara la limitacion en vez de imputar dato
# axial inventado para Vrancik.
# -------------------------------------------------------------------
print()
print("=" * 70)
print("(c) LOFO del modelo M3-analogo (Re + gap) ajustado con Pi_gap_axial")
print("    DESVIACION DECLARADA: solo 3 fuentes tienen dato axial ->")
print("    LOFO de 3 pliegues, NO el estandar de 4. Vrancik1968 excluido")
print("    por falta de dato, no imputado.")
print("=" * 70)

y_sub = np.log(sub["Cp"].values)
lRe_sub = np.log(sub["Re_Omega"].values)
lgap_radial_sub = np.log(sub["Pi_gap"].values)
lgap_axial_sub = np.log(sub["Pi_gap_axial"].values)
src_sub = sub["source"].values
facilities_sub = sorted(set(src_sub))
print(f"Fuentes en LOFO reducido: {facilities_sub}  (n={len(sub)})")

def fit_ols(X, yv):
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return coef

def lofo_two_predictor(l1, l2, label):
    all_resid = []
    per_facility = {}
    for f in facilities_sub:
        test_mask = src_sub == f
        train_mask = ~test_mask
        Xtr = np.column_stack([np.ones(train_mask.sum()), l1[train_mask], l2[train_mask]])
        ytr = y_sub[train_mask]
        coef = fit_ols(Xtr, ytr)
        Xte = np.column_stack([np.ones(test_mask.sum()), l1[test_mask], l2[test_mask]])
        yte = y_sub[test_mask]
        pred = Xte @ coef
        resid = yte - pred
        r2_f = (1 - np.sum(resid**2)/np.sum((yte-yte.mean())**2)) if test_mask.sum() > 1 else None
        per_facility[f] = {"n_test": int(test_mask.sum()), "coef": coef.tolist(),
                            "rmse_log": float(np.sqrt(np.mean(resid**2))), "r2_log": r2_f}
        all_resid.append(resid)
    all_resid = np.concatenate(all_resid)
    pooled_r2 = 1 - np.sum(all_resid**2) / np.sum((y_sub - y_sub.mean())**2)
    pooled_rmse = np.sqrt(np.mean(all_resid**2))
    print(f"\n-- LOFO(3-fold) {label} --")
    for f, r in per_facility.items():
        print(f"  {f:12s} n_test={r['n_test']:3d}  RMSE(log)={r['rmse_log']:.4f}  R2(log)={r['r2_log']}")
    print(f"  POOLED (3 folds)  RMSE(log)={pooled_rmse:.4f}  R2(log)={pooled_r2:.4f}")
    return per_facility, pooled_r2

# Modelo base (referencia): Re + Pi_gap RADIAL, mismo subconjunto de 3 fuentes
per_fac_radial, pooled_r2_radial_3fold = lofo_two_predictor(lRe_sub, lgap_radial_sub, "Re+Pi_gap RADIAL (subset 3 fuentes)")

# Modelo hipotesis: Re + Pi_gap_axial, mismo subconjunto
per_fac_axial, pooled_r2_axial_3fold = lofo_two_predictor(lRe_sub, lgap_axial_sub, "Re+Pi_gap_axial (subset 3 fuentes)")

# Tambien, para contexto directo pedido por el protocolo: el numero
# YA VERIFICADO de referencia (LOFO de 4 fuentes, modelo ganador M6,
# Pi_gap radial, 114 puntos) se cita tal cual del run anterior
# (results/scaling_law_search_results.json), NO se recalcula aqui
# para no duplicar trabajo ya hecho y validado.
import json
with open(_ROOT + "/results/scaling_law_search_results.json") as f:
    prev = json.load(f)
print(f"\n[Referencia ya verificada, NO recalculada aqui] LOFO 4-fuentes pooled "
      f"R2(log) del modelo ganador M6 (Pi_gap radial, 114 pts) = "
      f"{prev['lofo_cv']['pooled_r2_log']:.4f}")

print()
print("=" * 70)
print("(d) VEREDICTO HONESTO")
print("=" * 70)
print(f"""
1. Cobertura de dato axial: SOLO 73/114 filas (64%) tienen un gap axial
   no inventado disponible o derivable (Liu2024 directo; Guo2024 y
   Zheng2024 vía (H_chamber_m - h_rotor_m)/2, formula verificada
   exacta contra el valor real de Liu2024). Vrancik1968 (41/114, 36%
   del corpus, la fuente de MAYOR confianza geometrica 'high') NO
   tiene ningun dato de holgura axial en el CSV -> el proxy Pi_gap_axial
   es INDEFINIBLE para esa fuente sin inventar numeros.

2. Dentro de las 3 fuentes cubiertas, Pi_gap_axial tiene varianza
   intra-instalacion casi nula: es una CONSTANTE por fuente en
   Liu2024 y Zheng2024 (un solo valor cada una), y toma solo 2 valores
   discretos en Guo2024. Es decir, el mismo problema estructural que
   ya se diagnostico para Pi_gap radial (confundido con identidad de
   instalacion) se repite, e incluso se agrava, con el proxy axial.

3. Comparacion Daily-Nece regimen IV (mismo subconjunto de 73 filas):
   R2(log) radial={r2_radial_sub:.4f} vs R2(log) axial={r2_axial_sub:.4f}.

4. LOFO de 3 pliegues (Re+gap): pooled R2(log) radial={pooled_r2_radial_3fold:.4f}
   vs pooled R2(log) axial={pooled_r2_axial_3fold:.4f}.
""")

out = {
    "coverage": {src: int(cov_tab.loc[src, "n_with_Pi_gap_axial"]) for src in cov_tab.index},
    "n_rows_total": int(n_total),
    "n_rows_with_axial_proxy": int(df["Pi_gap_axial"].notna().sum()),
    "vrancik_excluded_pct_of_corpus": round(100*41/114, 1),
    "formula_verification_liu2024_exact_match": bool(match),
    "daily_nece_r2_log_radial_114_reference": float(r2_radial_full),
    "daily_nece_r2_log_radial_73subset": float(r2_radial_sub),
    "daily_nece_r2_log_axial_73subset": float(r2_axial_sub),
    "lofo_3fold_pooled_r2_log_radial_subset": float(pooled_r2_radial_3fold),
    "lofo_3fold_pooled_r2_log_axial_subset": float(pooled_r2_axial_3fold),
    "lofo_3fold_per_facility_radial": per_fac_radial,
    "lofo_3fold_per_facility_axial": per_fac_axial,
    "lofo_4fold_pooled_r2_log_reference_M6_radial_114pts": float(prev["lofo_cv"]["pooled_r2_log"]),
}
with open(_ROOT + "/results/axial_gap_hypothesis_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print("Resultados guardados en results/axial_gap_hypothesis_results.json")
