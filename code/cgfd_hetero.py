"""Follow-up to cgfd_discrimination_test.py: neither pure collinearity
artifact nor Gaussian per-facility offset (tau2_hat=0.073) reproduces the
real LOFO collapse magnitude for 3 of 4 facilities (real value falls
outside the full 500-replicate range of both). This script tests one
concrete candidate third mechanism: heteroscedastic noise matching this
corpus's OWN reported per-point measurement uncertainty
(data/cross_rotor_dataset_v3.csv, error_pct column), which is highly
unequal across facilities (Zheng2024 mean 15.6%, up to 83.1% on one
point, vs Guo2024/Liu2024/Vrancik1968 means 6-9%) -- Zheng2024 is also
the facility with the most extreme real LOFO collapse (-553.8).

Process 3: same true law (Re_Omega^-0.2, external, not fit to this
corpus) and same per-facility Gaussian offset as Process 2
(genuine confounding, tau2_hat=0.073), but noise standard deviation per
ROW now uses this corpus's own real reported error_pct instead of a
single pooled sigma2_hat -- i.e. heteroscedastic noise tied to actual
measurement uncertainty, not assumed homogeneous.

No fabricated numbers: everything below is computed from
data/cross_rotor_dataset_v3.csv.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "cross_rotor_dataset_v3.csv")
FEATURES = ["Re_Omega", "Pi_gap", "Pi_confinement", "Pi_aspect_axial"]

facilities = sorted(df["source"].unique())
log_Re = np.log(df["Re_Omega"].values)
log_gap = np.log(df["Pi_gap"].values)
log_conf = np.log(df["Pi_confinement"].values)
log_asp = np.log(df["Pi_aspect_axial"].values)
n = len(df)
fac_labels = df["source"].values

BETA_RE_TRUE = -0.2
INTERCEPT_TRUE = np.log(df["Cp"].values).mean() - BETA_RE_TRUE * log_Re.mean()
TAU2_HAT = 0.073
TAU_HAT = np.sqrt(TAU2_HAT)

# Per-row noise SD from this corpus's own reported fractional error,
# converted to an approximate log-space SD (log(1+x) ~= x for small
# fractional errors; used as-is for large ones, e.g. Zheng2024's 83.1%).
error_frac = df["error_pct"].values / 100.0
row_noise_sd = np.log1p(error_frac)
print("Row noise SD by facility (median, from real error_pct):")
for f in facilities:
    m = fac_labels == f
    print(f"  {f}: median={np.median(row_noise_sd[m]):.4f}, max={np.max(row_noise_sd[m]):.4f}")

N_REPLICATES = 500
X4 = np.column_stack([log_Re, log_gap, log_conf, log_asp])
rng = np.random.default_rng(2)


def lofo_pooled_and_per_facility(y, X, fac_labels):
    preds = np.full(n, np.nan)
    per_fac_r2 = {}
    for held_out in facilities:
        train_mask = fac_labels != held_out
        test_mask = ~train_mask
        Xtr = np.column_stack([np.ones(train_mask.sum()), X[train_mask]])
        ytr = y[train_mask]
        coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        Xte = np.column_stack([np.ones(test_mask.sum()), X[test_mask]])
        yhat = Xte @ coef
        preds[test_mask] = yhat
        yte = y[test_mask]
        ss_res = np.sum((yte - yhat) ** 2)
        ss_tot = np.sum((yte - yte.mean()) ** 2)
        per_fac_r2[held_out] = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    ss_res_pooled = np.sum((y - preds) ** 2)
    ss_tot_pooled = np.sum((y - y.mean()) ** 2)
    pooled_r2 = 1 - ss_res_pooled / ss_tot_pooled
    return pooled_r2, per_fac_r2


pooled_p3 = []
per_fac_p3 = {f: [] for f in facilities}
for rep in range(N_REPLICATES):
    noise = rng.normal(0.0, 1.0, size=n) * row_noise_sd
    offsets = {f: rng.normal(0.0, TAU_HAT) for f in facilities}
    offset_vec = np.array([offsets[f] for f in fac_labels])
    y3 = INTERCEPT_TRUE + BETA_RE_TRUE * log_Re + offset_vec + noise
    p3, pf3 = lofo_pooled_and_per_facility(y3, X4, fac_labels)
    pooled_p3.append(p3)
    for f in facilities:
        per_fac_p3[f].append(pf3[f])

pooled_p3 = np.array(pooled_p3)

real_per_facility = {
    "Guo2024": -8.986612059551168,
    "Liu2024": -118.63018606035955,
    "Vrancik1968": -1.8542073187456043,
    "Zheng2024": -553.7587044467027,
}
real_pooled_r2 = -0.8848004392997477

print(f"\nProcess 3 (heteroscedastic real error_pct + facility offset), pooled R2: "
      f"median={np.median(pooled_p3):.3f}, range=[{pooled_p3.min():.2f}, {pooled_p3.max():.2f}]")
print(f"Real pooled R2={real_pooled_r2:.3f}, "
      f"percentile under Process 3={float(np.mean(pooled_p3 <= real_pooled_r2))*100:.1f}%")

print("\nPer-facility, Process 3 vs real:")
out_per_fac = {}
for f in facilities:
    arr = np.array(per_fac_p3[f])
    real_val = real_per_facility[f]
    pct = float(np.mean(arr <= real_val)) * 100
    covers = bool(arr.min() <= real_val <= arr.max())
    print(f"  {f}: real={real_val:.2f}, synthetic range=[{arr.min():.2f}, {arr.max():.2f}], "
          f"median={np.median(arr):.2f}, percentile of real={pct:.1f}%, "
          f"real WITHIN synthetic range={covers}")
    out_per_fac[f] = {
        "real_r2": real_val, "synthetic_min": float(arr.min()), "synthetic_max": float(arr.max()),
        "synthetic_median": float(np.median(arr)), "percentile_of_real": pct,
        "real_within_synthetic_range": covers,
    }

out = {
    "purpose": "Test whether heteroscedastic noise (real per-point error_pct) "
               "plus facility offset reproduces the real LOFO collapse magnitude "
               "better than homoscedastic collinearity-only or homoscedastic "
               "confounding (cgfd_discrimination_test.py).",
    "n_replicates": N_REPLICATES,
    "pooled_r2_process3": {"median": float(np.median(pooled_p3)),
                             "min": float(pooled_p3.min()), "max": float(pooled_p3.max())},
    "real_pooled_r2": real_pooled_r2,
    "per_facility": out_per_fac,
    # Raw replicate distributions, persisted so the diagnostic figure is
    # reproducible from this JSON alone (see cgfd_test.py for the same note).
    "raw_per_facility": {f: [float(x) for x in per_fac_p3[f]] for f in facilities},
}
OUT_JSON = ROOT / "results" / "cgfd_hetero_results.json"
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to {OUT_JSON}")
