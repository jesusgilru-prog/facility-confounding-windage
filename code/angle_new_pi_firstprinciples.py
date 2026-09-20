"""
Angle: NEW first-principles dimensionless groups (agent-assigned angle,
2026-08-18), evaluated under the SAME mandatory LOFO-by-facility protocol
as the rest of the project.

This is DELIBERATELY distinct from the earlier exploratory script
`new_pi_groups_lofo_test.py`, which built log-additive interaction terms
of the ALREADY-EXISTING Pi columns (Pi_gap, Pi_confinement, Pi_aspect_axial,
M_tip, g_level) with forced unit coefficients. That script found nothing
beats the M6 baseline.

Here the three candidate groups are derived from actual rotor-stator /
Taylor-Couette boundary-layer theory and from raw geometric + kinematic
columns that are NOT already folded into Pi_gap / Pi_confinement /
Pi_aspect_axial as a *single new axis of information*, not merely a
recombination with a forced exponent ratio of 1:1 in log space (which is
what N9-N16 in the earlier script did).

STEP 0 (mandatory): verify the exact algebraic definition of the existing
Pi columns from the raw geometry, so any "new" group can be checked for
whether it's genuinely new information or a relabeling of something
already tested.

  Pi_gap        = gap_radial_m / R_m         (confirmed exact, max abs err ~0)
  Pi_confinement= R_m / R_chamber_m          (confirmed exact)
  Pi_aspect_axial = h_rotor_m / R_m          (confirmed exact)

  Raw columns fully populated across ALL 4 facilities (not facility-exclusive,
  unlike H_pole_m [Vrancik-only], gap_axial_m [Liu-only], H_chamber_m
  [missing for Vrancik]): R_chamber_m, gap_radial_m, h_rotor_m, R_m,
  omega_rad_s, rho_kgm3, mu_Pas, g_level (present everywhere but see below).

STEP 0b (a genuine, load-bearing data finding): g_level is NOT a real
physical measurement for 2 of the 4 facilities. Cross-checking
g_level against the actual centrifugal acceleration ratio Fr = omega^2 * R_m
/ g0 computed from the raw omega and R_m columns:
  - Guo2024 and Liu2024: g_level MATCHES omega^2*R_m/g0 almost exactly
    (these are genuine hypergravity centrifuge tests).
  - Vrancik1968 and Zheng2024: g_level is hardcoded to exactly 1.0 for
    ALL rows, while the TRUE omega^2*R_m/g0 for these rigs is actually
    ~900-20000 (they are small, very fast lab rigs, not hypergravity
    centrifuges at all -- g_level=1 is a "not applicable / standard bench
    rig" placeholder flag, not a measurement).
  This means every earlier script that used log(g_level) as a continuous
  regressor (N3_g_only, N4, N6, N9-N14, N16, N17 in the prior script) was
  regressing against a column that is EXACTLY CONSTANT (log(1)=0, zero
  within-facility variance) for 2 of the 4 held-out facilities, and a real,
  large-range physical quantity for the other 2. That is a latent
  facility-identity leak dressed up as a physical variable, not a fair
  physical test.

NEW GROUP 1 -- Fr_real: the REAL centrifugal-to-gravity ratio,
  Fr_real = omega_rad_s^2 * R_m / g0 (g0 = 9.81 m/s^2), computed identically
  from raw kinematics for ALL 114 rows (unlike the g_level column, this is
  an honest apples-to-apples physical quantity in every facility, including
  Vrancik/Zheng where it is large, not 1). Physical motivation: in a
  rotating cavity, the ratio of centrifugal body force to gravity sets the
  strength of buoyancy-driven secondary (Ekman-pumping-like) recirculation
  in the windage flow relative to the reference lab gravity -- the
  "hypergravity-ness" of the local flow regime.

NEW GROUP 2 -- Pi_Re_gap: the GAP Reynolds number (not the tip Reynolds
  number Re_Omega = rho*omega*R_m^2/mu already in the model). Re_gap =
  rho*omega*gap_radial_m^2/mu = Re_Omega * Pi_gap^2. Physical motivation:
  windage dissipation is generated in thin boundary/shear layers *inside
  the radial clearance*, whose local Reynolds number is built on the gap
  width, not the rotor tip radius; Daily & Nece-type rotor-stator windage
  correlations are organized by a gap-based Reynolds/G-number pair, not by
  tip Reynolds number alone. Tested as a SINGLE regressor (one free
  exponent on the combined group, i.e. lRe + 2*lgap fixed inside one
  coefficient) -- structurally different from M6, which fits Re and gap as
  two independent free exponents (a strictly more flexible model that
  already nests any linear combination of lRe and lgap separately, so the
  interesting question is whether the CONSTRAINED single-number version
  generalizes better, i.e. whether the physical Re_Omega*Pi_gap^2 grouping
  is closer to the "true" invariant than letting Re and gap float freely
  and overfit facility-specific idiosyncrasies).

NEW GROUP 3 -- Pi_curvature: the gap normalized by the OUTER CHAMBER
  radius rather than the rotor radius, Pi_curvature = gap_radial_m /
  R_chamber_m = Pi_gap * Pi_confinement. Physical motivation: for the
  enclosed_arm_in_chamber / enclosed_salient_in_stator / arm_in_cylinder
  geometries, the relevant curvature that sets how strongly the flow in
  the gap "feels" the outer stationary wall is set by the STATOR/CHAMBER
  radius of curvature, not by the rotor's own radius -- a thin arm deep
  inside a much larger chamber (Guo2024, Pi_confinement ~ small) has a
  very different outer-wall curvature relative to its own gap than a disk
  nearly filling its housing (Vrancik1968 disk_in_cylinder cases,
  Pi_confinement ~ 1). This SPECIFIC product (gap x confinement) was not
  tested in the prior exploratory script (which tried g_level x gap,
  g_level x confinement, M_tip x gap, M_tip x confinement, but never
  gap x confinement itself).

All three are tested (a) alone, (b) added to the M6 baseline structure
(Re, gap, confinement, aspect), and (c) replacing their most related
existing term, always under the SAME strict 4-fold leave-one-facility-out
protocol, with the SAME honest bar: pooled R2_log substantially positive
AND every individual held-out facility R2 >= 0.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd

DATA = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT = _ROOT + "/results/angle_new_pi_firstprinciples_results.json"

g0 = 9.81

df = pd.read_csv(DATA)

# ---------------------------------------------------------------
# STEP 0: verify exact algebraic identities of existing Pi columns
# ---------------------------------------------------------------
d0 = df.dropna(subset=["Pi_gap", "Pi_confinement", "Pi_aspect_axial",
                        "gap_radial_m", "R_m", "R_chamber_m", "h_rotor_m"]).copy()
check_gap = (d0["gap_radial_m"] / d0["R_m"] - d0["Pi_gap"]).abs().max()
check_conf = (d0["R_m"] / d0["R_chamber_m"] - d0["Pi_confinement"]).abs().max()
check_asp = (d0["h_rotor_m"] / d0["R_m"] - d0["Pi_aspect_axial"]).abs().max()

# ---------------------------------------------------------------
# STEP 0b: g_level vs real Fr = omega^2 R / g0, by facility
# ---------------------------------------------------------------
df["Fr_real"] = df["omega_rad_s"] ** 2 * df["R_m"] / g0
g_level_check = {}
for src, s in df.groupby("source"):
    g_level_check[src] = {
        "g_level_min": float(s["g_level"].min()),
        "g_level_max": float(s["g_level"].max()),
        "Fr_real_min": float(s["Fr_real"].min()),
        "Fr_real_max": float(s["Fr_real"].max()),
        "g_level_is_constant_placeholder": bool(s["g_level"].nunique() == 1 and s["g_level"].iloc[0] == 1.0),
    }

# ---------------------------------------------------------------
# Working subset: identical 114-row usable set as the rest of the project
# ---------------------------------------------------------------
d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "R_chamber_m", "gap_radial_m",
                       "R_m", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)
sources = d["source"].values
facilities = sorted(set(sources))

y = np.log(d["Cp"].values)
lRe = np.log(d["Re_Omega"].values)
lgap = np.log(d["Pi_gap"].values)
lconf = np.log(d["Pi_confinement"].values)
lasp = np.log(d["Pi_aspect_axial"].values)
lg_col = np.log(d["g_level"].clip(lower=1e-6).values)  # the flawed existing column, for comparison only

# NEW GROUP 1: real Froude/centrifugal number (honest across all facilities)
Fr_real = d["omega_rad_s"].values ** 2 * d["R_m"].values / g0
lFr = np.log(Fr_real)

# NEW GROUP 2: gap-based Reynolds number (single combined physical number)
Re_gap = d["Re_Omega"].values * d["Pi_gap"].values ** 2
lRegap = np.log(Re_gap)

# NEW GROUP 3: gap normalized by chamber (outer-wall curvature) radius
Pi_curv = d["gap_radial_m"].values / d["R_chamber_m"].values
lcurv = np.log(Pi_curv)

# sanity: is any new group degenerate / facility-collinear like geom_confidence?
per_facility_ranges = {}
for name, arr in [("Fr_real", Fr_real), ("Re_gap", Re_gap), ("Pi_curv", Pi_curv)]:
    per_facility_ranges[name] = {
        f: {"min": float(arr[sources == f].min()), "max": float(arr[sources == f].max())}
        for f in facilities
    }

# ---------------------------------------------------------------
# LOFO engine (identical protocol to the rest of the project: OLS in
# log space, train on 3 facilities, predict the 4th never seen at all,
# not even to fit an intercept)
# ---------------------------------------------------------------
def lofo_eval(cols_fn):
    all_resid = []
    per_facility = {}
    for f in facilities:
        test_mask = sources == f
        train_mask = ~test_mask
        Xtr = np.column_stack([np.ones(train_mask.sum())] + cols_fn(train_mask))
        ytr = y[train_mask]
        coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
        Xte = np.column_stack([np.ones(test_mask.sum())] + cols_fn(test_mask))
        yte = y[test_mask]
        resid = yte - Xte @ coef
        all_resid.append(resid)
        ss_tot = np.sum((yte - yte.mean()) ** 2)
        per_facility[f] = {
            "n_test": int(test_mask.sum()),
            "held_out_r2_log": float(1 - np.sum(resid ** 2) / ss_tot) if test_mask.sum() > 1 else None,
            "held_out_rmse_log": float(np.sqrt(np.mean(resid ** 2))),
        }
    all_resid = np.concatenate(all_resid)
    pooled_r2 = float(1 - np.sum(all_resid ** 2) / np.sum((y - y.mean()) ** 2))
    pooled_rmse = float(np.sqrt(np.mean(all_resid ** 2)))
    all_ge_zero = all(
        (pf["held_out_r2_log"] is not None and pf["held_out_r2_log"] >= 0)
        for pf in per_facility.values()
    )
    return {
        "per_facility": per_facility,
        "pooled_r2_log": pooled_r2,
        "pooled_rmse_log": pooled_rmse,
        "passes_honest_bar_all_facilities_r2_ge_0": all_ge_zero,
    }

structures = {
    # --- baselines, recomputed here for an apples-to-apples reference ---
    "M1_Re_only": lambda m: [lRe[m]],
    "M6_Re_gap_conf_asp_BASELINE": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m]],
    "M2_Re_glevel_flawed": lambda m: [lRe[m], lg_col[m]],

    # --- NEW GROUP 1: Fr_real (honest hypergravity number) ---
    "G1_Fr_real_only": lambda m: [lFr[m]],
    "G1_M6_plus_Fr_real": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], lFr[m]],
    "G1_Re_replaced_by_Fr_real": lambda m: [lFr[m], lgap[m], lconf[m], lasp[m]],
    "G1_M6_plus_Fr_real_minus_glevel_col": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], lFr[m]],  # same as above, kept for clarity in report

    # --- NEW GROUP 2: Re_gap (gap-based Reynolds number, single term) ---
    "G2_Re_gap_only": lambda m: [lRegap[m]],
    "G2_Re_gap_plus_conf_asp": lambda m: [lRegap[m], lconf[m], lasp[m]],
    "G2_M6_plus_Re_gap": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], lRegap[m]],
    "G2_Re_and_gap_replaced_by_Re_gap": lambda m: [lRegap[m], lconf[m], lasp[m]],  # same as Re_gap_plus_conf_asp

    # --- NEW GROUP 3: Pi_curvature = gap/R_chamber ---
    "G3_curv_only": lambda m: [lcurv[m]],
    "G3_M6_plus_curv": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], lcurv[m]],
    "G3_gap_conf_replaced_by_curv": lambda m: [lRe[m], lcurv[m], lasp[m]],

    # --- combined: all 3 new groups together on top of M6 ---
    "ALL3_M6_plus_Fr_Regap_curv": lambda m: [lRe[m], lgap[m], lconf[m], lasp[m], lFr[m], lRegap[m], lcurv[m]],
    "ALL3_Fr_Regap_curv_asp_only": lambda m: [lFr[m], lRegap[m], lcurv[m], lasp[m]],
}

lofo_all = {name: lofo_eval(fn) for name, fn in structures.items()}
lofo_ranking = sorted(lofo_all.items(), key=lambda kv: kv[1]["pooled_r2_log"], reverse=True)

baseline_pooled_r2 = lofo_all["M6_Re_gap_conf_asp_BASELINE"]["pooled_r2_log"]

any_new_beats_baseline = any(
    r["pooled_r2_log"] > baseline_pooled_r2
    for name, r in lofo_all.items()
    if name not in ("M6_Re_gap_conf_asp_BASELINE", "M1_Re_only", "M2_Re_glevel_flawed")
)
any_new_passes_honest_bar = any(
    r["passes_honest_bar_all_facilities_r2_ge_0"]
    for name, r in lofo_all.items()
    if name.startswith(("G1_", "G2_", "G3_", "ALL3_"))
)

out = {
    "n_points": int(n),
    "step0_pi_column_identity_check": {
        "Pi_gap_equals_gap_radial_over_R_m_max_abs_err": float(check_gap),
        "Pi_confinement_equals_R_m_over_R_chamber_m_max_abs_err": float(check_conf),
        "Pi_aspect_axial_equals_h_rotor_over_R_m_max_abs_err": float(check_asp),
    },
    "step0b_g_level_column_is_flawed_finding": {
        "description": (
            "g_level equals the TRUE omega^2*R_m/g0 for Guo2024 and Liu2024 "
            "(real hypergravity centrifuges) but is a hardcoded constant "
            "1.0 for Vrancik1968 and Zheng2024, whose TRUE omega^2*R_m/g0 "
            "is actually ~900-20000 (small fast lab rigs, not hypergravity "
            "at all). Any prior model term built from log(g_level) had "
            "exactly zero within-facility variance for 2 of 4 facilities."
        ),
        "per_facility_g_level_vs_real_Fr": g_level_check,
    },
    "new_group_definitions": {
        "Fr_real": "omega_rad_s**2 * R_m / 9.81  (real centrifugal/gravity ratio, all 4 facilities honest)",
        "Re_gap": "Re_Omega * Pi_gap**2  (= rho*omega*gap_radial_m**2/mu, gap-based Reynolds number)",
        "Pi_curvature": "gap_radial_m / R_chamber_m  (= Pi_gap * Pi_confinement, gap normalized by OUTER chamber radius)",
    },
    "new_group_ranges_per_facility": per_facility_ranges,
    "lofo_pooled_r2_by_structure_sorted_best_first": [(name, r["pooled_r2_log"]) for name, r in lofo_ranking],
    "lofo_detail": lofo_all,
    "baseline_M6_pooled_r2_log_recomputed_here": baseline_pooled_r2,
    "any_new_structure_beats_baseline_pooled_r2": any_new_beats_baseline,
    "any_new_structure_passes_honest_bar_pooled_AND_all_facilities_ge_0": any_new_passes_honest_bar,
    "best_new_structure": lofo_ranking[0],
}

with open(OUT, "w") as f:
    json.dump(out, f, indent=2, default=float)

print(f"n = {n}\n")
print("Step 0 identity checks (max abs error, should be ~0):")
print(f"  Pi_gap == gap_radial_m/R_m:         {check_gap:.3e}")
print(f"  Pi_confinement == R_m/R_chamber_m:  {check_conf:.3e}")
print(f"  Pi_aspect_axial == h_rotor_m/R_m:   {check_asp:.3e}")

print("\nStep 0b: g_level vs real Fr=omega^2*R_m/g0 by facility:")
for f, v in g_level_check.items():
    print(f"  {f:15s} g_level=[{v['g_level_min']:.3g},{v['g_level_max']:.3g}]  "
          f"real_Fr=[{v['Fr_real_min']:.3g},{v['Fr_real_max']:.3g}]  "
          f"g_level_is_placeholder={v['g_level_is_constant_placeholder']}")

print("\nNew group ranges by facility (checking for degenerate facility-collinearity):")
print(json.dumps(per_facility_ranges, indent=2))

print("\n" + "=" * 78)
print("LOFO-CV pooled R2(log), best to worst:")
print("=" * 78)
for name, r in lofo_ranking:
    ok = r["passes_honest_bar_all_facilities_r2_ge_0"]
    print(f"  {name:40s} pooled_R2={r['pooled_r2_log']:9.4f}  all_facilities_R2>=0: {ok}")
    for f in facilities:
        fr = r["per_facility"][f]
        print(f"       {f:15s} n={fr['n_test']:3d}  R2={fr['held_out_r2_log']}")

print(f"\nBaseline M6 pooled R2 (recomputed here) = {baseline_pooled_r2:.4f}")
print(f"Any new structure beats baseline pooled R2: {any_new_beats_baseline}")
print(f"Any new structure passes the FULL honest bar (pooled>>0 AND every facility R2>=0): {any_new_passes_honest_bar}")
print(f"Best new structure: {lofo_ranking[0]}")
