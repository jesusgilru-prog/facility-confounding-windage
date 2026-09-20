"""
ANGLE: first-principles (zero-fit) dimensional-analysis rescaling of Cp,
tried BEFORE any regression, to see if it collapses the 4 facilities onto
a common curve vs Re_Omega -- in the spirit of a boundary-layer similarity
variable (e.g. Blasius eta).

Executed 2026-08-18 on data/cross_rotor_dataset_v3.csv

METHOD
------
Pi definitions verified numerically from raw columns (not assumed):
  Pi_confinement = R_m / R_chamber_m            (rotor radius / chamber radius)
  Pi_gap         = gap_radial_m / R_m           (radial clearance / rotor radius)
  Pi_aspect_axial= h_rotor_m / R_m              (rotor axial height / rotor radius)
(Pi_blockage excluded throughout: proven exact algebraic identity of the
other two, R2=1.0, see verify_cp_equals_cm.py / scaling_law_search.py.)

Candidate rescalings, ALL WITH EXPONENTS FIXED FROM THEORY/LITERATURE OR
FROM AN EXACT GEOMETRIC IDENTITY -- NONE FIT TO THIS DATASET:

  RAW        : no correction (baseline)
  A_DN_IV    : Cp / Pi_gap**0.1     (Daily & Nece 1960 regime IV, turbulent
               separated boundary layers, G-exponent +0.1; source:
               Poncet et al. arXiv:1305.2882 eq.41 -- same source already
               used and cited in daily_nece_regime_check.py)
  B_DN_I     : Cp * Pi_gap**1.0     (Daily & Nece regime I, laminar merged
               Couette layers, G-exponent -1; undoing it means multiplying
               by Pi_gap**1)
  C_area_cyl : Cp * Pi_aspect_axial (reinterprets the characteristic drag
               area as the rotor's cylindrical lateral area ~R*h instead
               of a disk face ~R^2; since Cp is presumably already
               nondimensionalized with R^5, changing R^5 -> R^4*h_rotor
               multiplies Cp by h_rotor/R = Pi_aspect_axial)
  D_blockage : Cp / (Pi_aspect_axial * Pi_confinement**2) (undoes the
               EXACT geometric identity form found for Pi_blockage,
               (2/pi)*Pi_aspect*Pi_conf**2 -- i.e. treats Cp as if it
               should scale with the same exact "blocked/wetted area
               fraction" combination that Pi_blockage was proven to equal)
  E_lubrication: Cp * (1 - Pi_confinement) (classic lubrication-theory
               divergence of Couette drag torque as radial clearance
               (R_chamber - R) shrinks to zero, i.e. as confinement -> 1;
               (1-Pi_confinement) = (R_chamber-R)/R_chamber, a normalized
               clearance, NOT identical to Pi_gap but a related textbook
               near-wall-divergence argument)

For each candidate we test the COLLAPSE, not the absolute magnitude:
  1. Pool ALL 114 points, fit ONE common log-log slope+intercept vs Re_Omega
     -> R2_pooled (how well a single curve explains the corrected Cp)
  2. Fit the SAME common slope but a SEPARATE intercept per facility
     (ANCOVA / fixed effects) -> R2_fixed, and record the 4 per-facility
     intercepts.
  3. partial_R2_facility = (R2_fixed - R2_pooled) / (1 - R2_pooled)
     = fraction of the *pooled model's leftover variance* that is actually
     explained by which-facility-you-are. This is the real collapse
     metric: if the rescaling truly collapses the facilities onto one
     curve, this should shrink towards 0 (facility stops mattering once
     the physics-based correction is applied).
  4. intercept_spread = std across the 4 per-facility intercepts (in log
     units) -- a second, more visual measure of vertical mis-alignment
     between the 4 facility curves.

A negative control is also run: the SAME 5 candidates with the exponent
SIGN FLIPPED (wrong physics), to check whether "improvement" is a
generic artifact of multiplying by anything correlated with facility
identity (in which case both signs would "help" depending on luck) or is
specific to the correct physical direction.

An additional non-predictive ORACLE is computed per single Pi-group: the
exponent that *numerically minimizes* intercept_spread (i.e. best case,
fit to this exact data, NOT a real predictive rescaling) -- reported only
as a diagnostic ceiling to see whether the literature-fixed exponents are
even in the right ballpark/sign, not proposed as a result.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd

CSV = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT = _ROOT + "/results/angle_firstprinciples_collapse_results.json"

df = pd.read_csv(CSV)
d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_confinement", "Pi_aspect_axial", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)
assert n == 114, f"expected 114 rows after cleaning, got {n} -- investigate before trusting anything below"

# --- verify Pi definitions numerically from raw columns (do not trust docstring blindly) ---
verify = {}
chk_conf = d["R_m"] / d["R_chamber_m"]
verify["Pi_confinement_matches_R_over_Rchamber"] = bool(np.allclose(chk_conf, d["Pi_confinement"], rtol=1e-4))
chk_gap = d["gap_radial_m"] / d["R_m"]
verify["Pi_gap_matches_gapradial_over_R"] = bool(np.allclose(chk_gap, d["Pi_gap"], rtol=1e-4))
chk_asp = d["h_rotor_m"] / d["R_m"]
verify["Pi_aspect_axial_matches_hrotor_over_R"] = bool(np.allclose(chk_asp, d["Pi_aspect_axial"], rtol=1e-4))
for k, v in verify.items():
    print(k, "->", v)
    if not v:
        raise SystemExit(f"Pi definition assumption WRONG: {k}. Aborting rather than computing on a wrong basis.")

Cp = d["Cp"].values
Re = d["Re_Omega"].values
Gap = d["Pi_gap"].values
Conf = d["Pi_confinement"].values
Asp = d["Pi_aspect_axial"].values
source = d["source"].values
facilities = sorted(set(source))
lRe = np.log(Re)


def ancova_collapse_metrics(y, label):
    """Given y = log(Cp_variant), fit pooled single-line model and
    facility-fixed-effects model (common slope, separate intercepts).
    Returns collapse diagnostics."""
    # pooled: y = a + b*lRe
    Xp = np.column_stack([np.ones(n), lRe])
    coefp, *_ = np.linalg.lstsq(Xp, y, rcond=None)
    predp = Xp @ coefp
    rss_p = float(np.sum((y - predp) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2_p = 1 - rss_p / tss

    # fixed effects: y = a_f + b*lRe (common slope b, facility dummies)
    dummies = np.column_stack([(source == f).astype(float) for f in facilities])
    Xf = np.column_stack([dummies, lRe])
    coeff, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    predf = Xf @ coeff
    rss_f = float(np.sum((y - predf) ** 2))
    r2_f = 1 - rss_f / tss
    intercepts = dict(zip(facilities, coeff[:-1].tolist()))
    slope_common = float(coeff[-1])

    partial_r2_facility = (r2_f - r2_p) / (1 - r2_p) if (1 - r2_p) > 1e-12 else float("nan")
    intercept_vals = np.array(list(intercepts.values()))
    intercept_spread_std = float(np.std(intercept_vals))
    intercept_spread_range = float(np.ptp(intercept_vals))

    # per-facility residual std around the pooled (single-curve) fit, to
    # check whether WITHIN-facility scatter also changes (it should barely
    # move, since these corrections are near-constant within most facilities)
    resid_pooled = y - predp
    per_fac_resid_std = {f: float(np.std(resid_pooled[source == f])) for f in facilities}

    return {
        "label": label,
        "r2_pooled_single_curve": r2_p,
        "r2_fixed_effects_common_slope": r2_f,
        "partial_r2_of_facility_dummies": float(partial_r2_facility),
        "common_slope_in_fixed_effects_model": slope_common,
        "per_facility_intercept": intercepts,
        "intercept_spread_std": intercept_spread_std,
        "intercept_spread_range": intercept_spread_range,
        "per_facility_pooled_residual_std": per_fac_resid_std,
    }


results = {}
raw_metrics = ancova_collapse_metrics(np.log(Cp), "RAW (no geometric correction)")
results["RAW"] = raw_metrics

candidates = {
    "A_DN_IV_gap_pow_+0.1": Cp / (Gap ** 0.1),
    "B_DN_I_gap_pow_-1.0": Cp * (Gap ** 1.0),
    "C_area_cyl_asp_pow_+1.0": Cp * (Asp ** 1.0),
    "D_blockage_identity_form": Cp / (Asp * (Conf ** 2)),
    "E_lubrication_1_minus_conf": Cp * (1.0 - Conf),
}

for name, cp_variant in candidates.items():
    if np.any(cp_variant <= 0):
        raise SystemExit(f"{name} produced non-positive Cp_variant, cannot take log -- aborting")
    m = ancova_collapse_metrics(np.log(cp_variant), name)
    m["reduction_vs_raw_partial_r2_facility"] = float(
        1 - m["partial_r2_of_facility_dummies"] / raw_metrics["partial_r2_of_facility_dummies"]
    )
    m["reduction_vs_raw_intercept_spread_std"] = float(
        1 - m["intercept_spread_std"] / raw_metrics["intercept_spread_std"]
    )
    results[name] = m

# --- negative control: sign-flipped (wrong-physics) versions of the same candidates ---
neg_control = {
    "A_flip_gap_pow_-0.1": Cp * (Gap ** 0.1),
    "B_flip_gap_pow_+1.0": Cp / (Gap ** 1.0),
    "C_flip_asp_pow_-1.0": Cp / (Asp ** 1.0),
    "D_flip_blockage_form": Cp * (Asp * (Conf ** 2)),
    "E_flip_conf_direct": Cp / (1.0 - Conf),
}
neg_results = {}
for name, cp_variant in neg_control.items():
    if np.any(cp_variant <= 0):
        neg_results[name] = {"skipped": "non-positive values"}
        continue
    m = ancova_collapse_metrics(np.log(cp_variant), name)
    m["reduction_vs_raw_partial_r2_facility"] = float(
        1 - m["partial_r2_of_facility_dummies"] / raw_metrics["partial_r2_of_facility_dummies"]
    )
    m["reduction_vs_raw_intercept_spread_std"] = float(
        1 - m["intercept_spread_std"] / raw_metrics["intercept_spread_std"]
    )
    neg_results[name] = m

# --- non-predictive oracle: exponent per single Pi-group that MINIMIZES
# intercept_spread_std (best case, fit to this exact data -- NOT a real
# rescaling law, only a diagnostic ceiling / sign-check) ---
oracle = {}
for gname, gvals in [("Pi_gap", Gap), ("Pi_confinement", Conf), ("Pi_aspect_axial", Asp)]:
    exps = np.linspace(-3, 3, 601)
    best_exp, best_spread = None, np.inf
    for e in exps:
        variant = Cp * (gvals ** e)
        if np.any(variant <= 0) or not np.all(np.isfinite(variant)):
            continue
        m = ancova_collapse_metrics(np.log(variant), f"{gname}^{e:.3f}")
        if m["intercept_spread_std"] < best_spread:
            best_spread = m["intercept_spread_std"]
            best_exp = e
    oracle[gname] = {
        "best_fit_exponent_minimizing_intercept_spread": float(best_exp),
        "resulting_intercept_spread_std": float(best_spread),
        "raw_intercept_spread_std_for_reference": raw_metrics["intercept_spread_std"],
        "note": "This is a per-dataset numerical optimum (a fit), reported "
                "ONLY as a diagnostic of what magnitude/sign of exponent "
                "would even be needed -- it is NOT a proposed predictive "
                "law and was not evaluated out-of-sample.",
    }

# --- summary judgement against the angle's own honest bar ---
best_candidate_name = min(
    [k for k in results if k != "RAW"],
    key=lambda k: results[k]["intercept_spread_std"],
)
summary = {
    "raw_intercept_spread_std": raw_metrics["intercept_spread_std"],
    "raw_partial_r2_facility": raw_metrics["partial_r2_of_facility_dummies"],
    "best_theory_candidate_by_intercept_spread": best_candidate_name,
    "best_candidate_intercept_spread_std": results[best_candidate_name]["intercept_spread_std"],
    "best_candidate_reduction_vs_raw": results[best_candidate_name]["reduction_vs_raw_intercept_spread_std"],
    "best_candidate_partial_r2_facility": results[best_candidate_name]["partial_r2_of_facility_dummies"],
}

out = {
    "n": n,
    "pi_definition_verification": verify,
    "raw_baseline": raw_metrics,
    "theory_candidates_correct_sign": results,
    "negative_control_wrong_sign": neg_results,
    "non_predictive_oracle_per_group": oracle,
    "summary": summary,
}

with open(OUT, "w") as f:
    json.dump(out, f, indent=2, default=float)

print(json.dumps(summary, indent=2, default=float))
print("\nFull results written to", OUT)
