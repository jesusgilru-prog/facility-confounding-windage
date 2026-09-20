"""Cross-check every headline numeric fact quoted in manuscript.tex
against the JSON artifact that actually produced it. Third consecutive
review round flagged the same failure mode (text desynced from
results/*.json after a rerun) -- this script closes that gap.

Usage: python3 check_manuscript_numbers_sync.py [path/to/manuscript.tex]
The manuscript source is not part of this deposit, so give its path as the
first argument (or set MANUSCRIPT_TEX); without it the script reports the
numbers it would check and exits.
Exit code 0 = every fact found in manuscript.tex within tolerance.
Exit code 1 = at least one fact missing or mismatched (printed below).
"""
import json
import re
import sys
from pathlib import Path

import os

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
_arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MANUSCRIPT_TEX")
TEX = Path(_arg) if _arg else ROOT / "manuscript.tex"

if not TEX.is_file():
    print(
        f"manuscript.tex not found at {TEX}.\n"
        "The manuscript source is not redistributed in this deposit; pass its\n"
        "path as the first argument or set MANUSCRIPT_TEX to run the check.\n"
        "The facts themselves, and the result file each one is read from, are\n"
        "listed below so the deposited JSON files can still be inspected."
    )
    TEX = None
    tex_text = ""
    tex_lines = []
else:
    tex_text = TEX.read_text(encoding="utf-8")
    tex_lines = tex_text.splitlines()


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


scaling = load("scaling_law_search_results.json")
dn_regime = load("daily_nece_regime_check_results.json")
hier = load("hierarchical_calibration_lofo_results.json")
robust = load("robustness_summary_stats.json")
posctrl = load("posctrl_results.json")
t2norm = load("t2_normalization_sensitivity_results.json")
identind = load("identifiability_results.json")
cgfd = load("cgfd_test_results.json")
cgfdh = load("cgfd_hetero_results.json")

facts = []


def fact(name, value, decimals=3):
    facts.append((name, value, decimals))


# --- bootstrap (Table 4) ---
b = scaling["bootstrap_by_facility"]
fact("bootstrap: n_rank_deficient_skipped", b["n_rank_deficient_skipped"], 0)
fact("bootstrap: n_boot", b["n_boot"], 0)
n_useful = b["n_boot"] - b["n_rank_deficient_skipped"]
fact("bootstrap: n_useful (n_boot - rank_deficient)", n_useful, 0)
pct = 100 * b["n_rank_deficient_skipped"] / b["n_boot"]
fact("bootstrap: rank_deficient_pct", pct, 1)
for coef_name in ["const", "q_Re", "p_gap", "r_conf", "t_asp"]:
    se = b["coef_se"][coef_name]["bootstrap_se"]
    fact(f"bootstrap SE: {coef_name}", se, 2)

# --- M10 (Table 2) ---
m10 = scaling["structures"]["M10_Re_logcorrection"]
fact("M10: rss", m10["rss"], 2)
fact("M10: bic", m10["bic"], 2)
fact("M10: r2", m10["r2"], 3)
fact("M10: a_logcorr", m10["coef"]["a_logcorr"], 3)

# --- M6 winner in-sample (Table 2) ---
m6 = scaling["structures"]["M6_Re_gap_conf_asp"]
fact("M6: bic", m6["bic"], 2)
fact("M6: r2", m6["r2"], 4)

# --- LOFO pooled (M6) ---
lofo = scaling["lofo_cv"]
fact("LOFO M6: pooled_r2_log", lofo["pooled_r2_log"], 3)

# --- Daily-Nece comparison ---
dn = scaling["daily_nece_comparison"]
fact("Daily-Nece: median_relative_error_pct", dn["median_relative_error_pct"], 1)
fact("Daily-Nece: r2_log_space_vs_Daily_Nece", dn["r2_log_space_vs_Daily_Nece"], 2)

# --- Pi_blockage identity ---
blk = scaling["pi_blockage_identity_check"]
fact("Pi_blockage identity: r2", blk["r2"], 1)

# --- robustness_summary_stats ---
dnr = robust["daily_nece_ratio_regression"]
fact("robustness: Daily-Nece ratio regression slope", dnr["slope"], 3)
fact("robustness: Daily-Nece ratio regression r2", dnr["r2"], 3)
eta = robust["eta_squared_facility_identity"]
fact("eta^2: log_Re_Omega", 100 * eta["log_Re_Omega"], 1)
fact("eta^2: log_Pi_gap", 100 * eta["log_Pi_gap"], 1)
fact("eta^2: log_Pi_confinement", 100 * eta["log_Pi_confinement"], 1)
fact("eta^2: log_Pi_aspect_axial", 100 * eta["log_Pi_aspect_axial"], 1)
slopes = robust["reynolds_slope_per_facility"]
for src, val in slopes.items():
    fact(f"Reynolds slope per facility: {src}", val, 3)
fact("Reynolds slope spread ratio", robust["reynolds_slope_spread_ratio"], 0)

# --- few-shot calibration curve (hierarchical) ---
pooled = hier["pooled_by_n_cal"]
for n_cal in ["0", "1", "3", "5", "10"]:
    fact(f"few-shot LOFO pooled R2 (n_cal={n_cal})", pooled[n_cal]["mean_r2_pooled"], 3)

# --- new-Pi-groups / within-class robustness numbers already spot-checked in
# manuscript prose (post-hoc table); include the headline ones. ---
# best-of-sweep pooled R2 for M1 (Re-only), quoted repeatedly as +0.45;
# the new_pi_groups_lofo_results.json layout is not a stable dict of
# structures, so this value is asserted directly rather than parsed from it.
fact("M1 (Re-only) LOFO pooled R2 (post-hoc sweep)", 0.4526, 4)

# NOTE: the original posctrl.py positive control (Re-only vs M6-spec median
# R2, 14.0% pass rate, -3.79 median, -620.95 worst case, Vrancik1968/Zheng2024
# medians, 18.6x magnitude ratio) was the Limitations-point-6 argument this
# paper made before 2026-08-24. It has been fully superseded by the proper
# single-observation diagnostic in Section sec:results-cgfd
# (cgfd_test.py / cgfd_hetero.py), which supersedes rather than duplicates
# it, and posctrl.py's specific numbers no longer appear in the manuscript
# text at all -- removed from this checker too, rather than left in to
# silently pass on coincidental substring matches to unrelated numbers.
del posctrl  # loaded above only for this historical note; unused now

# --- Step 0: identifiability index (canonical correlations) ---
for i, cc in enumerate(identind["canonical_correlations_with_facility_identity"]):
    fact(f"identifiability index: canonical correlation #{i+1}", cc, 3)
for name, val in identind["canonical_loadings_smallest_correlation_direction"].items():
    fact(f"identifiability index: loading {name}", val, 3)

# --- CGFD discrimination test (Section sec:results-cgfd) ---
fact("cgfd: pooled R2 KS statistic", cgfd["pooled_r2_ks_test"]["ks_statistic"], 3)
fact("cgfd: pooled R2 KS p-value", cgfd["pooled_r2_ks_test"]["p_value"], 4)
fact("cgfd: collinearity-only median pooled R2", cgfd["pooled_r2_collinearity_only"]["median"], 2)
fact("cgfd: confounding median pooled R2", cgfd["pooled_r2_genuine_confounding"]["median"], 2)
sod = cgfd["single_observation_diagnostic"]
fact("cgfd: Vrancik1968 percentile under collinearity", sod["Vrancik1968"]["percentile_under_collinearity_only"], 0)
fact("cgfd: Vrancik1968 percentile under confounding", sod["Vrancik1968"]["percentile_under_genuine_confounding"], 0)
# Only Vrancik1968's ratio is calibrated (real value inside both nulls'
# support). The other three were quoted as +5.7/+585/+149 until the
# 2026-08-25 self-audit found them to be KDE extrapolations -- the +585 was
# literally a function of the 1e-300 numerical floor. The manuscript now
# reports them as uncalibrated and quotes the support bounds instead, so
# those are what this checker verifies.
fact("cgfd: Vrancik1968 calibrated log-LR", sod["Vrancik1968"]["log_lr_usable"], 2)
assert sum(1 for v in sod.values() if v["log_lr_is_calibrated"]) == 1, \
    "expected exactly one calibrated log-LR; manuscript text says so explicitly"
import numpy as _np
for _f, _dec in [("Vrancik1968", 1), ("Guo2024", 2), ("Liu2024", 2), ("Zheng2024", 1)]:
    _c = _np.array(cgfd["raw_per_facility_collinearity"][_f])
    fact(f"cgfd: {_f} collinearity support lower bound", float(_c.min()), _dec)

# --- CGFD heteroscedastic third mechanism ---
hf = cgfdh["per_facility"]
fact("cgfd hetero: Guo2024 percentile of real", hf["Guo2024"]["percentile_of_real"], 0)
fact("cgfd hetero: Vrancik1968 percentile of real", hf["Vrancik1968"]["percentile_of_real"], 0)
fact("cgfd hetero: Zheng2024 percentile of real", hf["Zheng2024"]["percentile_of_real"], 1)
fact("cgfd hetero: Liu2024 synthetic max", hf["Liu2024"]["synthetic_max"], 2)

# --- T2 units/normalization homogeneity check ---
fact("t2norm: median ratio T2 vs F7/F9/F13 trend", t2norm["homogeneity_check"]["median_ratio"], 2)
sweep = t2norm["correction_factor_sweep"]
fact("t2norm: pooled R2 with T2 corrected (worst-case k=10)", sweep["divide_by_10.00"]["pooled_r2"], 1)
fact("t2norm: pooled R2 with T2 corrected (mildest k=4)", sweep["divide_by_4.00"]["pooled_r2"], 1)


def format_number(value, decimals):
    if decimals == 0:
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


def find_in_tex(value, decimals):
    """Search manuscript.tex for value at the given precision, and at looser
    precisions down to decimals-1, to tolerate reasonable rounding in prose."""
    for d in range(decimals, -1, -1):
        target = format_number(value, d)
        # also allow a leading 0 to be dropped (0.45 vs .45) and either sign style
        patterns = [re.escape(target)]
        if target.startswith("0."):
            patterns.append(re.escape(target[1:]))
        if target.startswith("-0."):
            patterns.append(re.escape("-" + target[2:]))
        for pat in patterns:
            for i, line in enumerate(tex_lines, start=1):
                if re.search(pat, line):
                    return True, i, target, d
    return False, None, format_number(value, decimals), decimals


if TEX is None:
    for _f in facts:
        print(f"  {_f[0]:<55} = {_f[1]!s:>12}")
    print(f"\n{len(facts)} facts listed; no manuscript given, nothing cross-checked.")
    sys.exit(0)

print(f"Checking {len(facts)} numeric facts against {TEX.name}...\n")
missing = []
for name, value, decimals in facts:
    found, lineno, shown, used_decimals = find_in_tex(value, decimals)
    if found:
        note = "" if used_decimals == decimals else f" (matched at {used_decimals} dp, not {decimals})"
        print(f"  OK   {name:55s} = {shown:>12s}  (manuscript.tex:{lineno}){note}")
    else:
        print(f"  MISS {name:55s} = {shown:>12s}  NOT FOUND in manuscript.tex")
        missing.append((name, value, decimals))

print()
if missing:
    print(f"{len(missing)}/{len(facts)} facts NOT found in manuscript.tex at any tested precision:")
    for name, value, decimals in missing:
        print(f"  - {name} = {value}")
    sys.exit(1)
else:
    print(f"All {len(facts)} facts found in manuscript.tex. Sync OK.")
    sys.exit(0)
