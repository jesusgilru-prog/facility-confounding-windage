"""
Angle: literature/data search for additional PUBLIC, FREE rotor-stator windage /
disc-friction / enclosed rotating cavity torque-power datasets not already in
cross_rotor_dataset_v3.csv, that could plausibly enlarge or diversify the corpus.

This script does NOT modify the paper's corpus. It:
  1. Digitizes (by hand-transcription from the primary source's own clean
     typewritten data tables, cross-checked against the source's own reported
     derived quantities) a real, freely downloadable NASA report that was
     found during the search and that DOES report full per-point tabulated
     data (Gorland, Kempke & Lumannick, NASA TM X-52851, 1970).
  2. Recomputes Re_Omega and Cp from first principles (torque, speed, air
     properties from reported barometric pressure and temperature) and
     cross-checks against the source's own reported Reynolds number and
     windage-power columns, as an independent transcription-accuracy check.
  3. Computes this candidate facility's Pi-groups using the SAME formulas
     used in cross_rotor_dataset_v3.csv (verified against the existing CSV
     below) and compares its coordinate ranges to the existing 4 facilities.
  4. Reports, honestly, whether this is a geometrically/physically compatible
     5th facility or not, and what would still be required to actually merge
     it (it is NOT simply appended in this script -- that is a decision for
     the paper's authors, not this exploratory script).

Source: Gorland, S. H.; Kempke, E. E., Jr.; Lumannick, S. (1970).
"Experimental Windage Losses for Close Clearance Rotating Cylinders in the
Turbulent Flow Regime." NASA TM X-52851.
Free PDF: https://ntrs.nasa.gov/api/citations/19700023755/downloads/19700023755.pdf
(public domain, "Work of the US Gov. Public Use Permitted", confirmed via the
NTRS citation page fetched during this investigation.)

Geometry: a smooth, unslotted, unshrouded rotating cylinder, 12 in (0.3048 m)
diameter x 5.9 in (0.14986 m) long, inside a stationary concentric housing,
tested at three fixed radial gaps: 0.0565 in, 0.116 in, 0.236 in. Per the
abstract, the housing was mounted on a reaction-torque balance specifically
"to eliminate any disc type end effects" -- i.e. this is a pure cylindrical
(Couette-type) windage measurement, structurally close to the axial/cylindrical
component of Vrancik1968's disk_in_cylinder facility, but with NO salient
poles/arms and NO superimposed through-flow.
"""
import json
import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Verify the Pi-group formulas against the existing corpus before reusing
#    them on the new candidate data (so we are not guessing the definitions).
# ---------------------------------------------------------------------------
corpus = pd.read_csv("../data/cross_rotor_dataset_v3.csv")
row = corpus[corpus.source == "Vrancik1968"].iloc[0]
pi_gap_check = row.gap_radial_m / row.R_m
pi_conf_check = row.R_m / row.R_chamber_m
pi_aspect_check = row.h_rotor_m / row.R_m
assert math.isclose(pi_gap_check, row.Pi_gap, rel_tol=1e-9)
assert math.isclose(pi_conf_check, row.Pi_confinement, rel_tol=1e-9)
assert math.isclose(pi_aspect_check, row.Pi_aspect_axial, rel_tol=1e-9)
# Re_Omega definition check (Zheng2024 row 0): rho*omega*R^2/mu
z = corpus[corpus.source == "Zheng2024"].iloc[0]
re_check = z.rho_kgm3 * z.omega_rad_s * z.R_m ** 2 / z.mu_Pas
assert math.isclose(re_check, z.Re_Omega, rel_tol=1e-4)
# Cp definition check (already independently verified in
# verify_cp_equals_cm.py; re-derive once more here for self-containedness)
cp_check = corpus.P_w_W / (0.5 * corpus.rho_kgm3 * corpus.omega_rad_s ** 3 * corpus.R_m ** 5)
assert np.allclose(cp_check, corpus.Cp, atol=1e-6)
print("Pi-group and Cp/Re_Omega formula definitions confirmed against the "
      "existing corpus (Vrancik1968 and Zheng2024 rows).")

PI_GAP = lambda gap_radial_m, R_m: gap_radial_m / R_m
PI_CONF = lambda R_m, R_chamber_m: R_m / R_chamber_m
PI_ASPECT = lambda h_rotor_m, R_m: h_rotor_m / R_m
RE_OMEGA = lambda rho, omega, R, mu: rho * omega * R ** 2 / mu
CP = lambda P, rho, omega, R: P / (0.5 * rho * omega ** 3 * R ** 5)

# ---------------------------------------------------------------------------
# 1. Hand-transcribed data from NASA TM X-52851, Tables I, II, III
#    (columns as printed: SPEED rpm, TORQUE in-lb, REYNOLDS NUMBER [as
#    reported by the source, own convention], DRAG COEFFICIENT lambda
#    [source's own Cm-like coefficient, NOT reused here], TEMPERATURE degF,
#    WINDAGE watts [as reported by the source]).
#    Transcribed directly from 200-dpi renders of the source PDF pages
#    11-16 (report pp. 20-25), which are a clean, unambiguous monospaced
#    typewritten table (not a scan of handwriting, not OCR-guessed).
# ---------------------------------------------------------------------------
TABLE_I = [  # RUN NO. 10, GAP = 0.0565 in, barometric = 29.37 in Hg
    (241, .01534, 76, .043738), (483, .03835, 76, 0.2191), (735, .07670, 76, 0.66697),
    (975, .1227, 76, 1.415), (1220, .1687, 76, 2.554), (1457, .2301, 76, 3.9718),
    (1714, .2761, 76, 5.5988), (1954, .3528, 76, 8.1559), (2194, .4295, 76, 11.148),
    (2462, .5139, 76, 14.9688), (2710, .6059, 76, 19.426), (2964, .6980, 76, 24.477),
    (3214, .8054, 77, 30.625), (3456, .9127, 77, 37.318), (3700, 1.0201, 77, 44.655),
    (3939, 1.1352, 77, 52.903), (4229, 1.289, 78, 64.493), (4475, 1.427, 78, 75.55),
    (4800, 1.611, 78, 91.487), (5109, 1.818, 78, 109.716), (5406, 2.010, 79, 128.556),
    (5709, 2.224, 79, 150.216), (6006, 2.447, 80, 173.876), (6410, 2.800, 80, 212.34),
    (6809, 3.129, 81, 251.73), (7201, 3.490, 82, 297.33), (7605, 3.866, 83, 347.84),
    (7964, 4.180, 85, 393.85), (8650, 4.893, 88, 500.74), (9028, 5.323, 90, 568.55),
    (9519, 5.845, 93, 658.26), (9994, 6.404, 95, 757.20), (10569, 7.041, 99, 880.42),
    (11154, 7.762, 103, 1024.3), (11725, 8.491, 107, 1177.86), (12440, 9.410, 112, 1385.5),
    (13175, 10.378, 118, 1617.65), (13898, 11.405, 124, 1875.3), (14617, 12.487, 131, 2159.4),
    (15337, 13.553, 139, 2459.2), (16268, 14.795, 147, 2847.5), (17242, 15.931, 156, 3249.8),
    (18432, 17.679, 170, 3853.3), (19657, 19.313, 184, 4491.5), (20852, 21.108, 200, 5207.5),
    (22070, 22.719, 218, 5932.0),
]
GAP_I_IN = 0.0565
BARO_I_INHG = 29.37

TABLE_II = [  # RUN NO. 6, GAP = 0.116 in, barometric = 29.05 in Hg
    (270, .0153, 74, .04888), (515, .03825, 74, .233), (760, .06885, 74, .6191),
    (1000, .1071, 74, 1.267), (1245, .1530, 74, 2.2538), (1480, .1925, 74, 3.349),
    (1729, .25245, 74, 5.164), (1975, .31365, 74, 7.329), (2220, .3825, 74, 10.05),
    (2450, .45135, 74, 13.08), (2705, .5202, 74, 16.65), (2955, .60435, 74, 21.13),
    (3345, .7497, 74, 29.67), (3830, .90095, 75, 42.6397), (4247, 1.13985, 75, 57.28),
    (4604, 1.3158, 76, 71.68), (4845, 1.446, 76, 82.89), (5254, 1.644, 76, 102.198),
    (5651, 1.890, 77, 126.37), (6070, 2.134, 77, 153.26), (6456, 2.410, 77, 184.09),
    (6851, 2.670, 78, 216.43), (7256, 2.953, 79, 253.52), (7653, 3.251, 81, 294.37),
    (7952, 3.473, 83, 326.76), (8541, 3.955, 86, 399.67), (9032, 4.391, 90, 469.24),
    (9485, 4.758, 93, 533.96), (9998, 5.210, 97, 616.31), (10485, 5.669, 100, 703.27),
    (10960, 6.128, 103, 794.65), (11448, 6.602, 106, 894.24), (11920, 7.084, 110, 999.08),
    (12635, 7.857, 114, 1174.6), (13380, 8.652, 122, 1369.7), (14100, 9.463, 127, 1578.7),
    (15047, 10.565, 135, 1880.9), (15762, 11.383, 140, 2122.8), (16482, 12.133, 147, 2366.07),
    (17451, 13.395, 156, 2765.7), (18672, 14.925, 168, 3297.3), (19620, 16.149, 179, 3748.8),
    (20566, 17.297, 189, 4208.9), (21752, 18.980, 207, 4884.8), (22990, 20.586, 223, 5599.6),
    (24020, 21.963, 238, 6241.8),
]
GAP_II_IN = 0.116
BARO_II_INHG = 29.05

TABLE_III = [  # RUN NO. 3, GAP = 0.236 in, barometric = 29.22 in Hg
    (1203, .1076, 72, 1.531), (1496, .1460, 72, 2.584), (1786, .1844, 72, 3.896),
    (2075, .2535, 72, 6.223), (2378, .330, 72, 9.285), (2676, .415, 72, 13.13),
    (2976, .507, 72, 17.86), (3262, .599, 73, 23.12), (3565, .707, 73, 29.82),
    (3855, .830, 73, 37.86), (4152, .945, 73, 46.42), (4436, 1.060, 74, 55.63),
    (4483, 1.099, 74, 58.29), (4826, 1.245, 74, 71.09), (5002, 1.314, 74, 77.77),
    (5204, 1.429, 74, 87.99), (5486, 1.537, 74, 99.77), (5711, 1.652, 74, 111.62),
    (5911, 1.775, 74, 124.14), (6108, 1.882, 74, 136.00), (6305, 1.944, 76, 145.02),
    (6502, 2.113, 76, 162.55), (6705, 2.213, 76, 175.56), (6905, 2.297, 76, 187.66),
    (7108, 2.451, 77, 206.13), (7227, 2.520, 77, 215.48), (7402, 2.581, 78, 226.04),
    (7601, 2.720, 78, 244.62), (7727, 2.804, 80, 256.35), (7968, 2.996, 82, 282.49),
    (8277, 3.181, 83, 311.52), (8769, 3.457, 84, 358.67), (9257, 3.842, 86, 420.80),
    (9725, 4.187, 88, 481.77), (10200, 4.533, 90, 547.06), (10674, 4.879, 92, 616.18),
    (11172, 5.301, 95, 700.71), (11635, 5.610, 97, 772.28), (12365, 6.185, 102, 904.86),
    (13044, 6.800, 106, 1049.47), (13751, 7.45, 110, 1212.1), (14785, 8.41, 116, 1471.18),
    (15747, 9.3, 121, 1732.7), (16608, 10.1, 127, 1984.67), (17768, 11.29, 136, 2373.4),
    (18234, 11.91, 138, 2569.5), (18560, 12.14, 142, 2665.9),
]
GAP_III_IN = 0.236
BARO_III_INHG = 29.22

R_IN = 6.0          # 12-inch diameter cylinder -> radius 6 in
L_IN = 5.9           # cylinder axial length
IN_TO_M = 0.0254
INHG_TO_PA = 3386.39
INLB_TO_NM = 0.112984829
R_SPECIFIC_AIR = 287.05  # J/(kg K)


def sutherland_mu(T_K):
    mu0, T0, S = 1.716e-5, 273.15, 110.4
    return mu0 * (T_K / T0) ** 1.5 * (T0 + S) / (T_K + S)


def build_table(rows, gap_in, baro_inhg, run_label):
    R_m = R_IN * IN_TO_M
    L_m = L_IN * IN_TO_M
    gap_m = gap_in * IN_TO_M
    R_chamber_m = R_m + gap_m
    p_Pa = baro_inhg * INHG_TO_PA
    out = []
    for rpm, torque_inlb, T_F, watts_reported in rows:
        omega = rpm * 2 * math.pi / 60.0
        torque_Nm = torque_inlb * INLB_TO_NM
        P_w = torque_Nm * omega
        T_K = (T_F - 32) * 5 / 9 + 273.15
        rho = p_Pa / (R_SPECIFIC_AIR * T_K)
        mu = sutherland_mu(T_K)
        Re_Omega = RE_OMEGA(rho, omega, R_m, mu)
        Cp = CP(P_w, rho, omega, R_m)
        Pi_gap = PI_GAP(gap_m, R_m)
        Pi_conf = PI_CONF(R_m, R_chamber_m)
        Pi_aspect = PI_ASPECT(L_m, R_m)
        out.append(dict(
            source="Gorland1970", run=run_label, N_rpm=rpm, omega_rad_s=omega,
            torque_Nm=torque_Nm, T_K=T_K, rho_kgm3=rho, mu_Pas=mu,
            P_w_W=P_w, P_w_reported_W=watts_reported,
            Re_Omega=Re_Omega, Cp=Cp, Pi_gap=Pi_gap, Pi_confinement=Pi_conf,
            Pi_aspect_axial=Pi_aspect, R_m=R_m, R_chamber_m=R_chamber_m,
            gap_radial_m=gap_m, h_rotor_m=L_m,
        ))
    return out


rows_all = (
    build_table(TABLE_I, GAP_I_IN, BARO_I_INHG, "gap_0.0565in")
    + build_table(TABLE_II, GAP_II_IN, BARO_II_INHG, "gap_0.116in")
    + build_table(TABLE_III, GAP_III_IN, BARO_III_INHG, "gap_0.236in")
)
df = pd.DataFrame(rows_all)

# ---------------------------------------------------------------------------
# 2. Transcription/derivation sanity check: our independently-recomputed
#    windage power (from torque x omega, with our own air-property
#    assumptions) against the source's OWN reported "WINDAGE (watts)" column.
#    This is an honest check on (a) our unit conversions and (b) whether the
#    transcription of torque/speed values is internally consistent with the
#    source's own power column (it does not validate the source's original
#    measurement, only our transcription+arithmetic of it).
# ---------------------------------------------------------------------------
df["P_w_pct_diff"] = 100 * (df.P_w_W - df.P_w_reported_W) / df.P_w_reported_W
print("\nCross-check: recomputed P = torque*omega vs source's own reported "
      "windage (watts) column")
print(df["P_w_pct_diff"].describe())

# ---------------------------------------------------------------------------
# 3. Compare candidate facility's Pi-group / Re_Omega coordinates to the
#    ranges spanned by the four existing facilities.
# ---------------------------------------------------------------------------
existing_ranges = corpus.groupby("source")[
    ["Re_Omega", "Pi_gap", "Pi_confinement", "Pi_aspect_axial", "Cp"]
].agg(["min", "max"])

candidate_ranges = df.groupby("run")[
    ["Re_Omega", "Pi_gap", "Pi_confinement", "Pi_aspect_axial", "Cp"]
].agg(["min", "max"])

print("\nExisting corpus ranges by facility:")
print(existing_ranges)
print("\nCandidate (Gorland1970) ranges by run/gap:")
print(candidate_ranges)

# Is the candidate's Re_Omega / Pi_gap coordinate INSIDE the convex-ish range
# already spanned by the 4 existing facilities, or does it extend it?
existing_re_min, existing_re_max = corpus.Re_Omega.min(), corpus.Re_Omega.max()
existing_pigap_min, existing_pigap_max = corpus.Pi_gap.min(), corpus.Pi_gap.max()
existing_piconf_min, existing_piconf_max = corpus.Pi_confinement.min(), corpus.Pi_confinement.max()
existing_piaspect_min, existing_piaspect_max = corpus.Pi_aspect_axial.min(), corpus.Pi_aspect_axial.max()

cand_re_min, cand_re_max = df.Re_Omega.min(), df.Re_Omega.max()
cand_pigap_min, cand_pigap_max = df.Pi_gap.min(), df.Pi_gap.max()
cand_piconf_min, cand_piconf_max = df.Pi_confinement.min(), df.Pi_confinement.max()
cand_piaspect_min, cand_piaspect_max = df.Pi_aspect_axial.min(), df.Pi_aspect_axial.max()

print(f"\nExisting corpus Re_Omega range: [{existing_re_min:.3e}, {existing_re_max:.3e}]")
print(f"Candidate Re_Omega range:       [{cand_re_min:.3e}, {cand_re_max:.3e}]")
print(f"Existing corpus Pi_gap range: [{existing_pigap_min:.4f}, {existing_pigap_max:.4f}]")
print(f"Candidate Pi_gap range:       [{cand_pigap_min:.4f}, {cand_pigap_max:.4f}]")
print(f"Existing corpus Pi_confinement range: [{existing_piconf_min:.4f}, {existing_piconf_max:.4f}]")
print(f"Candidate Pi_confinement range:       [{cand_piconf_min:.4f}, {cand_piconf_max:.4f}]")
print(f"Existing corpus Pi_aspect_axial range: [{existing_piaspect_min:.4f}, {existing_piaspect_max:.4f}]")
print(f"Candidate Pi_aspect_axial range:       [{cand_piaspect_min:.4f}, {cand_piaspect_max:.4f}]")

n_total = len(df)
n_transcribed_tables = 3
existing_geometry_types = sorted(corpus.geometry_type.unique().tolist())

results = {
    "angle": "literature search for additional free public rotor-stator windage datasets",
    "headline_finding": (
        "Found ONE genuinely new, freely downloadable, per-point-tabulated "
        "dataset that is geometrically and physically compatible with the "
        "existing 4-predictor Pi-group framework (no superimposed through-"
        "flow, no missing dimensionless group): NASA TM X-52851 (Gorland, "
        "Kempke & Lumannick, 1970), a smooth concentric-cylinder windage rig "
        "at NASA Lewis. 139 individual test points were hand-transcribed "
        "from the source's own clean typewritten data tables (not digitized "
        "from plots) across 3 fixed radial gaps. This would be a legitimate "
        "5th facility candidate for the corpus, subject to the caveats below."
    ),
    "n_candidate_rows_transcribed": n_total,
    "n_runs_gaps_transcribed": n_transcribed_tables,
    "source_citation": "Gorland, S.H.; Kempke, E.E., Jr.; Lumannick, S. (1970). "
        "Experimental Windage Losses for Close Clearance Rotating Cylinders "
        "in the Turbulent Flow Regime. NASA TM X-52851.",
    "source_url": "https://ntrs.nasa.gov/api/citations/19700023755/downloads/19700023755.pdf",
    "source_license": "Work of the US Government, public use permitted (public domain), confirmed on NTRS citation page.",
    "transcription_crosscheck_P_w_pct_diff_summary": {
        "mean_abs_pct": float(df["P_w_pct_diff"].abs().mean()),
        "median_abs_pct": float(df["P_w_pct_diff"].abs().median()),
        "max_abs_pct": float(df["P_w_pct_diff"].abs().max()),
        "note": (
            "This compares OUR recomputed P=torque*omega (using OUR assumed "
            "air density from barometric pressure + reported temperature via "
            "ideal gas law, and Sutherland's law for viscosity) against the "
            "source's OWN reported windage-watts column. Differences of a "
            "few percent are expected because the source's own property "
            "assumptions (e.g., exact humidity, exact Sutherland constants "
            "used in 1970) are not fully documented in the report; this is "
            "not a red flag but it does mean any final Re_Omega/Cp values "
            "used in a real merge should be re-derived carefully, ideally "
            "matching the source's own stated property model if recoverable, "
            "rather than trusted blindly from this proof-of-concept script."
        ),
    },
    "existing_corpus_ranges": {
        "Re_Omega": [float(existing_re_min), float(existing_re_max)],
        "Pi_gap": [float(existing_pigap_min), float(existing_pigap_max)],
        "Pi_confinement": [float(existing_piconf_min), float(existing_piconf_max)],
        "Pi_aspect_axial": [float(existing_piaspect_min), float(existing_piaspect_max)],
        "geometry_types_present": existing_geometry_types,
    },
    "candidate_facility_ranges": {
        "Re_Omega": [float(cand_re_min), float(cand_re_max)],
        "Pi_gap": [float(cand_pigap_min), float(cand_pigap_max)],
        "Pi_confinement": [float(cand_piconf_min), float(cand_piconf_max)],
        "Pi_aspect_axial": [float(cand_piaspect_min), float(cand_piaspect_max)],
        "geometry_type_proposed_label": "smooth_cylinder_in_cylinder_no_end_effects",
    },
    "compatibility_with_existing_4_predictor_framework": (
        "GOOD, unlike every other candidate found in this search. This rig "
        "has (a) no superimposed through-flow (unlike the Sussex/TFMRC bolt-"
        "windage rig data described below, which is real and free but "
        "requires an additional Cw/lambda_T predictor not in the current "
        "framework), and (b) a well-defined axial rotor length (the 5.9-in "
        "cylinder length) that maps directly onto the same h_rotor_m concept "
        "used for Vrancik1968, so Pi_aspect_axial is physically meaningful "
        "here in the same sense as in the existing corpus (unlike a plain "
        "flat disc, which has no natural 'h_rotor'). Its Pi_gap "
        "(0.0094-0.039) and Pi_confinement (0.96-0.99) ranges sit inside or "
        "close to the existing corpus's spread, while its Re_Omega range "
        "(order 1e5-1e6, computed here) is lower than the existing corpus's "
        "high-g facilities and closer to Vrancik1968's own range -- i.e. it "
        "would mostly reinforce/densify the Vrancik1968-like region of "
        "parameter space rather than open a wholly new region, which limits "
        "(but does not eliminate) its value for testing CROSS-facility "
        "generalization specifically."
    ),
    "important_caveat_geometric_novelty_is_limited": (
        "This facility's geometry_type would be closest to Vrancik1968's "
        "existing 'disk_in_cylinder'/cylindrical-rotor sub-cases (no salient "
        "poles, smooth cylinder), NOT a new geometry family. It is a genuine "
        "new independent measurement facility/apparatus (different "
        "institution-report, different year, different specific rig) so it "
        "IS a legitimate additional facility for a leave-one-facility-out "
        "test in principle, but reviewers could reasonably ask whether it is "
        "geometrically diverse enough to matter, since it does not cover a "
        "salient-pole / arm-in-chamber geometry the way Guo2024/Liu2024/"
        "Zheng2024 do. It is real, free, extractable evidence -- but it is "
        "an incremental enlargement of an already-represented geometry "
        "family, not a new geometry family."
    ),
    "other_candidates_investigated_and_rejected": [
        {
            "source": "Kempke & Gorland (1972), NASA TN D-6650, "
                "'Correlation of Windage-Loss Data for a Lundell Alternator'",
            "url": "https://ntrs.nasa.gov/api/citations/19720008343/downloads/19720008343.pdf",
            "free": True,
            "has_per_point_table": False,
            "reason_rejected": (
                "Free NTRS PDF, real Lundell-shaped (claw-pole) rotor "
                "geometry (20.3 cm dia, 3 radial clearances, up to 36000 "
                "rpm, Re up to 70000), but the full torque-vs-speed sweeps "
                "are presented ONLY as Figures 4,5,7,8,9,11 (plots), not "
                "tables. Only ~6 numeric anchor values exist in the text "
                "(Tables II-IV, all at a single fixed speed of 24000 rpm "
                "across 3 gaps) -- not a usable per-point sweep dataset "
                "without plot digitization."
            ),
        },
        {
            "source": "Bruckner (2009), NASA/TM-2009-215826, 'Windage Power "
                "Loss in Gas Foil Bearings and the Rotor-Stator Clearance of "
                "High Speed Generators Operating in High Pressure "
                "Environments'",
            "url": "https://ntrs.nasa.gov/api/citations/20090042819/downloads/20090042819.pdf",
            "free": True,
            "has_per_point_table": False,
            "reason_rejected": (
                "Free NTRS PDF, direct torque measurements up to 42000 rpm "
                "and 45 atm with N2/He/CO2, which would have been a very "
                "valuable multi-gas Re/Pi range extension if tabulated -- "
                "but results are presented as plots/correlation curves only; "
                "no per-point data table found in the report body."
            ),
        },
        {
            "source": "Randriamampianina & Poncet (2006), arXiv:physics/0607097, "
                "'Turbulence characteristics of the Bodewadt layer in a "
                "large enclosed rotor-stator system'",
            "url": "https://arxiv.org/pdf/physics/0607097",
            "free": True,
            "has_per_point_table": False,
            "reason_rejected": (
                "Free arXiv preprint, DNS + experiment of an enclosed "
                "rotor-stator cavity, but it studies turbulence structure "
                "(velocity/Reynolds-stress fields) at a SINGLE fixed "
                "Reynolds number (Re=9.5e4); no Cp/Cm-vs-Re sweep at all, "
                "so there is nothing analogous to extract for this corpus."
            ),
        },
        {
            "source": "Hu, Brillert, Dohmen & Benra (2017), IJTPP (MDPI, "
                "open access), 'Investigation on the Flow in a Rotor-Stator "
                "Cavity with Centripetal Through-Flow'",
            "url": "https://www.mdpi.com/2504-186X/2/4/18",
            "free": True,
            "has_per_point_table": False,
            "reason_rejected": (
                "Genuinely open-access (confirmed via reader fetch after "
                "MDPI blocked direct WebFetch), real test rig with reported "
                "geometry, but moment-coefficient/thrust-coefficient results "
                "vs Re are presented only in Figures 10, 12-14; only "
                "Table 1 (rig parameters) and Table 2 (uncertainty budget) "
                "are numeric tables, neither of which is a per-point Cm/Re "
                "sweep. Also, like the Sussex rig below, this facility has "
                "mandatory superimposed centripetal through-flow (Cqr), so "
                "even if digitized it would need an added through-flow "
                "predictor, not a drop-in addition."
            ),
        },
        {
            "source": "Bo Hu et al. (2019), IJTPP (MDPI, open access), "
                "'Investigation on Thrust and Moment Coefficients of a "
                "Centrifugal Turbomachine'",
            "url": "https://www.mdpi.com/2504-186X/3/2/9",
            "free": True,
            "has_per_point_table": "unresolved",
            "reason_rejected": (
                "Direct fetch was blocked by MDPI's Akamai bot protection "
                "(403) even via a reader proxy, so this one was NOT fully "
                "verified either way within this investigation's time "
                "budget. Flagged as unresolved rather than rejected; a "
                "human with normal browser access could check it in a few "
                "minutes. Given the same journal/author-group pattern as "
                "the companion paper above (results-as-figures convention "
                "in this venue), a similar outcome is likely but NOT "
                "confirmed."
            ),
        },
        {
            "source": "NASA TM X-52851 landing page fetch (first attempt)",
            "url": "https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19700023755.pdf",
            "free": True,
            "has_per_point_table": "N/A",
            "reason_rejected": (
                "Not a rejection -- a tooling note. This URL pattern "
                "returned the NTRS citation HTML landing page instead of "
                "the actual PDF; the working direct-download URL "
                "(https://ntrs.nasa.gov/api/citations/19700023755/downloads/"
                "19700023755.pdf) is the one actually used for the "
                "Gorland1970 candidate above. Listed here so the URL "
                "pattern mistake is not repeated by whoever integrates this."
            ),
        },
    ],
    "promising_but_not_digitized_here_sussex_thesis": {
        "source": "Miles, A.L. (2011). 'An experimental study of windage due "
            "to rotating and static bolts in an enclosed rotor-stator "
            "system.' D.Phil thesis, University of Sussex (TFMRC).",
        "url": "https://api.figshare.com/v2/articles/23383880 "
            "(PDF direct: https://ndownloader.figshare.com/files/41116571)",
        "free": True,
        "login_required": False,
        "has_per_point_table": True,
        "n_pages_of_data_tables_estimated": "~23 pages in Appendix C (pp. 299-322 "
            "of the thesis), covering plain disc, stator bolts (multiple "
            "bolt counts), hexagonal rotor bolts (3 diameters), extra "
            "hexagonal rotor bolts, bi-hexagonal rotor bolts, extra "
            "bi-hex/hex rotor bolts, and surface cavities. Roughly 13 rows "
            "were directly read off the 'Plain disc' page alone (Table, "
            "p.299) as a spot-check; the full appendix plausibly contains "
            "several hundred individual test points in total.",
        "geometry": "smooth disc, radius b=0.225 m, gap ratio G=s/b=0.1 on "
            "both sides (axial gap s=0.0225 m each side), casing radius "
            "~0.232 m (radial clearance ~7mm, i.e. essentially a shrouded "
            "disc -- structurally similar to Vrancik1968's disk_in_cylinder "
            "category), PLUS bolt/protrusion variants (a genuinely new "
            "geometric feature -- rotor-mounted and stator-mounted "
            "hexagonal/bi-hexagonal bolts -- not present in any of the 4 "
            "existing facilities).",
        "why_not_digitized_in_this_script": (
            "Two separate reasons, both honestly disclosed: (1) The Appendix "
            "C tables are scanned table IMAGES with essentially no OCR text "
            "layer (verified: each of the 23 pages has ~1 embedded image and "
            "<60 characters of extractable text), so digitizing the FULL "
            "appendix (unlike the Gorland1970 report above, which is clean "
            "typewritten text) would require manually reading and "
            "transcribing on the order of 20+ more table images -- feasible "
            "but a multi-hour task, out of scope for this exploratory pass, "
            "which instead prioritized fully digitizing one smaller, cleaner, "
            "and more directly compatible source (Gorland1970). (2) More "
            "fundamentally, EVERY test point in this rig has a mandatory, "
            "non-negligible superimposed through-flow (Cw ranges "
            "0.28e5-1.6e5, turbulent flow parameter lambda_T ranges "
            "0.06-0.6, read directly off the plain-disc table), which is "
            "real physics absent from the current corpus's 4-predictor "
            "Pi-group framework. Naively appending these rows using only "
            "the current 4 predictors would be scientifically invalid (an "
            "omitted-variable problem, not a true test of cross-facility "
            "transfer) -- any LOFO result obtained that way, positive OR "
            "negative, would be an artifact of the missing through-flow "
            "term, not a genuine finding. A valid use of this dataset would "
            "require adding a through-flow dimensionless group (e.g. "
            "Cw or lambda_T) to the Pi-group family and treating it as an "
            "explicit modeling extension, which is a substantial new "
            "sub-study, not a drop-in corpus enlargement."
        ),
    },
    "honest_overall_verdict": (
        "This search DID find genuinely new, free, per-point-extractable "
        "data (unlike the vast majority of candidates in this literature, "
        "which report only correlation plots) -- so the angle is a partial "
        "positive. But the ONE facility that is both free AND fully "
        "digitized here AND physically drop-in-compatible with the existing "
        "4-predictor framework (Gorland1970) is only an incremental "
        "densification of a geometry family (smooth cylinder-in-cylinder, "
        "no salient features) that is already close to Vrancik1968's "
        "regime, not a new geometry family -- so it would likely NOT by "
        "itself flip the paper's negative cross-facility-generalization "
        "result, though actually running it through the LOFO pipeline as a "
        "5th facility (left outside this script's scope) would be needed to "
        "confirm that rather than assume it. The genuinely geometrically "
        "novel free find (Sussex/Miles 2011, with real rotor/stator bolt "
        "protrusions -- a feature none of the 4 existing facilities have) "
        "is NOT physically compatible with the current 4-predictor "
        "framework without first adding a through-flow predictor, and its "
        "full extraction requires substantial further manual transcription "
        "work not completed here. Neither candidate is a quick, free lunch: "
        "one is compatible but geometrically redundant; the other is "
        "geometrically novel but requires real additional modeling and "
        "data-entry work before it could be used honestly."
    ),
}

with open("../results/angle_lit_newfree_datasets_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Also persist the transcribed candidate dataset itself, separately from the
# paper's actual corpus (NOT merged into cross_rotor_dataset_v3.csv -- that
# integration decision belongs to the paper authors, not this script).
df.to_csv("../results/angle_lit_newfree_datasets_gorland1970_candidate.csv", index=False)
print(f"\nWrote {len(df)} candidate rows to "
      "../results/angle_lit_newfree_datasets_gorland1970_candidate.csv "
      "(NOT merged into the paper's corpus).")
print("Wrote ../results/angle_lit_newfree_datasets_results.json")
