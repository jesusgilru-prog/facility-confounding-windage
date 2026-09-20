# Angle: additional free/public rotor-stator windage datasets (literature search)

## Bottom line

The search DID find one genuinely new, free, fully per-point-tabulated dataset
(139 points, hand-transcribed and numerically cross-checked in
`code/angle_lit_newfree_datasets.py`), which is a real positive result for
this narrow angle — most of what exists in the literature is plot-only and
NOT usable without digitizing figures. But this specific find is only an
incremental densification of a geometry family the corpus already has
(Vrancik1968-like smooth cylinder, no salient poles), not a new geometry
family, and it has **not** been merged into `cross_rotor_dataset_v3.csv` or
run through the paper's LOFO pipeline — that would be a separate follow-up
decision. A second, geometrically genuinely novel free source was also found
(rotor/stator bolt protrusions) but is not physically compatible with the
current 4-predictor framework without adding a through-flow term, and its
full extraction requires substantial further manual work not done here.

## What was searched

NASA NTRS, arXiv, OSTI.gov, Zenodo, MDPI open-access journals (IJTPP,
Fluids), Google Scholar/general web search, and figshare-hosted UK
university thesis repositories (University of Sussex/TFMRC). Search terms
combined "rotor-stator", "windage", "disc friction", "moment coefficient",
"enclosed rotating cavity" with "dataset", "table", "csv", "thesis",
"NASA technical report".

## Positive find: NASA TM X-52851 (Gorland, Kempke & Lumannick, 1970)

*"Experimental Windage Losses for Close Clearance Rotating Cylinders in the
Turbulent Flow Regime."*
Free PDF: https://ntrs.nasa.gov/api/citations/19700023755/downloads/19700023755.pdf
(public domain, "Work of the US Gov, Public Use Permitted").

- Geometry: smooth, unslotted 12-in-diameter x 5.9-in-long cylinder inside a
  concentric stationary housing, 3 fixed radial gaps (0.0565, 0.116, 0.236
  in), end effects explicitly eliminated by the torque-balance design.
- Data: Tables I-III report SPEED (rpm), TORQUE (in-lb), REYNOLDS NUMBER,
  DRAG COEFFICIENT, TEMPERATURE (°F), WINDAGE (watts) for every individual
  test point — a clean typewritten table, not a plot, not a handwriting
  scan. 46+46+47 = **139 rows** transcribed directly from 200-dpi page
  renders (report pp. 20-25 / PDF pages 11-16).
- Independent verification done here: recomputed `P = torque x omega` from
  the transcribed torque/speed values and compared to the source's own
  reported windage-watts column. Median absolute difference ≈0.0009%, mean
  ≈-0.05%, only 2 of 139 points (both low-speed/low-torque) exceed 1%
  (max 4.65%) — strong evidence the transcription is accurate.
- Pi-group compatibility check (using the corpus's own verified formulas,
  `Pi_gap=gap_radial/R`, `Pi_confinement=R/R_chamber`,
  `Pi_aspect_axial=h_rotor/R`, `Re_Omega=rho*omega*R^2/mu`,
  `Cp=P/(0.5*rho*omega^3*R^5)`): candidate Pi_gap in [0.0094, 0.0393],
  Pi_confinement in [0.962, 0.991], Pi_aspect_axial = 0.983 (fixed, since L/R
  is fixed by the one rig), Cp in [0.0086, 0.057], Re_Omega in
  [3.7e4, 2.3e6] — all fall inside or adjacent to ranges already spanned by
  the existing corpus, mostly overlapping Vrancik1968's own range
  (Re_Omega 9.1e4-5.2e6, Cp 0.047-0.284). This is a genuinely compatible 5th
  facility from a physics/framework standpoint (unlike everything else found
  in this search) — but because it densifies an already-represented
  geometry/Re region rather than opening a new one, it is unlikely by itself
  to change whether cross-facility generalization succeeds; it would need to
  actually be run through the LOFO pipeline to know for sure.
- Caveat: air density/viscosity for Re_Omega/Cp were derived here from the
  report's own barometric pressure and per-point temperature via the ideal
  gas law + Sutherland's law — a reasonable standard approximation, but NOT
  necessarily identical to whatever property model the original 1970 authors
  used internally for their own "REYNOLDS NUMBER" and "DRAG COEFFICIENT"
  columns (which were not reverse-engineered here). A rigorous merge into
  the paper's corpus should re-derive this carefully.
- The transcribed data lives at
  `results/angle_lit_newfree_datasets_gorland1970_candidate.csv` as a
  standalone candidate file — it was deliberately NOT merged into
  `data/cross_rotor_dataset_v3.csv`, since integrating a new facility into
  the paper's actual corpus and reruning the 16-method LOFO benchmark is a
  substantive modeling decision for the paper authors, not something to do
  silently inside a literature-search script.

## Promising but NOT digitized here: A. Miles (2011) D.Phil thesis, U. Sussex

*"An experimental study of windage due to rotating and static bolts in an
enclosed rotor-stator system."* Free, no login, via figshare:
https://ndownloader.figshare.com/files/41116571 (341 pp., confirmed genuine
PDF, not a stub).

- Appendix C (~23 pages, thesis pp. 299-322) contains full test-matrix
  tables for: plain disc, stator bolts, hexagonal rotor bolts (3 diameters),
  extra hex rotor bolts, bi-hexagonal rotor bolts, extra bi-hex/hex rotor
  bolts, and surface cavities. A spot-check of the "plain disc" page alone
  yielded 13 clearly legible rows (measured torque, mass flow, omega, Re_phi,
  Cw, lambda_T, casing pressure); the full appendix plausibly holds several
  hundred points.
- Genuinely novel geometry feature versus the existing corpus: rotor- and
  stator-mounted bolt/protrusion features are not present in any of Guo2024,
  Vrancik1968, Liu2024, or Zheng2024.
- Two honest reasons this was NOT fully digitized in this pass:
  1. The Appendix C tables are scanned **table images** with almost no OCR
     text layer (verified: each page has 1 embedded image and <60 characters
     of extractable text) — unlike the clean typewritten Gorland1970 report,
     transcribing the full appendix would mean manually reading ~20+ more
     table images, a multi-hour task not attempted here.
  2. More fundamentally: every single test point in this rig has mandatory,
     non-negligible superimposed through-flow (Cw = 0.28-1.6 x 10^5,
     turbulent flow parameter lambda_T = 0.06-0.6, read directly off the
     plain-disc table). This is real physics the current 4-predictor
     Pi-group framework does not represent. Naively appending these rows
     under the existing 4 predictors would be an omitted-variable error, not
     a valid test — any LOFO result from doing so (positive or negative)
     would be an artifact of the missing through-flow term. Using this
     source honestly requires adding a through-flow dimensionless group to
     the framework first, which is a real modeling extension, not a
     drop-in corpus enlargement.

## Other candidates checked and rejected (plot-only, not usable as-is)

| Source | Free? | Per-point table? | Why rejected |
|---|---|---|---|
| Kempke & Gorland 1972, NASA TN D-6650, Lundell alternator | Yes (NTRS) | No | Torque-speed sweeps only in Figures 4/5/7/8/9/11; only ~6 anchor numbers in text tables at one fixed speed |
| Bruckner 2009, NASA/TM-2009-215826, gas foil bearing + generator rotor-stator | Yes (NTRS) | No | Results-as-plots only despite promising multi-gas/high-pressure test matrix |
| Randriamampianina & Poncet 2006, arXiv:physics/0607097, Bödewadt layer DNS | Yes (arXiv) | No | Single-Re turbulence structure study, not a Cp/Re sweep at all |
| Hu et al. 2017, MDPI IJTPP 2(4):18, rotor-stator cavity w/ through-flow | Yes (open access) | No | Cm/thrust results only in Figures 10,12-14; also has mandatory through-flow, same compatibility issue as the Sussex thesis |
| Hu et al. 2019(?), MDPI IJTPP 3(2):9, thrust/moment of centrifugal turbomachine | Yes (open access) | Unresolved | Blocked by Akamai bot protection even via reader proxy; not verified either way within time budget — worth a follow-up check with normal browser access |

Also confirmed (not a new source, a provenance check): NASA TN D-4849
(Vrancik, 1968), "Prediction of Windage Power Loss in Alternators," is
indeed the exact source already in the corpus as "Vrancik1968" — title,
author, and abstract match exactly.

## Recommendation for the paper

Do not claim this angle "solves" or meaningfully changes the negative
cross-facility-generalization headline. Do report it honestly as: (a)
evidence the authors searched the literature for additional free data
in good faith, (b) a concrete, reproducible, ready-to-use candidate 5th
facility (Gorland1970, 139 points, code+CSV available) for future corpus
expansion, explicitly flagged as not yet integrated or tested, and (c) an
honest note that the one genuinely novel-geometry free source found (Sussex
bolt-windage rig) is real but requires nontrivial additional modeling
(a through-flow predictor) and data-entry work before it could be used, so
it is left as future work rather than overclaimed as available data.
