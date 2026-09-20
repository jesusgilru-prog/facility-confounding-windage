# Angle: Literature search — do published multi-facility windage/disc-friction studies achieve genuine cross-facility generalization, and if so, how?

Slug: `lit_geomfamily`. Code: `code/angle_lit_geomfamily.py`. Results: `results/angle_lit_geomfamily_results.json`.

## Question
Search for real published multi-facility rotor-stator / disc-friction / windage /
turbomachinery scaling-law studies that DID achieve genuine cross-facility (or
cross-rig) predictive generalization, identify what made it work, and check
honestly whether those conditions are achievable with the current corpus
(114 rows, 4 facilities) or require new data.

## What I found in the literature (real sources, checked this session)

1. **Daily, J.W. & Nece, R.E. (1960)**, *Chamber dimension effects on induced
   flow and frictional resistance of enclosed rotating disks*, J. Basic
   Engineering 82(1):217-230, doi:10.1115/1.3662532. (Already cited in this
   paper as `daily1960`.) The classic 4-regime (I-IV) moment-coefficient map
   for a **plain, smooth disk** enclosed in a cylindrical cavity, parameterized
   by rotational Reynolds number and a single gap ratio G. This is the closest
   thing to a "universal" windage law in the field, and subsequent independent
   groups, on different rigs, over decades, reproduced its regime boundaries
   and turbulent exponents within engineering tolerance.

2. **Coren, D., Childs, P.R.N. & Long, C.A. (2009)**, *Windage sources in
   smooth-walled rotating disc systems*, Proc. IMechE Part C 223(4):873-888,
   doi:10.1243/09544062JMES1260. New rig, extends the smooth-disc correlation
   to Re_psi up to 1e7, and explicitly reports agreement with prior literature
   data from **other** rigs/labs "within 10 per cent" — i.e. a genuine,
   quantified cross-facility check that passed, again for the smooth,
   unbladed, unbolted disk family.

3. **Long, C.A. et al. (2014)**, *Windage Measurements in a Rotor-Stator
   System With Superimposed Cooling and Rotor-Mounted Protrusions*, J. Eng.
   Gas Turbines Power 136(4):042505 (conference version GT2012-69385). This is
   the single most useful reference for our paper: **the same research group,
   on the same rig**, tested a plain disk and a disk with 18 bolt-like
   protrusions. The plain-disk correlation did **not** transfer to the
   protrusion geometry — the excess windage is attributed to an added form-drag
   term, and a new empirical correlation with an explicit bolt/protrusion
   shape factor was required (Long et al. give Cm as a function of Re, a flow
   coefficient Cw, AND a bolt-shape factor). This is a **literature-internal
   negative control**: even holding the facility/rig fixed, changing the
   rotor's discrete geometric features (topology) breaks a previously-
   generalizing correlation.

4. **Da Soghe, R., Facchini, B., Innocenti, L. & Micio, M. (2011)**, *Analysis
   of Gas Turbine Rotating Cavities by a One-Dimensional Model: Definition of
   New Disk Friction Coefficient Correlations Set*, J. Turbomachinery
   133(2):021020, doi:10.1115/1.4000633. Achieves cross-case generality not by
   pooling raw heterogeneous geometries into one flat regression, but by
   decomposing the cavity into physically distinct flow zones (Ekman layers,
   core, shroud/rim-seal) with separate CFD-calibrated friction correlations
   per zone, combined in a 1D network model. Generalization is bought with
   more physics structure per geometric family, not by feeding a generic ML
   model raw dimensionless groups across families.

5. **Harmand, S., Pellé, J., Poncet, S. & Shevchuk, I.V. (2013)**, review, Int.
   J. Thermal Sciences 67:1-30, arXiv:1305.2882. (Already cited as
   `poncet2013`.) Confirms the Daily-Nece correlations have been reproduced
   across many independent studies, but the review itself organizes different
   cavity types (with through-flow, with impinging jets, with protrusions) as
   needing **separate** correlation sets — it never claims one law spans all
   topologies.

## The pattern across every success case

Every genuine cross-facility success we found shares the same recipe:
**hold the geometric topology fixed** (plain smooth disk in a cylindrical
cavity is the one topology with a 60+ year track record of transferring), and
let only continuous, smoothly-varying parameters (Re, gap ratio G, throughflow
coefficient Cw) differ across facilities. The instant the topology itself
changes — bolts, protrusions, blades, arms, salient poles — literature groups,
including ones using the *same rig*, needed a new geometry-specific term
(a shape factor, or a separate 1D-model zone), not a bigger/better-tuned
version of the same regression. We found no published case of a single flat
correlation (physics-based or ML) spanning genuinely different rotor
topologies with only 2-3 generic aspect-ratio Pi-groups.

## Honest check against THIS corpus (real numbers, computed this session)

The corpus's 4 "facilities" are perfectly confounded with 4 essentially
different geometric topologies (`enclosed_arm_in_chamber`, `arm_in_cylinder`,
and — within Vrancik1968 only — `disk_in_cylinder`, `salient_in_stator`,
`enclosed_salient_in_stator`). Vrancik1968 is the one facility that itself
contains **3 different geometry_type values on the same rig**, which lets us
test the literature's diagnosis (topology matters more than facility) without
any facility confound.

Result (`code/angle_lit_geomfamily.py`, n=41 Vrancik1968 rows):
- Fit log(Cp) ~ log(Re_Omega) (a single shared trend, slope = -0.44) across
  all of Vrancik1968.
- One-way ANOVA of the **residuals** by `geometry_type` (still within
  Vrancik1968, so facility is held constant): geometry_type alone explains
  **39.5%** of the residual variance (disk_in_cylinder: mean residual
  +0.065; enclosed_salient_in_stator: -0.140; salient_in_stator: +0.407 in
  log(Cp) units — a genuine, first-order, non-negligible systematic offset
  between topologies, of the same order of magnitude as the effect the
  16 previously-tried methods have been failing to close across facilities).
- The corpus also has no column encoding discrete-feature geometry (arm/pole
  count, protrusion frontal-area fraction, or a Long-et-al.-style shape
  factor) — confirmed by direct inspection of the column list. It only has
  continuous length-scale ratios (Pi_confinement, Pi_gap, Pi_aspect_axial),
  exactly the kind of information the literature's success cases show is
  *not* sufficient once topology varies.

## Verdict (honest, not oversold)

**Yes, genuine cross-facility generalization is a real, established thing in
this literature — but only under a condition this corpus violates by
construction.** The paper's negative result is not a failure of the 16 ML/
statistical methods tried; it is consistent with, and now has a literature-
grounded mechanistic explanation plus an internal (facility-controlled)
confirmatory check: **the corpus pools genuinely different rotor topologies
under too few generic Pi-groups, and geometric topology is a first-order,
non-negligible source of systematic Cp offset (~40% of residual variance)
even when facility is held fixed.**

This is a legitimate, citable, moderately novel contribution to add to the
paper's Related Work / Discussion: it reframes the negative result from "we
tried hard and nothing works" to "the literature's own success stories
predict exactly this failure mode, and we can show the mechanism directly in
our own data." It does **not**, however, turn the paper's headline into a
positive result under the project's own bar (pooled R2 substantially positive
AND every held-out facility R2 >= 0) — it explains the negative result, it
does not reverse it.

**What would actually be needed to fix it (new data, not reframing):**
1. Go back to the original sources (Vrancik 1968 NASA TN D-4849 in
   particular, since it already spans 3 topologies) and extract/digitize
   discrete geometric-feature descriptors (arm count, salient-pole count,
   frontal solidity / blockage area, a Long-et-al.-style shape factor) that
   are not currently in the CSV, then re-test whether adding them as explicit
   predictors closes some of the 39.5% within-Vrancik1968 gap and improves
   LOFO across the other three facilities.
2. Alternatively, honestly narrow the paper's scope: restrict any
   "cross-facility law" claim to the plain-disk subset only. This is
   scientifically honest but numerically hopeless here — collapsing to
   plain-disk points leaves only Vrancik1968's 12 `disk_in_cylinder` rows,
   i.e. a single facility, which cannot test cross-facility generalization at
   all (the same "geom_confidence is confounded with facility" trap the task
   brief already warns about, one level down).
3. Neither (1) nor (2) can be done with the current 114-row corpus as-is;
   both require new data collection/extraction, which is out of scope for a
   quick angle check and should be flagged as future work, not claimed as
   already achieved.

## Sources consulted this session (WebSearch/WebFetch)
- https://journals.sagepub.com/doi/10.1243/09544062JMES1260 (Coren, Childs, Long 2009 — fetched and read)
- https://asmedigitalcollection.asme.org/turbomachinery/article-abstract/133/2/021020/468953/ (Da Soghe et al. 2011 — abstract page blocked 403, bibliographic + content details corroborated via search snippets and citing literature)
- https://asmedigitalcollection.asme.org/gasturbinespower/article/136/4/042505/472316/ (Long et al. 2014 — content corroborated via search snippets, ResearchGate/Figshare mirrors, and the GT2012-69385 conference abstract)
- General searches on Daily-Nece regime-map cross-validation and windage/bolt/protrusion form-drag literature (see WebSearch queries used in this session).
