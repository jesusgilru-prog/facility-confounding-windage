# Angle: honest uncertainty (split conformal / CQR) under LOFO

Script: `code/angle_uncertainty_coverage.py`
Full numbers: `results/angle_uncertainty_coverage_results.json`

## Setup
Two independent, standard interval-construction methods, both strictly
LOFO (calibration data drawn only from the 3 training facilities, held-out
facility touched only at final scoring, 40 repeated random
proper-train/calibration splits averaged):

- **Method A**: OLS baseline (log Cp ~ features, standardized) + classic
  split conformal prediction (finite-sample-corrected residual quantile).
- **Method B**: GradientBoostingRegressor quantile regression (lower/upper
  quantile heads) + CQR calibration correction (Romano et al. 2019).

Two feature sets (RE_ONLY = log Re_Omega alone; FULL = + Pi_confinement,
Pi_gap, Pi_aspect_axial), two nominal levels (90%, 80%).

## Headline numbers (pooled across all 114 held-out points)

| Method | Features | Nominal | Pooled coverage | Gap |
|---|---|---|---|---|
| Split conformal (OLS) | RE_ONLY | 90% | 66.6% | **-23.4 pp** |
| Split conformal (OLS) | RE_ONLY | 80% | 62.2% | **-17.8 pp** |
| Split conformal (OLS) | FULL    | 90% | 16.3% | **-73.7 pp** |
| Split conformal (OLS) | FULL    | 80% | 10.1% | **-69.9 pp** |
| CQR (GBM quantile)    | RE_ONLY | 90% | 50.4% | **-39.6 pp** |
| CQR (GBM quantile)    | RE_ONLY | 80% | 31.8% | **-48.2 pp** |
| CQR (GBM quantile)    | FULL    | 90% | 34.9% | **-55.1 pp** |
| CQR (GBM quantile)    | FULL    | 80% | 19.4% | **-60.6 pp** |

Every combination undershoots its nominal target, in the worst cases by
~50-74 percentage points. There is no combination that clears the paper's
decision rule (pooled coverage within ~10pp of nominal AND no facility
catastrophically under).

## Per-facility pattern (best case: split conformal, RE_ONLY, 90% nominal)

| Facility (n) | Coverage | Interpretation |
|---|---|---|
| Guo2024 (45) | **17.8%** | catastrophic undercoverage — largest facility (39% of data) |
| Liu2024 (20) | 94.9% | near-nominal |
| Vrancik1968 (41) | 100% | over-covered (intervals too wide to be informative) |
| Zheng2024 (8) | 100% | over-covered |

Root cause traced directly (separate diagnostic, OLS bias check): the
RE_ONLY LOFO point-prediction bias (mean residual in log-space) is
**-1.31 for Guo2024** vs -0.14 / +0.47 / -0.10 for the other three. Split
conformal computes its quantile from calibration residuals drawn from the
*other* facilities, which have near-zero systematic bias among themselves;
that calibrated half-width (~0.66 in log space) is far smaller than
Guo2024's actual bias, so the interval systematically misses low. For the
FULL feature set the extrapolation is far worse still — e.g. leaving out
Vrancik1968, the OLS model (fit on the other 3 facilities in standardized
FULL-feature space) predicts log Cp in the range [17, 21] against a true
range of [-3.05, -1.26] (mean bias -21.7): a fitted-coefficient blow-up
from extrapolating standardized geometry ratios outside their training
range. No calibration-set-derived interval width can absorb a bias that
large, hence 0% coverage there.

Conversely, the two facilities that show ~100% coverage are not a success:
their apparent "good coverage" comes from an interval so wide (e.g.
pooled interval half-width of ~1.6 in log space for RE_ONLY/90%, i.e. a
multiplicative factor of roughly e^3.2 ≈ 25x from lower to upper bound)
that it is essentially uninformative, not because the model has correctly
identified those facilities as easy.

## Verdict

**Negative, and not a rescue of the paper's headline finding.** Neither
split conformal prediction nor CQR achieves honest cross-facility coverage
here. The failure mode is not random noise around the nominal level — it
mirrors, point for point, the same systematic per-facility bias structure
already documented for plain point-prediction LOFO (Guo2024 and Vrancik1968
FULL-feature extrapolation are the worst offenders in both the R2 results
and here). This is itself an informative, non-trivial addition to the
paper's negative-result story: it directly rules out the natural rebuttal
"maybe the point estimate is unreliable but a well-calibrated uncertainty
band would still be useful" — under strict LOFO, that band is *also*
unreliable, for the identical reason (each facility is a different
data-generating regime, so the standard interval-construction guarantee,
which assumes exchangeability between calibration and test data, does not
hold). No data leakage or cherry-picking was needed to get this negative
result — if anything the opposite effort was made (multiple feature sets,
methods, and nominal levels, all pointing the same direction).

This angle should be reported as a **confirmatory negative finding**
(strengthens the paper's core claim rather than opening a new positive
contribution): "the failure to generalize across facilities is not an
artifact of using point estimates; genuinely principled distribution-free
uncertainty quantification (split conformal, CQR) inherits the same
failure, with per-facility coverage gaps as large as 70+ percentage
points and a clear diagnosable cause (systematic, facility-specific bias
that calibration drawn from other facilities cannot see)."
