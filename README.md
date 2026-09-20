# Cross-facility windage power corpus, analysis code and results

Replication package for the manuscript *"Detecting and Diagnosing
Facility Confounding in Empirical Scaling Laws: A Cross-Facility
Validation Protocol Applied to Windage Power in Rotor-Stator Systems"*
by Jesús Gil Ruiz (Science and Aerospace Department, Universidad Europea
de Madrid), Yudith Cardinale and David Ariza Ruiz (Universidad
Internacional de Valencia).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22862639.svg)](https://doi.org/10.5281/zenodo.22862639)

This deposit contains **data, code, results and figures only**. It does
not contain the manuscript text, and is not a preprint.

## Contents

| Directory | Contents |
|---|---|
| `data/` | `cross_rotor_dataset_v3.csv` — the full 114-row corpus, one row per data point |
| `code/` | 59 analysis scripts (structure search, LOFO cross-validation, facility-cluster bootstrap, robustness checks, the 16-method learning sweep, the identifiability index, the confounding-versus-artifact diagnostic, and figure generation) |
| `results/` | 49 JSON result files, one per analysis, each written by the correspondingly named script, plus 5 Markdown/CSV notes for the literature-search analyses that produce no numeric fit |
| `figures/` | The 7 figures appearing in the manuscript |

## The dataset

`cross_rotor_dataset_v3.csv` has 114 rows from four independently built
facilities. Beyond the physical quantities, four columns carry
provenance information that the manuscript's Appendix A documents in
full:

- `source` — which of the four publications the row comes from
  (Guo2024 n=45, Vrancik1968 n=41, Liu2024 n=20, Zheng2024 n=8).
- `data_origin` — how the value was extracted: `digitized_from_figure`
  (93), `table_extraction` (8), `cfd_validated` (8),
  `computed_from_fitted_constant` (5).
- `geom_confidence` — how well the source documents the chamber
  geometry needed for the dimensionless groups (`high` for
  Vrancik1968, `medium` for Guo2024, `low` for Liu2024 and Zheng2024).
- `error_pct` — the per-point uncertainty reported by the source. This
  is not decorative: the manuscript's diagnostic section shows the
  unequal distribution of this quantity across facilities is what
  accounts for the collapse magnitude in three of four facilities.

Note that extraction route and physical nature are independent. All 45
Guo2024 rows are CFD outputs regardless of whether they were digitized
from a figure or read from a table, because that source study is
entirely computational. The 64/45/5 physical/CFD/reconstructed split
quoted in the manuscript abstract follows the physical axis.

## Known uncorrected defects

Four defects were identified and are **deliberately left uncorrected**
in this dataset, because correcting any of them requires source
information not recoverable from the published papers. All four are
quantified in Appendix A of the manuscript. In every case the
correction moves the paper's headline result further in the direction
it already reports, never back towards a more favourable one:

1. Eight Guo2024 "T2" rows appear misattributed to the CHIEF machine
   when their speeds match that paper's separate ZJU400 validation
   study. Excluding them moves pooled LOFO R²(log) from −0.885 to
   −6.811.
2. Those same 8 rows use a whole-machine power basis while the other 37
   Guo2024 rows use per-arm torque × ω. Correcting the scale gives
   pooled R²(log) between −10.4 and −24.2.
3. `gap_radial_m` is frozen at 0.5 m across the 7-row Guo2024 "F7"
   radius sweep, inconsistent with the paper's own Π_gap definition.
4. Liu2024 (20 rows, 17.5% of the corpus) is a preprint that had not
   completed peer review at the time of writing.

## Reproducing

Scripts are standalone Python 3. Each resolves its paths relative to
its own location, so they run unchanged from any working directory. Two are worth running first:

```
python3 code/verify_cp_equals_cm.py             # dataset self-consistency
python3 code/check_manuscript_numbers_sync.py path/to/manuscript.tex
```

The second cross-checks the 62 headline numeric claims of the manuscript
against the JSON file that produced each one and reports an exact match.
It is the fastest way to confirm that the reported numbers and the
deposited results agree. The manuscript source is not redistributed
here, so the script takes its path as an argument; run without one it
lists the 62 facts and the result files they come from.

Requirements: `numpy`, `pandas`, `scipy`, `matplotlib`. Individual
learning-method scripts additionally require `scikit-learn`, `torch`,
`gplearn`, `shap` or `tabpfn` depending on the method; each script
states its own dependency at the top and fails cleanly if it is absent.

## License

CC BY 4.0. The underlying measurements remain the intellectual property
of the cited source publications; this deposit contains a digitized,
documented compilation of published values together with original
analysis code.

## Citing this deposit

Cite the manuscript and this deposit. The concept DOI below always
resolves to the latest version; the version DOI pins the exact snapshot
used by the manuscript.

- Concept DOI (all versions): `10.5281/zenodo.22862639`

Each release also gets its own version DOI, listed on the concept DOI's
Zenodo page; the manuscript cites the exact version it used.

Author metadata for the archived record comes from `CITATION.cff`.
