"""Diagnostic figure for the confounding-versus-artifact discriminator
(Section sec:cgfd / sec:results-cgfd).

For each of the four facilities, plots the two synthetic null distributions
of held-out LOFO R^2 -- pure collinearity artifact (no facility effect) and
genuine facility confounding (per-facility offset ~ N(0, tau2_hat)) -- under
homoscedastic noise, plus the heteroscedastic-confounding distribution that
uses this corpus's own reported per-point measurement uncertainty, and marks
where the single REAL observed value falls in each.

This makes visible what the prose reports numerically: Vrancik1968's real
value sits centrally in both homoscedastic nulls (uninformative), the other
three sit outside both (neither homoscedastic model is complete), and the
heteroscedastic extension brings three of the four inside its range.

Reads only from results/*.json (no recomputation), so the figure is
reproducible without rerunning the 500-replicate simulations.
No fabricated data.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
cgfd = json.loads((ROOT / "results" / "cgfd_test_results.json").read_text())
hetero = json.loads((ROOT / "results" / "cgfd_hetero_results.json").read_text())

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

FACILITIES = ["Guo2024", "Vrancik1968", "Liu2024", "Zheng2024"]
N_FACILITY = {"Guo2024": 45, "Vrancik1968": 41, "Liu2024": 20, "Zheng2024": 8}

C_COLL = "#4c72b0"   # collinearity-only
C_CONF = "#c44e52"   # genuine confounding (homoscedastic)
C_HET = "#55a868"    # genuine confounding (heteroscedastic, real error_pct)
C_REAL = "#1a1a1a"


def slog(x):
    """Signed log transform: these R^2 values span 0 to -600+, strongly
    left-skewed, so a linear axis is unreadable."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log10(1.0 + np.abs(x))


fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))

for ax, fac in zip(axes.ravel(), FACILITIES):
    coll = np.array(cgfd["raw_per_facility_collinearity"][fac])
    conf = np.array(cgfd["raw_per_facility_confounded"][fac])
    het = np.array(hetero["raw_per_facility"][fac])
    real = cgfd["single_observation_diagnostic"][fac]["real_r2"]

    # Use the TRUE min/max across all three distributions and the real value,
    # not a trimmed percentile: the panel titles assert whether the real value
    # falls inside the heteroscedastic range, so clipping the axis would make
    # that claim unverifiable from the figure itself.
    allv = np.concatenate([slog(coll), slog(conf), slog(het), [slog(real)]])
    lo, hi = allv.min(), allv.max()
    pad = 0.06 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 46)

    for data, color, label in ((coll, C_COLL, "collinearity only"),
                                (conf, C_CONF, "confounding (homosc.)"),
                                (het, C_HET, "confounding (heterosc.)")):
        ax.hist(slog(data), bins=bins, color=color, alpha=0.45,
                label=label, density=True, edgecolor="none")

    ax.axvline(slog(real), color=C_REAL, lw=1.8, zorder=5)
    ax.annotate(f"real\n{real:.2f}", xy=(slog(real), ax.get_ylim()[1] * 0.86),
                xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                color=C_REAL, va="top", ha="left", fontweight="bold")

    covered = hetero["per_facility"][fac]["real_within_synthetic_range"]
    mark = "real value within heterosc. range" if covered else "real value outside all ranges"
    ax.set_title(f"{fac} ($n={N_FACILITY[fac]}$): {mark}",
                 fontsize=8.5, pad=4)
    ax.set_yticks([])
    ax.tick_params(labelsize=7.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

for ax in axes[1]:
    ax.set_xlabel(r"held-out LOFO $R^2_{\log}$   (signed-log scale)", fontsize=8)

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, 1.015))
fig.tight_layout(rect=[0, 0, 1, 0.945])

OUT = ROOT / "figures" / "fig6_cgfd_diagnostic.png"
fig.savefig(OUT, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print(f"Saved {OUT}")
