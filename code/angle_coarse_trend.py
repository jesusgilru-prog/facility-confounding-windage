"""Angle: does a deliberately COARSER target transfer better than exact
Cp regression, for the 'Windage Power' paper (2026-08-18)?

Two sub-questions, both under strict leave-one-facility-out (LOFO):

(a) MONOTONICITY / RANK-ORDER: for pairs of points drawn from the SAME
    held-out facility, can a model trained on the other 3 facilities
    correctly predict the SIGN of the Cp difference (i.e. does Cp go up
    or down between the two points)? This is a pairwise concordance /
    C-index test, evaluated per held-out facility and pooled.

(b) COARSE BINNING: label every point's Cp as low/medium/high TERCILE,
    computed SEPARATELY WITHIN EACH FACILITY (using that facility's own
    rank distribution, not a pooled threshold -- this avoids the trivial
    "scale leakage" failure mode where a global threshold just recovers
    facility identity because facilities occupy disjoint Cp ranges).
    Train a classifier via LOFO on the training facilities' own
    per-facility tercile labels, and test cross-facility classification
    accuracy on the held-out facility's own per-facility tercile labels.

Honesty check built into the design (per task instructions): within a
single facility, geometry (all 4 Pi-groups) is fixed or near-fixed --
the corpus's own diagnostics (scaling_law_search.py, robustness_summary
_stats.json) already establish that the Pi-groups behave as near-
constant per-facility "dummies". That means within-facility variation in
Cp is driven almost entirely by variation in Re_Omega, and the Reynolds
exponent's SIGN is already known to be robustly negative (facility-
cluster bootstrap 95% CI = [-0.524, -0.203], excludes zero -- see
robustness_summary_stats.json). So BOTH sub-questions here are at real
risk of being nothing more than a restatement of that already-
established finding, dressed up as a classification/ranking task. This
script computes the actual numbers and explicitly checks that
possibility rather than assuming either outcome.

Baselines used (mandatory, not optional):
  - 50% random guessing for (a); 1/(observed class balance) for (b).
  - A trivial, ZERO-TRAINING "physics prior" rule for (a): predict
    sign(Cp_j - Cp_i) = -sign(Re_j - Re_i) for every pair, i.e. just
    assume the ALREADY-ESTABLISHED global negative-Re-exponent finding
    holds locally, with no model fit at all. If this untrained rule
    matches or beats the LOFO-trained models, the "trained model"
    result is not adding anything beyond the pre-existing finding.
  - The EMPIRICAL ceiling: the actual Kendall tau / Spearman rho between
    Re_Omega and Cp computed directly within each facility (no model
    involved) -- this upper-bounds what ANY Re-based rule can achieve
    for that facility, trained or not.

Decision rule (same honesty bar as the rest of the paper, adapted to a
classification/ranking framing per task instructions): a genuinely
positive result requires pooled accuracy substantially above baseline
AND every individual held-out facility beating baseline, AND the result
must beat the untrained physics-prior baseline by a non-trivial margin
(otherwise it is just a restatement of the known Re-exponent sign, not
a new predictive capability).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RNG_SEED = 12345
DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT_PATH = _ROOT + "/results/angle_coarse_trend_results.json"

df = pd.read_csv(DATA_PATH)
d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)

y = np.log(d["Cp"].values)
lRe = np.log(d["Re_Omega"].values)
lgap = np.log(d["Pi_gap"].values)
lconf = np.log(d["Pi_confinement"].values)
lasp = np.log(d["Pi_aspect_axial"].values)
Re_raw = d["Re_Omega"].values
Cp_raw = d["Cp"].values
sources = d["source"].values
facilities = sorted(set(sources))

FEATURES = {
    "RE_ONLY": np.column_stack([lRe]),
    "FULL": np.column_stack([lRe, lgap, lconf, lasp]),
}

results = {"n_total": int(n), "facilities": {f: int((sources == f).sum()) for f in facilities}}

# ============================================================
# 0. EMPIRICAL CEILING: raw within-facility Re-Cp monotonicity
# ============================================================
empirical_monotonicity = {}
for f in facilities:
    mask = sources == f
    re_f = Re_raw[mask]
    cp_f = Cp_raw[mask]
    tau, tau_p = kendalltau(re_f, cp_f)
    rho, rho_p = spearmanr(re_f, cp_f)
    # pairwise concordance of raw data with a "Cp decreases as Re increases" rule
    pairs = list(combinations(range(len(re_f)), 2))
    n_pairs = 0
    n_consistent_with_decreasing = 0
    for i, j in pairs:
        dre = re_f[j] - re_f[i]
        dcp = cp_f[j] - cp_f[i]
        if dre == 0 or dcp == 0:
            continue
        n_pairs += 1
        if np.sign(dcp) == -np.sign(dre):
            n_consistent_with_decreasing += 1
    empirical_monotonicity[f] = {
        "n": int(mask.sum()),
        "n_pairs_nontied": n_pairs,
        "kendall_tau_Re_Cp": float(tau),
        "kendall_tau_p": float(tau_p),
        "spearman_rho_Re_Cp": float(rho),
        "frac_pairs_consistent_with_Cp_decreasing_in_Re": (
            n_consistent_with_decreasing / n_pairs if n_pairs else None),
    }
results["empirical_within_facility_monotonicity"] = empirical_monotonicity

# ============================================================
# (a) PAIRWISE RANK-ORDER / MONOTONICITY UNDER LOFO
# ============================================================
def fit_ols(X_train, y_train):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg

rank_order_results = {}
for feat_name, X in FEATURES.items():
    per_facility = {}
    pooled_correct = 0
    pooled_total = 0
    # also track the untrained physics-prior rule and the empirical ceiling,
    # evaluated on the SAME pairs, for direct comparison
    pooled_prior_correct = 0
    for f in facilities:
        test_mask = sources == f
        train_mask = ~test_mask
        reg = fit_ols(X[train_mask], y[train_mask])
        pred_log_cp_test = reg.predict(X[test_mask])
        re_f = Re_raw[test_mask]
        cp_f = Cp_raw[test_mask]
        idx = np.arange(test_mask.sum())
        pairs = list(combinations(idx, 2))
        n_pairs = 0
        n_correct = 0
        n_prior_correct = 0
        for i, j in pairs:
            dcp_actual = cp_f[j] - cp_f[i]
            if dcp_actual == 0:
                continue
            dpred = pred_log_cp_test[j] - pred_log_cp_test[i]
            dre = re_f[j] - re_f[i]
            if dpred == 0 and dre == 0:
                continue
            n_pairs += 1
            if dpred != 0 and np.sign(dpred) == np.sign(dcp_actual):
                n_correct += 1
            elif dpred == 0:
                pass  # tie in prediction -> counts as wrong (no info)
            if dre != 0 and np.sign(-dre) == np.sign(dcp_actual):
                n_prior_correct += 1
        acc = n_correct / n_pairs if n_pairs else None
        prior_acc = n_prior_correct / n_pairs if n_pairs else None
        per_facility[f] = {
            "n_test": int(test_mask.sum()),
            "n_pairs_nontied": n_pairs,
            "trained_model_coef": dict(zip(
                (["q_Re"] if feat_name == "RE_ONLY" else ["q_Re", "p_gap", "r_conf", "t_asp"]),
                reg.coef_.tolist())),
            "pairwise_accuracy_trained_model": acc,
            "pairwise_accuracy_untrained_physics_prior": prior_acc,
            "beats_random_50pct": (acc - 0.5) if acc is not None else None,
            "beats_untrained_prior": (acc - prior_acc) if (acc is not None and prior_acc is not None) else None,
        }
        if n_pairs:
            pooled_correct += n_correct
            pooled_prior_correct += n_prior_correct
            pooled_total += n_pairs
    rank_order_results[feat_name] = {
        "per_facility": per_facility,
        "pooled_pairwise_accuracy_trained_model": pooled_correct / pooled_total if pooled_total else None,
        "pooled_pairwise_accuracy_untrained_physics_prior": pooled_prior_correct / pooled_total if pooled_total else None,
        "pooled_n_pairs": pooled_total,
        "all_facilities_beat_50pct": all(
            (v["pairwise_accuracy_trained_model"] is not None and v["pairwise_accuracy_trained_model"] > 0.5)
            for v in per_facility.values()
        ),
        "all_facilities_beat_prior": all(
            (v["beats_untrained_prior"] is not None and v["beats_untrained_prior"] > 0)
            for v in per_facility.values()
        ),
    }
results["pairwise_rank_order_LOFO"] = rank_order_results

# ============================================================
# (b) COARSE TERCILE BINNING UNDER LOFO
# ============================================================
def per_facility_tercile_labels(cp_values):
    """Rank-based tercile split, robust to small n and ties (no qcut
    duplicate-edge failures). Returns integer labels 0=low,1=med,2=high
    and the majority-class fraction (trivial baseline) for this facility."""
    order = np.argsort(cp_values, kind="mergesort")
    labels = np.empty(len(cp_values), dtype=int)
    groups = np.array_split(order, 3)
    for lbl, grp in enumerate(groups):
        labels[grp] = lbl
    counts = np.bincount(labels, minlength=3)
    majority_frac = counts.max() / counts.sum()
    return labels, majority_frac

tercile_labels_all = np.empty(n, dtype=int)
majority_frac_by_facility = {}
for f in facilities:
    mask = sources == f
    labels_f, maj_f = per_facility_tercile_labels(Cp_raw[mask])
    tercile_labels_all[mask] = labels_f
    majority_frac_by_facility[f] = maj_f

binning_results = {}
for feat_name, X in FEATURES.items():
    for clf_name, clf_ctor in [
        ("logreg", lambda: LogisticRegression(max_iter=2000)),
        ("rf", lambda: RandomForestClassifier(n_estimators=300, max_depth=4, random_state=RNG_SEED)),
    ]:
        key = f"{feat_name}__{clf_name}"
        per_facility = {}
        pooled_correct = 0
        pooled_total = 0
        for f in facilities:
            test_mask = sources == f
            train_mask = ~test_mask
            Xtr, ytr = X[train_mask], tercile_labels_all[train_mask]
            Xte, yte = X[test_mask], tercile_labels_all[test_mask]
            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
            # guard: training labels must have >=2 classes present, else classifier is degenerate
            if len(set(ytr.tolist())) < 2:
                per_facility[f] = {"n_test": int(test_mask.sum()), "skipped": "degenerate_train_labels"}
                continue
            clf = clf_ctor()
            clf.fit(Xtr_s, ytr)
            pred = clf.predict(Xte_s)
            acc = float((pred == yte).mean())
            per_facility[f] = {
                "n_test": int(test_mask.sum()),
                "accuracy": acc,
                "majority_class_baseline_this_facility": float(majority_frac_by_facility[f]),
                "beats_majority_baseline": acc - float(majority_frac_by_facility[f]),
                "confusion_pred_vs_actual": {
                    "pred_counts": {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
                    "actual_counts": {int(k): int(v) for k, v in zip(*np.unique(yte, return_counts=True))},
                },
            }
            pooled_correct += int((pred == yte).sum())
            pooled_total += len(yte)
        valid = {k: v for k, v in per_facility.items() if "accuracy" in v}
        binning_results[key] = {
            "per_facility": per_facility,
            "pooled_accuracy": pooled_correct / pooled_total if pooled_total else None,
            "pooled_n": pooled_total,
            "random_baseline_3class": 1.0 / 3,
            "all_facilities_beat_own_majority_baseline": (
                len(valid) == len(facilities) and
                all(v["beats_majority_baseline"] > 0 for v in valid.values())
            ),
        }
results["coarse_tercile_binning_LOFO"] = binning_results

# ============================================================
# Overall verdict (computed from the numbers above, not assumed)
# ============================================================
best_rank_order = max(
    rank_order_results.items(),
    key=lambda kv: (kv[1]["pooled_pairwise_accuracy_trained_model"] or 0)
)
best_binning = max(
    binning_results.items(),
    key=lambda kv: (kv[1]["pooled_accuracy"] or 0)
)

verdict = {
    "best_rank_order_variant": best_rank_order[0],
    "best_rank_order_pooled_acc": best_rank_order[1]["pooled_pairwise_accuracy_trained_model"],
    "best_rank_order_all_facilities_beat_50pct": best_rank_order[1]["all_facilities_beat_50pct"],
    "best_rank_order_all_facilities_beat_untrained_prior": best_rank_order[1]["all_facilities_beat_prior"],
    "best_binning_variant": best_binning[0],
    "best_binning_pooled_acc": best_binning[1]["pooled_accuracy"],
    "best_binning_all_facilities_beat_majority": best_binning[1]["all_facilities_beat_own_majority_baseline"],
}
results["verdict"] = verdict

with open(OUT_PATH, "w") as fh:
    json.dump(results, fh, indent=2, default=str)

print(json.dumps(verdict, indent=2))
print("\nEmpirical within-facility Re-Cp Kendall tau (ceiling):")
for f, v in empirical_monotonicity.items():
    print(f"  {f}: tau={v['kendall_tau_Re_Cp']:.3f} (p={v['kendall_tau_p']:.3g}), "
          f"frac_decreasing={v['frac_pairs_consistent_with_Cp_decreasing_in_Re']}")
print("\nRank-order pairwise accuracy (trained model, LOFO), pooled and per facility:")
for feat_name, res in rank_order_results.items():
    print(f"  [{feat_name}] pooled={res['pooled_pairwise_accuracy_trained_model']:.4f} "
          f"(untrained prior pooled={res['pooled_pairwise_accuracy_untrained_physics_prior']:.4f})")
    for f, v in res["per_facility"].items():
        print(f"    {f}: model_acc={v['pairwise_accuracy_trained_model']}, "
              f"prior_acc={v['pairwise_accuracy_untrained_physics_prior']}")
print("\nCoarse tercile binning accuracy (LOFO), pooled and per facility:")
for key, res in binning_results.items():
    print(f"  [{key}] pooled={res['pooled_accuracy']}")
    for f, v in res["per_facility"].items():
        print(f"    {f}: {v}")
print(f"\nSaved to {OUT_PATH}")
