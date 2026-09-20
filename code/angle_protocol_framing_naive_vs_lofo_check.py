"""
angle_protocol_framing: supporting empirical check for the framing/literature angle.

Question: does this dataset actually exhibit the "naive validation looks great,
rigorous facility-held-out validation collapses" pattern that is the load-bearing
empirical signature of the precedent papers we're citing (DomainBed / WILDS /
Underspecification / the MSSP bearing-fault leakage paper)? If yes, that pattern
is real, computed evidence for the "protocol is the contribution" framing. If the
naive number is ALSO bad, the framing angle loses its strongest rhetorical device
(there'd be no "before/after protocol" contrast to show a CS reviewer).

Method: same 4 predictors (Re_Omega, Pi_confinement, Pi_gap, Pi_aspect_axial) ->
log(Cp), same RandomForest/GradientBoosting family already used elsewhere in the
project. Compare:
  (a) naive = ordinary shuffled K-fold CV, ignoring facility identity (rows from
      the same facility can appear in both train and test folds) -- this is what
      an unwary practitioner following standard ML tutorials would do.
  (b) strict LOFO = GroupKFold by facility (already the project standard).
Repeated over multiple seeds/fold-counts for (a) since shuffled-fold R2 is noisy
at n=114.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict

df = pd.read_csv(_ROOT + "/data/cross_rotor_dataset_v3.csv")
preds = ["Re_Omega", "Pi_confinement", "Pi_gap", "Pi_aspect_axial"]
X = np.log(df[preds].values)
y = np.log(df["Cp"].values)
groups = df["source"].values


def r2_log(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)


def naive_kfold_r2(model_fn, n_splits, seeds):
    vals = []
    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pred = cross_val_predict(model_fn(), X, y, cv=kf)
        vals.append(r2_log(y, pred))
    return float(np.mean(vals)), float(np.std(vals)), vals


def lofo_r2(model_fn):
    gkf = GroupKFold(n_splits=4)
    pred = cross_val_predict(model_fn(), X, y, cv=gkf, groups=groups)
    per_facility = {}
    for fac in np.unique(groups):
        m = groups == fac
        per_facility[fac] = float(r2_log(y[m], pred[m]))
    return float(r2_log(y, pred)), per_facility


results = {}
for name, model_fn in [
    ("random_forest", lambda: RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0)),
    ("gradient_boosting", lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0)),
]:
    naive_mean5, naive_std5, naive_vals5 = naive_kfold_r2(model_fn, 5, seeds=range(20))
    naive_mean10, naive_std10, naive_vals10 = naive_kfold_r2(model_fn, 10, seeds=range(20))
    lofo_pooled, lofo_per_fac = lofo_r2(model_fn)
    results[name] = {
        "naive_shuffled_kfold5_r2_log_mean": naive_mean5,
        "naive_shuffled_kfold5_r2_log_std": naive_std5,
        "naive_shuffled_kfold10_r2_log_mean": naive_mean10,
        "naive_shuffled_kfold10_r2_log_std": naive_std10,
        "strict_lofo_pooled_r2_log": lofo_pooled,
        "strict_lofo_per_facility_r2_log": lofo_per_fac,
        "gap_kfold5_minus_lofo": naive_mean5 - lofo_pooled,
    }
    print(f"\n=== {name} ===")
    print(f"  naive shuffled KFold(5), 20 seeds: R2_log = {naive_mean5:.3f} +/- {naive_std5:.3f}")
    print(f"  naive shuffled KFold(10), 20 seeds: R2_log = {naive_mean10:.3f} +/- {naive_std10:.3f}")
    print(f"  strict LOFO (GroupKFold by facility): pooled R2_log = {lofo_pooled:.3f}")
    print(f"  strict LOFO per-facility: {lofo_per_fac}")

with open(_ROOT + "/results/angle_protocol_framing_naive_vs_lofo_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved to results/angle_protocol_framing_naive_vs_lofo_results.json")
