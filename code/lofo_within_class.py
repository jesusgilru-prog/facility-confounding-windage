import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import pandas as pd
import numpy as np
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

class LinearRegression:
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xd = np.column_stack([X, np.ones(len(X))])
        coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        self.coef_ = coef[:-1]
        self.intercept_ = coef[-1]
        return self
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_

df = pd.read_csv(_ROOT + '/data/cross_rotor_dataset_v3.csv')

json_out = {"shared_geometry_types": [], "results_by_class": {}}

print("="*70)
print("PART (a): geometry_type x source crosstab")
print("="*70)
ct = pd.crosstab(df['source'], df['geometry_type'])
print(ct)
print()
gt_nsources = df.groupby('geometry_type')['source'].nunique()
print("Distinct sources per geometry_type:")
print(gt_nsources)
shared = gt_nsources[gt_nsources >= 2].index.tolist()
print()
print("geometry_type(s) shared by 2+ sources:", shared)
print()

json_out["shared_geometry_types"] = shared
json_out["testable"] = bool(shared)

if not shared:
    print("No shared geometry_type across sources -> hypothesis not testable with this dataset.")
else:
    for gt in shared:
        sub = df[df['geometry_type'] == gt].copy()
        srcs = sorted(sub['source'].unique())
        print("="*70)
        print(f"PART (b)/(c): restricted LOFO within geometry_type = '{gt}'")
        print(f"Sources sharing this class: {srcs}, n_total={len(sub)}")
        print("="*70)

        # check within-source variance of geometric Pi terms
        for col in ['Pi_gap','Pi_confinement','Pi_aspect_axial','Pi_blockage']:
            nun = sub.groupby('source')[col].nunique()
            print(f"  unique values of {col} per source: {dict(nun)}")
        print()

        # log-space target and predictors
        sub['logCp'] = np.log(sub['Cp'])
        sub['logRe'] = np.log(sub['Re_Omega'])

        results_simple = {}
        results_full = {}
        all_true_simple, all_pred_simple = [], []
        all_true_full, all_pred_full = [], []

        for test_src in srcs:
            train = sub[sub['source'] != test_src]
            test = sub[sub['source'] == test_src]
            print(f"  -- Fold: train on {sorted(train['source'].unique())} (n={len(train)}), test on {test_src} (n={len(test)})")

            # Model 1: simple log(Cp) ~ log(Re) only (geometric Pi's are constant per source -> not identifiable per-facility)
            X_tr = train[['logRe']].values
            y_tr = train['logCp'].values
            X_te = test[['logRe']].values
            y_te = test['logCp'].values

            m = LinearRegression().fit(X_tr, y_tr)
            pred = m.predict(X_te)
            r2 = r2_score(y_te, pred)
            print(f"     simple log(Cp)~log(Re): slope={m.coef_[0]:.4f}, intercept={m.intercept_:.4f}, held-out R2(log)={r2:.4f}")
            results_simple[test_src] = r2
            all_true_simple.extend(y_te.tolist())
            all_pred_simple.extend(pred.tolist())

            # Model 2: try full formula incl Pi terms -- check identifiability
            cols_full = ['logRe','Pi_gap','Pi_confinement','Pi_aspect_axial']
            X_tr_full = train[cols_full].values
            # check rank
            rank = np.linalg.matrix_rank(np.column_stack([X_tr_full, np.ones(len(X_tr_full))]))
            print(f"     full design matrix (incl Pi terms) rank={rank} out of {X_tr_full.shape[1]+1} columns -> {'RANK DEFICIENT (Pi terms constant per single-source train fold)' if rank < X_tr_full.shape[1]+1 else 'full rank'}")
        print()
        pooled_r2_simple = r2_score(all_true_simple, all_pred_simple)
        print(f"  POOLED (combining both folds) held-out R2(log), simple Re-only model, class '{gt}': {pooled_r2_simple:.4f}")
        print(f"  Individual per-source R2: {results_simple}")
        print()

        json_out["results_by_class"][gt] = {
            "sources": srcs,
            "n_total": int(len(sub)),
            "model": "log(Cp) ~ log(Re_Omega) only -- full model with Pi terms is RANK DEFICIENT with 2 single-geometry sources per fold, not identifiable",
            "pooled_r2_log_space": float(pooled_r2_simple),
            "per_source_r2": {k: float(v) for k, v in results_simple.items()},
            "fair_comparison_note": (
                "Compare ONLY against the global M1_Re_only (pooled "
                "R2=+0.4526, 4 facilities, same Re-only model), NOT "
                "against M6 (-0.885), which uses 4 geometric predictors "
                "and 4 facilities -- that would be an apples-to-oranges "
                "comparison. Homogeneous comparison: within-class Re-only "
                f"({pooled_r2_simple:.4f}) vs. global Re-only (+0.4526)."
            ),
        }

with open(_ROOT + '/results/lofo_within_class_results.json', 'w') as f:
    json.dump(json_out, f, indent=2)
print("Escrito: results/lofo_within_class_results.json")
