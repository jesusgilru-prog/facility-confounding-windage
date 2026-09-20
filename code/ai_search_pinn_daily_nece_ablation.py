"""
Ablation companion to ai_search_pinn_daily_nece.py: evaluate FIXED lambda_phys (no nested search)
directly against the true held-out facility, for lambda=0.0 (plain MLP, no physics term) and
lambda=1.0 (fixed moderate physics prior), to see whether the physics-informed regularization
helps or hurts relative to the plain network, using the identical architecture/training code.
Not the headline number (headline uses the honest nested-search-selected lambda per fold) -- this
is diagnostic context only.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, _ROOT + "/code")
from ai_search_pinn_daily_nece import (
    FEATURES, RE_IDX, standardize, train_pinn, predict, r2_log, SEEDS_FINAL, EPOCHS_FINAL, DATA_PATH
)

df = pd.read_csv(DATA_PATH)
log_X = pd.DataFrame({f: np.log(df[f].to_numpy()) for f in FEATURES}, index=df.index)
log_y = pd.Series(np.log(df["Cp"].to_numpy()), index=df.index, name="log_Cp")
facilities = df["source"].unique().tolist()
global_ref_mean = log_y.to_numpy().mean()

out = {}
for fixed_lambda in [0.0, 1.0]:
    per_fac = {}
    all_true, all_pred = [], []
    for held_out in facilities:
        train_idx = df.index[df["source"] != held_out]
        test_idx = df.index[df["source"] == held_out]
        Xtr_log = log_X.loc[train_idx].to_numpy()
        Xte_log = log_X.loc[test_idx].to_numpy()
        ytr = log_y.loc[train_idx].to_numpy()
        yte = log_y.loc[test_idx].to_numpy()
        Xtr_std, Xte_std, mean, std = standardize(Xtr_log, Xte_log)
        re_std = std[RE_IDX]
        preds_seeds = []
        for seed in SEEDS_FINAL:
            model = train_pinn(Xtr_std, ytr, re_std, fixed_lambda, seed, EPOCHS_FINAL)
            preds_seeds.append(predict(model, Xte_std))
        pred_mean = np.mean(preds_seeds, axis=0)
        per_fac[held_out] = float(r2_log(yte, pred_mean, yte.mean()))
        all_true.append(yte)
        all_pred.append(pred_mean)
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    pooled = float(r2_log(all_true, all_pred, global_ref_mean))
    out[str(fixed_lambda)] = {"per_facility_r2_log": per_fac, "pooled_r2_log": pooled}
    print(fixed_lambda, "pooled:", pooled, "per-facility:", per_fac)

with open(_ROOT + "/results/ai_search_pinn_daily_nece_ablation.json", "w") as f:
    json.dump(out, f, indent=2)
