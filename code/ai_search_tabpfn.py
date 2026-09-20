"""
Windage Power paper - AI search approach: TabPFN (pretrained tabular
foundation-model transformer, in-context regression, no gradient training
on the target task).

Protocol (fixed by the paper, non-negotiable):
  Leave-One-FACILITY-Out (LOFO) cross-validation over the 4 facilities in
  cross_rotor_dataset_v3.csv (Guo2024, Vrancik1968, Liu2024, Zheng2024).
  For each held-out facility: train (= "give as in-context examples") on
  the other 3 facilities' points, predict the held-out facility's points,
  in LOG space (log(Cp) as target, log of the four Pi-predictors as
  features). Pool all 4 held-out predictions together and compute ONE
  pooled R2 in log space (pooled_r2_log = headline number). Also report
  per-facility R2.

Decision rule: pooled R2 substantially positive AND every per-facility
R2 >= 0.

Model: TabPFNRegressor from the `tabpfn` package (v8.3.0), used purely at
inference time (fit() just caches the context set; no gradient training
happens on our data). This is a genuine in-context/meta-learning approach,
not a proxy.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

CSV_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT_JSON = _ROOT + "/results/ai_search_tabpfn_results.json"

FEATURES = ["Re_Omega", "Pi_confinement", "Pi_gap", "Pi_aspect_axial"]
TARGET = "Cp"
FACILITIES = ["Guo2024", "Vrancik1968", "Liu2024", "Zheng2024"]


def load_log_data():
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in FEATURES + [TARGET, "source"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    df = df.dropna(subset=FEATURES + [TARGET, "source"]).copy()
    # all predictors and target are strictly positive Pi-groups / Re / Cp
    for c in FEATURES + [TARGET]:
        if (df[c] <= 0).any():
            bad = df[df[c] <= 0]
            raise ValueError(f"Non-positive values found in {c}, cannot log-transform: {bad[[c,'source']]}")
    for c in FEATURES:
        df[f"log_{c}"] = np.log(df[c])
    df["log_target"] = np.log(df[TARGET])
    return df


def main():
    substituted_proxy = False
    proxy_explanation = ""
    backend_used = "unknown"

    try:
        import torch
        from tabpfn import TabPFNRegressor
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        backend_used = f"tabpfn v8.3.0 (torch {torch.__version__}, device={device})"
        USE_TABPFN = True
    except Exception as e:  # pragma: no cover - fallback path
        print(f"[WARN] Could not import tabpfn/torch: {e}", file=sys.stderr)
        USE_TABPFN = False
        device = "cpu"

    df = load_log_data()
    log_feat_cols = [f"log_{c}" for c in FEATURES]

    per_facility = {}
    all_true = []
    all_pred = []
    all_source = []

    for held_out in FACILITIES:
        train_df = df[df["source"] != held_out]
        test_df = df[df["source"] == held_out]
        X_train = train_df[log_feat_cols].values.astype(np.float64)
        y_train = train_df["log_target"].values.astype(np.float64)
        X_test = test_df[log_feat_cols].values.astype(np.float64)
        y_test = test_df["log_target"].values.astype(np.float64)

        if USE_TABPFN:
            try:
                reg = TabPFNRegressor(device=device, random_state=0, ignore_pretraining_limits=True)
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
            except Exception as e:
                print(f"[WARN] TabPFN failed at inference for held-out {held_out}: {e}", file=sys.stderr)
                print("[WARN] Falling back to CPU device for this fold.", file=sys.stderr)
                reg = TabPFNRegressor(device="cpu", random_state=0, ignore_pretraining_limits=True)
                reg.fit(X_train, y_train)
                y_pred = reg.predict(X_test)
        else:
            # Should not trigger given tabpfn import above succeeded; kept for safety.
            raise RuntimeError("tabpfn unavailable and no fallback implemented in this branch")

        r2 = r2_score(y_test, y_pred)
        per_facility[held_out] = float(r2)
        all_true.extend(y_test.tolist())
        all_pred.extend(np.asarray(y_pred).tolist())
        all_source.extend([held_out] * len(y_test))
        print(f"Held-out {held_out}: n={len(y_test)}, R2_log={r2:.4f}")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    pooled_r2 = float(r2_score(all_true, all_pred))
    print(f"\nPooled LOFO R2 (log space): {pooled_r2:.4f}")

    decision_rule_pass = bool(pooled_r2 > 0.1 and all(v >= 0 for v in per_facility.values()))
    print(f"Decision rule (pooled substantially positive AND all per-facility >= 0): {decision_rule_pass}")

    results = {
        "slug": "tabpfn",
        "approach": "TabPFN (pretrained tabular foundation-model transformer, in-context regression)",
        "backend_used": backend_used,
        "substituted_proxy": substituted_proxy,
        "proxy_explanation": proxy_explanation,
        "features": FEATURES,
        "target": TARGET,
        "space": "log",
        "n_total": int(len(df)),
        "facility_counts": {f: int((df["source"] == f).sum()) for f in FACILITIES},
        "per_facility_r2_log": per_facility,
        "pooled_r2_log": pooled_r2,
        "decision_rule_pass": decision_rule_pass,
        "decision_rule_text": "pooled R2 substantially positive AND every per-facility R2 >= 0",
        "predictions_log": {
            "y_true": all_true.tolist(),
            "y_pred": all_pred.tolist(),
            "source": all_source,
        },
        "known_baselines_for_comparison": {
            "best_linear_4pred_pooled_r2_log": -0.885,
            "reynolds_only_linear_pooled_r2_log": 0.4526,
            "reynolds_only_linear_per_facility_note": "positive pooled (0.4526) but per-facility R2 = -2.36, -61.7, -9.59, +0.468 -> FAILS decision rule (artifact of pooling)",
            "already_tried_and_failed": [
                "plain MLP", "Random Forest", "Gradient Boosting",
                "kernel ridge regression (2 variants)", "plain (non-hierarchical) Gaussian Process",
                "symbolic regression (genetic search)", "Huber/robust regression"
            ],
            "already_tried_range_pooled_r2_log": [-2.67, -0.55],
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {OUT_JSON}")


if __name__ == "__main__":
    main()
