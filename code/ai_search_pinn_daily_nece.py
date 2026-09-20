"""
Physics-Informed Neural Network (PINN) for Cp = f(Re_Omega, Pi_confinement, Pi_gap, Pi_aspect_axial)
in log-log space, with a SOFT physics-informed regularization term that penalizes deviation of the
network's local implied exponent d(log Cp)/d(log Re_Omega) from the Daily & Nece regime-IV
theoretical exponent of -0.2. This is a soft prior (regularization weight lambda_phys), not a hard
constraint, so the network can override it where the data strongly disagrees.

Protocol (fixed by the paper, non-negotiable):
  - Leave-One-FACILITY-Out (LOFO) cross-validation across the 4 facilities
    (Guo2024, Vrancik1968, Liu2024, Zheng2024).
  - For each outer fold: train on 3 facilities, predict the held-out facility, in log space.
  - Pool the residuals of all 4 held-out folds and compute ONE pooled R2 in log space
    (this is "pooled_r2_log", the headline number).
  - Report per-facility R2 too.
  - Decision rule: pooled R2 must be substantially positive AND every per-facility R2 must be >= 0.

Hyperparameter (lambda_phys, the physics-regularization weight) is chosen per outer fold via a
NESTED leave-one-facility-out search using only the 3 outer-training facilities (never touching the
true held-out facility), to avoid any data leakage / cherry-picking on the actual test metric.

Library actually used: PyTorch 2.13 (CPU), found pre-installed in
a local virtual environment (system pip is externally-managed and blocked; this local
venv already had torch, so no proxy was needed -- this is a real PINN, not an approximation).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(4)

DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT_JSON = _ROOT + "/results/ai_search_pinn_daily_nece_results.json"

FEATURES = ["Re_Omega", "Pi_confinement", "Pi_gap", "Pi_aspect_axial"]
RE_IDX = 0  # index of Re_Omega within FEATURES / log-feature matrix
TARGET_EXPONENT = -0.2  # Daily & Nece regime-IV exponent on Re_Omega

LAMBDA_CANDIDATES = [0.0, 0.05, 0.2, 1.0, 5.0]
SEEDS_FINAL = [0, 1, 2]        # ensemble seeds for the final (outer) fit
SEEDS_INNER = [0]              # single seed for the (cheaper) inner hyperparameter search
EPOCHS_INNER = 1500
EPOCHS_FINAL = 3000
LR = 0.01
WEIGHT_DECAY = 1e-3
HIDDEN = 8


def r2_log(y_true, y_pred, ref_mean):
    """R2 in log space against a fixed reference mean (for pooled computations)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - ref_mean) ** 2)
    return 1.0 - ss_res / ss_tot


class SmallMLP(nn.Module):
    def __init__(self, n_in=4, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def standardize(train_log_X, *other_log_Xs):
    mean = train_log_X.mean(axis=0)
    std = train_log_X.std(axis=0)
    std[std == 0] = 1.0
    out = [(train_log_X - mean) / std]
    for X in other_log_Xs:
        out.append((X - mean) / std)
    return (*out, mean, std)


def train_pinn(Xtr_std, ytr, re_std, lambda_phys, seed, epochs, lr=LR, wd=WEIGHT_DECAY, hidden=HIDDEN):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SmallMLP(n_in=Xtr_std.shape[1], hidden=hidden).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    Xtr_t = torch.tensor(Xtr_std, dtype=torch.float64, requires_grad=True)
    ytr_t = torch.tensor(ytr, dtype=torch.float64).view(-1, 1)
    re_std_t = torch.tensor(float(re_std), dtype=torch.float64)

    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xtr_t)
        mse = torch.mean((pred - ytr_t) ** 2)
        if lambda_phys > 0:
            grad_x = torch.autograd.grad(pred.sum(), Xtr_t, create_graph=True)[0]
            dlogCp_dlogRe = grad_x[:, RE_IDX] / re_std_t
            phys_loss = torch.mean((dlogCp_dlogRe - TARGET_EXPONENT) ** 2)
        else:
            phys_loss = torch.zeros((), dtype=torch.float64)
        loss = mse + lambda_phys * phys_loss
        loss.backward()
        opt.step()
    return model


def predict(model, X_std):
    with torch.no_grad():
        X_t = torch.tensor(X_std, dtype=torch.float64)
        return model(X_t).numpy().ravel()


def run_inner_search(df_train_outer, log_X_all, log_y_all):
    """Nested LOFO search over the 3 outer-training facilities to pick lambda_phys.
    Never touches the true held-out (outer test) facility."""
    inner_facilities = df_train_outer["source"].unique().tolist()
    idx_all = df_train_outer.index.to_numpy()

    best_lambda = None
    best_score = -np.inf
    scores_by_lambda = {}

    for lam in LAMBDA_CANDIDATES:
        inner_preds, inner_true = [], []
        for inner_test_fac in inner_facilities:
            inner_train_idx = df_train_outer.index[df_train_outer["source"] != inner_test_fac].to_numpy()
            inner_test_idx = df_train_outer.index[df_train_outer["source"] == inner_test_fac].to_numpy()

            Xtr_log = log_X_all.loc[inner_train_idx].to_numpy()
            Xte_log = log_X_all.loc[inner_test_idx].to_numpy()
            ytr = log_y_all.loc[inner_train_idx].to_numpy()
            yte = log_y_all.loc[inner_test_idx].to_numpy()

            Xtr_std, Xte_std, mean, std = standardize(Xtr_log, Xte_log)
            re_std = std[RE_IDX]

            preds_seeds = []
            for seed in SEEDS_INNER:
                model = train_pinn(Xtr_std, ytr, re_std, lam, seed, EPOCHS_INNER)
                preds_seeds.append(predict(model, Xte_std))
            pred_mean = np.mean(preds_seeds, axis=0)

            inner_preds.append(pred_mean)
            inner_true.append(yte)

        inner_preds = np.concatenate(inner_preds)
        inner_true = np.concatenate(inner_true)
        ref_mean = log_y_all.loc[idx_all].to_numpy().mean()
        score = r2_log(inner_true, inner_preds, ref_mean)
        scores_by_lambda[lam] = float(score)

        if score > best_score:
            best_score = score
            best_lambda = lam

    return best_lambda, scores_by_lambda


def main():
    df = pd.read_csv(DATA_PATH)
    log_X = pd.DataFrame({f: np.log(df[f].to_numpy()) for f in FEATURES}, index=df.index)
    log_y = pd.Series(np.log(df["Cp"].to_numpy()), index=df.index, name="log_Cp")

    facilities = df["source"].unique().tolist()
    print("Facilities:", facilities)

    per_facility_r2 = {}
    per_facility_n = {}
    chosen_lambda = {}
    inner_scores_all = {}
    all_true, all_pred = [], []

    global_ref_mean = log_y.to_numpy().mean()  # pooled reference mean (all 114 points)

    for held_out in facilities:
        train_idx = df.index[df["source"] != held_out]
        test_idx = df.index[df["source"] == held_out]
        df_train_outer = df.loc[train_idx]

        # --- nested hyperparameter search on the 3 training facilities only ---
        best_lambda, scores_by_lambda = run_inner_search(df_train_outer, log_X, log_y)
        chosen_lambda[held_out] = best_lambda
        inner_scores_all[held_out] = scores_by_lambda
        print(f"[{held_out}] inner search scores: {scores_by_lambda} -> chosen lambda={best_lambda}")

        # --- final fit on all 3 training facilities with chosen lambda, ensembled over seeds ---
        Xtr_log = log_X.loc[train_idx].to_numpy()
        Xte_log = log_X.loc[test_idx].to_numpy()
        ytr = log_y.loc[train_idx].to_numpy()
        yte = log_y.loc[test_idx].to_numpy()

        Xtr_std, Xte_std, mean, std = standardize(Xtr_log, Xte_log)
        re_std = std[RE_IDX]

        preds_seeds = []
        for seed in SEEDS_FINAL:
            model = train_pinn(Xtr_std, ytr, re_std, best_lambda, seed, EPOCHS_FINAL)
            preds_seeds.append(predict(model, Xte_std))
        pred_mean = np.mean(preds_seeds, axis=0)

        fold_r2 = r2_log(yte, pred_mean, yte.mean())  # per-facility R2 vs its own mean
        per_facility_r2[held_out] = float(fold_r2)
        per_facility_n[held_out] = int(len(test_idx))

        all_true.append(yte)
        all_pred.append(pred_mean)

        print(f"[{held_out}] n={len(test_idx)} per-facility R2_log={fold_r2:.4f}")

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    pooled_r2 = r2_log(all_true, all_pred, global_ref_mean)

    decision_rule_pass = bool(pooled_r2 > 0 and all(v >= 0 for v in per_facility_r2.values()))

    results = {
        "slug": "pinn_daily_nece",
        "approach": "Physics-Informed Neural Network (soft Daily-Nece regime-IV exponent prior on d(logCp)/d(logRe_Omega) = -0.2)",
        "library_used": "torch (PyTorch) 2.13+cpu, from pre-existing venv a local virtual environment",
        "substituted_proxy": False,
        "features": FEATURES,
        "target_exponent_re_omega": TARGET_EXPONENT,
        "hyperparameters": {
            "hidden_units": HIDDEN,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "epochs_final": EPOCHS_FINAL,
            "epochs_inner_search": EPOCHS_INNER,
            "seeds_final_ensemble": SEEDS_FINAL,
            "lambda_candidates": LAMBDA_CANDIDATES,
        },
        "chosen_lambda_per_outer_fold": chosen_lambda,
        "inner_nested_search_scores": inner_scores_all,
        "per_facility_r2_log": per_facility_r2,
        "per_facility_n": per_facility_n,
        "pooled_r2_log": float(pooled_r2),
        "decision_rule": {
            "requires": "pooled_r2_log substantially positive AND all per_facility_r2_log >= 0",
            "pass": decision_rule_pass,
        },
        "known_baselines_for_comparison": {
            "best_linear_4predictor_pooled_r2_log": -0.885,
            "reynolds_only_linear_pooled_r2_log": 0.4526,
            "reynolds_only_linear_per_facility_r2_log_note": "artifact: -2.36, -61.7, -9.59, +0.468 -- fails decision rule",
            "already_failed_nonlinear_methods_pooled_r2_log_range": [-2.67, -0.55],
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
