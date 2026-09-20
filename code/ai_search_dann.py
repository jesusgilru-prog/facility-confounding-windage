"""
Domain-Adversarial Neural Network (DANN-style) for LOFO Cp prediction.

Approach:
  - Shared trunk MLP maps log-space predictors -> hidden representation.
  - Regression head predicts log(Cp) from the representation.
  - Domain (facility) classification head, fed through a Gradient
    Reversal Layer (GRL), tries to predict which of the *training*
    facilities a sample came from. Because the GRL flips the sign of
    the gradient during backprop, the shared trunk is pushed to produce
    a representation from which facility identity becomes HARD to
    recover -> a facility-invariant representation, which is the whole
    point of DANN (Ganin & Lempitsky 2016) applied here as a
    domain-generalization aid for the held-out 4th facility.
  - Strict Leave-One-FACILITY-Out (LOFO): train on 3 facilities
    (domain head only ever sees these 3 as classes), predict the 4th,
    exactly as required by the paper's protocol.
  - Predictions on the held-out facility are ensembled over several
    random seeds to reduce the variance any single tiny-data NN fit
    has, then pooled across all 4 folds for the headline pooled R2 in
    log space.

Environment note: torch is not part of the base interpreter used across
this shared project folder, so a local venv
(a local virtual environment) was created and torch (CPU
wheel) installed there. Run this script with that venv's python.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
import torch.nn as nn
from sklearn.metrics import r2_score

DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
RESULTS_PATH = _ROOT + "/results/ai_search_dann_results.json"

FEATURES = ["Re_Omega", "Pi_confinement", "Pi_gap", "Pi_aspect_axial"]
TARGET = "Cp"
FACILITY_COL = "source"

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
N_EPOCHS = 400
LR = 3e-3
WEIGHT_DECAY = 3e-3
HIDDEN1 = 16
HIDDEN2 = 8
LAMBDA_MAX = 0.5   # ceiling of the GRL adversarial weight
GRL_GAMMA = 10.0   # standard Ganin & Lempitsky schedule steepness


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


class DANN(nn.Module):
    def __init__(self, in_dim, n_domains, h1=HIDDEN1, h2=HIDDEN2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(h2, h2), nn.ReLU(),
            nn.Linear(h2, 1),
        )
        self.domain_clf = nn.Sequential(
            nn.Linear(h2, h2), nn.ReLU(),
            nn.Linear(h2, n_domains),
        )

    def forward(self, x, lambd):
        z = self.trunk(x)
        y_pred = self.regressor(z).squeeze(-1)
        d_logits = self.domain_clf(grad_reverse(z, lambd))
        return y_pred, d_logits


def load_data():
    df = pd.read_csv(DATA_PATH)
    X_log = np.log(df[FEATURES].values.astype(np.float64))
    y_log = np.log(df[TARGET].values.astype(np.float64))
    fac = df[FACILITY_COL].values
    return X_log, y_log, fac


def run_fold(X_log, y_log, fac, held_out, seeds):
    train_mask = fac != held_out
    test_mask = fac == held_out

    X_train_raw, y_train_raw = X_log[train_mask], y_log[train_mask]
    X_test_raw, y_test_raw = X_log[test_mask], y_log[test_mask]

    train_facilities = sorted(set(fac[train_mask]))
    n_domains = len(train_facilities)
    fac_to_idx = {f: i for i, f in enumerate(train_facilities)}
    d_train = np.array([fac_to_idx[f] for f in fac[train_mask]], dtype=np.int64)

    # Standardize features and target using TRAIN stats only (no leakage).
    x_mean, x_std = X_train_raw.mean(0), X_train_raw.std(0) + 1e-8
    y_mean, y_std = y_train_raw.mean(), y_train_raw.std() + 1e-8

    Xtr = (X_train_raw - x_mean) / x_std
    Xte = (X_test_raw - x_mean) / x_std
    ytr = (y_train_raw - y_mean) / y_std

    # Class weights for the (often heavily imbalanced) domain classifier.
    counts = np.bincount(d_train, minlength=n_domains).astype(np.float64)
    class_w = counts.sum() / (n_domains * np.maximum(counts, 1))
    class_w_t = torch.tensor(class_w, dtype=torch.float32)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    dtr_t = torch.tensor(d_train, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    preds_per_seed = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = DANN(in_dim=X_log.shape[1], n_domains=n_domains)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss(weight=class_w_t)

        for epoch in range(N_EPOCHS):
            p = epoch / N_EPOCHS
            lambd = LAMBDA_MAX * (2.0 / (1.0 + math.exp(-GRL_GAMMA * p)) - 1.0)

            model.train()
            opt.zero_grad()
            y_pred, d_logits = model(Xtr_t, lambd)
            loss_reg = mse(y_pred, ytr_t)
            loss_dom = ce(d_logits, dtr_t)
            loss = loss_reg + loss_dom
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            y_test_pred_std, _ = model(Xte_t, 0.0)
            y_test_pred = y_test_pred_std.numpy() * y_std + y_mean
        preds_per_seed.append(y_test_pred)

    y_pred_ens = np.mean(np.stack(preds_per_seed, axis=0), axis=0)
    r2_fold = r2_score(y_test_raw, y_pred_ens)
    return y_test_raw, y_pred_ens, r2_fold


def main():
    X_log, y_log, fac = load_data()
    facilities = sorted(set(fac))

    per_facility_r2 = {}
    all_true, all_pred = [], []

    for held_out in facilities:
        y_true, y_pred, r2_fold = run_fold(X_log, y_log, fac, held_out, SEEDS)
        per_facility_r2[held_out] = float(r2_fold)
        all_true.append(y_true)
        all_pred.append(y_pred)
        print(f"held-out={held_out:12s} n={len(y_true):3d}  R2_log={r2_fold:+.4f}")

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    pooled_r2 = float(r2_score(all_true, all_pred))

    all_nonneg = all(v >= 0 for v in per_facility_r2.values())
    clears = bool(pooled_r2 > 0.0 and all_nonneg)

    print(f"\nPooled LOFO R2 (log space): {pooled_r2:+.4f}")
    print(f"All per-facility R2 >= 0: {all_nonneg}")
    print(f"Clears decision rule: {clears}")

    results = {
        "approach": "domain_adversarial_neural_network (DANN, gradient reversal)",
        "slug": "dann",
        "library": f"torch {torch.__version__} (CPU, local venv .venv_dann)",
        "substituted_proxy": False,
        "features": FEATURES,
        "target": TARGET,
        "log_space": True,
        "protocol": "Leave-One-Facility-Out, pooled residual R2 in log(Cp) space, "
                    "predictions ensembled over 10 random seeds per fold",
        "hyperparameters": {
            "seeds": SEEDS,
            "n_epochs": N_EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "hidden_dims": [HIDDEN1, HIDDEN2],
            "lambda_max": LAMBDA_MAX,
            "grl_gamma": GRL_GAMMA,
        },
        "pooled_r2_log": pooled_r2,
        "per_facility_r2_log": per_facility_r2,
        "all_facilities_r2_nonneg": all_nonneg,
        "clears_decision_rule": clears,
        "baselines_for_reference": {
            "best_linear_4pred_pooled_r2": -0.885,
            "reynolds_only_pooled_r2": 0.4526,
            "reynolds_only_per_facility": {
                "note": "positive pooled R2 was an artifact; per-facility R2 was "
                        "-2.36, -61.7, -9.59, +0.468 -> fails decision rule"
            },
            "already_failed_methods": [
                "plain MLP", "Random Forest", "Gradient Boosting",
                "kernel ridge (2 variants)", "plain (non-hierarchical) GP",
                "symbolic regression (genetic search)", "Huber/robust regression"
            ],
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
