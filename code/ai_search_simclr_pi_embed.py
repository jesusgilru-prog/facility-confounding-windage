"""
Windage Power paper -- AI search candidate: self-supervised contrastive
representation learning (SimCLR-style) on the log-Pi feature space,
followed by a frozen-embedding ridge regression to log(Cp), evaluated
under strict Leave-One-FACILITY-Out (LOFO) cross-validation.

Protocol (must match the paper's established bar exactly):
  - 4 facilities: Guo2024 (n=45), Vrancik1968 (n=41), Liu2024 (n=20),
    Zheng2024 (n=8).
  - For each facility, train on the OTHER 3, predict the held-out one,
    compute R2 in log space on that held-out facility.
  - Pool the residuals (true, pred) of all 4 held-out folds and compute
    ONE pooled R2_log -- the headline number.
  - Decision rule: pooled R2 must be substantially positive AND every
    per-facility held-out R2 must be >= 0.

Method:
  1. For each LOFO fold, standardize the 4 log-Pi predictors using ONLY
     the 3 training facilities (no leakage from the held-out facility,
     not even its raw features).
  2. Pretrain a small MLP encoder with a SimCLR / NT-Xent contrastive
     objective on the pooled TRAINING points only (3 facilities mixed
     together, facility labels NOT used, Cp NOT used at all). Positive
     pairs = two independently-noised augmentations of the same physical
     point (small Gaussian jitter in standardized log-Pi space, i.e. a
     physically-plausible measurement-noise augmentation). Negatives =
     all other points in the batch.
  3. Freeze the encoder. Encode the (unaugmented) training and held-out
     points into the learned embedding.
  4. Fit a RidgeCV linear model from the frozen embedding to log(Cp)
     using only the training facilities; predict on the held-out
     facility's embedding.
  5. Record per-facility R2 and pool all held-out (true, pred) pairs for
     the pooled R2.

This is a REAL torch implementation (torch 2.x CPU wheel installed for
this run), not a proxy -- confirmed working before this script was
finalized.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV

torch.set_num_threads(2)

DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT_PATH = _ROOT + "/results/ai_search_simclr_pi_embed_results.json"

FEATURES = ["Re_Omega", "Pi_confinement", "Pi_gap", "Pi_aspect_axial"]
TARGET = "Cp"
FACILITY_COL = "source"

SEED = 0
EMBED_DIM = 8
HIDDEN = 32
NOISE_STD = 0.12          # gaussian jitter std in standardized log-Pi space
EPOCHS = 600
LR = 2e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.3


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def nt_xent_loss(z1, z2, temperature):
    """Standard SimCLR NT-Xent loss for a batch of paired augmentations."""
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2n, d)
    z = nn.functional.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temperature  # (2n, 2n)

    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    positives = torch.cat([
        torch.arange(n, 2 * n),
        torch.arange(0, n),
    ]).to(z.device)

    loss = nn.functional.cross_entropy(sim, positives)
    return loss


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def pretrain_encoder(X_train_std, seed):
    set_seed(seed)
    device = torch.device("cpu")
    encoder = Encoder(X_train_std.shape[1], HIDDEN, EMBED_DIM).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    X_t = torch.tensor(X_train_std, dtype=torch.float32, device=device)
    n = X_t.shape[0]

    encoder.train()
    losses = []
    for epoch in range(EPOCHS):
        if epoch % 100 == 0:
            print(f"  epoch {epoch}/{EPOCHS}", flush=True)
        noise1 = torch.randn_like(X_t) * NOISE_STD
        noise2 = torch.randn_like(X_t) * NOISE_STD
        view1 = X_t + noise1
        view2 = X_t + noise2

        z1 = encoder(view1)
        z2 = encoder(view2)
        loss = nt_xent_loss(z1, z2, TEMPERATURE)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    encoder.eval()
    return encoder, losses


def encode(encoder, X_std):
    with torch.no_grad():
        X_t = torch.tensor(X_std, dtype=torch.float32)
        z = encoder(X_t)
        z = nn.functional.normalize(z, dim=1)
    return z.numpy()


def run():
    df = pd.read_csv(DATA_PATH)
    facilities = sorted(df[FACILITY_COL].unique().tolist())
    assert set(facilities) == {"Guo2024", "Vrancik1968", "Liu2024", "Zheng2024"}, facilities

    # log-space predictors and target (standard for this corpus)
    X_log = np.log(df[FEATURES].values.astype(float))
    y_log = np.log(df[TARGET].values.astype(float))
    fac = df[FACILITY_COL].values

    per_facility_r2 = {}
    pooled_true = []
    pooled_pred = []
    final_loss_by_fold = {}

    ridge_alphas = np.logspace(-3, 3, 25)

    for held_out in facilities:
        print(f"=== fold held_out={held_out} ===", flush=True)
        train_mask = fac != held_out
        test_mask = fac == held_out

        X_train_raw = X_log[train_mask]
        X_test_raw = X_log[test_mask]
        y_train = y_log[train_mask]
        y_test = y_log[test_mask]

        # standardize using ONLY the 3 training facilities -- no leakage
        mu = X_train_raw.mean(axis=0)
        sigma = X_train_raw.std(axis=0)
        sigma[sigma == 0] = 1.0
        X_train_std = (X_train_raw - mu) / sigma
        X_test_std = (X_test_raw - mu) / sigma

        # 1-2: contrastive pretraining on training facilities only,
        # no facility labels, no Cp
        encoder, losses = pretrain_encoder(X_train_std, seed=SEED)
        final_loss_by_fold[held_out] = float(np.mean(losses[-20:]))

        # 3: freeze, encode
        Z_train = encode(encoder, X_train_std)
        Z_test = encode(encoder, X_test_std)

        # 4: ridge regression from frozen embedding -> log(Cp)
        n_train = Z_train.shape[0]
        cv_folds = min(5, n_train)
        reg = RidgeCV(alphas=ridge_alphas, cv=cv_folds)
        reg.fit(Z_train, y_train)
        y_pred = reg.predict(Z_test)

        fold_r2 = float(r2_score(y_test, y_pred))
        per_facility_r2[held_out] = fold_r2

        pooled_true.extend(y_test.tolist())
        pooled_pred.extend(y_pred.tolist())

    pooled_r2 = float(r2_score(np.array(pooled_true), np.array(pooled_pred)))

    all_nonneg = bool(all(v >= 0.0 for v in per_facility_r2.values()))
    pooled_positive = bool(pooled_r2 > 0.0)
    # "substantially positive" -- use a modest explicit threshold for the paper's language
    substantially_positive = bool(pooled_r2 >= 0.3)
    clears_decision_rule = bool(substantially_positive and all_nonneg)

    results = {
        "slug": "simclr_pi_embed",
        "method": "SimCLR-style contrastive pretraining (real torch, NT-Xent) on log-Pi features "
                  "-> frozen embedding -> RidgeCV to log(Cp), strict per-fold LOFO with fold-specific "
                  "standardization (no leakage of held-out facility into pretraining or scaling).",
        "libraries_used": ["torch", "numpy", "pandas", "scikit-learn"],
        "torch_version": torch.__version__,
        "substituted_proxy": False,
        "hyperparameters": {
            "embed_dim": EMBED_DIM,
            "hidden": HIDDEN,
            "noise_std": NOISE_STD,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "temperature": TEMPERATURE,
            "seed": SEED,
        },
        "final_contrastive_loss_by_fold": final_loss_by_fold,
        "per_facility_r2_log": per_facility_r2,
        "pooled_r2_log": pooled_r2,
        "pooled_n": len(pooled_true),
        "decision_rule": {
            "pooled_positive": pooled_positive,
            "pooled_substantially_positive_ge_0.3": substantially_positive,
            "all_facilities_nonneg": all_nonneg,
            "clears_decision_rule": clears_decision_rule,
        },
        "known_baselines_for_reference": {
            "best_linear_4pred_pooled_r2_log": -0.885,
            "reynolds_only_pooled_r2_log": 0.4526,
            "reynolds_only_per_facility_r2_log_note": "positive pooled but -2.36/-61.7/-9.59/+0.468 per facility -- FAILS decision rule",
            "already_failed_family": ["MLP", "RandomForest", "GradientBoosting",
                                       "KernelRidge (2 variants)", "plain GaussianProcess",
                                       "symbolic regression (genetic)", "Huber/robust regression"],
        },
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run()
