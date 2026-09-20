"""
Deep Kernel Learning Gaussian Process (DKL-GP) for windage Cp prediction.

Approach: a small neural-network feature extractor (torch) feeds a learned
low-dimensional representation into a GP kernel (gpytorch ExactGP, RBF
kernel with ARD), trained end-to-end by maximizing the exact marginal log
likelihood. This is NOT a plain GP (already tried and failed, pooled
R2=-0.92) -- the point of DKL is that the network can learn a nonlinear
warping of the four Pi-groups before the GP's stationary kernel acts on it,
which a plain GP with a fixed kernel on raw (or log) predictors cannot do.

Protocol (fixed by the paper, non-negotiable):
  - Leave-One-FACILITY-Out (LOFO) cross-validation over the 4 facilities.
  - Fit in log space: target = log10(Cp), features = log10 of the 4 Pi-groups
    (Re_Omega, Pi_confinement, Pi_gap, Pi_aspect_axial). Pi_blockage excluded
    (exact algebraic identity of Pi_aspect_axial & Pi_confinement).
  - Per-facility R2 computed on the held-out facility's log-Cp values.
  - Pooled R2: concatenate the 4 held-out folds' (true, pred) log-Cp pairs
    into ONE set of 114 points and compute a single R2 over that pooled set.
  - Decision rule: pooled R2 substantially positive AND every per-facility
    R2 >= 0. A positive pooled R2 masking a negative facility does not count.

Implementation notes / honesty:
  - torch + gpytorch were successfully pip-installed in this environment
    (--user --break-system-packages), so this is a REAL deep-kernel GP,
    not a proxy: an nn.Module feature extractor is registered as part of
    the gpytorch ExactGP model and its parameters are optimized jointly
    with the GP's kernel hyperparameters and noise via Adam on the exact
    marginal log likelihood (ExactMarginalLogLikelihood).
  - Given the very small sample sizes per training fold (71-106 points) and
    especially the tiny held-out folds (8-45 points), the feature extractor
    is kept deliberately small (4 -> 8 -> 4 -> 2, Tanh) with weight decay,
    and results are averaged over multiple random seeds/restarts (selected
    by TRAIN marginal log likelihood only, never by peeking at the held-out
    facility) to reduce the variance any single random init would add to a
    number that goes into a paper.
  - Standardization of inputs (z-score) is fit on the training fold only
    and applied to the held-out fold -- no leakage across the LOFO split.
  - Latent-feature min-max normalization (the standard Wilson et al. DKL
    trick to keep the RBF kernel's lengthscale well-conditioned) is
    likewise computed from training-fold data only and re-used, frozen, for
    the held-out facility's predictions.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import os
import warnings
from pathlib import Path

# Machine is shared with several other concurrent training jobs (heavy load
# average observed). Cap thread usage per the project's known thread-
# oversubscription pitfall (n_jobs=-1 / unbounded BLAS threads degrade badly
# on a loaded box) instead of grabbing all cores.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gpytorch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

warnings.filterwarnings("ignore")

ROOT = Path(_ROOT + "")
DATA = ROOT / "data" / "cross_rotor_dataset_v3.csv"
OUT_JSON = ROOT / "results" / "ai_search_dklgp_results.json"

FEATURES = ["Re_Omega", "Pi_confinement", "Pi_gap", "Pi_aspect_axial"]
TARGET = "Cp"
FACILITIES = ["Guo2024", "Vrancik1968", "Liu2024", "Zheng2024"]

SEEDS = [0, 1, 2, 3, 4]
LATENT_DIM = 2
HIDDEN = (8, 4)
N_EPOCHS = 400
LR = 0.01
WEIGHT_DECAY = 1e-3
torch.set_default_dtype(torch.float64)


class FeatureExtractor(nn.Module):
    def __init__(self, in_dim, hidden, latent_dim):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers += [nn.Linear(d, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DKLModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor, latent_dim):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim)
        )
        self.feature_extractor = feature_extractor
        self.register_buffer("feat_min", torch.zeros(latent_dim))
        self.register_buffer("feat_max", torch.ones(latent_dim))

    def project(self, x, update_stats=False):
        z = self.feature_extractor(x)
        if update_stats:
            zmin = z.min(dim=0).values.detach()
            zmax = z.max(dim=0).values.detach()
            zmax = torch.where(zmax - zmin < 1e-8, zmin + 1.0, zmax)
            self.feat_min.copy_(zmin)
            self.feat_max.copy_(zmax)
        z = (z - self.feat_min) / (self.feat_max - self.feat_min)
        z = 2.0 * z - 1.0
        return z

    def forward(self, x):
        z = self.project(x, update_stats=self.training)
        mean_x = self.mean_module(z)
        covar_x = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def fit_one_seed(train_x, train_y, seed):
    torch.manual_seed(seed)
    fe = FeatureExtractor(train_x.shape[1], HIDDEN, LATENT_DIM)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
    )
    model = DKLModel(train_x, train_y, likelihood, fe, LATENT_DIM)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(N_EPOCHS):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    # freeze latent normalization stats using full training set, eval mode
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        _ = model.project(train_x, update_stats=True)  # lock stats from train data

    return model, likelihood, final_loss


def predict(model, likelihood, x):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(x))
        return pred.mean.numpy()


def run_lofo():
    df = pd.read_csv(DATA)
    df = df[df["source"].isin(FACILITIES)].copy()

    X_log = np.log10(df[FEATURES].values.astype(np.float64))
    y_log = np.log10(df[TARGET].values.astype(np.float64))
    source = df["source"].values

    per_facility = {}
    pooled_true = []
    pooled_pred = []
    seed_diagnostics = {}

    for held_out in FACILITIES:
        train_mask = source != held_out
        test_mask = source == held_out

        X_train_raw = X_log[train_mask]
        y_train = y_log[train_mask]
        X_test_raw = X_log[test_mask]
        y_test = y_log[test_mask]

        mu = X_train_raw.mean(axis=0)
        sigma = X_train_raw.std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        X_train = (X_train_raw - mu) / sigma
        X_test = (X_test_raw - mu) / sigma

        train_x = torch.from_numpy(X_train)
        train_y = torch.from_numpy(y_train)
        test_x = torch.from_numpy(X_test)

        seed_results = []
        for seed in SEEDS:
            model, likelihood, final_loss = fit_one_seed(train_x, train_y, seed)
            test_pred = predict(model, likelihood, test_x)
            train_pred = predict(model, likelihood, train_x)
            train_r2 = r2_score(y_train, train_pred)
            seed_results.append(
                {
                    "seed": seed,
                    "final_neg_mll": final_loss,
                    "train_r2": train_r2,
                    "test_pred": test_pred,
                }
            )

        # Model selection by TRAIN marginal log likelihood only (no test peeking).
        # Average the predictions of the 3 best-trained (lowest final loss) seeds
        # for a more stable estimate, still selected without touching y_test.
        seed_results.sort(key=lambda r: r["final_neg_mll"])
        best = seed_results[: min(3, len(seed_results))]
        avg_pred = np.mean([r["test_pred"] for r in best], axis=0)

        fold_r2 = r2_score(y_test, avg_pred)
        per_facility[held_out] = fold_r2
        pooled_true.extend(y_test.tolist())
        pooled_pred.extend(avg_pred.tolist())

        seed_diagnostics[held_out] = [
            {"seed": r["seed"], "final_neg_mll": r["final_neg_mll"], "train_r2": r["train_r2"]}
            for r in seed_results
        ]

        print(f"{held_out}: n_train={train_mask.sum()}, n_test={test_mask.sum()}, "
              f"held-out R2_log={fold_r2:.4f}", flush=True)

    pooled_r2 = r2_score(np.array(pooled_true), np.array(pooled_pred))
    print(f"\nPooled LOFO R2_log = {pooled_r2:.4f}")
    for f in FACILITIES:
        print(f"  {f}: {per_facility[f]:.4f}")

    clears = pooled_r2 > 0.1 and all(per_facility[f] >= 0 for f in FACILITIES)

    results = {
        "approach": "Deep Kernel Learning GP (torch nn.Module feature extractor -> gpytorch ExactGP RBF-ARD kernel, jointly trained on exact marginal log likelihood)",
        "slug": "dklgp",
        "substituted_proxy": False,
        "libraries_used": ["torch", "gpytorch", "numpy", "pandas"],
        "hyperparameters": {
            "hidden_layers": HIDDEN,
            "latent_dim": LATENT_DIM,
            "n_epochs": N_EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seeds_per_fold": SEEDS,
            "seed_selection": "best 3 of 5 seeds by lowest final train negative-MLL, averaged (no test-set peeking)",
        },
        "features": FEATURES,
        "target": TARGET,
        "space": "log10",
        "pooled_r2_log": pooled_r2,
        "per_facility_r2_log": per_facility,
        "n_per_facility": {f: int((source == f).sum()) for f in FACILITIES},
        "decision_rule": "pooled R2 substantially positive AND every per-facility R2 >= 0",
        "clears_decision_rule": bool(clears),
        "seed_diagnostics": seed_diagnostics,
        "known_baselines": {
            "best_linear_4pred_pooled_r2_log": -0.885,
            "reynolds_only_linear_pooled_r2_log": 0.4526,
            "reynolds_only_linear_per_facility_note": "positive pooled R2 hides -2.36, -61.7, -9.59 on three facilities -- fails decision rule",
            "already_failed_methods_range": "-0.55 to -2.67 pooled R2_log (plain MLP, RF, GBM, KRR x2, plain GP, symbolic regression, Huber)",
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUT_JSON}")
    return results


if __name__ == "__main__":
    run_lofo()
