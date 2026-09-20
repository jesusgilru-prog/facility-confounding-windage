"""Set-Transformer / attention-pooling domain-invariant regressor under real
leave-one-facility-out (LOFO) validation, for the 'Windage Power' paper
(2026-08-18). Assigned slug: set_attention.

IDEA (assigned approach, not a repeat of anything already tried): treat each
LOFO training fold as an unordered SET of (x, log Cp) points drawn from up to
3 facilities. A shared encoder MLP embeds every point. A self-attention block
(a single-head "SAB", Set-Transformer style) lets the context points attend
to EACH OTHER, mixing information across facilities inside one embedding
space -- the hoped-for effect is a facility-invariant representation of the
"shared physical structure" (the idea explicitly named in the task). A held-
out query point is then embedded with the SAME encoder and predicted by
CROSS-ATTENDING over the (self-attended) context points: the prediction is a
convex combination (softmax attention weights) of the ACTUAL observed
log(Cp) values of the context points, i.e. a learned, embedding-space kernel
regression / matching-network-style in-context predictor. This is different
in kind from every approach already tried (plain MLP, RF, GBM, kernel ridge,
plain GP, symbolic regression, Huber/robust regression): the prediction
mechanism itself is non-parametric attention over training instances, not a
global parametric function fit.

Library: torch IS available in this project's .venv (2.13.0+cpu, confirmed
by `python -c "import torch"` inside .venv before writing this script) --
this is a genuine implementation, not a proxy substitution.

Inputs (log-log space, matching the corpus convention stated in task
context): log(Re_Omega), log(Pi_confinement), log(Pi_gap), log(Pi_aspect_axial).
Target: log(Cp). Pi_blockage is excluded per task instructions (exact
algebraic identity of Pi_aspect_axial and Pi_confinement).

Protocol (mandatory, matches the paper's established bar exactly):
  - True LOFO: for each of the 4 facilities, train on the other 3, predict
    the held-out one.
  - Feature standardization fit on the training fold ONLY.
  - Hyperparameters (embedding dim, weight decay, epochs) selected per outer
    fold via a NESTED inner-LOFO loop over the 3 training facilities only,
    exactly as in mlp_lofo.py -- the held-out outer facility is NEVER used
    for any decision.
  - Stochastic init -> every outer-fold evaluation is repeated over multiple
    seeds and reported as mean +/- std (not a single lucky draw).
  - Pooled R2_log = 1 - RSS/TSS over the residuals of all 4 held-out
    facilities concatenated, TSS taken about the GLOBAL log(Cp) mean.

Training regime for the attention model (this is the part specific to this
architecture): within one outer training fold, we do NOT fit one global
function. Instead, on every epoch we run a "leave-one-out-within-the-set"
forward pass: every training point in turn plays the role of the query,
context = all OTHER training points in that same fold (diagonal masked out
of both the self-attention block and the prediction attention), and the
model is trained to predict that point's log(Cp) from attention over the
rest of the set. This is exactly the in-context/metric-learning regime the
task asks for, and it is what teaches the encoder+attention weights to
generalize to a genuinely new facility at test time (where context = ALL 3
training facilities, unmasked, and the query is the held-out facility).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json
import warnings

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_OK = False
    TORCH_VERSION = None

warnings.filterwarnings("ignore")

RESULTS_PATH = _ROOT + "/results/ai_search_set_attention_results.json"
DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"

RNG_SEED = 20260818
N_SEEDS = 8          # outer-fold final evaluation repeats (stochastic init)
N_SEEDS_INNER = 2    # inner hyperparam-selection repeats (kept cheap)

torch.manual_seed(RNG_SEED) if TORCH_OK else None

# ---------------------------------------------------------------- data ----
df = pd.read_csv(DATA_PATH)
d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)
n = len(d)

y_all = np.log(d["Cp"].values).astype(np.float64)
X_all = np.column_stack([
    np.log(d["Re_Omega"].values),
    np.log(d["Pi_confinement"].values),
    np.log(d["Pi_gap"].values),
    np.log(d["Pi_aspect_axial"].values),
]).astype(np.float64)
sources = d["source"].values
facilities = sorted(set(sources))

FEATURE_NAMES = ["log_Re_Omega", "log_Pi_confinement", "log_Pi_gap", "log_Pi_aspect_axial"]


# ---------------------------------------------------------- model def -----
class SetAttentionRegressor(nn.Module):
    """Shared encoder -> self-attention context refinement (SAB) ->
    cross-attention prediction as a convex combination of context targets.
    Deliberately small given n<=106 points per training fold."""

    def __init__(self, in_dim=4, hidden=24, embed_dim=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, embed_dim),
        )
        # SAB (set self-attention block), single head, explicit so the
        # "value" path used by the prediction head can stay raw targets.
        self.Wq1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                 nn.Linear(embed_dim, embed_dim))
        self.ln2 = nn.LayerNorm(embed_dim)
        # prediction (cross) attention: query embeds attend over refined
        # context embeds; the VALUES combined are the raw scalar targets,
        # not a learned projection, so the prediction is honestly a
        # metric-learned weighted average of observed log(Cp).
        self.Wq2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(1))  # learnable softmax temperature

    def _sab(self, h, mask=None):
        # h: (n, d). mask: (n, n) additive mask (0 or -inf), or None.
        q, k, v = self.Wq1(h), self.Wk1(h), self.Wv1(h)
        scores = q @ k.t() / np.sqrt(self.embed_dim)
        if mask is not None:
            scores = scores + mask
        w = torch.softmax(scores, dim=-1)
        ctx = w @ v
        h1 = self.ln1(h + ctx)
        h2 = self.ln2(h1 + self.ff(h1))
        return h2

    def forward(self, X_ctx, y_ctx, X_query, loo_mask_ctx=None, loo_mask_pred=None):
        """X_ctx: (n_ctx, in_dim) context inputs. y_ctx: (n_ctx,) context
        targets (raw log Cp). X_query: (n_query, in_dim) query inputs.
        loo_mask_ctx: additive (n_ctx, n_ctx) mask for the SAB (diagonal
        -inf during leave-one-out training). loo_mask_pred: additive
        (n_query, n_ctx) mask for the prediction attention (diagonal -inf
        when query points are literally the same points as the context,
        i.e. during LOO training; None at real test time)."""
        h_ctx = self.encoder(X_ctx)
        h_ctx_ref = self._sab(h_ctx, mask=loo_mask_ctx)
        h_q = self.encoder(X_query)
        qv = self.Wq2(h_q)
        kv = self.Wk2(h_ctx_ref)
        scores = qv @ kv.t() / np.sqrt(self.embed_dim) * torch.exp(self.log_tau)
        if loo_mask_pred is not None:
            scores = scores + loo_mask_pred
        w = torch.softmax(scores, dim=-1)
        y_hat = w @ y_ctx
        return y_hat


def standardize_fit(X):
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def train_model(X_train, y_train, embed_dim, weight_decay, epochs, seed, lr=1e-2, hidden=24):
    """Trains via leave-one-out-within-the-training-set attention regression."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_tr = X_train.shape[0]
    model = SetAttentionRegressor(in_dim=X_train.shape[1], hidden=hidden, embed_dim=embed_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)

    diag_mask = torch.eye(n_tr) * (-1e9)  # used both for SAB and for prediction (ctx==query set here)

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        y_hat = model(Xt, yt, Xt, loo_mask_ctx=diag_mask, loo_mask_pred=diag_mask)
        loss = torch.mean((y_hat - yt) ** 2)
        loss.backward()
        opt.step()
    return model


def predict(model, X_train, y_train, X_test):
    model.eval()
    with torch.no_grad():
        Xtr_t = torch.tensor(X_train, dtype=torch.float32)
        ytr_t = torch.tensor(y_train, dtype=torch.float32)
        Xte_t = torch.tensor(X_test, dtype=torch.float32)
        y_hat = model(Xtr_t, ytr_t, Xte_t, loo_mask_ctx=None, loo_mask_pred=None)
    return y_hat.numpy()


def fit_eval_once(train_idx, test_idx, embed_dim, weight_decay, epochs, seed, hidden=24):
    Xtr_raw, ytr = X_all[train_idx], y_all[train_idx]
    Xte_raw, yte = X_all[test_idx], y_all[test_idx]
    mu, sd = standardize_fit(Xtr_raw)
    Xtr = standardize_apply(Xtr_raw, mu, sd)
    Xte = standardize_apply(Xte_raw, mu, sd)
    model = train_model(Xtr, ytr, embed_dim, weight_decay, epochs, seed, hidden=hidden)
    pred = predict(model, Xtr, ytr, Xte)
    return yte - pred


HIDDEN = 24
EMBED_DIM_GRID = [8, 12]
WEIGHT_DECAY_GRID = [0.0, 1e-3]
EPOCHS_GRID = [150, 300]


def inner_select_hyperparams(train_facility_mask):
    """Nested LOFO restricted to the 3 outer-training facilities. Selects
    (embed_dim, weight_decay, epochs) by mean pooled R2_log across the
    inner folds (2 seeds each, averaged), never touching the outer held-out
    facility."""
    inner_facilities = sorted(set(sources[train_facility_mask]))
    train_idx_global = np.where(train_facility_mask)[0]
    if len(inner_facilities) < 2:
        return 8, 0.0, 150, {"note": "inner LOFO skipped, <2 facilities; used default hyperparams"}

    best_r2, best_combo = -np.inf, None
    search_log = []
    for embed_dim in EMBED_DIM_GRID:
        for wd in WEIGHT_DECAY_GRID:
            for ep in EPOCHS_GRID:
                resids_seed_avg = []
                for f in inner_facilities:
                    inner_test_mask = sources[train_idx_global] == f
                    inner_test_idx = train_idx_global[inner_test_mask]
                    inner_train_idx = train_idx_global[~inner_test_mask]
                    if len(inner_train_idx) < 3 or len(inner_test_idx) < 1:
                        continue
                    seed_resids = []
                    for s in range(N_SEEDS_INNER):
                        r = fit_eval_once(inner_train_idx, inner_test_idx, embed_dim, wd, ep,
                                          seed=RNG_SEED + 1000 * s + 7, hidden=HIDDEN)
                        seed_resids.append(r)
                    resids_seed_avg.append(np.mean(seed_resids, axis=0))
                if not resids_seed_avg:
                    continue
                resids = np.concatenate(resids_seed_avg)
                y_train_full = y_all[train_idx_global]
                tss = np.sum((y_train_full - y_train_full.mean()) ** 2)
                r2 = float(1 - np.sum(resids ** 2) / tss) if tss > 0 else -np.inf
                search_log.append({"embed_dim": embed_dim, "weight_decay": wd, "epochs": ep,
                                    "inner_lofo_pooled_r2": r2})
                if r2 > best_r2:
                    best_r2, best_combo = r2, (embed_dim, wd, ep)
    if best_combo is None:
        return 8, 0.0, 150, {"note": "inner search produced no valid combo, used default", "search_log": search_log}
    return best_combo[0], best_combo[1], best_combo[2], {
        "inner_lofo_pooled_r2_at_selection": best_r2, "search_log": search_log}


def pooled_r2_log(resid_concat, y_full):
    tss = np.sum((y_full - y_full.mean()) ** 2)
    rss = np.sum(resid_concat ** 2)
    return float(1 - rss / tss)


def run():
    outer_results = {}
    all_resid_mean_over_seeds = []
    selection_log = {}

    for f in facilities:
        test_mask = sources == f
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        embed_dim, wd, epochs, sel_info = inner_select_hyperparams(train_mask)
        selection_log[f] = {"selected_embed_dim": embed_dim, "selected_weight_decay": wd,
                             "selected_epochs": epochs, **sel_info}

        seed_resids = []
        seed_r2s = []
        for s in range(N_SEEDS):
            seed = RNG_SEED + s
            resid = fit_eval_once(train_idx, test_idx, embed_dim, wd, epochs, seed, hidden=HIDDEN)
            seed_resids.append(resid)
            y_test = y_all[test_idx]
            tss_f = np.sum((y_test - y_test.mean()) ** 2)
            r2_f = float(1 - np.sum(resid ** 2) / tss_f) if tss_f > 0 else None
            seed_r2s.append(r2_f)
        seed_resids = np.array(seed_resids)
        mean_resid = seed_resids.mean(axis=0)
        all_resid_mean_over_seeds.append(mean_resid)

        outer_results[f] = {
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
            "selected_embed_dim": embed_dim,
            "selected_weight_decay": wd,
            "selected_epochs": epochs,
            "held_out_r2_log_per_seed": seed_r2s,
            "held_out_r2_log_mean": float(np.mean(seed_r2s)),
            "held_out_r2_log_std": float(np.std(seed_r2s)),
            "held_out_rmse_log_mean_of_seed_means": float(np.sqrt(np.mean(mean_resid ** 2))),
        }
        print(f"  {f:15s} n_test={test_mask.sum():3d}  embed_dim={embed_dim} wd={wd} epochs={epochs}  "
              f"R2_mean={outer_results[f]['held_out_r2_log_mean']:.4f}  "
              f"R2_std={outer_results[f]['held_out_r2_log_std']:.4f}")

    resid_concat = np.concatenate(all_resid_mean_over_seeds)
    pooled_r2 = pooled_r2_log(resid_concat, y_all)
    pooled_rmse = float(np.sqrt(np.mean(resid_concat ** 2)))
    return outer_results, selection_log, pooled_r2, pooled_rmse


if not TORCH_OK:
    out = {
        "error": "torch not available even inside .venv -- unexpected, could not run the "
                 "attention model. No proxy substitution attempted for this failure mode "
                 "because torch import succeeded in the pre-flight check.",
        "torch_available": False,
    }
    with open(RESULTS_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    raise SystemExit(0)

print(f"torch {TORCH_VERSION} OK. n={n} points, facilities: "
      f"{dict(pd.Series(sources).value_counts())}")
print("=== Running Set-Attention (Set-Transformer-style) LOFO regression ===")
outer_results, selection_log, pooled_r2, pooled_rmse = run()

worst_facility_r2 = min(r["held_out_r2_log_mean"] for r in outer_results.values())
clears_decision_rule = bool(pooled_r2 > 0.1 and all(r["held_out_r2_log_mean"] >= 0 for r in outer_results.values()))

linear_baseline_reference = {
    "note": "Reproduced from prior work / task context, NOT recomputed in this script. "
            "Shown only as a comparison anchor.",
    "linear_full_4pred_pooled_r2_log": -0.885,
    "linear_re_only_pooled_r2_log": 0.4526,
    "linear_re_only_per_facility_r2_log": {
        "Liu2024": -61.7, "Guo2024": -2.36, "Zheng2024": -9.59, "Vrancik1968": 0.468,
    },
    "already_failed_decision_rule": {
        "approaches": ["plain MLP", "Random Forest", "Gradient Boosting",
                       "kernel ridge regression (2 variants)", "plain (non-hierarchical) GP",
                       "symbolic regression (genetic search)", "Huber/robust regression"],
        "pooled_r2_log_range": [-2.67, -0.55],
    },
}

out = {
    "slug": "set_attention",
    "approach": "Set-Transformer-style attention pooling: shared MLP encoder + single-head "
                "self-attention (SAB) refines a facility-mixed context embedding; a query "
                "point's prediction is a softmax-attention-weighted convex combination of the "
                "context points' OBSERVED log(Cp) values (in-context / metric-learning "
                "regression), trained via leave-one-out-within-the-training-set.",
    "n_points": int(n),
    "sources": {k: int(v) for k, v in pd.Series(sources).value_counts().to_dict().items()},
    "features": FEATURE_NAMES,
    "target": "log(Cp)",
    "library_used": "torch",
    "torch_version": TORCH_VERSION,
    "substituted_proxy": False,
    "n_seeds_outer": N_SEEDS,
    "n_seeds_inner": N_SEEDS_INNER,
    "rng_seed_base": RNG_SEED,
    "hyperparam_search_grid": {"embed_dim": EMBED_DIM_GRID, "weight_decay": WEIGHT_DECAY_GRID,
                                "epochs": EPOCHS_GRID, "hidden": HIDDEN},
    "validation_protocol": "True LOFO across all 4 facilities. StandardScaler fit on training "
                            "fold only. Hyperparameters selected per outer fold via NESTED "
                            "inner-LOFO over the 3 training facilities only (2 seeds averaged "
                            "per inner fold), never touching the outer held-out facility. Final "
                            "outer-fold evaluation repeated over 8 seeds (stochastic init), "
                            "mean residual per point used to build the pooled residual vector.",
    "per_facility": outer_results,
    "hyperparam_selection_per_outer_fold": selection_log,
    "pooled_r2_log": pooled_r2,
    "pooled_rmse_log": pooled_rmse,
    "worst_facility_r2_log": worst_facility_r2,
    "clears_decision_rule": clears_decision_rule,
    "decision_rule": "pooled R2_log substantially positive AND every per-facility held-out "
                      "R2_log >= 0.",
    "linear_baseline_reference_from_prior_work": linear_baseline_reference,
}

with open(RESULTS_PATH, "w") as fh:
    json.dump(out, fh, indent=2, default=float)

print("\n=== SUMMARY ===")
print(f"Pooled LOFO R2 (log-space, mean over {N_SEEDS} seeds): {pooled_r2:.4f}")
print(f"Worst per-facility R2: {worst_facility_r2:.4f}")
print(f"Clears decision rule (pooled>>0 AND all facilities>=0): {clears_decision_rule}")
for f, r in outer_results.items():
    print(f"  {f:15s} n={r['n_test']:3d}  R2_mean={r['held_out_r2_log_mean']:.4f}  "
          f"R2_std={r['held_out_r2_log_std']:.4f}")
print("\nResults written to", RESULTS_PATH)
