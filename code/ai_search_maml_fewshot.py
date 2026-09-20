"""Few-shot gradient-based meta-learning (MAML) LOFO probe (2026-08-18).

Assigned approach: proper gradient-based meta-learning (MAML-style), NOT the
simple empirical-Bayes intercept shrinkage of hierarchical_calibration_lofo.py
and NOT the plain-pretrain-then-fine-tune MLP of mlp_finetune_lofo.py.

Difference vs. mlp_finetune_lofo.py (already tried, already failed):
  mlp_finetune_lofo.py pre-trains a plain MLP by pooling the 3 training
  facilities together (ordinary supervised training, single task), then
  fine-tunes with a few SGD steps on the held-out facility's calibration
  points. That initialization was never explicitly optimized to BE fast to
  adapt -- it is just a good pooled fit.

  Here we run actual model-agnostic meta-learning (Finn et al. 2017, MAML):
  each of the 3 training facilities is treated as a separate few-shot TASK.
  In every meta-iteration, for each training-facility task we (a) sample a
  support set of n_shot points, (b) take INNER_STEPS gradient-descent steps
  on the support set starting from the shared meta-parameters theta
  (with create_graph=True so the adaptation is differentiable), producing
  adapted parameters theta', and (c) evaluate theta' on the task's query
  set (the rest of that facility's points). The QUERY loss is what drives
  the OUTER (meta) gradient update on theta itself, back-propagated through
  the inner-loop steps -- full second-order MAML, not first-order/Reptile
  approximation, since the network is tiny (4->8->1, 49 params) and this is
  computationally trivial on CPU.

  theta after meta-training is therefore explicitly optimized so that a
  HANDFUL of gradient steps on a NEW facility's calibration points produces
  a good fit on the rest of that facility -- the actual MAML promise, as
  opposed to mlp_finetune_lofo.py's incidental transfer from pooled
  pretraining.

Library: PyTorch (CPU). Not pre-installed in this environment; installed
via `pip install --user --break-system-packages torch --index-url
https://download.pytorch.org/whl/cpu` (succeeded, torch 2.13.0+cpu). No
proxy/approximation was needed -- this is real MAML with real
second-order gradients through the inner loop.

Protocol (LOFO, matches the paper's established bar exactly):
  For each held-out facility f:
    - Meta-train theta using ONLY the other 3 facilities as tasks (f is
      NEVER seen during meta-training, at any n_shot).
    - n_cal=0: zero-shot evaluation of theta (no adaptation) on ALL of f.
    - For n_cal in {1,3,5,10}, 20 repeats: sample n_cal calibration points
      from f (rng.choice, no replacement, deterministic seed), take
      INNER_STEPS gradient steps from theta on those points (same inner
      recipe used at meta-train time), evaluate R2_log on the REMAINING
      points of f only. Folds where n_f - n_cal < 2 (Zheng2024, n=8, at
      n_cal=10) are marked infeasible, never filled with an invented
      number, exactly as in the sibling scripts.
    - Pooled LOFO R2_log per n_cal: residuals of all 4 held-out facilities
      concatenated within each repetition, then averaged over repetitions
      (same coupled-sampling construction as mlp_finetune_lofo.py).

Features (identical to sibling few-shot scripts, so the comparison is
apples-to-apples): log(Re_Omega), log(Pi_gap), log(Pi_confinement),
log(Pi_aspect_axial) -> log(Cp). Standardized (z-score) using ONLY the 3
training facilities of each fold, pooled.

Direct comparison requested (existing James-Stein pooled LOFO R2_log,
from hierarchical_calibration_lofo_results.json):
  n_cal = 0, 1, 3, 5, 10 -> -0.885, -0.859, -0.818, -0.743, -0.683

Hyperparameters (H=8 hidden units, inner steps, inner/outer lr, meta
iterations) were fixed a priori by standard ML judgment (small net for
~100 total data points, few inner steps as is standard for MAML, outer
Adam) -- NOT tuned by looking at any held-out facility's LOFO R2, which
would violate the protocol. Meta-training convergence (query-set MSE
decreasing over meta-iterations, measured on the 3 training-facility
tasks only) is logged as a diagnostic before trusting the LOFO numbers.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json

import numpy as np
import pandas as pd
import torch

RNG_SEED = 20260818
N_REPEATS = 20
N_CAL_GRID = [1, 3, 5, 10]
MIN_TEST_POINTS = 2

HIDDEN = 8
INNER_STEPS = 5
INNER_LR = 0.1
META_LR = 0.01
META_WEIGHT_DECAY = 1e-3
META_ITERS = 800

torch.set_default_dtype(torch.float64)

DATA_PATH = _ROOT + "/data/cross_rotor_dataset_v3.csv"
OUT_PATH = _ROOT + "/results/ai_search_maml_fewshot_results.json"

df = pd.read_csv(DATA_PATH)
d = df.dropna(subset=["Cp", "Re_Omega", "Pi_gap", "Pi_confinement",
                       "Pi_aspect_axial", "source"]).copy()
d = d[d["Cp"] > 0].reset_index(drop=True)

y_raw = np.log(d["Cp"].values)
lRe = np.log(d["Re_Omega"].values)
lgap = np.log(d["Pi_gap"].values)
lconf = np.log(d["Pi_confinement"].values)
lasp = np.log(d["Pi_aspect_axial"].values)
sources = d["source"].values
facilities = sorted(set(sources))

X_raw = np.column_stack([lRe, lgap, lconf, lasp])  # (n,4)


def r2_log(y_true, y_pred):
    if len(y_true) < 2:
        return None
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return None
    return float(1 - ss_res / ss_tot)


def standardize_fit(X, y):
    mu_X, sd_X = X.mean(axis=0), X.std(axis=0)
    sd_X[sd_X == 0] = 1.0
    mu_y, sd_y = y.mean(), y.std()
    if sd_y == 0:
        sd_y = 1.0
    return mu_X, sd_X, mu_y, sd_y


def standardize_apply(X, y, mu_X, sd_X, mu_y, sd_y):
    Xs = (X - mu_X) / sd_X
    ys = None if y is None else (y - mu_y) / sd_y
    return Xs, ys


def unstandardize_y(y_std, mu_y, sd_y):
    return y_std * sd_y + mu_y


# ---------------- functional MLP (raw tensors, differentiable inner loop) ----------------

def init_meta_params(seed, n_in=4, hidden=HIDDEN):
    g = torch.Generator().manual_seed(seed)
    W1 = (torch.randn(n_in, hidden, generator=g) * np.sqrt(1.0 / n_in)).clone().requires_grad_(True)
    b1 = torch.zeros(hidden, requires_grad=True)
    W2 = (torch.randn(hidden, 1, generator=g) * np.sqrt(1.0 / hidden)).clone().requires_grad_(True)
    b2 = torch.zeros(1, requires_grad=True)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward(params, X_t):
    z1 = X_t @ params["W1"] + params["b1"]
    h = torch.tanh(z1)
    yhat = (h @ params["W2"] + params["b2"]).squeeze(-1)
    return yhat


def inner_adapt(params, X_supp_t, y_supp_t, steps, inner_lr, create_graph):
    p = dict(params)
    for _ in range(steps):
        yhat = forward(p, X_supp_t)
        loss = torch.mean((yhat - y_supp_t) ** 2)
        grads = torch.autograd.grad(loss, list(p.values()), create_graph=create_graph)
        p = {k: v - inner_lr * g for (k, v), g in zip(p.items(), grads)}
    return p


def feasible_shots_for(n_t, grid):
    return [s for s in grid if n_t - s >= MIN_TEST_POINTS]


results = {
    "per_facility": {},
    "pooled_by_n_cal": {},
    "meta_train_diagnostics": {},
    "hyperparameters": {
        "hidden": HIDDEN, "inner_steps": INNER_STEPS, "inner_lr": INNER_LR,
        "meta_lr": META_LR, "meta_weight_decay": META_WEIGHT_DECAY,
        "meta_iters": META_ITERS, "n_repeats": N_REPEATS, "n_cal_grid": N_CAL_GRID,
        "method": "second-order MAML (Finn et al. 2017), full backprop through "
                  "the inner loop via create_graph=True, PyTorch autograd.grad"},
}

meta_theta_cache = {}
pretrained_cache = {}

for f in facilities:
    test_mask = sources == f
    train_mask = ~test_mask
    f_idx_all = np.where(test_mask)[0]
    n_f = len(f_idx_all)
    train_facilities = [t for t in facilities if t != f]

    Xtr, ytr = X_raw[train_mask], y_raw[train_mask]
    mu_X, sd_X, mu_y, sd_y = standardize_fit(Xtr, ytr)

    # per-task (per training-facility) standardized arrays, as torch tensors
    task_data = {}
    for t in train_facilities:
        t_idx = np.where(sources == t)[0]
        Xt_s, yt_s = standardize_apply(X_raw[t_idx], y_raw[t_idx], mu_X, sd_X, mu_y, sd_y)
        task_data[t] = {
            "X": torch.tensor(Xt_s), "y": torch.tensor(yt_s), "n": len(t_idx),
            "feasible_shots": feasible_shots_for(len(t_idx), N_CAL_GRID),
        }

    fold_seed = RNG_SEED + sum(ord(c) for c in f) * 97
    meta_rng = np.random.default_rng(fold_seed + 1)
    theta = init_meta_params(fold_seed)
    optimizer = torch.optim.Adam(list(theta.values()), lr=META_LR, weight_decay=META_WEIGHT_DECAY)

    loss_trace = []
    for it in range(META_ITERS):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        for t in train_facilities:
            td = task_data[t]
            shots = td["feasible_shots"]
            n_shot = int(meta_rng.choice(shots))
            perm = meta_rng.permutation(td["n"])
            supp_idx = perm[:n_shot]
            query_idx = perm[n_shot:]
            X_supp = td["X"][supp_idx]
            y_supp = td["y"][supp_idx]
            X_query = td["X"][query_idx]
            y_query = td["y"][query_idx]
            theta_adapted = inner_adapt(theta, X_supp, y_supp, INNER_STEPS, INNER_LR,
                                         create_graph=True)
            yhat_q = forward(theta_adapted, X_query)
            loss_q = torch.mean((yhat_q - y_query) ** 2)
            total_loss = total_loss + loss_q
        total_loss = total_loss / len(train_facilities)
        total_loss.backward()
        optimizer.step()
        if it % 50 == 0 or it == META_ITERS - 1:
            loss_trace.append({"iter": it, "mean_query_mse_std_space": float(total_loss.item())})

    theta_frozen = {k: v.detach().clone() for k, v in theta.items()}
    meta_theta_cache[f] = (theta_frozen, mu_X, sd_X, mu_y, sd_y)
    results["meta_train_diagnostics"][f] = {
        "loss_trace_every_50_iters": loss_trace,
        "note": "mean query-set MSE (standardized log-space) across the 3 training-"
                "facility tasks, after adaptation, during meta-training. Should "
                "decrease and stabilize; this is NOT LOFO, only a convergence check.",
    }

    # ---- n_cal = 0: zero-shot on ALL of f ----
    def leaf(p):
        return {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}

    Xf_s, _ = standardize_apply(X_raw[f_idx_all], None, mu_X, sd_X, mu_y, sd_y)
    with torch.no_grad():
        yhat0_s = forward(theta_frozen, torch.tensor(Xf_s)).numpy()
    yhat0 = unstandardize_y(yhat0_s, mu_y, sd_y)
    r2_0 = r2_log(y_raw[f_idx_all], yhat0)

    results["per_facility"][f] = {"n_total_facility": int(n_f), "by_n_cal": {}}
    results["per_facility"][f]["by_n_cal"]["0"] = {
        "feasible": True, "mean_r2": r2_0, "std_r2": 0.0, "n_reps_used": 1,
        "note": "zero-shot: meta-learned theta, no inner-loop adaptation, evaluated on ALL of f."}

    eval_rng = np.random.default_rng(RNG_SEED + sum(ord(c) for c in f) * 13)
    for n_cal in N_CAL_GRID:
        n_test_after = n_f - n_cal
        if n_test_after < MIN_TEST_POINTS:
            results["per_facility"][f]["by_n_cal"][str(n_cal)] = {
                "feasible": False,
                "reason": f"facility {f} has only {n_f} points; holding out "
                          f"{n_cal} for calibration leaves {n_test_after} < "
                          f"{MIN_TEST_POINTS} for testing",
                "mean_r2": None, "std_r2": None, "n_reps_used": 0}
            continue
        rep_r2 = []
        for rep in range(N_REPEATS):
            cal_idx = eval_rng.choice(f_idx_all, size=n_cal, replace=False)
            test_idx = np.setdiff1d(f_idx_all, cal_idx, assume_unique=False)
            Xcal_s, ycal_s = standardize_apply(X_raw[cal_idx], y_raw[cal_idx], mu_X, sd_X, mu_y, sd_y)
            theta_ft = inner_adapt(leaf(theta_frozen), torch.tensor(Xcal_s), torch.tensor(ycal_s),
                                    INNER_STEPS, INNER_LR, create_graph=False)
            Xtest_s, _ = standardize_apply(X_raw[test_idx], None, mu_X, sd_X, mu_y, sd_y)
            with torch.no_grad():
                yhat_test_s = forward(theta_ft, torch.tensor(Xtest_s)).numpy()
            yhat_test = unstandardize_y(yhat_test_s, mu_y, sd_y)
            r2 = r2_log(y_raw[test_idx], yhat_test)
            if r2 is not None:
                rep_r2.append(r2)
        results["per_facility"][f]["by_n_cal"][str(n_cal)] = {
            "feasible": True,
            "mean_r2": float(np.mean(rep_r2)) if rep_r2 else None,
            "std_r2": float(np.std(rep_r2, ddof=1)) if len(rep_r2) > 1 else 0.0,
            "n_reps_used": len(rep_r2)}

# ---------------- pooled R2 per n_cal (coupled sampling across facilities) ----------------

# n_cal = 0 pooled (deterministic, single "repetition")
all_y0, all_pred0 = [], []
for f in facilities:
    f_idx_all = np.where(sources == f)[0]
    theta_frozen, mu_X, sd_X, mu_y, sd_y = meta_theta_cache[f]
    Xf_s, _ = standardize_apply(X_raw[f_idx_all], None, mu_X, sd_X, mu_y, sd_y)
    with torch.no_grad():
        yhat0_s = forward(theta_frozen, torch.tensor(Xf_s)).numpy()
    yhat0 = unstandardize_y(yhat0_s, mu_y, sd_y)
    all_y0.append(y_raw[f_idx_all]); all_pred0.append(yhat0)
all_y0 = np.concatenate(all_y0); all_pred0 = np.concatenate(all_pred0)
results["pooled_by_n_cal"]["0"] = {
    "mean_r2_pooled": r2_log(all_y0, all_pred0), "std_r2_pooled": 0.0,
    "n_reps_used": 1, "note": "zero-shot meta-learned theta, all 4 held-out facilities pooled."}

rep_pooled_by_ncal = {nc: [] for nc in N_CAL_GRID}
for f in facilities:
    f_idx_all = np.where(sources == f)[0]
    n_f = len(f_idx_all)
    theta_frozen, mu_X, sd_X, mu_y, sd_y = meta_theta_cache[f]
    eval_rng = np.random.default_rng(RNG_SEED + sum(ord(c) for c in f) * 13)  # identical to per-facility loop above
    for n_cal in N_CAL_GRID:
        n_test_after = n_f - n_cal
        if n_test_after < MIN_TEST_POINTS:
            continue
        for rep in range(N_REPEATS):
            cal_idx = eval_rng.choice(f_idx_all, size=n_cal, replace=False)
            test_idx = np.setdiff1d(f_idx_all, cal_idx, assume_unique=False)
            Xcal_s, ycal_s = standardize_apply(X_raw[cal_idx], y_raw[cal_idx], mu_X, sd_X, mu_y, sd_y)
            theta_ft = inner_adapt({k: v.detach().clone().requires_grad_(True) for k, v in theta_frozen.items()},
                                    torch.tensor(Xcal_s), torch.tensor(ycal_s),
                                    INNER_STEPS, INNER_LR, create_graph=False)
            Xtest_s, _ = standardize_apply(X_raw[test_idx], None, mu_X, sd_X, mu_y, sd_y)
            with torch.no_grad():
                yhat_test_s = forward(theta_ft, torch.tensor(Xtest_s)).numpy()
            yhat_test = unstandardize_y(yhat_test_s, mu_y, sd_y)
            rep_pooled_by_ncal[n_cal].append((rep, f, y_raw[test_idx], yhat_test))

for n_cal in N_CAL_GRID:
    per_rep = {}
    for rep, f, yt, yp in rep_pooled_by_ncal[n_cal]:
        per_rep.setdefault(rep, {"y": [], "p": [], "facilities": set()})
        per_rep[rep]["y"].append(yt)
        per_rep[rep]["p"].append(yp)
        per_rep[rep]["facilities"].add(f)
    rep_r2_vals = []
    n_facilities_contributing = len(set(f for _, f, _, _ in rep_pooled_by_ncal[n_cal]))
    for rep, dd in per_rep.items():
        yy = np.concatenate(dd["y"]); pp = np.concatenate(dd["p"])
        r2 = r2_log(yy, pp)
        if r2 is not None:
            rep_r2_vals.append(r2)
    zheng_skipped = n_facilities_contributing < len(facilities)
    results["pooled_by_n_cal"][str(n_cal)] = {
        "mean_r2_pooled": float(np.mean(rep_r2_vals)) if rep_r2_vals else None,
        "std_r2_pooled": float(np.std(rep_r2_vals, ddof=1)) if len(rep_r2_vals) > 1 else 0.0,
        "n_reps_used": len(rep_r2_vals),
        "any_rep_skipped_zheng2024": bool(zheng_skipped),
        "note": "all 4 facilities contribute" if not zheng_skipped else
                "Zheng2024 (n=8) cannot supply test points for this pooled round when "
                "n_f - n_cal < 2; excluded ONLY from this pooled round."}

js_baseline = {"0": -0.8848004392997484, "1": -0.8594553795566364,
               "3": -0.8179778180758814, "5": -0.7428989686137879,
               "10": -0.683270527499946}
comparison = {}
for k in ["0", "1", "3", "5", "10"]:
    maml_r2 = results["pooled_by_n_cal"][k]["mean_r2_pooled"]
    comparison[k] = {"james_stein_pooled_r2": js_baseline[k],
                      "maml_pooled_r2": maml_r2,
                      "maml_minus_js": (maml_r2 - js_baseline[k]) if maml_r2 is not None else None}
results["comparison_vs_james_stein"] = comparison

# ---- decision-rule check at n_cal=5: the largest grid point where ALL 4 facilities
# (including Zheng2024, n=8) still have >=2 held-out test points, i.e. the fairest
# "most information used" headline that still includes every facility. ----
headline_n_cal = "5"
per_facility_headline = {f: results["per_facility"][f]["by_n_cal"][headline_n_cal]["mean_r2"]
                          for f in facilities}
pooled_headline = results["pooled_by_n_cal"][headline_n_cal]["mean_r2_pooled"]
clears_rule = (pooled_headline is not None and pooled_headline > 0 and
               all(v is not None and v >= 0 for v in per_facility_headline.values()))
results["headline"] = {
    "n_cal": headline_n_cal,
    "pooled_r2_log": pooled_headline,
    "per_facility_r2_log": per_facility_headline,
    "decision_rule_clears": bool(clears_rule),
    "decision_rule": "pooled R2 substantially positive AND every per-facility "
                      "held-out R2 >= 0; chosen at n_cal=5, the largest grid "
                      "point where all 4 facilities (incl. Zheng2024, n=8) "
                      "still have >=2 held-out test points.",
}

with open(OUT_PATH, "w") as fh:
    json.dump(results, fh, indent=2)

print(json.dumps(results["meta_train_diagnostics"], indent=2, default=str)[:2000])
print(json.dumps(results["pooled_by_n_cal"], indent=2))
print(json.dumps(comparison, indent=2))
print(json.dumps(results["headline"], indent=2))
