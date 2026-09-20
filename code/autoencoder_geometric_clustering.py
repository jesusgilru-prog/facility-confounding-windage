"""
Diagnostic (NOT predictive) analysis: does a small nonlinear autoencoder's
latent space separate the 4 facilities (Guo2024/Xia2024, Vrancik1968, Liu2024,
Zheng2024) any differently than a linear PCA does, when trained on the
log-transformed dimensionless geometric Pi-groups (Pi_confinement, Pi_gap,
Pi_aspect_axial, Pi_blockage)?

This does NOT touch the target P_w_W / Cp at all -- it is purely an
unsupervised geometry-clustering diagnostic, following directly from the
already-established finding that these Pi groups are near-constant within
each facility (source-dummy behavior).

Environment: conda env 'base' has torch 2.10 + scikit-learn 1.7.2 + pandas/numpy.
The project-specific 'windage' conda env has scikit-learn/pandas but NOT torch.
So this script is run with `conda run -n base python ...`.
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness

torch.manual_seed(0)
np.random.seed(0)

DATA = _ROOT + "/data/cross_rotor_dataset_v3.csv"
GEOM_COLS = ["Pi_confinement", "Pi_gap", "Pi_aspect_axial", "Pi_blockage"]
LATENT_DIM = 2

df = pd.read_csv(DATA)
assert df[GEOM_COLS].isna().sum().sum() == 0
assert (df[GEOM_COLS] > 0).all().all()

X_log = np.log(df[GEOM_COLS].values)
scaler = StandardScaler()
X = scaler.fit_transform(X_log)
sources = df["source"].values
uniq_sources = sorted(pd.unique(sources))
n, d = X.shape
print(f"n={n} rows, d={d} features (log-Pi groups), sources={uniq_sources}")

# ---------------------------------------------------------------------------
# 1) Linear PCA baseline (2 components)
# ---------------------------------------------------------------------------
pca = PCA(n_components=LATENT_DIM, random_state=0)
Z_pca = pca.fit_transform(X)
pca_recon = pca.inverse_transform(Z_pca)
pca_mse = float(np.mean((X - pca_recon) ** 2))
print(f"\nPCA explained variance ratio (2 comps): {pca.explained_variance_ratio_}")
print(f"PCA reconstruction MSE (standardized log space): {pca_mse:.5f}")

# ---------------------------------------------------------------------------
# 2) Small nonlinear autoencoder (bottleneck = 2), pytorch
# ---------------------------------------------------------------------------
class AE(nn.Module):
    def __init__(self, d_in, d_latent=2, hidden=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, hidden), nn.Tanh(),
            nn.Linear(hidden, d_latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.Tanh(),
            nn.Linear(hidden, d_in),
        )

    def forward(self, x):
        z = self.enc(x)
        xhat = self.dec(z)
        return xhat, z


X_t = torch.tensor(X, dtype=torch.float32)

best_state = None
best_loss = np.inf
# Multiple random restarts (small data, small net -> can land in bad local minima)
for restart in range(10):
    torch.manual_seed(restart)
    model = AE(d_in=d, d_latent=LATENT_DIM, hidden=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for epoch in range(2000):
        opt.zero_grad()
        xhat, z = model(X_t)
        loss = loss_fn(xhat, X_t)
        loss.backward()
        opt.step()
    final_loss = loss.item()
    if final_loss < best_loss:
        best_loss = final_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

model = AE(d_in=d, d_latent=LATENT_DIM, hidden=8)
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    Xhat, Z_ae_t = model(X_t)
    ae_mse = loss_fn(Xhat, X_t).item()
Z_ae = Z_ae_t.numpy()

print(f"\nAutoencoder (best of 10 restarts, 2000 epochs each) reconstruction MSE: {ae_mse:.5f}")
print(f"PCA reconstruction MSE:                                                {pca_mse:.5f}")
print(f"AE improves reconstruction over linear PCA by: {(1 - ae_mse/pca_mse)*100:.1f}%")

# ---------------------------------------------------------------------------
# 3) Cluster separation diagnostics in each latent space
# ---------------------------------------------------------------------------
def cluster_summary(Z, label, sources, uniq_sources):
    print(f"\n--- {label} latent coordinates (mean +/- std per facility) ---")
    centroids = {}
    for s in uniq_sources:
        m = sources == s
        pts = Z[m]
        cen = pts.mean(axis=0)
        centroids[s] = cen
        std = pts.std(axis=0)
        rng0 = (pts[:, 0].min(), pts[:, 0].max())
        rng1 = (pts[:, 1].min(), pts[:, 1].max())
        print(f"  {s:14s} n={m.sum():3d}  centroid=({cen[0]:+.3f},{cen[1]:+.3f})  "
              f"std=({std[0]:.3f},{std[1]:.3f})  range_d0=({rng0[0]:+.3f},{rng0[1]:+.3f})  "
              f"range_d1=({rng1[0]:+.3f},{rng1[1]:+.3f})")
    # pairwise centroid distances
    print("  pairwise centroid distances:")
    for i, s1 in enumerate(uniq_sources):
        for s2 in uniq_sources[i + 1:]:
            dist = np.linalg.norm(centroids[s1] - centroids[s2])
            print(f"    {s1} <-> {s2}: {dist:.3f}")
    return centroids


def separation_score(Z, sources, uniq_sources):
    """Ratio of between-cluster to within-cluster scatter (like a simple
    Calinski-Harabasz numerator/denominator ratio), higher = better separated."""
    overall_mean = Z.mean(axis=0)
    between = 0.0
    within = 0.0
    for s in uniq_sources:
        m = sources == s
        pts = Z[m]
        cen = pts.mean(axis=0)
        between += m.sum() * np.sum((cen - overall_mean) ** 2)
        within += np.sum((pts - cen) ** 2)
    if within == 0:
        return np.inf
    k = len(uniq_sources)
    n_ = len(Z)
    return (between / (k - 1)) / (within / (n_ - k))


centroids_pca = cluster_summary(Z_pca, "PCA (linear)", sources, uniq_sources)
centroids_ae = cluster_summary(Z_ae, "Autoencoder (nonlinear)", sources, uniq_sources)

ch_pca = separation_score(Z_pca, sources, uniq_sources)
ch_ae = separation_score(Z_ae, sources, uniq_sources)
print(f"\nPseudo Calinski-Harabasz separation score (higher=more separated clusters):")
print(f"  PCA:          {ch_pca:.2f}")
print(f"  Autoencoder:  {ch_ae:.2f}")

# Trustworthiness: does the AE 2D embedding preserve the local neighborhood
# structure of the original 4D space any better than linear PCA does?
# (only meaningful measure of "did nonlinearity buy us anything")
tw_pca = trustworthiness(X, Z_pca, n_neighbors=5)
tw_ae = trustworthiness(X, Z_ae, n_neighbors=5)
print(f"\nTrustworthiness of 2D embedding (5-NN, 1.0=perfect local structure preserved):")
print(f"  PCA:          {tw_pca:.4f}")
print(f"  Autoencoder:  {tw_ae:.4f}")

# Correlation between PCA and AE latent axes (are they just a rotation/rescale
# of each other, i.e. did the AE learn something PCA didn't?)
from numpy.linalg import lstsq
# Best linear map from PCA space to AE space
A, *_ = lstsq(np.hstack([Z_pca, np.ones((n, 1))]), Z_ae, rcond=None)
Z_ae_pred_linear = np.hstack([Z_pca, np.ones((n, 1))]) @ A
resid = Z_ae - Z_ae_pred_linear
r2_linear_map = 1 - np.sum(resid ** 2) / np.sum((Z_ae - Z_ae.mean(axis=0)) ** 2)
print(f"\nR^2 of BEST LINEAR MAP from PCA(2D) -> AE(2D) latent space: {r2_linear_map:.4f}")
print("  (if close to 1.0, the AE latent space is essentially a linear")
print("   transform of the PCA space -> no real nonlinear structure found)")

print("\n=== SUMMARY ===")
print(f"AE reconstruction MSE vs PCA: {ae_mse:.5f} vs {pca_mse:.5f} "
      f"({'AE better' if ae_mse < pca_mse else 'PCA better or tied'})")
print(f"Separation score AE vs PCA: {ch_ae:.2f} vs {ch_pca:.2f}")
print(f"Trustworthiness AE vs PCA: {tw_ae:.4f} vs {tw_pca:.4f}")
print(f"Linear-map R^2 (PCA->AE): {r2_linear_map:.4f}")
