import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(_ROOT + '/data/cross_rotor_dataset_v3.csv')

print("N rows:", len(df))
print("Sources:", df['source'].value_counts().to_dict())

features = ['Pi_gap', 'Pi_confinement', 'Pi_aspect_axial', 'Re_Omega', 'M_tip']

# check for missing / non-positive values before log
sub = df[features].copy()
print("\nMissing per feature:\n", sub.isna().sum())
print("\nMin per feature (check positivity for log):\n", sub.min())

X = np.log(sub.values)
print("\nAny inf/nan after log?", np.isinf(X).any(), np.isnan(X).any())

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

true_labels = df['source'].values
true_codes = pd.Categorical(true_labels).codes

results = {}
print("\n=== GMM clustering on log-space geometric+flow features ===")
print("Features:", features)
for k in [2, 3, 4, 5]:
    best_bic = np.inf
    best_gmm = None
    best_labels = None
    # multiple inits for stability
    for seed in range(10):
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               random_state=seed, n_init=1, reg_covar=1e-6)
        gmm.fit(Xs)
        bic = gmm.bic(Xs)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_labels = gmm.predict(Xs)
    ari = adjusted_rand_score(true_codes, best_labels)
    nmi = normalized_mutual_info_score(true_codes, best_labels)
    results[k] = dict(ari=ari, nmi=nmi, bic=best_bic, labels=best_labels)
    print(f"\nk={k}: ARI vs source = {ari:.4f}, NMI = {nmi:.4f}, BIC = {best_bic:.2f}")
    ct = pd.crosstab(pd.Series(best_labels, name='gmm_cluster'),
                      pd.Series(true_labels, name='source'))
    print(ct)

# Also check: does GMM cluster align with Re_Omega ranges (regime) instead of source?
print("\n=== Re_Omega distribution by source (log10) ===")
print(df.groupby('source')['Re_Omega'].agg(lambda x: (np.log10(x.min()), np.log10(x.max()))))

# Best k by BIC overall
best_k_overall = min(results, key=lambda k: results[k]['bic'])
print(f"\nBest k by BIC overall: {best_k_overall}")

# Save summary
print("\n=== Summary table ===")
for k, r in results.items():
    print(f"k={k}: ARI={r['ari']:.4f} NMI={r['nmi']:.4f} BIC={r['bic']:.2f}")

# Additional: cluster on ONLY geometric Pi features (no Re, no M_tip) to isolate "geometry only" latent structure
print("\n=== GMM on geometric-Pi-only features (Pi_gap, Pi_confinement, Pi_aspect_axial) ===")
geom_features = ['Pi_gap', 'Pi_confinement', 'Pi_aspect_axial']
Xg = np.log(df[geom_features].values)
Xgs = StandardScaler().fit_transform(Xg)
for k in [2, 3, 4, 5]:
    best_bic = np.inf
    best_labels = None
    for seed in range(10):
        gmm = GaussianMixture(n_components=k, covariance_type='full',
                               random_state=seed, n_init=1, reg_covar=1e-6)
        gmm.fit(Xgs)
        bic = gmm.bic(Xgs)
        if bic < best_bic:
            best_bic = bic
            best_labels = gmm.predict(Xgs)
    ari = adjusted_rand_score(true_codes, best_labels)
    print(f"k={k}: ARI vs source = {ari:.4f}, BIC={best_bic:.2f}")
    ct = pd.crosstab(pd.Series(best_labels, name='gmm_cluster'),
                      pd.Series(true_labels, name='source'))
    print(ct)

# === Robustness check: covariance_type='diag' (more stable than 'full' when the
# smallest facility, Zheng2024, has only n=8 points -- full covariance in 3-5 dims
# is numerically fragile there) ===
print("\n=== ROBUSTNESS: GMM with covariance_type='diag' ===")
for feat_name, feats in [("geom-only", geom_features), ("all5", features)]:
    Xr = np.log(df[feats].values)
    Xrs = StandardScaler().fit_transform(Xr)
    print(f"\n--- {feat_name}: {feats} ---")
    for k in [2, 3, 4, 5]:
        best_bic = np.inf
        best_labels = None
        for seed in range(15):
            gmm = GaussianMixture(n_components=k, covariance_type='diag',
                                   random_state=seed, n_init=1, reg_covar=1e-6)
            gmm.fit(Xrs)
            bic = gmm.bic(Xrs)
            if bic < best_bic:
                best_bic = bic
                best_labels = gmm.predict(Xrs)
        ari = adjusted_rand_score(true_codes, best_labels)
        print(f"k={k}: ARI={ari:.4f} BIC={best_bic:.2f}")
        if k in (4, 5):
            ct = pd.crosstab(pd.Series(best_labels, name='cluster'),
                              pd.Series(true_labels, name='source'))
            print(ct)

# Stability across random seeds for the headline k=4 geom-only full-covariance result
print("\n=== Stability across 30 seeds: k=4, geom-only, full covariance ===")
Xr = np.log(df[geom_features].values)
Xrs = StandardScaler().fit_transform(Xr)
aris = []
for seed in range(30):
    gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=seed,
                           n_init=1, reg_covar=1e-6)
    gmm.fit(Xrs)
    aris.append(adjusted_rand_score(true_codes, gmm.predict(Xrs)))
aris = np.array(aris)
print(f"mean={aris.mean():.4f} std={aris.std():.4f} min={aris.min():.4f} max={aris.max():.4f}")
