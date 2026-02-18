import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

# --- GEOMETRIC FUNCTIONS (RECAP) ---

def calculate_contrast(distances):
    return np.std(distances) / np.mean(distances)

def log_map_projection(X):
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    mu = np.mean(X_norm, axis=0)
    mu /= np.linalg.norm(mu)
    projections = []
    for x in X_norm:
        cos_theta = np.clip(np.dot(x, mu), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        v = (theta / np.sin(theta)) * (x - cos_theta * mu) if theta > 1e-9 else np.zeros_like(x)
        projections.append(v)
    return np.array(projections)

# --- 1. DATA GENERATION ---
dim = 100
n_samples = 800
# Generate points in a tight cone around the first axis
mean_vec = np.zeros(dim); mean_vec[0] = 1.0
# Small standard deviation creates the "narrow" cone effect
X = mean_vec + np.random.normal(0, 0.05, (n_samples, dim))

# --- 2. APPLY TRANSFORMATIONS ---

# Baseline: Raw Cosine Distances
dist_raw = pdist(X, metric='cosine')

# Solution 1: Binarization (Centered Hamming)
X_centered = X - np.mean(X, axis=0)
X_bin = (X_centered > 0).astype(int)
dist_bin = pdist(X_bin, metric='hamming')

# Solution 2: Mahalanobis (Whitening via PCA)
pca_white = PCA(whiten=True)
X_white = pca_white.fit_transform(X)
dist_white = pdist(X_white, metric='euclidean')

# Solution 3: Log-Map (Tangent Space)
X_tangent = log_map_projection(X)
dist_tangent = pdist(X_tangent, metric='euclidean')

# --- 3. PLOTTING THE "DISTANCE SPREAD" ---

plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

# We normalize distances to [0, 1] for visual comparison of the "spread"
def scale(d): return (d - np.min(d)) / (np.max(d) - np.min(d))

data_to_plot = [
    ("Raw Cosine (The Problem)", dist_raw),
    ("Hamming (Binarized)", dist_bin),
    ("Whitened (Mahalanobis)", dist_white),
    ("Log-Map (Tangent Space)", dist_tangent)
]

for i, (title, d) in enumerate(data_to_plot):
    plt.subplot(2, 2, i+1)
    sns.histplot(scale(d), kde=True, color='teal', bins=50)
    contrast = calculate_contrast(d)
    plt.title(f"{title}\nContrast Ratio: {contrast:.4f}")
    plt.xlabel("Normalized Distance")
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# --- 4. 2D VISUALIZATION (PCA PROJECTION) ---
plt.figure(figsize=(12, 5))

# Original tight cone (First 2 PCA components)
plt.subplot(1, 2, 1)
coords_raw = PCA(n_components=2).fit_transform(X)
plt.scatter(coords_raw[:, 0], coords_raw[:, 1], alpha=0.5, s=10, c='crimson')
plt.title("Original Hypercone Projection\n(Points are 'Suffocated')")

# Log-Map "Unfolded"
plt.subplot(1, 2, 2)
coords_tangent = PCA(n_components=2).fit_transform(X_tangent)
plt.scatter(coords_tangent[:, 0], coords_tangent[:, 1], alpha=0.5, s=10, color='royalblue')
plt.title("Log-Map Tangent Projection\n(Internal Structure Revealed)")

plt.show()
