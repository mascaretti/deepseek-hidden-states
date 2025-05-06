#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jax
from jax import random
from jax import numpy as jnp
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
from scipy.linalg import toeplitz


# In[2]:


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


# In[3]:


from src.modules.information_imbalance import compute_imbalance_and_error


# In[4]:


def toeplitz_covariance(p: int, rho: float) -> np.ndarray:
    """Generates a Toeplitz covariance matrix with elements rho^|i-j|."""
    if not (0 <= rho < 1):
        raise ValueError("rho must be between 0 and 1 (exclusive of 1 for strict positive definiteness in theory).")
    # Create a vector for the first row/column
    first_row = rho**np.arange(p)
    # Use numpy.linalg.toeplitz to construct the matrix
    cov_matrix = toeplitz(first_row)
    return cov_matrix


# In[5]:


def compute_information(key, Y):
    n, p = Y.shape
    n_rows = jnp.floor(n * 0.2).astype(int)
    n_cols = n - n_rows
    indices = jnp.arange(n)
    key, subkey = random.split(key)
    indices_rows = random.choice(key=subkey, a=indices, shape=(n_rows,), replace=False)
    key, subkey = random.split(key)
    remaining_indices = jnp.setdiff1d(indices, indices_rows)
    indices_columns = random.choice(key=subkey, a=remaining_indices, shape=(n_cols,), replace=False)
    inf_imb = []
    for jj in tqdm(jnp.arange(1, p + 1)):
        X = Y[:, :jj]
        inf_imb.append(compute_imbalance_and_error(X, Y, indices_rows, indices_columns, 1)[0])
    return inf_imb


# In[6]:


def make_samples(key, p, rho, samples):
    inf_imb = {}
    for n_samples in samples:
        cov_matrix = jnp.asarray(toeplitz_covariance(p, rho))
        key, subkey_data, subkey_inf = random.split(key, 3)
        Y = random.multivariate_normal(subkey, mean, cov_matrix, shape=(n_samples,))
        inf_imb[str(n_samples)] = compute_information(subkey_inf, Y)
    return inf_imb


# In[7]:


key = random.key(42)
print(key)


# In[8]:


p = 100
mean = np.zeros(p)


# In[9]:


samples = [100, 500, 1_000]


# In[10]:


correlation_levels = [0.0, 0.5, 0.995]


# In[11]:


key, subkey = random.split(key)
inf_imb_id = make_samples(subkey, p, correlation_levels[0], samples)


# In[12]:


key, subkey = random.split(key)
inf_imb_corr = make_samples(subkey, p, correlation_levels[2], samples)


# In[55]:


plt.style.use(['nature', 'science'])
plt.figure(figsize=(16, 16))

fig, axes = plt.subplots(nrows=2, ncols=2)
y_min = 0.0
y_max = 1.1
fontsize=4

for curr_key in inf_imb_id.keys():
    axes[0,0].plot(jnp.arange(p) / p, inf_imb_id[curr_key], label=f"n/p: {int(curr_key)/p}")

axes[0,0].set_ylabel("Information Imbalance", fontsize=fontsize)
axes[0,0].set_xlabel(r"\% Shared features", fontsize=fontsize)
axes[0,0].legend(fontsize=fontsize)
axes[0,0].set_title(r"$\rho$=0.0", fontsize=fontsize)
axes[0,0].set_ylim(y_min, y_max)

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

for curr_key in inf_imb_corr.keys():
    axes[0,1].plot(jnp.arange(p) / p, inf_imb_corr[curr_key], label=f"n/p: {int(curr_key)/p}")

axes[0,1].set_ylabel("", fontsize=fontsize)
axes[0,1].set_xlabel(r"\% Shared features", fontsize=fontsize)
axes[0,1].legend(fontsize=fontsize)
axes[0,1].set_title(r"$\rho$=0.95", fontsize=fontsize)
axes[0,1].set_ylim(y_min, y_max)

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

correlation_levels = [0.0, 0.95]

for i, rho in enumerate(correlation_levels):
    cov_matrix = toeplitz_covariance(p, rho)
    im = axes[1, i].imshow(cov_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ## axes[1, i].set_title(rf"$\rho$ = {rho}")
    axes[1, i].set_xlabel("Feature Index", fontsize=fontsize)
    axes[1, i].set_ylabel("Feature Index", fontsize=fontsize)
    axes[1, i].set_aspect('equal', adjustable='box')

# fig.colorbar(im, ax=axes.ravel().tolist(), label='Correlation')
# plt.tight_layout()
## plt.show()
## plt.legend()
plt.suptitle(f"Number of features p: {p}", fontsize=fontsize)
plt.savefig("inf-imb.png", dpi=500)


# In[14]:


toeplitz_covariance(3., 0.)


# In[15]:


toeplitz_covariance(3., 0.95)


# In[28]:


# You could also visualize the correlation matrices or sample distributions
# For example, visualize the theoretical correlation matrices:
correlation_levels = [0.0, 0.95]
fig, axes = plt.subplots(1, len(correlation_levels), figsize=(7.5, 4))
fig.suptitle("Theoretical Correlation Matrices for Different Rho Values")

for i, rho in enumerate(correlation_levels):
    cov_matrix = toeplitz_covariance(p, rho)
    im = axes[i].imshow(cov_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[i].set_title(rf"$\rho$ = {rho}")
    axes[i].set_xlabel("Feature Index")
    axes[i].set_ylabel("Feature Index")
    axes[i].set_aspect('equal', adjustable='box')

# fig.colorbar(im, ax=axes.ravel().tolist(), label='Correlation')
# plt.tight_layout()
plt.show()


# In[ ]:




