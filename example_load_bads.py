# %%
"""
Example script showing how to load and use saved BADS optimization results.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# %%
# Load the saved results
# Replace with your actual pickle filename
pkl_filename = 'bads_optimization_results_YYYYMMDD_HHMMSS.pkl'  # Update this!

with open(pkl_filename, 'rb') as f:
    results = pickle.load(f)

print(f"Results loaded from: {pkl_filename}")
print(f"Timestamp: {results['timestamp']}")

# %%
# Extract key components
optimize_result = results['bads_result']
x_opt = results['optimized_params']['x_opt']
N_opt = results['optimized_params']['N']
c_opt = results['optimized_params']['c']
k_opt = results['optimized_params']['k']

print("\nOptimized Parameters:")
print(f"  N: {N_opt}")
print(f"  r1: {x_opt[1]:.6f}")
print(f"  r2: {x_opt[2]:.6f}")
print(f"  k: {k_opt:.6f}")
print(f"  c: {c_opt:.8f}")
print(f"  theta: {x_opt[4]:.6f}")

print(f"\nOptimization KS-statistic: {optimize_result['fval']:.8f}")
print(f"Function evaluations: {optimize_result['func_count']}")

# %%
# Access DDM and Poisson data
ddm_rts = results['ddm_simulation']['ddm_rts_decided']
poisson_rts = results['validation']['poisson_rts_val']

print(f"\nDDM trials: {len(ddm_rts)}")
print(f"Poisson trials: {len(poisson_rts)}")
print(f"DDM mean RT: {np.mean(ddm_rts):.6f} s")
print(f"Poisson mean RT: {np.mean(poisson_rts):.6f} s")

# %%
# Re-compute validation statistics if needed
ks_stat, ks_pval = ks_2samp(ddm_rts, poisson_rts)
print(f"\nValidation KS-statistic: {ks_stat:.8f}")
print(f"KS p-value: {ks_pval:.8e}")

# %%
# Visualize the results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RT distributions
ax1 = axes[0]
bins = np.linspace(0, min(np.percentile(ddm_rts, 99), 
                           np.percentile(poisson_rts, 99)), 50)
ax1.hist(ddm_rts, bins=bins, alpha=0.5, density=True, 
         label=f'DDM (n={len(ddm_rts)})', color='blue')
ax1.hist(poisson_rts, bins=bins, alpha=0.5, density=True, 
         label=f'Poisson (n={len(poisson_rts)})', color='red')
ax1.set_xlabel('Reaction Time (s)')
ax1.set_ylabel('Density')
ax1.set_title(f'RT Distribution Comparison\nKS-stat: {ks_stat:.6f}')
ax1.legend()
ax1.grid(alpha=0.3)

# Q-Q plot
ax2 = axes[1]
ddm_quantiles = np.percentile(ddm_rts, np.linspace(0, 100, 100))
poisson_quantiles = np.percentile(poisson_rts, np.linspace(0, 100, 100))
ax2.scatter(ddm_quantiles, poisson_quantiles, alpha=0.5, s=20)
lims = [min(ddm_quantiles.min(), poisson_quantiles.min()),
        max(ddm_quantiles.max(), poisson_quantiles.max())]
ax2.plot(lims, lims, 'k--', alpha=0.5, label='Perfect match')
ax2.set_xlabel('DDM RT Quantiles (s)')
ax2.set_ylabel('Poisson RT Quantiles (s)')
ax2.set_title('Q-Q Plot: DDM vs Poisson')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loaded_bads_result_comparison.png', dpi=150)
print("\n✓ Plot saved: loaded_bads_result_comparison.png")
plt.show()

# %%
# Access other saved data if needed
ddm_params = results['ddm_params']
validation_params = results['validation']['validation_params']

print("\nAll available keys in results:")
for key in results.keys():
    print(f"  - {key}")
