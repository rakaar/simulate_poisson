# %%
"""
Plot KS-statistic for V3 results (varying c with fixed corr_factor and noise).
Creates:
1) A simple plot of KS statistic vs c
2) Raw RTD comparisons for each c value
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy import stats

# %%
# Setup parameters matching the V3 simulation
c_array = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
corr_factor_array = np.array([1.1])
exponential_noise_array = np.array([0])  # 0ms

results_folder = Path('results_V3')
output_folder = Path('figures_V3')
output_folder.mkdir(exist_ok=True)

print(f"Reading from: {results_folder}/")
print(f"Saving to: {output_folder}/")
print(f"Parameter space: {len(c_array)} c values, {len(corr_factor_array)} corr_factor, {len(exponential_noise_array)} noise level")

# %%
# Function to load data for a specific parameter combination
def load_data(c, corr_factor, noise):
    """Load pickle file for given parameter combination."""
    filename = f"c_{c}_corrfactor_{corr_factor}_noise_{noise*1000:.1f}ms.pkl"
    filepath = results_folder / filename
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data

# %%
# Function to compute KS statistic between two RTD distributions
def compute_ks_statistic(poisson_rts, ddm_rts):
    """
    Compute KS statistic (max CDF difference) between Poisson and DDM RTDs.
    """
    ks_stat, p_value = stats.ks_2samp(poisson_rts, ddm_rts)
    return ks_stat, p_value

# %%
# Collect KS statistics
ks_stats = []
c_values = []

corr_factor = corr_factor_array[0]
noise = exponential_noise_array[0]

for c in c_array:
    try:
        # Load data
        data = load_data(c, corr_factor, noise)
        
        # Extract RTs (all choices combined)
        poisson_results = data['poisson']['results']
        ddm_results = data['ddm']['results']
        
        poisson_rts = poisson_results[poisson_results[:, 1] != 0, 0]
        ddm_rts = ddm_results[ddm_results[:, 1] != 0, 0]
        
        # Compute KS statistic
        ks_stat, p_value = compute_ks_statistic(poisson_rts, ddm_rts)
        
        ks_stats.append(ks_stat)
        c_values.append(c)
        
        print(f"c={c}, corr_factor={corr_factor}, noise={noise*1000:.1f}ms: KS={ks_stat:.4f}, p={p_value:.4e}")
        
    except FileNotFoundError:
        print(f"Warning: File not found for c={c}")
        ks_stats.append(np.nan)
        c_values.append(c)

# %%
# FIGURE 1: KS Statistic vs c
print("\nCreating KS statistic plot...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(c_values, ks_stats, 
        color='#1f77b4', 
        marker='o',
        linewidth=2.5,
        markersize=10,
        label=f'corr_factor={corr_factor}, noise={noise*1000:.1f}ms')

# Formatting
ax.set_xlabel('c (coherence)', fontsize=13)
ax.set_ylabel('KS Statistic (max CDF difference)', fontsize=13)
ax.set_title('KS Statistic: Poisson vs DDM RTDs', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)
ax.set_xticks(c_array)

plt.tight_layout()

# Save figure
output_filename = output_folder / 'ks_statistic_vs_c.png'
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"Saved: {output_filename}")

plt.close(fig)

# %%
# FIGURE 2: Raw RTD Comparisons (2x3 subplots for 6 c values)
print("\nCreating raw RTD comparison plot...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten()

for c_idx, c in enumerate(c_array):
    try:
        # Load data
        data = load_data(c, corr_factor, noise)
        
        # Extract data
        params = data['params']
        poisson_results = data['poisson']['results']
        ddm_results = data['ddm']['results']
        
        N_right_and_left = params['N_right_and_left']
        
        # Get all decided RTs (both choices combined)
        poisson_rts = poisson_results[poisson_results[:, 1] != 0, 0]
        ddm_rts = ddm_results[ddm_results[:, 1] != 0, 0]
        
        # Define bins: 0 to 2 in steps of 0.05
        bins_rt = np.arange(0, 2.05, 0.05)
        
        # Plot histograms
        ax = axes[c_idx]
        ax.hist(poisson_rts, bins=bins_rt, density=True, histtype='step',
                label='Poisson', color='blue', linewidth=2)
        ax.hist(ddm_rts, bins=bins_rt, density=True, histtype='step',
                label='DDM', color='red', linewidth=2)
        
        # Formatting
        ax.set_xlabel("RT (s)", fontsize=11)
        ax.set_title(f'c = {c}\nN={N_right_and_left}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim(0, 2)
        ax.tick_params(labelsize=10)
        
        # Add ylabel to leftmost subplots
        if c_idx % 3 == 0:
            ax.set_ylabel("Density", fontsize=11)
        
        ax.legend(loc='upper right', fontsize=9)
        
    except FileNotFoundError:
        ax = axes[c_idx]
        ax.text(0.5, 0.5, 'Data not found', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(0, 2)

# Main title
fig.suptitle(f'Poisson vs DDM Combined RTD Comparison (corr_factor={corr_factor}, noise={noise*1000:.1f}ms)', 
            fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()

# Save figure
output_filename = output_folder / 'rtd_comparison_all_c.png'
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"Saved: {output_filename}")

plt.close(fig)

# %%
print("\n" + "="*70)
print("ALL FIGURES CREATED FOR V3 DATA")
print("="*70)
print(f"Saved to: {output_folder}/")
print(f"  1) ks_statistic_vs_c.png")
print(f"  2) rtd_comparison_all_c.png")
