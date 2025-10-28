# %%
"""
Plot KS-statistic (max CDF difference) between Poisson and DDM RTDs.
Creates a single 1x4 figure showing how KS-statistic varies with correlation factor
for different noise levels across different c values.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy import stats

# %%
# Setup parameters matching the simulation
c_array = np.array([0.01, 0.05, 0.1, 0.2])
corr_factor_array = np.array([1.1, 2, 5, 10, 20])
exponential_noise_array = np.array([0, 1e-3, 2.5e-3, 5e-3])  # 0ms, 1ms, 2.5ms, 5ms in seconds

results_folder = Path('results')
output_folder = Path('ks_statistic_plots')
output_folder.mkdir(exist_ok=True)

print(f"Creating 1x4 figure for all c values")

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
# Collect KS statistics for all parameter combinations
ks_data = {}

for c in c_array:
    ks_data[c] = {}
    
    for noise in exponential_noise_array:
        ks_data[c][noise] = {
            'ks_stats': [],
            'p_values': [],
            'corr_factors': []
        }
        
        for corr_factor in corr_factor_array:
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
                
                # Store results
                ks_data[c][noise]['ks_stats'].append(ks_stat)
                ks_data[c][noise]['p_values'].append(p_value)
                ks_data[c][noise]['corr_factors'].append(corr_factor)
                
                print(f"c={c}, noise={noise*1000:.1f}ms, corr_factor={corr_factor}: KS={ks_stat:.4f}, p={p_value:.4e}")
                
            except FileNotFoundError:
                print(f"Warning: File not found for c={c}, corr_factor={corr_factor}, noise={noise*1000:.1f}ms")
                ks_data[c][noise]['ks_stats'].append(np.nan)
                ks_data[c][noise]['p_values'].append(np.nan)
                ks_data[c][noise]['corr_factors'].append(corr_factor)

# %%
# Define colors for different noise levels
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

# Create single figure with 1x4 subplots (one for each c value)
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for c_idx, c in enumerate(c_array):
    ax = axes[c_idx]
    
    # Plot KS statistic vs corr_factor for each noise level
    for noise_idx, noise in enumerate(exponential_noise_array):
        corr_factors = np.array(ks_data[c][noise]['corr_factors'])
        ks_stats = np.array(ks_data[c][noise]['ks_stats'])
        
        ax.plot(corr_factors, ks_stats, 
                color=colors[noise_idx], 
                marker=markers[noise_idx],
                linewidth=2,
                markersize=8,
                label=f'noise = {noise*1000:.1f}ms')
    
    # Formatting
    ax.set_xlabel('Correlation Factor', fontsize=11)
    ax.set_title(f'c = {c}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.set_xticks(corr_factor_array)
    
    # Add legend only to the first subplot
    if c_idx == 0:
        ax.set_ylabel('KS Statistic (max CDF difference)', fontsize=11)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

# Main title
fig.suptitle('KS Statistic: Poisson vs DDM RTDs vs Correlation Factor', 
            fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure
output_filename = output_folder / 'ks_statistic_vs_corrfactor_all_c.png'
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_filename}")

plt.close(fig)

# %%
print("\n" + "="*70)
print("KS STATISTIC VS CORRELATION FACTOR FIGURE CREATED")
print("="*70)
print(f"Saved to: {output_folder}/ks_statistic_vs_corrfactor_all_c.png")
