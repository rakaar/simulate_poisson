# %%
"""
Plot KS-statistic (max CDF difference) between Poisson and DDM RTDs.
For each correlation factor, creates two figures:
1) KS vs noise for different c values
2) KS vs c for different noise levels
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

print(f"Creating 2 figures with 1x5 subplots each")

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
# Collect KS statistics organized by correlation factor
ks_data_by_corrfactor = {}

for corr_factor in corr_factor_array:
    ks_data_by_corrfactor[corr_factor] = {}
    
    # Organize by c values (for plotting vs noise)
    ks_data_by_corrfactor[corr_factor]['by_c'] = {}
    for c in c_array:
        ks_data_by_corrfactor[corr_factor]['by_c'][c] = {
            'ks_stats': [],
            'noise_levels': []
        }
    
    # Organize by noise levels (for plotting vs c)
    ks_data_by_corrfactor[corr_factor]['by_noise'] = {}
    for noise in exponential_noise_array:
        ks_data_by_corrfactor[corr_factor]['by_noise'][noise] = {
            'ks_stats': [],
            'c_values': []
        }

# Collect data
for corr_factor in corr_factor_array:
    for c in c_array:
        for noise in exponential_noise_array:
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
                
                # Store in both organizations
                ks_data_by_corrfactor[corr_factor]['by_c'][c]['ks_stats'].append(ks_stat)
                ks_data_by_corrfactor[corr_factor]['by_c'][c]['noise_levels'].append(noise)
                
                ks_data_by_corrfactor[corr_factor]['by_noise'][noise]['ks_stats'].append(ks_stat)
                ks_data_by_corrfactor[corr_factor]['by_noise'][noise]['c_values'].append(c)
                
                print(f"corr_factor={corr_factor}, c={c}, noise={noise*1000:.1f}ms: KS={ks_stat:.4f}")
                
            except FileNotFoundError:
                print(f"Warning: File not found for corr_factor={corr_factor}, c={c}, noise={noise*1000:.1f}ms")
                ks_data_by_corrfactor[corr_factor]['by_c'][c]['ks_stats'].append(np.nan)
                ks_data_by_corrfactor[corr_factor]['by_c'][c]['noise_levels'].append(noise)
                
                ks_data_by_corrfactor[corr_factor]['by_noise'][noise]['ks_stats'].append(np.nan)
                ks_data_by_corrfactor[corr_factor]['by_noise'][noise]['c_values'].append(c)

# %%
# Define colors and markers
colors_c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # For c values
colors_noise = ['#e377c2', '#8c564b', '#bcbd22', '#17becf']  # For noise levels
markers_c = ['o', 's', '^', 'D']
markers_noise = ['v', 'p', '*', 'h']

# %%
# FIGURE 1: KS vs Noise for each corr_factor (lines for different c values)
print("\nCreating Figure 1: KS vs Noise for each corr_factor...")
fig1, axes1 = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

for cf_idx, corr_factor in enumerate(corr_factor_array):
    ax = axes1[cf_idx]
    
    # Plot KS statistic vs noise for each c value
    for c_idx, c in enumerate(c_array):
        noise_levels = np.array(ks_data_by_corrfactor[corr_factor]['by_c'][c]['noise_levels']) * 1000  # Convert to ms
        ks_stats = np.array(ks_data_by_corrfactor[corr_factor]['by_c'][c]['ks_stats'])
        
        ax.plot(noise_levels, ks_stats, 
                color=colors_c[c_idx], 
                marker=markers_c[c_idx],
                linewidth=2,
                markersize=8,
                label=f'c = {c}')
    
    # Formatting
    ax.set_xlabel('Exponential Noise (ms)', fontsize=11)
    ax.set_title(f'corr_factor = {corr_factor}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.set_xticks(exponential_noise_array * 1000)
    
    # Add legend and ylabel to first subplot
    if cf_idx == 0:
        ax.set_ylabel('KS Statistic (max CDF difference)', fontsize=11)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

# Main title
fig1.suptitle('KS Statistic vs Noise for each Correlation Factor', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure
output_filename1 = output_folder / 'ks_vs_noise_by_corrfactor.png'
plt.savefig(output_filename1, dpi=150, bbox_inches='tight')
print(f"Saved: {output_filename1}")

plt.close(fig1)

# %%
# FIGURE 2: KS vs c for each corr_factor (lines for different noise levels)
print("\nCreating Figure 2: KS vs c for each corr_factor...")
fig2, axes2 = plt.subplots(1, 5, figsize=(25, 5), sharey=True)

for cf_idx, corr_factor in enumerate(corr_factor_array):
    ax = axes2[cf_idx]
    
    # Plot KS statistic vs c for each noise level
    for noise_idx, noise in enumerate(exponential_noise_array):
        c_values = np.array(ks_data_by_corrfactor[corr_factor]['by_noise'][noise]['c_values'])
        ks_stats = np.array(ks_data_by_corrfactor[corr_factor]['by_noise'][noise]['ks_stats'])
        
        ax.plot(c_values, ks_stats, 
                color=colors_noise[noise_idx], 
                marker=markers_noise[noise_idx],
                linewidth=2,
                markersize=8,
                label=f'noise = {noise*1000:.1f}ms')
    
    # Formatting
    ax.set_xlabel('c', fontsize=11)
    ax.set_title(f'corr_factor = {corr_factor}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.set_xticks(c_array)
    
    # Add legend and ylabel to first subplot
    if cf_idx == 0:
        ax.set_ylabel('KS Statistic (max CDF difference)', fontsize=11)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

# Main title
fig2.suptitle('KS Statistic vs c for each Correlation Factor', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure
output_filename2 = output_folder / 'ks_vs_c_by_corrfactor.png'
plt.savefig(output_filename2, dpi=150, bbox_inches='tight')
print(f"Saved: {output_filename2}")

plt.close(fig2)

# %%
print("\n" + "="*70)
print("BOTH FIGURES CREATED")
print("="*70)
print(f"Saved to: {output_folder}/")
print(f"  1) ks_vs_noise_by_corrfactor.png")
print(f"  2) ks_vs_c_by_corrfactor.png")
