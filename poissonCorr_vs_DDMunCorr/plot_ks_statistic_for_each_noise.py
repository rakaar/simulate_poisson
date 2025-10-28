# %%
"""
Plot KS-statistic (max CDF difference) between Poisson and DDM RTDs.
For each noise level, creates two figures:
1) KS vs corr_factor for different c values
2) KS vs c for different corr_factors
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

print(f"Creating 2 figures with 1x4 subplots each")

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
# Collect KS statistics organized by noise level
ks_data_by_noise = {}

for noise in exponential_noise_array:
    ks_data_by_noise[noise] = {}
    
    # Organize by c values (for plotting vs corr_factor)
    ks_data_by_noise[noise]['by_c'] = {}
    for c in c_array:
        ks_data_by_noise[noise]['by_c'][c] = {
            'ks_stats': [],
            'corr_factors': []
        }
    
    # Organize by corr_factor (for plotting vs c)
    ks_data_by_noise[noise]['by_corrfactor'] = {}
    for corr_factor in corr_factor_array:
        ks_data_by_noise[noise]['by_corrfactor'][corr_factor] = {
            'ks_stats': [],
            'c_values': []
        }

# Collect data
for noise in exponential_noise_array:
    for c in c_array:
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
                
                # Store in both organizations
                ks_data_by_noise[noise]['by_c'][c]['ks_stats'].append(ks_stat)
                ks_data_by_noise[noise]['by_c'][c]['corr_factors'].append(corr_factor)
                
                ks_data_by_noise[noise]['by_corrfactor'][corr_factor]['ks_stats'].append(ks_stat)
                ks_data_by_noise[noise]['by_corrfactor'][corr_factor]['c_values'].append(c)
                
                print(f"noise={noise*1000:.1f}ms, c={c}, corr_factor={corr_factor}: KS={ks_stat:.4f}")
                
            except FileNotFoundError:
                print(f"Warning: File not found for noise={noise*1000:.1f}ms, c={c}, corr_factor={corr_factor}")
                ks_data_by_noise[noise]['by_c'][c]['ks_stats'].append(np.nan)
                ks_data_by_noise[noise]['by_c'][c]['corr_factors'].append(corr_factor)
                
                ks_data_by_noise[noise]['by_corrfactor'][corr_factor]['ks_stats'].append(np.nan)
                ks_data_by_noise[noise]['by_corrfactor'][corr_factor]['c_values'].append(c)

# %%
# Define colors and markers
colors_c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # For c values
colors_cf = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']  # For corr_factors
markers_c = ['o', 's', '^', 'D']
markers_cf = ['v', 'p', '*', 'h', 'X']

# %%
# FIGURE 1: KS vs corr_factor for each noise level (lines for different c values)
print("\nCreating Figure 1: KS vs corr_factor for each noise level...")
fig1, axes1 = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for noise_idx, noise in enumerate(exponential_noise_array):
    ax = axes1[noise_idx]
    
    # Plot KS statistic vs corr_factor for each c value
    for c_idx, c in enumerate(c_array):
        corr_factors = np.array(ks_data_by_noise[noise]['by_c'][c]['corr_factors'])
        ks_stats = np.array(ks_data_by_noise[noise]['by_c'][c]['ks_stats'])
        
        ax.plot(corr_factors, ks_stats, 
                color=colors_c[c_idx], 
                marker=markers_c[c_idx],
                linewidth=2,
                markersize=8,
                label=f'c = {c}')
    
    # Formatting
    ax.set_xlabel('Correlation Factor', fontsize=11)
    ax.set_title(f'noise = {noise*1000:.1f}ms', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.set_xticks(corr_factor_array)
    
    # Add legend and ylabel to first subplot
    if noise_idx == 0:
        ax.set_ylabel('KS Statistic (max CDF difference)', fontsize=11)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

# Main title
fig1.suptitle('KS Statistic vs Correlation Factor for each Noise Level', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure
output_filename1 = output_folder / 'ks_vs_corrfactor_by_noise.png'
plt.savefig(output_filename1, dpi=150, bbox_inches='tight')
print(f"Saved: {output_filename1}")

plt.close(fig1)

# %%
# FIGURE 2: KS vs c for each noise level (lines for different corr_factors)
print("\nCreating Figure 2: KS vs c for each noise level...")
fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5), sharey=True)

for noise_idx, noise in enumerate(exponential_noise_array):
    ax = axes2[noise_idx]
    
    # Plot KS statistic vs c for each corr_factor
    for cf_idx, corr_factor in enumerate(corr_factor_array):
        c_values = np.array(ks_data_by_noise[noise]['by_corrfactor'][corr_factor]['c_values'])
        ks_stats = np.array(ks_data_by_noise[noise]['by_corrfactor'][corr_factor]['ks_stats'])
        
        ax.plot(c_values, ks_stats, 
                color=colors_cf[cf_idx], 
                marker=markers_cf[cf_idx],
                linewidth=2,
                markersize=8,
                label=f'corr_factor = {corr_factor}')
    
    # Formatting
    ax.set_xlabel('c', fontsize=11)
    ax.set_title(f'noise = {noise*1000:.1f}ms', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)
    ax.set_xticks(c_array)
    
    # Add legend and ylabel to first subplot
    if noise_idx == 0:
        ax.set_ylabel('KS Statistic (max CDF difference)', fontsize=11)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

# Main title
fig2.suptitle('KS Statistic vs c for each Noise Level', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure
output_filename2 = output_folder / 'ks_vs_c_by_noise.png'
plt.savefig(output_filename2, dpi=150, bbox_inches='tight')
print(f"Saved: {output_filename2}")

plt.close(fig2)

# %%
print("\n" + "="*70)
print("BOTH FIGURES CREATED")
print("="*70)
print(f"Saved to: {output_folder}/")
print(f"  1) ks_vs_corrfactor_by_noise.png")
print(f"  2) ks_vs_c_by_noise.png")
