# %%
"""
Plot combined RTD comparisons for Poisson vs DDM (without separating by choice).
Creates 4 figures (one per c value) with subplots showing combined RTDs for each 
corr_factor and exponential noise combination.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path

# %%
# Setup parameters matching the simulation
c_array = np.array([0.01, 0.05, 0.1, 0.2])
corr_factor_array = np.array([1.1, 2, 5, 10, 20])
exponential_noise_array = np.array([0, 1e-3, 2.5e-3, 5e-3])  # 0ms, 1ms, 2.5ms, 5ms in seconds

results_folder = Path('results')
output_folder = Path('raw_rtds_combined_choice')
output_folder.mkdir(exist_ok=True)

print(f"Creating {len(c_array)} figures (one per c value)")
print(f"Each figure: {len(corr_factor_array)} corr_factors × {len(exponential_noise_array)} noise levels")

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
# Function to plot combined RTD in a single subplot
def plot_rtd_combined(ax, data, show_xlabel=False, show_ylabel=False, show_title=True):
    """
    Plot combined Poisson vs DDM RTD comparison (all choices together).
    """
    # Extract data
    params = data['params']
    poisson_results = data['poisson']['results']
    ddm_results = data['ddm']['results']
    
    c = params['c']
    corr_factor = params['corr_factor']
    exponential_noise = params['exponential_noise_to_spk_time']
    N_right_and_left = params['N_right_and_left']
    
    mu = data['ddm']['params']['mu']
    sigma = data['ddm']['params']['sigma']
    
    # Get all decided RTs (both choices combined)
    poisson_rts = poisson_results[poisson_results[:, 1] != 0, 0]
    ddm_rts = ddm_results[ddm_results[:, 1] != 0, 0]
    
    # Define bins: 0 to 2 in steps of 0.05
    bins_rt = np.arange(0, 2.05, 0.05)
    
    # Plot histograms with step style
    ax.hist(poisson_rts, bins=bins_rt, density=True, histtype='step',
            label='Poisson', color='blue', linewidth=1.5)
    ax.hist(ddm_rts, bins=bins_rt, density=True, histtype='step',
            label='DDM', color='red', linewidth=2)
    
    if show_xlabel:
        ax.set_xlabel("RT (s)", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=9)
    
    if show_title:
        ax.set_title(
            f'corr_f={corr_factor}, noise={exponential_noise*1000:.1f}ms\n'
            f'N={N_right_and_left}',
            fontsize=8
        )
    
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(0, 2)
    ax.tick_params(labelsize=7)
    
    return ax

# %%
# Create figures - one for each c value
for c_idx, c in enumerate(c_array):
    print(f"\nCreating figure for c = {c}")
    
    # Create figure with subplots
    n_rows = len(exponential_noise_array)
    n_cols = len(corr_factor_array)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    
    # Iterate over parameter combinations
    for noise_idx, noise in enumerate(exponential_noise_array):
        for corr_idx, corr_factor in enumerate(corr_factor_array):
            
            # Load data
            try:
                data = load_data(c, corr_factor, noise)
                
                # Get axis
                ax = axes[noise_idx, corr_idx]
                
                # Determine labels
                show_xlabel = (noise_idx == n_rows - 1)
                show_ylabel = (corr_idx == 0)
                
                # Plot
                plot_rtd_combined(ax, data, 
                                 show_xlabel=show_xlabel, 
                                 show_ylabel=show_ylabel,
                                 show_title=True)
                
                # Add legend only to top-left subplot
                if noise_idx == 0 and corr_idx == 0:
                    ax.legend(loc='upper right', fontsize=8)
                
            except FileNotFoundError:
                print(f"  Warning: File not found for c={c}, corr_factor={corr_factor}, noise={noise*1000:.1f}ms")
                ax = axes[noise_idx, corr_idx]
                ax.text(0.5, 0.5, 'Data not found', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xlim(0, 2)
    
    # Add column labels at top
    for corr_idx, corr_factor in enumerate(corr_factor_array):
        axes[0, corr_idx].text(0.5, 1.15, f'corr_factor = {corr_factor}', 
                              ha='center', va='bottom', 
                              transform=axes[0, corr_idx].transAxes,
                              fontsize=10, fontweight='bold')
    
    # Add row labels on right
    for noise_idx, noise in enumerate(exponential_noise_array):
        axes[noise_idx, -1].text(1.15, 0.5, f'noise = {noise*1000:.1f}ms', 
                                rotation=-90, ha='left', va='center',
                                transform=axes[noise_idx, -1].transAxes,
                                fontsize=10, fontweight='bold')
    
    # Main title
    fig.suptitle(f'Poisson vs DDM Combined RTD Comparison (c = {c})', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.97, 0.98])
    
    # Save figure
    output_filename = output_folder / f'rtd_combined_c_{c}.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_filename}")
    
    plt.close(fig)

# %%
print("\n" + "="*70)
print("ALL FIGURES CREATED")
print("="*70)
print(f"Saved {len(c_array)} figures to: {output_folder}/")
print(f"Files:")
for c in c_array:
    print(f"  - rtd_combined_c_{c}.png")
