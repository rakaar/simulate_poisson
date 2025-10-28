# %%
"""
Plot raw RTD comparisons for Poisson vs DDM across parameter combinations.
Creates 4 figures (one per c value) with subplots showing RTDs for each 
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
output_folder = Path('raw_rtds')
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
# Function to plot RTD in a single subplot
def plot_rtd_comparison(ax, data, show_xlabel=False, show_ylabel=False, show_title=True):
    """
    Plot Poisson vs DDM RTD comparison in the given axes.
    Mirrors the style from combined_poisson_ddm_jump_analysis_V2_gaussian_noise_to_spk_times.py
    """
    # Extract data
    params = data['params']
    poisson_results = data['poisson']['results']
    ddm_results = data['ddm']['results']
    
    c = params['c']
    corr_factor = params['corr_factor']
    exponential_noise = params['exponential_noise_to_spk_time']
    theta = params['theta']
    theta_scaled = params['theta_scaled']
    r_right = params['r_right']
    r_left = params['r_left']
    r_right_scaled = params['r_right_scaled']
    r_left_scaled = params['r_left_scaled']
    N_right_and_left = params['N_right_and_left']
    
    mu = data['ddm']['params']['mu']
    sigma = data['ddm']['params']['sigma']
    
    # Prepare RT histograms
    max_T_plot = np.max(poisson_results[(poisson_results[:, 1] == 1), 0])
    bins_rt = np.arange(0, max_T_plot, max_T_plot / 1000)
    
    # Poisson RTDs
    pos_rts_poisson = poisson_results[(poisson_results[:, 1] == 1), 0]
    neg_rts_poisson = poisson_results[(poisson_results[:, 1] == -1), 0]
    pos_hist_poisson, _ = np.histogram(pos_rts_poisson, bins=bins_rt, density=True)
    neg_hist_poisson, _ = np.histogram(neg_rts_poisson, bins=bins_rt, density=True)
    
    # DDM RTDs
    pos_rts_ddm = ddm_results[(ddm_results[:, 1] == 1), 0]
    neg_rts_ddm = ddm_results[(ddm_results[:, 1] == -1), 0]
    pos_hist_ddm, _ = np.histogram(pos_rts_ddm, bins=bins_rt, density=True)
    neg_hist_ddm, _ = np.histogram(neg_rts_ddm, bins=bins_rt, density=True)
    
    bin_centers = (bins_rt[:-1] + bins_rt[1:]) / 2
    
    # Calculate fractions
    poisson_up = len(pos_rts_poisson)
    poisson_down = len(neg_rts_poisson)
    poisson_frac_up = poisson_up / (poisson_up + poisson_down)
    poisson_frac_down = poisson_down / (poisson_up + poisson_down)
    
    ddm_up = len(pos_rts_ddm)
    ddm_down = len(neg_rts_ddm)
    ddm_frac_up = ddm_up / (ddm_up + ddm_down)
    ddm_frac_down = ddm_down / (ddm_up + ddm_down)
    
    # Plot
    ax.plot(bin_centers, pos_hist_poisson * poisson_frac_up, 
            label='Poisson +', color='blue', linestyle='-', linewidth=1.5)
    ax.plot(bin_centers, -neg_hist_poisson * poisson_frac_down, 
            label='Poisson -', color='blue', linestyle='-', linewidth=1.5)
    ax.plot(bin_centers, pos_hist_ddm * ddm_frac_up, 
            label='DDM +', color='red', linestyle='-', linewidth=3, alpha=0.3)
    ax.plot(bin_centers, -neg_hist_ddm * ddm_frac_down, 
            label='DDM -', color='red', linestyle='-', linewidth=3, alpha=0.3)
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
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
                plot_rtd_comparison(ax, data, 
                                  show_xlabel=show_xlabel, 
                                  show_ylabel=show_ylabel,
                                  show_title=True)
                
                # Add legend only to top-left subplot
                if noise_idx == 0 and corr_idx == 0:
                    ax.legend(loc='upper right', fontsize=7)
                
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
    fig.suptitle(f'Poisson vs DDM RTD Comparison (c = {c})', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.97, 0.98])
    
    # Save figure
    output_filename = output_folder / f'rtd_comparison_c_{c}.png'
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
    print(f"  - rtd_comparison_c_{c}.png")
