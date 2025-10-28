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
from params import theta

# %%
# CONFIGURATION - Change these as needed
version_number = '13'
results_folder_name = f'results_V{version_number}'
output_folder_name = f'raw_rtds_combined_choice_V{version_number}'

# %%
# Get the directory where this script is located
script_dir = Path(__file__).parent

results_folder = script_dir / results_folder_name
output_folder = script_dir / output_folder_name
output_folder.mkdir(exist_ok=True)

if not results_folder.exists():
    print(f"ERROR: Results folder '{results_folder}' does not exist!")
    import sys
    sys.exit(1)

print(f"Reading from: {results_folder}/")
print(f"Saving to: {output_folder}/")

# %%
# Load baseline RTD data (theta=2, no correlation)
baseline_path = script_dir.parent / "theta_2_no_corr_baseline_data.pkl"
baseline_sk_2 = None
baseline_dd_2 = None

if baseline_path.exists():
    try:
        with open(baseline_path, 'rb') as f:
            baseline_data = pickle.load(f)
        baseline_sk_2 = baseline_data['sk_2']
        baseline_dd_2 = baseline_data['dd_2']
        print(f"\nLoaded baseline RTD data (θ=2, no correlation)")
        print(f"  Baseline Poisson (sk_2): {len(baseline_sk_2)} trials")
        print(f"  Baseline DDM (dd_2): {len(baseline_dd_2)} trials")
    except Exception as e:
        print(f"\nWarning: Could not load baseline data: {e}")
else:
    print(f"\nWarning: Baseline file not found at {baseline_path}")

# %%
# Auto-detect parameter arrays by scanning pickle files
def detect_parameters_from_files(folder):
    """
    Scan all pickle files in the folder and extract unique parameter values.
    Returns: (c_array, corr_factor_array, noise_array)
    """
    c_values = set()
    corr_factor_values = set()
    noise_values = set()
    
    pkl_files = list(folder.glob('*.pkl'))
    
    if len(pkl_files) == 0:
        print(f"ERROR: No pickle files found in {folder}/")
        import sys
        sys.exit(1)
    
    print(f"Found {len(pkl_files)} pickle files. Detecting parameters...")
    
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            params = data['params']
            c_values.add(params['c'])
            corr_factor_values.add(params['corr_factor'])
            noise_values.add(params['exponential_noise_to_spk_time'])
            
        except Exception as e:
            print(f"Warning: Could not read {pkl_file.name}: {e}")
            continue
    
    c_array = np.array(sorted(c_values))
    corr_factor_array = np.array(sorted(corr_factor_values))
    noise_array = np.array(sorted(noise_values))
    
    return c_array, corr_factor_array, noise_array

# Detect parameters from files
c_array, corr_factor_array, exponential_noise_array = detect_parameters_from_files(results_folder)

print(f"\nDetected parameters:")
print(f"  c values: {c_array}")
print(f"  corr_factor values: {corr_factor_array}")
print(f"  exponential_noise values (s): {exponential_noise_array}")
print(f"  exponential_noise values (ms): {exponential_noise_array*1000}")

print(f"\nCreating {len(c_array)} figures (one per c value)")
print(f"Each figure: {len(corr_factor_array)} corr_factors × {len(exponential_noise_array)} noise levels")
# %%
# print(f'cf * theta = {corr_factor_array[1]} * {theta} = {corr_factor_array[1]*theta}')
# print(f' 4 ==  corr_factor_array[1]*theta: {4 == corr_factor_array[1]*theta} ')

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
def plot_rtd_combined(ax, data, baseline_sk=None, baseline_dd=None, 
                     show_xlabel=False, show_ylabel=False, show_title=True):
    """
    Plot combined Poisson vs DDM RTD comparison (all choices together).
    Optionally includes baseline RTDs in lighter colors.
    """
    # Extract data
    params = data['params']
    poisson_results = data['poisson']['results']
    ddm_results = data['ddm']['results']
    
    c = params['c']
    corr_factor = params['corr_factor']
    exponential_noise = params['exponential_noise_to_spk_time']
    N_right_and_left = params['N_right_and_left']
    noise_mean_subtraction = params.get('noise_mean_subtraction', False)
    
    mu = data['ddm']['params']['mu']
    sigma = data['ddm']['params']['sigma']
    
    # Get all decided RTs (both choices combined)
    poisson_rts = poisson_results[poisson_results[:, 1] != 0, 0]
    ddm_rts = ddm_results[ddm_results[:, 1] != 0, 0]
    
    # Define bins based on noise_mean_subtraction flag
    if noise_mean_subtraction:
        # Allow for negative RTs when mean subtraction is enabled
        all_rts = np.concatenate([poisson_rts, ddm_rts])
        if len(all_rts) > 0:
            min_rt = min(0, np.min(all_rts))
            max_rt = max(2, np.max(all_rts))
            bins_rt = np.arange(-2, 2 + 0.01, 0.01)
            xlim_range = (-2, 2)
        else:
            bins_rt = np.arange(0, 2.01, 0.01)
            xlim_range = (0, 2)
    else:
        # Standard bins: 0 to 2 in steps of 0.01
        bins_rt = np.arange(0, 2.01, 0.01)
        xlim_range = (0, 2)
    
    # Plot baseline RTDs in lighter colors (if provided)
    if baseline_sk is not None:
        ax.hist(baseline_sk, bins=bins_rt, density=True, histtype='step',
                label='Baseline Poisson', color='green', alpha=0.75, lw = 3)
    if baseline_dd is not None:
        ax.hist(baseline_dd, bins=bins_rt, density=True, histtype='step',
                label='Baseline DDM', color='k', ls='--', alpha=0.75, lw=3)
    
    # Plot histograms with step style
    ax.hist(poisson_rts, bins=bins_rt, density=True, histtype='step',
            label='Poisson', color='blue', linewidth=1)
    ax.hist(ddm_rts, bins=bins_rt, density=True, histtype='step',
            label='DDM', color='red', linewidth=1)
    
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
    
    # Set x-axis limits
    # ax.set_xlim(xlim_range)
    ax.set_xlim(0, 1.2)
    
    ax.tick_params(labelsize=7)
    
    return ax

# %%
# Create figures - one for each c value
for c_idx, c in enumerate(c_array):
    print(f"\nCreating figure for c = {c}")
    
    # Create figure with subplots
    n_rows = len(exponential_noise_array)
    n_cols = len(corr_factor_array)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12), squeeze=False)
    
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
                                 baseline_sk=baseline_sk_2,
                                 baseline_dd=baseline_dd_2,
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

# %%
