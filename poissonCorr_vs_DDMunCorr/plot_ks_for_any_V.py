# %%
"""
Generic plotting script for KS-statistic analysis.
Automatically detects parameter arrays from pickle files in the results folder.
Can be used for any version (V2, V3, V4, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path
from scipy import stats

# %%
# CONFIGURATION - Change these as needed
version_number = '12'
results_folder_name = f'results_V{version_number}'  # Change this to your results folder
output_folder_name = f'figures_V{version_number}'   # Change this to your output folder

# %%
results_folder = Path(results_folder_name)
output_folder = Path(output_folder_name)
output_folder.mkdir(exist_ok=True)

if not results_folder.exists():
    print(f"ERROR: Results folder '{results_folder}' does not exist!")
    sys.exit(1)

print(f"Reading from: {results_folder}/")
print(f"Saving to: {output_folder}/")

# %%
# Load baseline KS statistic (theta=2, no correlation)
baseline_path = Path("../theta_2_no_corr_baseline_data.pkl")
baseline_ks_stat = None

if baseline_path.exists():
    try:
        with open(baseline_path, 'rb') as f:
            baseline_data = pickle.load(f)
        baseline_ks_stat = baseline_data['ks_statistic']
        print(f"\nLoaded baseline KS statistic (θ=2, no correlation): {baseline_ks_stat:.4f}")
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

# %%
# Detect parameters
c_array, corr_factor_array, exponential_noise_array = detect_parameters_from_files(results_folder)

print("\nDetected parameter space:")
print(f"  c_array: {c_array}")
print(f"  corr_factor_array: {corr_factor_array}")
print(f"  exponential_noise_array (ms): {exponential_noise_array * 1000}")
print(f"  Total combinations: {len(c_array)} × {len(corr_factor_array)} × {len(exponential_noise_array)} = {len(c_array) * len(corr_factor_array) * len(exponential_noise_array)}")

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
# Collect ALL KS statistics
print("\nComputing KS statistics for all parameter combinations...")
ks_data = {}

for c in c_array:
    for corr_factor in corr_factor_array:
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
                
                # Store in nested dict
                if c not in ks_data:
                    ks_data[c] = {}
                if corr_factor not in ks_data[c]:
                    ks_data[c][corr_factor] = {}
                ks_data[c][corr_factor][noise] = {
                    'ks_stat': ks_stat,
                    'p_value': p_value
                }
                
                print(f"  c={c}, corr_factor={corr_factor}, noise={noise*1000:.1f}ms: KS={ks_stat:.4f}")
                
            except FileNotFoundError:
                print(f"  Warning: File not found for c={c}, corr_factor={corr_factor}, noise={noise*1000:.1f}ms")

# %%
# DETERMINE WHAT TYPE OF PLOTS TO CREATE based on parameter space
print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

# Case 1: Only c varies (single corr_factor, single noise)
if len(corr_factor_array) == 1 and len(exponential_noise_array) == 1:
    print(f"\nCase: Only c varies (corr_factor={corr_factor_array[0]}, noise={exponential_noise_array[0]*1000:.1f}ms)")
    
    corr_factor = corr_factor_array[0]
    noise = exponential_noise_array[0]
    
    # FIGURE 1: KS vs c
    print("Creating: ks_statistic_vs_c.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    c_vals = []
    ks_vals = []
    for c in c_array:
        if c in ks_data and corr_factor in ks_data[c] and noise in ks_data[c][corr_factor]:
            c_vals.append(c)
            ks_vals.append(ks_data[c][corr_factor][noise]['ks_stat'])
    
    ax.plot(c_vals, ks_vals, color='#1f77b4', marker='o', linewidth=2.5, markersize=10,
            label=f'corr_factor={corr_factor}, noise={noise*1000:.1f}ms')
    
    # Add baseline KS statistic as horizontal line
    if baseline_ks_stat is not None:
        ax.axhline(y=baseline_ks_stat, color='red', linestyle='--', linewidth=2,
                   label=f'Baseline (θ=2, no corr): {baseline_ks_stat:.4f}', alpha=0.7)
    
    ax.set_xlabel('c (correlation)', fontsize=13)
    ax.set_ylabel('KS Statistic (max CDF difference)', fontsize=13)
    ax.set_title('KS Statistic: Poisson vs DDM RTDs', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(c_array)
    
    plt.tight_layout()
    plt.savefig(output_folder / 'ks_statistic_vs_c.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # FIGURE 2: RTD comparisons
    print("Creating: rtd_comparison_all_c.png")
    n_c = len(c_array)
    
    # Determine subplot layout
    if n_c <= 4:
        nrows, ncols = 1, n_c
        figsize = (5*n_c, 5)
    elif n_c <= 6:
        nrows, ncols = 2, 3
        figsize = (18, 10)
    else:
        ncols = 4
        nrows = int(np.ceil(n_c / ncols))
        figsize = (20, 5*nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, squeeze=False)
    axes = axes.flatten()
    
    for idx, c in enumerate(c_array):
        try:
            data = load_data(c, corr_factor, noise)
            params = data['params']
            poisson_results = data['poisson']['results']
            ddm_results = data['ddm']['results']
            
            poisson_rts = poisson_results[poisson_results[:, 1] != 0, 0]
            ddm_rts = ddm_results[ddm_results[:, 1] != 0, 0]
            
            # Dynamically determine bin range to include negative times
            min_rt = min(np.min(poisson_rts), np.min(ddm_rts))
            max_rt = max(np.max(poisson_rts), np.max(ddm_rts))
            min_rt = min(0, min_rt)  # Ensure we at least start at 0 or below
            max_rt = max(2, max_rt)  # Ensure we at least go to 2 or above
            bins_rt = np.arange(min_rt, max_rt + 0.05, 0.05)
            
            ax = axes[idx]
            ax.hist(poisson_rts, bins=bins_rt, density=True, histtype='step',
                    label='Poisson', color='blue', linewidth=2)
            ax.hist(ddm_rts, bins=bins_rt, density=True, histtype='step',
                    label='DDM', color='red', linewidth=2)
            
            ax.set_xlabel("RT (s)", fontsize=11)
            ax.set_title(f'c = {c}\nN={params["N_right_and_left"]}', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_xlim(min_rt, max_rt)
            ax.legend(loc='upper right', fontsize=9)
            
            if idx % ncols == 0:
                ax.set_ylabel("Density", fontsize=11)
                
        except FileNotFoundError:
            ax = axes[idx]
            ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 2)
    
    # Hide extra subplots
    for idx in range(n_c, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Poisson vs DDM RTD Comparison (corr_factor={corr_factor}, noise={noise*1000:.1f}ms)',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_folder / 'rtd_comparison_all_c.png', dpi=150, bbox_inches='tight')
    plt.close()

# Case 2: Multiple parameters vary - create summary KS plot
else:
    print(f"\nCase: Multiple parameters vary")
    print("Creating: ks_statistic_summary.png")
    
    # Create subplots - one for each correlation factor
    n_corr_factors = len(corr_factor_array)
    
    # Determine subplot layout
    if n_corr_factors <= 3:
        nrows, ncols = 1, n_corr_factors
        figsize = (6*n_corr_factors, 5)
    elif n_corr_factors <= 6:
        nrows, ncols = 2, 3
        figsize = (18, 10)
    else:
        ncols = 4
        nrows = int(np.ceil(n_corr_factors / ncols))
        figsize = (24, 5*nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharey=True)
    axes = axes.flatten()
    
    # Color map for different noise values
    colors = plt.cm.tab10(np.linspace(0, 1, len(exponential_noise_array)))
    
    for idx, corr_factor in enumerate(corr_factor_array):
        ax = axes[idx]
        
        for noise_idx, noise in enumerate(exponential_noise_array):
            c_vals = []
            ks_vals = []
            
            for c in c_array:
                if c in ks_data and corr_factor in ks_data[c] and noise in ks_data[c][corr_factor]:
                    c_vals.append(c)
                    ks_vals.append(ks_data[c][corr_factor][noise]['ks_stat'])
            
            if len(c_vals) > 0:
                ax.plot(c_vals, ks_vals, marker='o', linewidth=2, markersize=8,
                       color=colors[noise_idx],
                       label=f'noise={noise*1000:.1f}ms')
        
        # Add baseline KS statistic as horizontal line
        if baseline_ks_stat is not None:
            ax.axhline(y=baseline_ks_stat, color='red', linestyle='--', linewidth=1.5,
                       label=f'Baseline (θ=2)', alpha=0.7)
        
        ax.set_xlabel('c (correlation)', fontsize=12)
        if idx % ncols == 0:  # Only leftmost subplots get y-axis label
            ax.set_ylabel('KS Statistic', fontsize=12)
        ax.set_title(f'Corr Factor = {corr_factor}', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots if any
    for idx in range(n_corr_factors, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('KS Statistic: Poisson vs DDM RTDs', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_folder / 'ks_statistic_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create additional plot: corr_factor on x-axis, different lines for each c
    print("Creating: ks_statistic_vs_corrfactor.png")
    
    n_noise = len(exponential_noise_array)
    
    # Determine subplot layout for noise values
    if n_noise <= 3:
        nrows, ncols = 1, n_noise
        figsize = (6*n_noise, 5)
    elif n_noise <= 6:
        nrows, ncols = 2, 3
        figsize = (18, 10)
    else:
        ncols = 4
        nrows = int(np.ceil(n_noise / ncols))
        figsize = (24, 5*nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharey=True)
    axes = axes.flatten()
    
    # Color map for different c values
    colors_c = plt.cm.tab10(np.linspace(0, 1, len(c_array)))
    
    for idx, noise in enumerate(exponential_noise_array):
        ax = axes[idx]
        
        for c_idx, c in enumerate(c_array):
            corr_factor_vals = []
            ks_vals = []
            
            for corr_factor in corr_factor_array:
                if c in ks_data and corr_factor in ks_data[c] and noise in ks_data[c][corr_factor]:
                    corr_factor_vals.append(corr_factor)
                    ks_vals.append(ks_data[c][corr_factor][noise]['ks_stat'])
            
            if len(corr_factor_vals) > 0:
                ax.plot(corr_factor_vals, ks_vals, marker='o', linewidth=2, markersize=8,
                       color=colors_c[c_idx],
                       label=f'c={c}')
        
        # Add baseline KS statistic as horizontal line
        if baseline_ks_stat is not None:
            ax.axhline(y=baseline_ks_stat, color='red', linestyle='--', linewidth=1.5,
                       label=f'Baseline (θ=2)', alpha=0.7)
        
        ax.set_xlabel('Correlation Factor', fontsize=12)
        if idx % ncols == 0:  # Only leftmost subplots get y-axis label
            ax.set_ylabel('KS Statistic', fontsize=12)
        ax.set_title(f'Noise = {noise*1000:.1f}ms', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots if any
    for idx in range(n_noise, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle('KS Statistic vs Correlation Factor', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_folder / 'ks_statistic_vs_corrfactor.png', dpi=150, bbox_inches='tight')
    plt.close()

# %%
print("\n" + "="*70)
print("ALL FIGURES CREATED")
print("="*70)
print(f"Output folder: {output_folder}/")
print("\nGenerated files:")
for fname in sorted(output_folder.glob('*.png')):
    print(f"  - {fname.name}")

