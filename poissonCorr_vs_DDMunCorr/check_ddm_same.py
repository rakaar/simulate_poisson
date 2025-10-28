# %%
"""
Check DDM simulation results by plotting RTDs for all parameter combinations.
Reads DDM params and results from pkl files and plots RT distributions
with area proportional to fraction of up/down trials.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# %%
def plot_ddm_rtd(ddm_data, ddm_params, file_params, ax, max_rt=None):
    """
    Plot DDM RT distribution with area proportional to fraction of trials.
    
    Parameters:
    -----------
    ddm_data : np.ndarray
        Shape (N, 2) with column 0 = RTs, column 1 = choices (1, -1, or 0)
    ddm_params : dict
        DDM parameters (mu, sigma, theta_ddm, dt, dB)
    file_params : dict
        File-specific parameters (c, corr_factor, noise, etc.)
    ax : matplotlib axis
        Axis to plot on
    max_rt : float, optional
        Maximum RT for binning. If None, uses max RT from positive choices.
    """
    # Filter decided trials (exclude 0 choices)
    decided_mask = ddm_data[:, 1] != 0
    ddm_decided = ddm_data[decided_mask]
    
    # Separate positive and negative choices
    pos_rts_ddm = ddm_decided[ddm_decided[:, 1] == 1, 0]
    neg_rts_ddm = ddm_decided[ddm_decided[:, 1] == -1, 0]
    
    # Set up bins
    if max_rt is None:
        max_rt = np.max(pos_rts_ddm) if len(pos_rts_ddm) > 0 else 10
    bins_rt = np.arange(0, max_rt, max_rt / 1000)
    
    # Calculate histograms
    pos_hist_ddm, _ = np.histogram(pos_rts_ddm, bins=bins_rt, density=True)
    neg_hist_ddm, _ = np.histogram(neg_rts_ddm, bins=bins_rt, density=True)
    
    bin_centers = (bins_rt[:-1] + bins_rt[1:]) / 2
    
    # Calculate fractions
    ddm_up = len(pos_rts_ddm)
    ddm_down = len(neg_rts_ddm)
    ddm_frac_up = ddm_up / (ddm_up + ddm_down) if (ddm_up + ddm_down) > 0 else 0
    ddm_frac_down = ddm_down / (ddm_up + ddm_down) if (ddm_up + ddm_down) > 0 else 0
    
    # Plot with area proportional to fraction of trials
    ax.plot(bin_centers, pos_hist_ddm * ddm_frac_up, 
            label=f'Positive Choice ({ddm_frac_up:.3f})', 
            color='red', linestyle='-', linewidth=2)
    ax.plot(bin_centers, -neg_hist_ddm * ddm_frac_down, 
            label=f'Negative Choice ({ddm_frac_down:.3f})', 
            color='blue', linestyle='-', linewidth=2)
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Reaction Time (s)", fontsize=10)
    ax.set_ylabel("Density × Fraction", fontsize=10)
    
    # Create title with parameters
    c = file_params['c']
    corr_factor = file_params['corr_factor']
    noise = file_params['exponential_noise_to_spk_time']
    mu = ddm_params['mu']
    sigma = ddm_params['sigma']
    theta = ddm_params['theta_ddm']
    
    ax.set_title(
        f'c={c:.2f}, cf={corr_factor:.1f}, noise={noise*1000:.1f}ms\n'
        f'μ={mu:.3f}, σ={sigma:.3f}, θ={theta:.2f}',
        fontsize=9
    )
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    return ddm_frac_up, ddm_frac_down


# %%
# Get all pkl files from results folder
script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
results_dir = script_dir / 'results'
output_dir = script_dir / 'figures'  # Save figures here instead of in results/
output_dir.mkdir(exist_ok=True)

pkl_files = sorted(glob.glob(str(results_dir / '*.pkl')))

print(f"Found {len(pkl_files)} pkl files")

# %%
# Organize files by correlation coefficient for plotting
files_by_c = {}
for pkl_file in pkl_files:
    # Extract c value from filename
    filename = Path(pkl_file).name
    c_str = filename.split('_')[1]
    c_val = float(c_str)
    
    if c_val not in files_by_c:
        files_by_c[c_val] = []
    files_by_c[c_val].append(pkl_file)

# %%
# Process each c value separately
for c_val in sorted(files_by_c.keys()):
    files = files_by_c[c_val]
    print(f"\nProcessing c={c_val} ({len(files)} files)")
    
    # Create figure with subplots
    n_files = len(files)
    ncols = 5  # 5 corr_factors
    nrows = 4  # 4 noise values
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 12))
    fig.suptitle(f'DDM RT Distributions: c={c_val}', fontsize=16, fontweight='bold')
    
    # Sort files to ensure consistent ordering
    files_sorted = sorted(files)
    
    for idx, pkl_file in enumerate(files_sorted):
        # Load data
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract DDM data
        ddm_data = data['ddm']['results']
        ddm_params = data['ddm']['params']
        file_params = data['params']
        
        # Determine subplot position
        row = idx // ncols
        col = idx % ncols
        
        if nrows == 1:
            ax = axes[col] if ncols > 1 else axes
        elif ncols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        
        # Plot
        frac_up, frac_down = plot_ddm_rtd(ddm_data, ddm_params, file_params, ax)
        
        # Print summary
        filename = Path(pkl_file).name
        print(f"  {filename}: frac_up={frac_up:.3f}, frac_down={frac_down:.3f}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure
    output_file = output_dir / f'ddm_rtd_c_{c_val}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    # plt.show()  # Uncomment to display interactively
    plt.close()

print("\n=== All plots generated ===")

# %%
# Create a single summary plot showing one example from each c value
print("\nCreating summary plot...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DDM RT Distributions: Summary (corr_factor=5.0, noise=1.0ms)', 
             fontsize=14, fontweight='bold')

c_values = sorted(files_by_c.keys())
for idx, c_val in enumerate(c_values):
    # Find file with corr_factor=5.0 and noise=1.0ms
    target_file = None
    for pkl_file in files_by_c[c_val]:
        if 'corrfactor_5.0_noise_1.0ms' in pkl_file:
            target_file = pkl_file
            break
    
    if target_file:
        with open(target_file, 'rb') as f:
            data = pickle.load(f)
        
        ax = axes.flat[idx]
        plot_ddm_rtd(data['ddm']['results'], data['ddm']['params'], 
                    data['params'], ax)

plt.tight_layout(rect=[0, 0, 1, 0.97])
summary_file = output_dir / 'ddm_rtd_summary.png'
plt.savefig(summary_file, dpi=150, bbox_inches='tight')
print(f"Saved: {summary_file}")
plt.close()

print("\n=== Complete ===")
