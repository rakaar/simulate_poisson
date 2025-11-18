# %%
"""
Analysis script for BADS Rate Scaling and Bound Increment Optimization Results

This script loads the results from find_bound_incr_rate_scale_bads.py and provides
detailed analysis and visualizations of the optimized parameters.
"""

# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
from mgf_helper_utils import poisson_fc_dt, ddm_fc_dt
from corr_poisson_utils_subtractive import run_poisson_trial

# %%
# Find and load the most recent optimization results
# Try to find combined file first, then fall back to intermediate files
pkl_pattern_combined = 'bads_rate_bound_optimization_results_*_all.pkl'
pkl_pattern_old = 'bads_rate_bound_optimization_results_*.pkl'

pkl_files_combined = glob.glob(pkl_pattern_combined)
pkl_files_old = [f for f in glob.glob(pkl_pattern_old) if not f.endswith('_all.pkl') and 'theta' not in f]

# Prefer combined file if available
if pkl_files_combined:
    latest_pkl = max(pkl_files_combined, key=lambda x: Path(x).stat().st_mtime)
    print(f"Loading combined results from: {latest_pkl}")
elif pkl_files_old:
    latest_pkl = max(pkl_files_old, key=lambda x: Path(x).stat().st_mtime)
    print(f"Loading results from: {latest_pkl}")
else:
    # Try to load from intermediate files
    pkl_pattern_intermediate = 'bads_rate_bound_optimization_results_*_theta_*.pkl'
    pkl_files_intermediate = glob.glob(pkl_pattern_intermediate)
    
    if not pkl_files_intermediate:
        raise FileNotFoundError(f"No optimization result files found. Looking for:\n"
                              f"  - {pkl_pattern_combined}\n"
                              f"  - {pkl_pattern_old}\n"
                              f"  - {pkl_pattern_intermediate}")
    
    # Load all intermediate files and combine them
    print(f"Found {len(pkl_files_intermediate)} intermediate files. Combining them...")
    results_dict = {}
    for pkl_file in pkl_files_intermediate:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            results_dict[data['original_theta']] = data['result']
    
    # Reconstruct the saved_data structure
    # Get fixed_params and timestamp from first file
    with open(pkl_files_intermediate[0], 'rb') as f:
        first_data = pickle.load(f)
        
    saved_data = {
        'results_dict': results_dict,
        'original_theta_values': sorted(results_dict.keys()),
        'fixed_params': first_data['fixed_params'],
        'timestamp': first_data['timestamp'],
    }
    
    print(f"Successfully combined {len(results_dict)} intermediate result files")
    latest_pkl = "Combined from intermediate files"

# Load data if we haven't already (from intermediate files case)
if 'saved_data' not in locals():
    with open(latest_pkl, 'rb') as f:
        saved_data = pickle.load(f)

# Extract data
results_dict = saved_data['results_dict']
original_theta_values = saved_data['original_theta_values']
fixed_params = saved_data['fixed_params']
timestamp = saved_data['timestamp']

# Extract fixed parameters
lam = fixed_params['lam']
l = fixed_params['l']
Nr0_base = fixed_params['Nr0_base']
N = fixed_params['N']
rho = fixed_params['rho']
dt = fixed_params['dt']
ABL_range = fixed_params['ABL_range']
ILD_range = fixed_params['ILD_range']

print("\n" + "="*70)
print("BADS OPTIMIZATION RESULTS ANALYSIS")
print("="*70)
print(f"\nLoaded results for {len(original_theta_values)} original_theta values")
print(f"Original theta values: {original_theta_values}")

# %%
# Display summary table
print(f"\n{'='*70}")
print("SUMMARY TABLE")
print(f"{'='*70}")
print(f"\n{'Original Theta':<15} {'Rate Scaling':<15} {'Theta Inc':<12} {'Theta Poisson':<15} {'Sq. Err':<12}")
print("-" * 80)

for original_theta in original_theta_values:
    result = results_dict[original_theta]
    print(f"{original_theta:<15} "
          f"{result['rate_scaling_factor_opt']:<15.4f} "
          f"{result['theta_increment_opt']:<12} "
          f"{result['theta_poisson_opt']:<15} "
          f"{result['final_objective_value']:<12.6f} ")

# %%
# Create 3-panel plot based on summary table
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Rate scaling factor vs original theta
axes[0].plot(original_theta_values, 
             [results_dict[theta]['rate_scaling_factor_opt'] for theta in original_theta_values],
             marker='o', linewidth=2.5, markersize=10, color='#2E86AB')
axes[0].set_xlabel('Original Theta')
axes[0].set_ylabel('Rate Scaling Factor')
axes[0].set_title('Rate Scaling Factor vs Original Theta')
axes[0].set_xticks(original_theta_values)

# Plot 2: Theta increment vs original theta
axes[1].plot(original_theta_values, 
             [results_dict[theta]['theta_increment_opt'] for theta in original_theta_values],
             marker='s', linewidth=2.5, markersize=10, color='#A23B72')
axes[1].set_xlabel('Original Theta')
axes[1].set_ylabel('Theta Increment')
axes[1].set_title('Theta Increment vs Original Theta')
axes[1].set_xticks(original_theta_values)
axes[1].set_yticks(range(0, max([results_dict[theta]['theta_increment_opt'] for theta in original_theta_values]) + 2))

# Plot 3: Final squared error vs original theta
axes[2].plot(original_theta_values, 
             [results_dict[theta]['final_objective_value'] for theta in original_theta_values],
             marker='^', linewidth=2.5, markersize=10, color='#F18F01')
axes[2].set_xlabel('Original Theta')
axes[2].set_ylabel('Squared Error)')
axes[2].set_title('Sq. Err vs Original Theta(log scale)')
axes[2].set_yscale('log')
axes[2].set_xticks(original_theta_values)

plt.tight_layout()
summary_plot_filename = f'bads_rate_bound_summary_3panel_{timestamp}.png'
plt.savefig(summary_plot_filename, dpi=150, bbox_inches='tight')
print(f"\n✓ Summary plots saved to: {summary_plot_filename}")
plt.show()

# %%
# Create comprehensive visualization
fig = plt.figure(figsize=(18, 10))

# Plot 1: Rate scaling factor vs original theta
ax1 = plt.subplot(2, 3, 1)
ax1.plot(original_theta_values, 
         [results_dict[theta]['rate_scaling_factor_opt'] for theta in original_theta_values],
         marker='o', linewidth=2, markersize=10, color='#2E86AB')
ax1.set_xlabel('Original Theta', fontsize=12, fontweight='bold')
ax1.set_ylabel('Optimal Rate Scaling Factor', fontsize=12, fontweight='bold')
ax1.set_title('Rate Scaling Factor vs Original Theta', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(original_theta_values)

# Plot 2: Theta increment vs original theta
ax2 = plt.subplot(2, 3, 2)
ax2.plot(original_theta_values, 
         [results_dict[theta]['theta_increment_opt'] for theta in original_theta_values],
         marker='s', linewidth=2, markersize=10, color='#A23B72')
ax2.set_xlabel('Original Theta', fontsize=12, fontweight='bold')
ax2.set_ylabel('Optimal Theta Increment', fontsize=12, fontweight='bold')
ax2.set_title('Theta Increment vs Original Theta', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(original_theta_values)

# Plot 3: Final MSE vs original theta
ax3 = plt.subplot(2, 3, 3)
ax3.plot(original_theta_values, 
         [results_dict[theta]['final_objective_value'] for theta in original_theta_values],
         marker='^', linewidth=2, markersize=10, color='#F18F01')
ax3.set_xlabel('Original Theta', fontsize=12, fontweight='bold')
ax3.set_ylabel('Final MSE', fontsize=12, fontweight='bold')
ax3.set_title('Final MSE vs Original Theta', fontsize=13, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xticks(original_theta_values)

# Plot 4: Poisson theta vs original theta
ax4 = plt.subplot(2, 3, 4)
ax4.plot(original_theta_values, 
         [results_dict[theta]['theta_poisson_opt'] for theta in original_theta_values],
         marker='D', linewidth=2, markersize=10, color='#6A994E', label='Poisson Theta')
ax4.plot(original_theta_values, original_theta_values, 
         'k--', linewidth=2, alpha=0.5, label='Original Theta')
ax4.set_xlabel('Original Theta', fontsize=12, fontweight='bold')
ax4.set_ylabel('Theta Value', fontsize=12, fontweight='bold')
ax4.set_title('Original vs Poisson Theta', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(fontsize=10)
ax4.set_xticks(original_theta_values)

# Plot 5: Function evaluations vs original theta
ax5 = plt.subplot(2, 3, 5)
ax5.bar(original_theta_values, 
        [results_dict[theta]['func_count'] for theta in original_theta_values],
        color='#BC4B51', alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.set_xlabel('Original Theta', fontsize=12, fontweight='bold')
ax5.set_ylabel('Number of Function Evaluations', fontsize=12, fontweight='bold')
ax5.set_title('BADS Function Evaluations', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
ax5.set_xticks(original_theta_values)

# Plot 6: Optimization time vs original theta
ax6 = plt.subplot(2, 3, 6)
ax6.bar(original_theta_values, 
        [results_dict[theta]['optimization_time']/60 for theta in original_theta_values],
        color='#5F0F40', alpha=0.7, edgecolor='black', linewidth=1.5)
ax6.set_xlabel('Original Theta', fontsize=12, fontweight='bold')
ax6.set_ylabel('Optimization Time (minutes)', fontsize=12, fontweight='bold')
ax6.set_title('Optimization Time', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
ax6.set_xticks(original_theta_values)

plt.tight_layout()
plt.savefig(f'bads_rate_bound_analysis_{timestamp}.png', dpi=150, bbox_inches='tight')
print(f"\nComprehensive analysis plot saved to: bads_rate_bound_analysis_{timestamp}.png")
plt.show()

# %%
# Detailed comparison for a specific original_theta
def analyze_single_theta(original_theta, n_sim_validate=int(50e3)):
    """
    Detailed analysis for a specific original_theta value.
    Generates high-fidelity simulations to validate the optimization results.
    """
    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS FOR ORIGINAL THETA = {original_theta}")
    print(f"{'='*70}")
    
    result = results_dict[original_theta]
    rate_scaling_factor_opt = result['rate_scaling_factor_opt']
    theta_increment_opt = result['theta_increment_opt']
    theta_poisson_opt = result['theta_poisson_opt']
    
    print(f"\nOptimized parameters:")
    print(f"  Rate scaling factor: {rate_scaling_factor_opt:.4f}")
    print(f"  Theta increment: {theta_increment_opt}")
    print(f"  Poisson theta: {theta_poisson_opt}")
    
    # Get DDM data
    ddm_rt_data = result['ddm_rt_data']
    ddm_acc_data = result['ddm_acc_data']
    
    # Compute Poisson predictions with optimized parameters
    Nr0_scaled = Nr0_base * rate_scaling_factor_opt
    
    poisson_rt_data = {}
    poisson_acc_data = {}
    
    print(f"\nComputing Poisson predictions for all stimuli...")
    for ABL in tqdm(ABL_range):
        for ILD in ILD_range:
            # Calculate rates
            r0 = Nr0_scaled / N
            r_db = (2*ABL + ILD)/2
            l_db = (2*ABL - ILD)/2
            pr = (10 ** (r_db/20))
            pl = (10 ** (l_db/20))
            
            den = (pr ** (lam * l)) + (pl ** (lam * l))
            rr = (pr ** lam) / den
            rl = (pl ** lam) / den
            r_right = r0 * rr
            r_left = r0 * rl
            
            # Estimate bound offset
            poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
                delayed(run_poisson_trial)(N, rho, r_right, r_left, theta_poisson_opt) 
                for _ in range(int(10e3))
            )
            bound_offsets = np.array([data[2] for data in poisson_data])
            bound_offsets = bound_offsets[~np.isnan(bound_offsets)]
            bound_offset_mean = np.mean(bound_offsets) if len(bound_offsets) > 0 else 0.0
            
            # Effective theta
            theta_poisson_eff = theta_poisson_opt + bound_offset_mean
            
            # Get Poisson predictions
            poisson_acc, poisson_mean_rt = poisson_fc_dt(
                N, rho, theta_poisson_eff, lam, l, Nr0_scaled, ABL, ILD, dt
            )
            
            poisson_rt_data[(ABL, ILD)] = poisson_mean_rt
            poisson_acc_data[(ABL, ILD)] = poisson_acc
    
    # Calculate errors
    squared_error_rt = sum(
        (poisson_rt_data[(ABL, ILD)] - ddm_rt_data[(ABL, ILD)])**2 
        for ABL in ABL_range for ILD in ILD_range
    )
    
    squared_error_acc = sum(
        (poisson_acc_data[(ABL, ILD)] - ddm_acc_data[(ABL, ILD)])**2 
        for ABL in ABL_range for ILD in ILD_range
    )
    
    print(f"\nValidation errors:")
    print(f"  Squared Error (RT): {squared_error_rt:.6f}")
    print(f"  Squared Error (Acc): {squared_error_acc:.6f}")
    print(f"  Total MSE: {squared_error_rt + squared_error_acc:.6f}")
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chronometric plot
    ABL_color_map = {20: '#2E86AB', 40: '#A23B72', 60: '#F18F01'}
    ax1 = axes[0]
    for ABL in ABL_range:
        ax1.plot([ILD for ILD in ILD_range], 
                [ddm_rt_data[(ABL, ILD)] for ILD in ILD_range], 
                label=f'DDM ABL={ABL}', marker='o', markersize=8, 
                linewidth=2, color=ABL_color_map[ABL])
        ax1.plot([ILD for ILD in ILD_range], 
                [poisson_rt_data[(ABL, ILD)] for ILD in ILD_range], 
                label=f'Poisson ABL={ABL}', marker='x', markersize=10, 
                linewidth=2, linestyle='--', color=ABL_color_map[ABL])
    ax1.set_xlabel('ILD', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Mean RT', fontsize=13, fontweight='bold')
    ax1.set_title(f'Chronometric: Original θ={original_theta}, Rate×{rate_scaling_factor_opt:.2f}, θ+{theta_increment_opt}\nSq.Err={squared_error_rt:.6f}', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Psychometric plot
    ax2 = axes[1]
    for a_idx, ABL in enumerate(ABL_range):
        offset = a_idx * 0.15
        x_ddm = np.array(ILD_range) - offset
        x_poisson = np.array(ILD_range) + offset
        
        ax2.plot(x_ddm, 
                [ddm_acc_data[(ABL, ILD)] for ILD in ILD_range], 
                label=f'DDM ABL={ABL}', marker='o', markersize=8, 
                linewidth=2, linestyle='-', color=ABL_color_map[ABL])
        ax2.plot(x_poisson, 
                [poisson_acc_data[(ABL, ILD)] for ILD in ILD_range], 
                label=f'Poisson ABL={ABL}', marker='x', markersize=10, 
                linewidth=2, linestyle='--', color=ABL_color_map[ABL])
    
    ax2.set_xlabel('ILD', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title(f'Psychometric: Original θ={original_theta}, Rate×{rate_scaling_factor_opt:.2f}, θ+{theta_increment_opt}\nSq.Err={squared_error_acc:.6f}', 
                 fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0.45, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'detailed_comparison_theta_{original_theta}_{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"\nDetailed comparison plot saved to: detailed_comparison_theta_{original_theta}_{timestamp}.png")
    plt.show()
    
    return poisson_rt_data, poisson_acc_data


# %%
# Run detailed analysis for a selected theta (change this to analyze different values)
selected_theta = original_theta_values[0]  # Change index to analyze different theta
print(f"\nRunning detailed analysis for original_theta = {selected_theta}...")
poisson_rt, poisson_acc = analyze_single_theta(selected_theta)

