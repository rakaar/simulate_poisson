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
# Validation parameters - expanded stimulus set
VALIDATION_ABL_RANGE = [20, 40, 60]
VALIDATION_ILD_RANGE = [1, 2, 4, 8, 16]
N_TRIALS_VALIDATE = int(10e3)  # Number of trials per stimulus

print(f"\n{'='*70}")
print("VALIDATION STIMULUS PARAMETERS")
print(f"{'='*70}")
print(f"ABL values: {VALIDATION_ABL_RANGE}")
print(f"ILD values: {VALIDATION_ILD_RANGE}")
print(f"Total stimuli: {len(VALIDATION_ABL_RANGE) * len(VALIDATION_ILD_RANGE)}")
print(f"Trials per stimulus: {N_TRIALS_VALIDATE}")


# %%
def run_ddm_trial_single(trial_idx, N, r_right, r_left, theta, dt_sim=1e-4):
    """
    Simulate a single DDM trial.
    Returns: (rt, choice) where choice is 1 for right, -1 for left, 0 for no decision
    """
    mu = N * (r_right - r_left)  # Drift rate
    sigma = np.sqrt(N * (r_right + r_left))  # Noise standard deviation
    dB = np.sqrt(dt_sim)
    
    # Simulation parameters
    max_steps = int(50 / dt_sim)  # 50 seconds max
    
    # Initialize
    position = 0
    time = 0
    
    # Run the diffusion process
    for step in range(max_steps):
        # Update position
        position += mu * dt_sim + sigma * np.random.normal(0, dB)
        time += dt_sim
        
        # Check for threshold crossing
        if position >= theta:
            return (time, 1)  # Hit upper bound
        elif position <= -theta:
            return (time, -1)  # Hit lower bound
    
    # Time limit reached without decision
    return (np.nan, 0)


def compute_quantiles_from_rt_data(rt_data, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Compute quantiles from RT data, filtering out NaN values.
    """
    valid_rts = rt_data[~np.isnan(rt_data)]
    
    if len(valid_rts) == 0:
        return np.array([np.nan] * len(quantiles))
    
    return np.quantile(valid_rts, quantiles)


def simulate_ddm_for_stimulus(ABL, ILD, theta_ddm, n_trials=N_TRIALS_VALIDATE, show_progress=True):
    """
    Simulate DDM for a single stimulus and compute quantiles and accuracy.
    
    Returns:
        quantiles: Array of [q10, q30, q50, q70, q90]
        accuracy: Proportion of correct (right) choices
    """
    # Calculate rates
    r0 = Nr0_base / N
    r_db = (2*ABL + ILD)/2
    l_db = (2*ABL - ILD)/2
    pr = (10 ** (r_db/20))
    pl = (10 ** (l_db/20))
    
    den = (pr ** (lam * l)) + (pl ** (lam * l))
    rr = (pr ** lam) / den
    rl = (pl ** lam) / den
    r_right = r0 * rr
    r_left = r0 * rl
    
    # Run DDM simulations
    if show_progress:
        ddm_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
            delayed(run_ddm_trial_single)(i, N, r_right, r_left, theta_ddm, dt) 
            for i in tqdm(range(n_trials), desc=f'  DDM ABL={ABL}, ILD={ILD}', leave=False)
        )
    else:
        ddm_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
            delayed(run_ddm_trial_single)(i, N, r_right, r_left, theta_ddm, dt) 
            for i in range(n_trials)
        )
    
    ddm_array = np.array(ddm_data)
    rts = ddm_array[:, 0]
    choices = ddm_array[:, 1]
    
    # Compute accuracy (proportion of correct choices, assuming right is correct)
    valid_choices = choices[~np.isnan(rts)]
    if len(valid_choices) > 0:
        accuracy = np.sum(valid_choices == 1) / len(valid_choices)
    else:
        accuracy = 0.5
    
    # Compute quantiles
    quantiles = compute_quantiles_from_rt_data(rts, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])
    
    return quantiles, accuracy


def simulate_poisson_for_stimulus(ABL, ILD, Nr0_scaled, theta_poisson, n_trials=N_TRIALS_VALIDATE, show_progress=True):
    """
    Simulate Poisson for a single stimulus and compute quantiles and accuracy.
    
    Returns:
        quantiles: Array of [q10, q30, q50, q70, q90]
        accuracy: Proportion of correct (right) choices
    """
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
    
    # Run Poisson simulations
    if show_progress:
        poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
            delayed(run_poisson_trial)(N, rho, r_right, r_left, theta_poisson) 
            for _ in tqdm(range(n_trials), desc=f'  Poisson ABL={ABL}, ILD={ILD}', leave=False)
        )
    else:
        poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
            delayed(run_poisson_trial)(N, rho, r_right, r_left, theta_poisson) 
            for _ in range(n_trials)
        )
    
    poisson_array = np.array(poisson_data)
    rts = poisson_array[:, 0]
    choices = poisson_array[:, 1]
    
    # Compute accuracy
    valid_choices = choices[~np.isnan(rts)]
    if len(valid_choices) > 0:
        accuracy = np.sum(valid_choices == 1) / len(valid_choices)
    else:
        accuracy = 0.5
    
    # Compute quantiles
    quantiles = compute_quantiles_from_rt_data(rts, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])
    
    return quantiles, accuracy


# %%
def simulate_validation_data(original_theta):
    """
    Generate validation data by simulating DDM and Poisson for all stimuli.
    This is the expensive operation - separate from plotting.
    
    Returns:
        dict with keys: ddm_quantiles_dict, ddm_acc_dict, poisson_quantiles_dict, 
                        poisson_acc_dict, rate_scaling_factor_opt, theta_poisson_opt
    """
    print(f"\n{'='*70}")
    print(f"GENERATING VALIDATION DATA FOR ORIGINAL THETA = {original_theta}")
    print(f"{'='*70}")
    
    result = results_dict[original_theta]
    rate_scaling_factor_opt = result['rate_scaling_factor_opt']
    theta_increment_opt = result['theta_increment_opt']
    theta_poisson_opt = result['theta_poisson_opt']
    
    print(f"\nOptimized parameters:")
    print(f"  Rate scaling factor: {rate_scaling_factor_opt:.4f}")
    print(f"  Theta increment: {theta_increment_opt}")
    print(f"  DDM theta: {original_theta}")
    print(f"  Poisson theta: {theta_poisson_opt}")
    
    Nr0_scaled = Nr0_base * rate_scaling_factor_opt
    
    # Storage for results
    ddm_quantiles_dict = {}
    ddm_acc_dict = {}
    poisson_quantiles_dict = {}
    poisson_acc_dict = {}
    
    # Simulate for all stimuli
    total_stimuli = len(VALIDATION_ABL_RANGE) * len(VALIDATION_ILD_RANGE)
    print(f"\nSimulating DDM and Poisson for {total_stimuli} stimuli...")
    print(f"  {N_TRIALS_VALIDATE} trials per stimulus")
    
    # Create list of all stimuli for progress tracking
    stimuli_list = [(ABL, ILD) for ABL in VALIDATION_ABL_RANGE for ILD in VALIDATION_ILD_RANGE]
    
    for stim_idx, (ABL, ILD) in enumerate(tqdm(stimuli_list, desc="Overall progress"), 1):
        print(f"\nStimulus {stim_idx}/{total_stimuli}: ABL={ABL}, ILD={ILD}")
        
        # Simulate DDM
        ddm_q, ddm_acc = simulate_ddm_for_stimulus(ABL, ILD, original_theta, n_trials=N_TRIALS_VALIDATE, show_progress=True)
        ddm_quantiles_dict[(ABL, ILD)] = ddm_q
        ddm_acc_dict[(ABL, ILD)] = ddm_acc
        
        # Simulate Poisson
        poisson_q, poisson_acc = simulate_poisson_for_stimulus(ABL, ILD, Nr0_scaled, theta_poisson_opt, n_trials=N_TRIALS_VALIDATE, show_progress=True)
        poisson_quantiles_dict[(ABL, ILD)] = poisson_q
        poisson_acc_dict[(ABL, ILD)] = poisson_acc
    
    print("\n✓ All simulations complete")
    
    return {
        'ddm_quantiles_dict': ddm_quantiles_dict,
        'ddm_acc_dict': ddm_acc_dict,
        'poisson_quantiles_dict': poisson_quantiles_dict,
        'poisson_acc_dict': poisson_acc_dict,
        'rate_scaling_factor_opt': rate_scaling_factor_opt,
        'theta_poisson_opt': theta_poisson_opt,
        'original_theta': original_theta
    }


def plot_validation_results(validation_data, timestamp):
    """
    Create validation plots from pre-generated data.
    This is the cheap operation - can be run multiple times to adjust styling.
    
    Parameters:
        validation_data: dict returned by simulate_validation_data()
        timestamp: timestamp string for filename
    """
    # Extract data
    ddm_quantiles_dict = validation_data['ddm_quantiles_dict']
    ddm_acc_dict = validation_data['ddm_acc_dict']
    poisson_quantiles_dict = validation_data['poisson_quantiles_dict']
    poisson_acc_dict = validation_data['poisson_acc_dict']
    rate_scaling_factor_opt = validation_data['rate_scaling_factor_opt']
    theta_poisson_opt = validation_data['theta_poisson_opt']
    original_theta = validation_data['original_theta']
    
    # Plotting style configuration
    ABL_color_map = {20: '#2E86AB', 40: '#A23B72', 60: '#F18F01'}
    quantile_labels = ['Q10', 'Q30', 'Q50', 'Q70', 'Q90']
    
    # ===== QUANTILE PLOTS (1x3) =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax_idx, ABL in enumerate(VALIDATION_ABL_RANGE):
        ax = axes[ax_idx]
        
        for q_idx in range(5):
            # DDM quantiles
            ddm_q_values = [ddm_quantiles_dict[(ABL, ILD)][q_idx] for ILD in VALIDATION_ILD_RANGE]
            # Poisson quantiles
            poisson_q_values = [poisson_quantiles_dict[(ABL, ILD)][q_idx] for ILD in VALIDATION_ILD_RANGE]
            
            # Plot with different line styles for different quantiles
            alpha_val = 0.5 + (q_idx * 0.1)
            ax.plot(VALIDATION_ILD_RANGE, ddm_q_values, marker='o', markersize=6, 
                   label=f'{quantile_labels[q_idx]}', color=ABL_color_map[ABL], 
                   alpha=alpha_val, linewidth=1.5)
            ax.plot(VALIDATION_ILD_RANGE, poisson_q_values, marker='x', markersize=8, 
                   linestyle='--', color=ABL_color_map[ABL], alpha=alpha_val, linewidth=1.5)
        
        ax.set_xlabel('ILD', fontsize=12, fontweight='bold')
        ax.set_ylabel('RT (s)', fontsize=12, fontweight='bold')
        ax.set_title(f'ABL = {ABL}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        # ax.legend(fontsize=9, loc='best')
        # ax.set_xscale('log')
        ax.set_xticks(VALIDATION_ILD_RANGE)
        ax.set_xticklabels(VALIDATION_ILD_RANGE)
    
    fig.suptitle(f'RT Quantiles: θ_DDM={original_theta}, θ_Poisson={theta_poisson_opt}, Rate×{rate_scaling_factor_opt:.2f}\n(dots=DDM, x=Poisson)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'validation_quantiles_theta_{original_theta}_{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Quantile plot saved: validation_quantiles_theta_{original_theta}_{timestamp}.png")
    plt.show()
    
    # ===== PSYCHOMETRIC PLOTS (1x3) =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax_idx, ABL in enumerate(VALIDATION_ABL_RANGE):
        ax = axes[ax_idx]
        
        # DDM accuracy
        ddm_acc_values = [ddm_acc_dict[(ABL, ILD)] for ILD in VALIDATION_ILD_RANGE]
        # Poisson accuracy
        poisson_acc_values = [poisson_acc_dict[(ABL, ILD)] for ILD in VALIDATION_ILD_RANGE]
        
        ax.plot(VALIDATION_ILD_RANGE, ddm_acc_values, marker='o', markersize=10, 
               label='DDM', color=ABL_color_map[ABL], linewidth=2.5)
        ax.plot(VALIDATION_ILD_RANGE, poisson_acc_values, marker='x', markersize=12, 
               linestyle='--', label='Poisson', color=ABL_color_map[ABL], linewidth=2.5)
        
        ax.set_xlabel('ILD', fontsize=12, fontweight='bold')
        ax.set_ylabel('P(Right)', fontsize=12, fontweight='bold')
        ax.set_title(f'ABL = {ABL}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best')
        ax.set_ylim([0.45, 1.05])
        # ax.set_xscale('log')
        ax.set_xticks(VALIDATION_ILD_RANGE)
        ax.set_xticklabels(VALIDATION_ILD_RANGE)
    
    fig.suptitle(f'Psychometric Functions: θ_DDM={original_theta}, θ_Poisson={theta_poisson_opt}, Rate×{rate_scaling_factor_opt:.2f}\n(dots=DDM, x=Poisson)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'validation_psychometric_theta_{original_theta}_{timestamp}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Psychometric plot saved: validation_psychometric_theta_{original_theta}_{timestamp}.png")
    plt.show()


# %%
# Run validation analysis for a selected theta
selected_theta = original_theta_values[0]
print(f"\nRunning validation analysis for original_theta = {selected_theta}...")

# Step 1: Generate data (expensive - run once)
validation_data = simulate_validation_data(selected_theta)
#%%
# Step 2: Create plots (cheap - can rerun to adjust styling)
plot_validation_results(validation_data, timestamp)

# %%