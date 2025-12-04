# %%
"""
Analysis for BADS optimization with separate left/right rate scaling factors.
FAST VERSION: Uses vectorized DDM and optimized Poisson (no pandas).

Mirrors analyse_bads_rate_left_right_seperate_bound_optim_quantiles.py but with
optimized simulation functions for ~10x speedup.
"""

# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from pathlib import Path
from tqdm import tqdm
from corr_poisson_utils_subtractive import generate_correlated_pool

# %%
# Find and load the most recent optimization results
# Try to find combined file first, then fall back to intermediate files
pkl_pattern_combined = 'bads_rate_bound_optimization_results_left_right_seperate_*_all.pkl'
pkl_pattern_old = 'bads_rate_bound_optimization_results_left_right_seperate_*.pkl'

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
    pkl_pattern_intermediate = 'bads_rate_bound_optimization_results_left_right_seperate_*_theta_*.pkl'
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
# N = fixed_params['N']
# rho = fixed_params['rho']
N = 10
rho = 1e-3
dt = fixed_params['dt']
ABL_range = fixed_params['ABL_range']
ILD_range = fixed_params['ILD_range']

print("\n" + "="*70)
print("BADS OPTIMIZATION RESULTS ANALYSIS (LEFT/RIGHT SEPARATE) - FAST")
print("="*70)
print(f"\nLoaded results for {len(original_theta_values)} original_theta values")
print(f"Original theta values: {original_theta_values}")

# %%
# Display summary table
print(f"\n{'='*70}")
print("SUMMARY TABLE")
print(f"{'='*70}")
print(f"\n{'Original Theta':<15} {'Rate Scale R':<15} {'Rate Scale L':<15} {'Theta Inc':<12} {'Theta Poisson':<15} {'Sq. Err':<12}")
print("-" * 95)

for original_theta in original_theta_values:
    result = results_dict[original_theta]
    print(f"{original_theta:<15} "
          f"{result['rate_scaling_right_opt']:<15.4f} "
          f"{result['rate_scaling_left_opt']:<15.4f} "
          f"{result['theta_increment_opt']:<12} "
          f"{result['theta_poisson_opt']:<15} "
          f"{result['final_objective_value']:<12.6f} ")
##################################
print('------------------------------------------------------')
result['rate_scaling_right_opt'] = 2.1
result['rate_scaling_left_opt'] = 2.1
result['theta_increment_opt'] = 1
result['theta_poisson_opt'] = 3
print(f'{result['rate_scaling_right_opt']}')
print(f'{result['rate_scaling_left_opt']}')
print(f'{result['theta_increment_opt']}')
print(f'{result['theta_poisson_opt']}')
print('------------------------------------------------------')
##################################


# %%
# Create 3-panel plot based on summary table
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Rate scaling factors vs original theta
axes[0].plot(
    original_theta_values,
    [results_dict[theta]['rate_scaling_right_opt'] for theta in original_theta_values],
    marker='o',
    linewidth=2.5,
    markersize=10,
    color='#2E86AB',
    label='Right'
)
axes[0].plot(
    original_theta_values,
    [results_dict[theta]['rate_scaling_left_opt'] for theta in original_theta_values],
    marker='s',
    linewidth=2.5,
    markersize=10,
    color='#A23B72',
    label='Left'
)
axes[0].set_xlabel('Original Theta')
axes[0].set_ylabel('Rate Scaling Factor')
axes[0].set_title('Rate Scaling Factor vs Original Theta')
axes[0].set_xticks(original_theta_values)
axes[0].legend()

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
summary_plot_filename = f'bads_rate_bound_summary_left_right_seperate_3panel_{timestamp}.png'
plt.savefig(summary_plot_filename, dpi=150, bbox_inches='tight')
print(f"\n✓ Summary plots saved to: {summary_plot_filename}")
plt.show()

# %%
# Create comprehensive visualization
fig = plt.figure(figsize=(18, 10))

# Plot 1: Rate scaling factors vs original theta
ax1 = plt.subplot(2, 3, 1)
ax1.plot(
    original_theta_values,
    [results_dict[theta]['rate_scaling_right_opt'] for theta in original_theta_values],
    marker='o',
    linewidth=2,
    markersize=10,
    color='#2E86AB',
    label='Right'
)
ax1.plot(
    original_theta_values,
    [results_dict[theta]['rate_scaling_left_opt'] for theta in original_theta_values],
    marker='s',
    linewidth=2,
    markersize=10,
    color='#A23B72',
    label='Left'
)
ax1.set_xlabel('Original Theta', fontsize=12, fontweight='bold')
ax1.set_ylabel('Optimal Rate Scaling Factor', fontsize=12, fontweight='bold')
ax1.set_title('Rate Scaling Factor vs Original Theta', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(original_theta_values)
ax1.legend(fontsize=10)

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
plt.savefig(f'bads_rate_bound_analysis_left_right_seperate_{timestamp}.png', dpi=150, bbox_inches='tight')
print(f"\nComprehensive analysis plot saved to: bads_rate_bound_analysis_left_right_seperate_{timestamp}.png")
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
# ============================================================================
# OPTIMIZED SIMULATION FUNCTIONS
# ============================================================================

def compute_quantiles_from_rt_data(rt_data, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Compute quantiles from RT data, filtering out NaN values.
    """
    valid_rts = rt_data[~np.isnan(rt_data)]
    return np.quantile(valid_rts, quantiles)


def simulate_ddm_for_stimulus(ABL, ILD, theta_ddm, n_trials=N_TRIALS_VALIDATE, show_progress=True):
    """
    VECTORIZED DDM simulation - all trials updated simultaneously.
    ~5x faster than original joblib parallel approach.
    
    Returns:
        quantiles: Array of [q10, q30, q50, q70, q90]
        accuracy: Proportion of correct (right) choices
        rts: Raw RT data (for distribution plotting)
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
    
    # DDM parameters
    mu = N * (r_right - r_left)  # Drift rate
    sigma = np.sqrt(N * (r_right + r_left))  # Noise standard deviation
    dt_sim = dt  # Use global dt
    dB = np.sqrt(dt_sim)
    T_max = 5  # Max simulation time
    max_steps = int(T_max / dt_sim)
    
    # State arrays - vectorized across all trials
    position = np.zeros(n_trials)
    rts = np.full(n_trials, np.nan)
    choices = np.zeros(n_trials)
    active = np.ones(n_trials, dtype=bool)
    
    # Optional progress bar for the time steps
    step_iterator = range(max_steps)
    if show_progress:
        step_iterator = tqdm(step_iterator, desc=f'  DDM ABL={ABL}, ILD={ILD}', leave=False)
    
    for step in step_iterator:
        n_active = active.sum()
        if n_active == 0:
            break
        
        # Update all active trials at once (vectorized)
        position[active] += mu * dt_sim + sigma * np.random.normal(0, dB, size=n_active)
        time_now = (step + 1) * dt_sim
        
        # Check thresholds (vectorized)
        hit_upper = active & (position >= theta_ddm)
        hit_lower = active & (position <= -theta_ddm)
        
        rts[hit_upper] = time_now
        choices[hit_upper] = 1
        rts[hit_lower] = time_now
        choices[hit_lower] = -1
        
        active[hit_upper | hit_lower] = False
    
    # Compute accuracy (proportion of correct choices, assuming right is correct)
    valid_choices = choices[~np.isnan(rts)]
    if len(valid_choices) > 0:
        accuracy = np.sum(valid_choices == 1) / len(valid_choices)
    else:
        accuracy = 0.5
    
    # Compute quantiles
    quantiles = compute_quantiles_from_rt_data(rts, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])
    
    return quantiles, accuracy, rts


def simulate_poisson_for_stimulus(ABL, ILD, Nr0_scaled_right, Nr0_scaled_left, theta_poisson, n_trials=N_TRIALS_VALIDATE, show_progress=True):
    """
    OPTIMIZED Poisson simulation - no pandas, uses numpy argsort instead.
    ~12x faster than original approach.
    
    Returns:
        quantiles: Array of [q10, q30, q50, q70, q90]
        accuracy: Proportion of correct (right) choices
        rts: Raw RT data (for distribution plotting)
    """
    # Calculate rates with separate left/right scaling
    r0_right = Nr0_scaled_right / N
    r0_left = Nr0_scaled_left / N
    r_db = (2*ABL + ILD)/2
    l_db = (2*ABL - ILD)/2
    pr = (10 ** (r_db/20))
    pl = (10 ** (l_db/20))
    
    den = (pr ** (lam * l)) + (pl ** (lam * l))
    rr = (pr ** lam) / den
    rl = (pl ** lam) / den
    r_right = r0_right * rr
    r_left = r0_left * rl
    
    T = 50  # Max time for spike generation
    
    # Storage for results
    rts = np.full(n_trials, np.nan)
    choices = np.zeros(n_trials)
    
    # Run trials
    trial_iterator = range(n_trials)
    if show_progress:
        trial_iterator = tqdm(trial_iterator, desc=f'  Poisson ABL={ABL}, ILD={ILD}', leave=False)
    
    for trial in trial_iterator:
        rng = np.random.default_rng()
        
        # Generate spikes - optimized for low correlation (rho ≈ 0)
        if rho < 0.01:
            # Essentially independent - faster generation
            right_spikes = []
            left_spikes = []
            for i in range(N):
                n_r = rng.poisson(r_right * T)
                n_l = rng.poisson(r_left * T)
                right_spikes.append(rng.random(n_r) * T)
                left_spikes.append(rng.random(n_l) * T)
            all_right = np.concatenate(right_spikes) if right_spikes else np.array([])
            all_left = np.concatenate(left_spikes) if left_spikes else np.array([])
        else:
            # Use correlated spike generation
            right_pool = generate_correlated_pool(N, rho, r_right, T, rng)
            left_pool = generate_correlated_pool(N, rho, r_left, T, rng)
            all_right = np.concatenate(list(right_pool.values()))
            all_left = np.concatenate(list(left_pool.values()))
        
        if len(all_right) == 0 and len(all_left) == 0:
            continue
        
        # Combine and sort (NO PANDAS - this is the key optimization)
        all_times = np.concatenate([all_right, all_left])
        all_evidence = np.concatenate([np.ones(len(all_right)), -np.ones(len(all_left))])
        
        # Sort by time
        sort_idx = np.argsort(all_times)
        sorted_times = all_times[sort_idx]
        sorted_evidence = all_evidence[sort_idx]
        
        # Cumsum and find first crossing
        dv = np.cumsum(sorted_evidence)
        
        pos_cross = np.where(dv >= theta_poisson)[0]
        neg_cross = np.where(dv <= -theta_poisson)[0]
        
        first_pos = pos_cross[0] if len(pos_cross) > 0 else np.inf
        first_neg = neg_cross[0] if len(neg_cross) > 0 else np.inf
        
        if first_pos < first_neg:
            rts[trial] = sorted_times[int(first_pos)]
            choices[trial] = 1
        elif first_neg < first_pos:
            rts[trial] = sorted_times[int(first_neg)]
            choices[trial] = -1
    
    # Compute accuracy
    valid_choices = choices[~np.isnan(rts)]
    if len(valid_choices) > 0:
        accuracy = np.sum(valid_choices == 1) / len(valid_choices)
    else:
        accuracy = 0.5
    
    # Compute quantiles
    quantiles = compute_quantiles_from_rt_data(rts, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])
    
    return quantiles, accuracy, rts


# %%
def simulate_validation_data(original_theta):
    """
    Generate validation data by simulating DDM and Poisson for all stimuli.
    This is the expensive operation - separate from plotting.
    
    Returns:
        dict with keys: ddm_quantiles_dict, ddm_acc_dict, ddm_rts_dict,
                        poisson_quantiles_dict, poisson_acc_dict, poisson_rts_dict,
                        rate_scaling_right_opt, rate_scaling_left_opt,
                        theta_poisson_opt, original_theta
    """
    print(f"\n{'='*70}")
    print(f"GENERATING VALIDATION DATA FOR ORIGINAL THETA = {original_theta}")
    print(f"{'='*70}")
    
    result = results_dict[original_theta]
    rate_scaling_right_opt = result['rate_scaling_right_opt']
    rate_scaling_left_opt = result['rate_scaling_left_opt']
    theta_increment_opt = result['theta_increment_opt']
    theta_poisson_opt = result['theta_poisson_opt']
    
    print(f"\nOptimized parameters:")
    print(f"  Rate scaling right: {rate_scaling_right_opt:.4f}")
    print(f"  Rate scaling left: {rate_scaling_left_opt:.4f}")
    print(f"  Theta increment: {theta_increment_opt}")
    print(f"  DDM theta: {original_theta}")
    print(f"  Poisson theta: {theta_poisson_opt}")
    
    Nr0_scaled_right = Nr0_base * rate_scaling_right_opt
    Nr0_scaled_left = Nr0_base * rate_scaling_left_opt
    
    # Storage for results
    ddm_quantiles_dict = {}
    ddm_acc_dict = {}
    ddm_rts_dict = {}  # Store raw RTs for distribution plotting
    poisson_quantiles_dict = {}
    poisson_acc_dict = {}
    poisson_rts_dict = {}  # Store raw RTs for distribution plotting
    
    # Simulate for all stimuli
    total_stimuli = len(VALIDATION_ABL_RANGE) * len(VALIDATION_ILD_RANGE)
    print(f"\nSimulating DDM and Poisson for {total_stimuli} stimuli...")
    print(f"  {N_TRIALS_VALIDATE} trials per stimulus")
    print(f"  Using FAST vectorized DDM and optimized Poisson (no pandas)")
    
    # Create list of all stimuli for progress tracking
    stimuli_list = [(ABL, ILD) for ABL in VALIDATION_ABL_RANGE for ILD in VALIDATION_ILD_RANGE]
    
    for stim_idx, (ABL, ILD) in enumerate(tqdm(stimuli_list, desc="Overall progress"), 1):
        print(f"\nStimulus {stim_idx}/{total_stimuli}: ABL={ABL}, ILD={ILD}")
        
        # Simulate DDM (VECTORIZED - fast)
        print(f'DDM data for ABL={ABL}, ILD={ILD}')
        ddm_q, ddm_acc, ddm_rts = simulate_ddm_for_stimulus(ABL, ILD, original_theta, n_trials=N_TRIALS_VALIDATE, show_progress=False)
        ddm_quantiles_dict[(ABL, ILD)] = ddm_q
        ddm_acc_dict[(ABL, ILD)] = ddm_acc
        ddm_rts_dict[(ABL, ILD)] = ddm_rts
        
        # Simulate Poisson (OPTIMIZED - no pandas)
        print(f'Poisson data for ABL={ABL}, ILD={ILD}')
        poisson_q, poisson_acc, poisson_rts = simulate_poisson_for_stimulus(
            ABL,
            ILD,
            Nr0_scaled_right,
            Nr0_scaled_left,
            theta_poisson_opt,
            n_trials=N_TRIALS_VALIDATE,
            show_progress=False
        )
        poisson_quantiles_dict[(ABL, ILD)] = poisson_q
        poisson_acc_dict[(ABL, ILD)] = poisson_acc
        poisson_rts_dict[(ABL, ILD)] = poisson_rts
    
    print("\n✓ All simulations complete")
    
    return {
        'ddm_quantiles_dict': ddm_quantiles_dict,
        'ddm_acc_dict': ddm_acc_dict,
        'ddm_rts_dict': ddm_rts_dict,
        'poisson_quantiles_dict': poisson_quantiles_dict,
        'poisson_acc_dict': poisson_acc_dict,
        'poisson_rts_dict': poisson_rts_dict,
        'rate_scaling_right_opt': rate_scaling_right_opt,
        'rate_scaling_left_opt': rate_scaling_left_opt,
        'theta_poisson_opt': theta_poisson_opt,
        'original_theta': original_theta
    }



# %%
# Run validation analysis for a selected theta
selected_theta = original_theta_values[0]
print(f"\nRunning validation analysis for original_theta = {selected_theta}...")

# Step 1: Generate data (expensive - run once)
validation_data = simulate_validation_data(selected_theta)
#%%
# plot_quantiles, psychometric
ddm_quantiles_dict = validation_data['ddm_quantiles_dict']
ddm_acc_dict = validation_data['ddm_acc_dict']
poisson_quantiles_dict = validation_data['poisson_quantiles_dict']
poisson_acc_dict = validation_data['poisson_acc_dict']
rate_scaling_right_opt = validation_data['rate_scaling_right_opt']
rate_scaling_left_opt = validation_data['rate_scaling_left_opt']
theta_poisson_opt = validation_data['theta_poisson_opt']
original_theta = validation_data['original_theta']

# Plotting style configuration
ABL_color_map = {20: '#2E86AB', 40: '#A23B72', 60: '#F18F01'}
quantile_labels = ['Q10', 'Q30', 'Q50', 'Q70', 'Q90']

# ===== QUANTILE PLOTS (1x3) =====
fig, axes = plt.subplots(1, 3, figsize=(18, 10))
# NOTE: just to test effect of delay. 
# NDT = 0*1e-3
NDT_ABL_map = {20: 20e-3, 40: 15e-3, 60: 10e-3}
# NDT_ILD_map = {1: 0, 2: 5e-3, 4: 8e-3, 8: 10e-3, 16: 6e-3}


for ax_idx, ABL in enumerate(VALIDATION_ABL_RANGE):
    ax = axes[ax_idx]
    
    for q_idx in range(5):
        # DDM quantiles
        ddm_q_values = [ddm_quantiles_dict[(ABL, ILD)][q_idx] for ILD in VALIDATION_ILD_RANGE]
        # Poisson quantiles
        poisson_q_values = [poisson_quantiles_dict[(ABL, ILD)][q_idx] for ILD in VALIDATION_ILD_RANGE]
        # Plot with different line styles for different quantiles
        alpha_val = 0.5 + (q_idx * 0.1)
        ax.plot(VALIDATION_ILD_RANGE, np.array(ddm_q_values), marker='o', markersize=6, 
                label=f'{quantile_labels[q_idx]}', color=ABL_color_map[ABL], 
                alpha=alpha_val, linewidth=1.5)
        ax.plot(VALIDATION_ILD_RANGE, np.array(poisson_q_values) + NDT_ABL_map[ABL], marker='x', markersize=8, 
                linestyle='--', color=ABL_color_map[ABL], alpha=alpha_val, linewidth=1.5)
    
    ax.set_xlabel('ILD', fontsize=12, fontweight='bold')
    ax.set_ylabel('RT (s)', fontsize=12, fontweight='bold')
    ax.set_title(f'ABL = {ABL}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    # ax.legend(fontsize=9, loc='best')
    # ax.set_xscale('log')
    ax.set_xticks(VALIDATION_ILD_RANGE)
    ax.set_xticklabels(VALIDATION_ILD_RANGE)

fig.suptitle(
    f'RT Quantiles: θ_DDM={original_theta}, θ_Poisson={theta_poisson_opt}, '
    f'RateR×{rate_scaling_right_opt:.2f}, RateL×{rate_scaling_left_opt:.2f}\n(dots=DDM, x=Poisson). NDT_Poisson={NDT_ABL_map.values()}', 
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

fig.suptitle(
    f'Psychometric Functions: θ_DDM={original_theta}, θ_Poisson={theta_poisson_opt}, '
    f'RateR×{rate_scaling_right_opt:.2f}, RateL×{rate_scaling_left_opt:.2f}\n(dots=DDM, x=Poisson)', 
    fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'validation_psychometric_theta_{original_theta}_{timestamp}.png', dpi=150, bbox_inches='tight')
print(f"✓ Psychometric plot saved: validation_psychometric_theta_{original_theta}_{timestamp}.png")
plt.show()


# %%
# def plot_rt_distributions(validation_data, timestamp):
"""
Plot RT distributions for all ABL×ILD combinations (3×5 grid).
Shows both DDM and Poisson distributions overlaid.

Parameters:
    validation_data: dict returned by simulate_validation_data()
    timestamp: timestamp string for filename
"""
# Extract data
# NDT = 20*1e-3
NDT_ABL_map = {20: 5e-3, 40: 5e-3, 60: 5e-3}
# NDT_ILD_map = {1: 0, 2: 5e-3, 4: 8e-3, 8: 10e-3, 16: 6e-3}


ddm_rts_dict = validation_data['ddm_rts_dict']
poisson_rts_dict = validation_data['poisson_rts_dict']
ddm_acc_dict = validation_data['ddm_acc_dict']
poisson_acc_dict = validation_data['poisson_acc_dict']
rate_scaling_right_opt = validation_data['rate_scaling_right_opt']
rate_scaling_left_opt = validation_data['rate_scaling_left_opt']
theta_poisson_opt = validation_data['theta_poisson_opt']
original_theta = validation_data['original_theta']

# Create 3×5 subplot grid
fig, axes = plt.subplots(3, 5, figsize=(25, 16))

for row_idx, ABL in enumerate(VALIDATION_ABL_RANGE):
    for col_idx, ILD in enumerate(VALIDATION_ILD_RANGE):
        ax = axes[row_idx, col_idx]
        
        # Get RT data for this stimulus
        ddm_rts = ddm_rts_dict[(ABL, ILD)]
        poisson_rts = poisson_rts_dict[(ABL, ILD)]
        
        # Filter out NaNs
        ddm_rts_valid = ddm_rts[~np.isnan(ddm_rts)]
        poisson_rts_valid = poisson_rts[~np.isnan(poisson_rts)]

        # Add delay to poisson rts
        # poisson_rts_valid += NDT_ILD_map[ILD]
        poisson_rts_valid += NDT_ABL_map[ABL]

        
        # Determine bin edges (use same bins for both)
        # max_rt = max(np.max(ddm_rts_valid) if len(ddm_rts_valid) > 0 else 1,
        #             np.max(poisson_rts_valid) if len(poisson_rts_valid) > 0 else 1)
        # bins = np.linspace(0, min(max_rt, 2), 50)  # Cap at 2s for visibility
        bins = np.arange(0,2,0.005)
        # Plot histograms
        ax.hist(ddm_rts_valid, bins=bins, label='DDM', 
                color='blue', density=True, histtype='step')
        ax.hist(poisson_rts_valid, bins=bins, label='Poisson', 
                color='red', density=True, histtype='step')
        
        # Add title with stimulus info and accuracy
        ddm_acc = ddm_acc_dict[(ABL, ILD)]
        poisson_acc = poisson_acc_dict[(ABL, ILD)]
        ax.set_title(f'ABL={ABL}, ILD={ILD}\nDDM acc={ddm_acc:.3f}, Poisson acc={poisson_acc:.3f}',
                    fontsize=9)
        
        # Labels only on edges
        if col_idx == 0:
            ax.set_ylabel('Density', fontsize=9)
        if row_idx == 2:
            ax.set_xlabel('RT (s)', fontsize=9)
        
        # Legend only on first subplot
        if row_idx == 0 and col_idx == 0:
            ax.legend(fontsize=8, loc='upper right')
        
        # ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=8)
        ax.set_xlim(0,0.8)
        ax.set_ylim(0,17)
fig.suptitle(
    f'RT Distributions: θ_DDM={original_theta}, θ_Poisson={theta_poisson_opt}, '
    f'RateR×{rate_scaling_right_opt:.2f}, RateL×{rate_scaling_left_opt:.2f}\n'
    f'{len(ddm_rts_valid)} trials per stimulus. DELAY_poisson={NDT_ABL_map.values()}',
    fontsize=14,
    fontweight='bold'
)
plt.tight_layout()
plt.savefig(f'validation_rt_distributions_theta_{original_theta}_{timestamp}.png', dpi=150, bbox_inches='tight')
print(f"\n✓ RT distributions plot saved: validation_rt_distributions_theta_{original_theta}_{timestamp}.png")
plt.show()


# %%
# Plot RT distributions (3×5 grid)
# plot_rt_distributions(validation_data, timestamp)
# %%
# Does adding NDT improve MSE
NDT_arr = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])*1e-3
mse_err = np.zeros(len(NDT_arr))
for n_idx, NDT in enumerate(NDT_arr):
    print(f'NDT = {NDT}')
    for row_idx, ABL in enumerate(VALIDATION_ABL_RANGE):
        for col_idx, ILD in enumerate(VALIDATION_ILD_RANGE):
            
            
            # Get RT data for this stimulus
            ddm_rts = ddm_rts_dict[(ABL, ILD)]
            poisson_rts = poisson_rts_dict[(ABL, ILD)]
        
            # Filter out NaNs
            ddm_rts_valid = ddm_rts[~np.isnan(ddm_rts)]
            poisson_rts_valid = poisson_rts[~np.isnan(poisson_rts)]

            # Add delay to poisson rts
            poisson_rts_valid += NDT
        
            bins = np.arange(0,2,0.01)

            poisson_hist, _ = np.histogram(poisson_rts_valid, bins=bins, density=True)
            ddm_hist, _ = np.histogram(ddm_rts_valid, bins=bins, density=True)

            mse_err[n_idx] += np.sum((poisson_hist - ddm_hist)**2)        
# %%
mse_err
# %%
plt.plot(1000*NDT_arr, mse_err/1000,'-o')
plt.xlabel('NDT (ms)')
plt.ylabel('Sq. Err')
plt.title('SE vs NDT')
plt.show()

# %%
# check why RTD and quantiles mis-match
ABL_test = 20
ILD_test = 16
delay_test = NDT_ABL_map[ABL_test]

ddm_rts = ddm_rts_dict[(ABL_test, ILD_test)]
poisson_rts = np.array(poisson_rts_dict[(ABL_test, ILD_test)]) + delay_test

bins = np.arange(0,2,0.005)

quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
ddm_q = np.quantile(ddm_rts, quantiles)
poisson_q = np.quantile(poisson_rts, quantiles)

plt.figure(figsize=(20,8))
plt.hist(ddm_rts, bins=bins,density=True,histtype='step',label='ddm', color='b')
plt.hist(poisson_rts, bins=bins, density=True, histtype='step', label='poisson', color='r')
for i in range(5):
    plt.axvline(ddm_q[i], color='b', alpha=0.5)
    plt.axvline(poisson_q[i], color='r', alpha=0.5)
plt.legend()
plt.xlim(0,1)
plt.title(f'ABL={ABL_test}, ILD = {ILD_test}')
plt.xlabel('rts')


        
