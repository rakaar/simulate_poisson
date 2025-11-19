# %%
"""
BADS Optimization to Find Optimal Rate Scaling Factor and Bound Increment

For each original_theta value (2, 3, 4, 5, 6, 7), this script uses BADS optimization
to find the best:
- rate_scaling_factor: How much to scale up firing rates (range: 1x to 10x)
- theta_increment: How much to increase the bound (range: +1 to +20, integer values)

The objective is to minimize squared error between DDM and Poisson predictions
for both:
- RT quantiles (10th, 50th, 90th percentiles) - computed from simulations
- Psychometric (accuracy) data

This version uses trial-level simulations to compute RT quantiles instead of
analytical mean RT formulas.
"""

# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from pybads import BADS
from mgf_helper_utils import poisson_fc_dt, ddm_fc_dt
from corr_poisson_utils_subtractive import run_poisson_trial

# %%
# Fixed parameters (shared across all optimizations)
lam = 1.3
l = 0.9
Nr0_base = 13.3  # Base Nr0 before multiplication
dt = 1e-6  # Time step for continuous DDM simulation
rho = 2e-3
N = 500  # Fixed number of neurons

# Stimuli to test
ABL_range = [20, 60]
ILD_range = [2,4]

# Number of trials for bound offset estimation
N_sim_bound_offset = int(10e3)

# Number of trials for DDM quantile estimation
N_sim_ddm_quantiles = int(10e3)

# Number of trials for Poisson quantile estimation in objective function
N_sim_poisson_quantiles = int(10e3)

# Original theta values to iterate over
original_theta_values = [2]

print("="*70)
print("BADS OPTIMIZATION: RATE SCALING AND BOUND INCREMENT")
print("="*70)
print(f"\nFixed parameters:")
print(f"  lambda: {lam}")
print(f"  ell: {l}")
print(f"  Nr0_base: {Nr0_base}")
print(f"  N: {N}")
print(f"  rho: {rho}")
print(f"  dt: {dt}")
print(f"\nStimuli:")
print(f"  ABL_range: {ABL_range}")
print(f"  ILD_range: {ILD_range}")
print(f"  Total stimuli: {len(ABL_range) * len(ILD_range)}")
print(f"\nOriginal theta values to optimize: {original_theta_values}")
print(f"Bound offset estimation trials: {N_sim_bound_offset}")
print(f"DDM quantile estimation trials: {N_sim_ddm_quantiles}")
print(f"Poisson quantile estimation trials: {N_sim_poisson_quantiles}")

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


def compute_quantiles_from_rt_data(rt_data, quantiles=[0.1, 0.5, 0.9]):
    """
    Compute quantiles from RT data, filtering out NaN values.
    
    Parameters:
        rt_data: Array of reaction times (may contain NaN)
        quantiles: List of quantiles to compute (default: 10th, 50th, 90th percentiles)
    
    Returns:
        Array of quantile values
    """
    # Filter out NaN values and choice==0 trials
    valid_rts = rt_data[~np.isnan(rt_data)]
    
    if len(valid_rts) == 0:
        return np.array([np.nan] * len(quantiles))
    
    return np.quantile(valid_rts, quantiles)


def compute_ddm_predictions(original_theta, n_trials=N_sim_ddm_quantiles):
    """
    Compute DDM predictions (accuracy and RT quantiles) for all stimuli with given original_theta.
    Uses simulations to compute 10th, 50th, and 90th percentile RTs.
    
    Parameters:
        original_theta: DDM threshold parameter
        n_trials: Number of trials to simulate per stimulus
    
    Returns:
        ddm_quantiles_data: dict with keys (ABL, ILD) and values = array of [q10, q50, q90] RT quantiles
        ddm_acc_data: dict with keys (ABL, ILD) and values = accuracy
    """
    ddm_quantiles_data = {}
    ddm_acc_data = {}
    
    for ABL in ABL_range:
        for ILD in ILD_range:
            # Calculate rates for this stimulus
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
            
            # Run DDM simulations in parallel
            ddm_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
                delayed(run_ddm_trial_single)(i, N, r_right, r_left, original_theta, dt) 
                for i in range(n_trials)
            )
            
            ddm_array = np.array(ddm_data)
            rts = ddm_array[:, 0]
            choices = ddm_array[:, 1]
            
            # Compute accuracy (proportion of correct choices, assuming right is correct)
            valid_choices = choices[~np.isnan(rts)]
            if len(valid_choices) > 0:
                ddm_acc = np.sum(valid_choices == 1) / len(valid_choices)
            else:
                ddm_acc = 0.5  # Default if no decisions
            
            # Compute quantiles
            quantiles = compute_quantiles_from_rt_data(rts, quantiles=[0.1, 0.5, 0.9])
            
            ddm_quantiles_data[(ABL, ILD)] = quantiles
            ddm_acc_data[(ABL, ILD)] = ddm_acc
    
    return ddm_quantiles_data, ddm_acc_data


def run_poisson_simulations_get_all(Nr0_scaled, theta_poisson, ABL, ILD, n_trials=N_sim_poisson_quantiles):
    """
    Run Poisson simulations once and compute bound offset, RT quantiles, and accuracy.
    
    Parameters:
        Nr0_scaled: Scaled firing rate
        theta_poisson: Poisson threshold
        ABL: Average binaural level
        ILD: Interaural level difference
        n_trials: Number of trials to simulate
    
    Returns:
        bound_offset_mean: Mean bound offset
        quantiles: Array of [q10, q50, q90] RT quantiles
        accuracy: Proportion of correct choices
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
    
    # Run simulations once
    poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
        delayed(run_poisson_trial)(N, rho, r_right, r_left, theta_poisson) 
        for _ in range(n_trials)
    )
    
    poisson_array = np.array(poisson_data)
    rts = poisson_array[:, 0]
    choices = poisson_array[:, 1]
    bound_offsets = poisson_array[:, 2]
    
    # Compute bound offset mean
    valid_offsets = bound_offsets[~np.isnan(bound_offsets)]
    if len(valid_offsets) == 0:
        bound_offset_mean = 0.0  # Default if no valid offsets
    else:
        bound_offset_mean = np.mean(valid_offsets)
    
    # Compute accuracy (proportion of correct choices, assuming right is correct)
    valid_choices = choices[~np.isnan(rts)]
    if len(valid_choices) > 0:
        accuracy = np.sum(valid_choices == 1) / len(valid_choices)
    else:
        accuracy = 0.5  # Default if no decisions
    
    # Compute quantiles
    quantiles = compute_quantiles_from_rt_data(rts, quantiles=[0.1, 0.5, 0.9])
    
    return bound_offset_mean, quantiles, accuracy


def objective_function_rate_bound(x, original_theta, ddm_quantiles_data, ddm_acc_data, verbose=False):
    """
    Objective function for BADS optimization using RT quantiles.
    
    Parameters:
        x: [rate_scaling_factor, theta_increment]
        original_theta: The original DDM theta value
        ddm_quantiles_data: dict of DDM RT quantiles [q10, q50, q90] for each stimulus
        ddm_acc_data: dict of DDM accuracies for each stimulus
        verbose: Print details if True
    
    Returns:
        total_squared_error: Sum of squared errors for quantiles and accuracy across all stimuli
    """
    rate_scaling_factor = x[0]
    theta_increment_raw = x[1]
    
    # Theta increment must be an integer (round to nearest integer)
    theta_increment = int(np.round(theta_increment_raw))
    
    # Ensure theta_increment is at least 1
    if theta_increment < 1:
        theta_increment = 1
    
    # Calculate scaled Nr0 and poisson theta
    Nr0_scaled = Nr0_base * rate_scaling_factor
    theta_poisson = original_theta + theta_increment
    
    if verbose:
        print(f"\n  rate_scaling_factor: {rate_scaling_factor:.4f}")
        print(f"  theta_increment (rounded): {theta_increment}")
        print(f"  theta_poisson: {theta_poisson}")
    
    poisson_quantiles_data = {}
    poisson_acc_data = {}
    
    # Compute Poisson predictions for all stimuli
    for ABL in ABL_range:
        for ILD in ILD_range:
            # Run simulations once to get quantiles and accuracy
            # Note: bound offset not needed since we're using simulations, not analytical formulas
            _, poisson_quantiles, poisson_acc = run_poisson_simulations_get_all(
                Nr0_scaled, theta_poisson, ABL, ILD, n_trials=N_sim_poisson_quantiles
            )
            
            poisson_quantiles_data[(ABL, ILD)] = poisson_quantiles
            poisson_acc_data[(ABL, ILD)] = poisson_acc
    
    # Calculate squared error for quantiles
    # Sum of squared errors across all 3 quantiles (10th, 50th, 90th) for all stimuli
    squared_error_quantiles = 0.0
    for ABL in ABL_range:
        for ILD in ILD_range:
            ddm_q = ddm_quantiles_data[(ABL, ILD)]
            poisson_q = poisson_quantiles_data[(ABL, ILD)]
            
            # Sum squared errors for all 3 quantiles
            for i in range(3):
                if not (np.isnan(ddm_q[i]) or np.isnan(poisson_q[i])):
                    squared_error_quantiles += (poisson_q[i] - ddm_q[i])**2
    
    # Calculate squared error for accuracy
    squared_error_acc = sum(
        (poisson_acc_data[(ABL, ILD)] - ddm_acc_data[(ABL, ILD)])**2 
        for ABL in ABL_range for ILD in ILD_range
    )
    
    total_squared_error = squared_error_quantiles + squared_error_acc
    
    if verbose:
        print(f"  Squared Error (Quantiles): {squared_error_quantiles:.6f}")
        print(f"  Squared Error (Acc): {squared_error_acc:.6f}")
        print(f"  Total Squared Error: {total_squared_error:.6f}")
    
    return total_squared_error


# %%
# Main optimization loop over original_theta values
results_dict = {}

# Create timestamped output files for progressive saving
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
text_log_filename = f'bads_optimization_log_{timestamp}.txt'
pickle_filename_base = f'bads_rate_bound_optimization_results_{timestamp}'

# Initialize text log file
with open(text_log_filename, 'w') as log_file:
    log_file.write("="*70 + "\n")
    log_file.write("BADS OPTIMIZATION: RATE SCALING AND BOUND INCREMENT\n")
    log_file.write("="*70 + "\n")
    log_file.write(f"\nTimestamp: {timestamp}\n")
    log_file.write(f"Original theta values: {original_theta_values}\n")
    log_file.write(f"\nFixed parameters:\n")
    log_file.write(f"  lambda: {lam}\n")
    log_file.write(f"  ell: {l}\n")
    log_file.write(f"  Nr0_base: {Nr0_base}\n")
    log_file.write(f"  N: {N}\n")
    log_file.write(f"  rho: {rho}\n")
    log_file.write(f"  dt: {dt}\n")
    log_file.write(f"  ABL_range: {ABL_range}\n")
    log_file.write(f"  ILD_range: {ILD_range}\n")
    log_file.write("="*70 + "\n\n")

print(f"\nProgress will be saved to: {text_log_filename}")
print(f"Intermediate results will be saved as: {pickle_filename_base}_theta_X.pkl\n")

for original_theta in original_theta_values:
    print("\n" + "="*70)
    print(f"OPTIMIZING FOR ORIGINAL THETA = {original_theta}")
    print("="*70)
    
    # Compute DDM predictions once for this original_theta
    print(f"\nComputing DDM predictions for original_theta = {original_theta}...")
    print(f"Running {N_sim_ddm_quantiles} DDM trials per stimulus to compute quantiles...")
    ddm_quantiles_data, ddm_acc_data = compute_ddm_predictions(original_theta)
    print(f"DDM predictions computed for {len(ddm_quantiles_data)} stimuli")
    
    # BADS optimization setup
    # Parameters: [rate_scaling_factor, theta_increment]
    
    # Hard bounds (actual optimization constraints)
    lower_bounds = np.array([1.0, 1.0])  # rate: [1, 10], theta_inc: [1, 20]
    upper_bounds = np.array([3.0, 7.0])
    
    # Plausible bounds (where we expect the solution to be)
    plausible_lower_bounds = np.array([1.1, 2.0])
    plausible_upper_bounds = np.array([2, 5.0])
    
    # Initial guess (midpoint of plausible range)
    x0 = (plausible_lower_bounds + plausible_upper_bounds) / 2
    
    print(f"\nBADS setup:")
    print(f"  Parameter structure: [rate_scaling_factor, theta_increment]")
    print(f"  Lower bounds: {lower_bounds}")
    print(f"  Upper bounds: {upper_bounds}")
    print(f"  Plausible lower bounds: {plausible_lower_bounds}")
    print(f"  Plausible upper bounds: {plausible_upper_bounds}")
    print(f"  Initial guess: {x0}")
    
    # Define objective function wrapper for this original_theta
    def obj_func(x):
        return objective_function_rate_bound(x, original_theta, ddm_quantiles_data, ddm_acc_data, verbose=False)
    
    # BADS options
    options = {
        'uncertainty_handling': True,  # Using stochastic simulations to compute quantiles
    }
    
    # Initialize BADS
    print(f"\nInitializing BADS optimizer...")
    start_time = time.time()
    
    bads = BADS(obj_func, x0, lower_bounds, upper_bounds, 
                plausible_lower_bounds, plausible_upper_bounds, options=options)
    
    # Run optimization
    print("Starting BADS optimization...")
    optimize_result = bads.optimize()
    
    end_time = time.time()
    optimization_time = end_time - start_time
    
    # Extract optimized parameters
    x_opt = optimize_result['x']
    rate_scaling_factor_opt = x_opt[0]
    theta_increment_opt = int(np.round(x_opt[1]))  # Round to integer
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE FOR ORIGINAL THETA = {original_theta}")
    print(f"{'='*70}")
    print(f"Optimization time: {optimization_time:.2f} seconds")
    print(f"\nOptimized parameters:")
    print(f"  rate_scaling_factor: {rate_scaling_factor_opt:.6f}")
    print(f"  theta_increment: {theta_increment_opt}")
    print(f"  theta_poisson: {original_theta + theta_increment_opt}")
    print(f"\nOptimization statistics:")
    print(f"  Final objective value: {optimize_result['fval']:.6f}")
    print(f"  Function evaluations: {optimize_result['func_count']}")
    print(f"  Success: {optimize_result['success']}")
    
    # Store results for this original_theta
    # Note: We don't save the full 'bads_result' because it contains unpicklable function references
    results_dict[original_theta] = {
        'original_theta': original_theta,
        'rate_scaling_factor_opt': rate_scaling_factor_opt,
        'theta_increment_opt': theta_increment_opt,
        'theta_poisson_opt': original_theta + theta_increment_opt,
        'x_opt': x_opt,
        'final_objective_value': optimize_result['fval'],
        'func_count': optimize_result['func_count'],
        'success': optimize_result['success'],
        'optimization_time': optimization_time,
        'ddm_quantiles_data': ddm_quantiles_data,
        'ddm_acc_data': ddm_acc_data,
    }
    
    # IMMEDIATE SAVE #1: Append results to text log file
    with open(text_log_filename, 'a') as log_file:
        log_file.write(f"\n{'='*70}\n")
        log_file.write(f"RESULTS FOR ORIGINAL THETA = {original_theta}\n")
        log_file.write(f"{'='*70}\n")
        log_file.write(f"Optimization time: {optimization_time:.2f} seconds ({optimization_time/60:.2f} minutes)\n")
        log_file.write(f"\nOptimized parameters:\n")
        log_file.write(f"  rate_scaling_factor: {rate_scaling_factor_opt:.8f}\n")
        log_file.write(f"  theta_increment: {theta_increment_opt}\n")
        log_file.write(f"  theta_poisson: {original_theta + theta_increment_opt}\n")
        log_file.write(f"\nOptimization statistics:\n")
        log_file.write(f"  Final objective value (MSE): {optimize_result['fval']:.8f}\n")
        log_file.write(f"  Function evaluations: {optimize_result['func_count']}\n")
        log_file.write(f"  Success: {optimize_result['success']}\n")
        log_file.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'='*70}\n")
    
    print(f"✓ Results appended to text log: {text_log_filename}")
    
    # IMMEDIATE SAVE #2: Save intermediate pickle file for this theta
    intermediate_pickle = f"{pickle_filename_base}_theta_{original_theta}.pkl"
    intermediate_save_dict = {
        'original_theta': original_theta,
        'result': results_dict[original_theta],
        'fixed_params': {
            'lam': lam,
            'l': l,
            'Nr0_base': Nr0_base,
            'N': N,
            'rho': rho,
            'dt': dt,
            'ABL_range': ABL_range,
            'ILD_range': ILD_range,
            'N_sim_bound_offset': N_sim_bound_offset,
        },
        'timestamp': timestamp,
    }
    
    try:
        with open(intermediate_pickle, 'wb') as f:
            pickle.dump(intermediate_save_dict, f)
        print(f"✓ Intermediate results saved to: {intermediate_pickle}")
    except Exception as e:
        print(f"⚠ Warning: Could not save intermediate pickle: {e}")
        print(f"  (Results are still safe in text log: {text_log_filename})")

# %%
# Save final combined pickle file with all results
final_pickle_filename = f'{pickle_filename_base}_all.pkl'

save_dict = {
    'results_dict': results_dict,
    'original_theta_values': original_theta_values,
    'fixed_params': {
        'lam': lam,
        'l': l,
        'Nr0_base': Nr0_base,
        'N': N,
        'rho': rho,
        'dt': dt,
        'ABL_range': ABL_range,
        'ILD_range': ILD_range,
        'N_sim_bound_offset': N_sim_bound_offset,
    },
    'timestamp': timestamp,
}

print(f"\n{'='*70}")
print("ALL OPTIMIZATIONS COMPLETE")
print(f"{'='*70}")

# Try to save the final combined pickle file
try:
    with open(final_pickle_filename, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"✓ Final combined results saved to: {final_pickle_filename}")
except Exception as e:
    print(f"⚠ Warning: Could not save final combined pickle file: {e}")
    print(f"  Individual results are still available in:")
    print(f"  - Text log: {text_log_filename}")
    print(f"  - Intermediate pickles: {pickle_filename_base}_theta_X.pkl")

# %%
# Print and save summary table
print(f"\n{'='*70}")
print("SUMMARY TABLE")
print(f"{'='*70}")
print(f"\n{'Original Theta':<15} {'Rate Scaling':<15} {'Theta Inc':<12} {'Theta Poisson':<15} {'Final MSE':<12}")
print("-" * 70)

# Also write summary to text log
with open(text_log_filename, 'a') as log_file:
    log_file.write(f"\n\n{'='*70}\n")
    log_file.write("FINAL SUMMARY TABLE\n")
    log_file.write(f"{'='*70}\n")
    log_file.write(f"\n{'Original Theta':<15} {'Rate Scaling':<15} {'Theta Inc':<12} {'Theta Poisson':<15} {'Final MSE':<12}\n")
    log_file.write("-" * 70 + "\n")

for original_theta in original_theta_values:
    result = results_dict[original_theta]
    summary_line = (f"{original_theta:<15} "
                   f"{result['rate_scaling_factor_opt']:<15.4f} "
                   f"{result['theta_increment_opt']:<12} "
                   f"{result['theta_poisson_opt']:<15} "
                   f"{result['final_objective_value']:<12.6f}")
    print(summary_line)
    
    # Also write to text log
    with open(text_log_filename, 'a') as log_file:
        log_file.write(summary_line + "\n")

# Final timestamp in text log
with open(text_log_filename, 'a') as log_file:
    log_file.write("-" * 70 + "\n")
    log_file.write(f"\nAll optimizations completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"\nResults saved in:\n")
    log_file.write(f"  - Text log: {text_log_filename}\n")
    log_file.write(f"  - Intermediate pickles: {pickle_filename_base}_theta_X.pkl\n")
    log_file.write(f"  - Final combined pickle: {final_pickle_filename}\n")
    log_file.write(f"\n{'='*70}\n")

print(f"\n✓ Summary appended to text log: {text_log_filename}")

# %%
# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Rate scaling factor vs original theta
axes[0].plot(original_theta_values, 
             [results_dict[theta]['rate_scaling_factor_opt'] for theta in original_theta_values],
             marker='o', linewidth=2, markersize=8)
axes[0].set_xlabel('Original Theta', fontsize=12)
axes[0].set_ylabel('Optimal Rate Scaling Factor', fontsize=12)
axes[0].set_title('Rate Scaling Factor vs Original Theta', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Plot 2: Theta increment vs original theta
axes[1].plot(original_theta_values, 
             [results_dict[theta]['theta_increment_opt'] for theta in original_theta_values],
             marker='s', linewidth=2, markersize=8, color='orange')
axes[1].set_xlabel('Original Theta', fontsize=12)
axes[1].set_ylabel('Optimal Theta Increment', fontsize=12)
axes[1].set_title('Theta Increment vs Original Theta', fontsize=12)
axes[1].grid(True, alpha=0.3)

# Plot 3: Final MSE vs original theta
axes[2].plot(original_theta_values, 
             [results_dict[theta]['final_objective_value'] for theta in original_theta_values],
             marker='^', linewidth=2, markersize=8, color='green')
axes[2].set_xlabel('Original Theta', fontsize=12)
axes[2].set_ylabel('Final MSE', fontsize=12)
axes[2].set_title('Final MSE vs Original Theta', fontsize=12)
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = f'bads_optimization_summary_{timestamp}.png'
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
print(f"\n✓ Summary plot saved to: {plot_filename}")
plt.show()

# Final message
print(f"\n{'='*70}")
print("ALL FILES SAVED SUCCESSFULLY")
print(f"{'='*70}")
print(f"Text log:        {text_log_filename}")
print(f"Combined pickle: {final_pickle_filename}")
print(f"Summary plot:    {plot_filename}")
print(f"Intermediate:    {pickle_filename_base}_theta_X.pkl (one per theta)")
print(f"{'='*70}\n")

# %%
