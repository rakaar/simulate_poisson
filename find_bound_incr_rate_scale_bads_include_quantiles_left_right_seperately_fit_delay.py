# %%
"""
BADS Optimization to Find Optimal Right/Left Rate Scaling Factors, Bound Increment, and Poisson Delay

For each original_theta value (2, 3, 4, 5, 6, 7), this script uses BADS optimization
to find the best:
- rate_scaling_right: How much to scale up right firing rate (shared bounds with left; default hard bounds 0.01-3)
- rate_scaling_left: How much to scale up left firing rate (shared bounds with right; default hard bounds 0.01-3)
- theta_increment: How much to increase the bound (hard bounds 0–7, cast to integer)
- poisson_delay: Delay to add to Poisson RTs (hard bounds 0-30ms, plausible 5-15ms)

The objective is to minimize squared error between DDM and Poisson predictions
for both:
- RT quantiles (10th, 30th, 50th, 70th, 90th percentiles) - computed from simulations
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
rho = 1e-3
N = 100  # Fixed number of neurons

# Stimuli to test
ABL_range = [20,40, 60]
ILD_range = [1,2,4]

# Number of trials for bound offset estimation
N_sim_bound_offset = int(10e3)

# Number of trials for DDM quantile estimation
N_sim_ddm_quantiles = int(10e3)

# Number of trials for Poisson quantile estimation in objective function
N_sim_poisson_quantiles = int(10e3)

# Original theta values to iterate over
original_theta_values = [2]

print("="*70)
print("BADS OPTIMIZATION: RATE SCALING, BOUND INCREMENT, AND POISSON DELAY")
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


def compute_quantiles_from_rt_data(rt_data, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Compute quantiles from RT data, filtering out NaN values.
    """
    valid_rts = rt_data[~np.isnan(rt_data)]
    if len(valid_rts) == 0:
        return np.array([np.nan] * len(quantiles))
    return np.quantile(valid_rts, quantiles)


def compute_ddm_predictions(original_theta, n_trials=N_sim_ddm_quantiles):
    """
    Compute DDM predictions (accuracy and RT quantiles) for all stimuli with given original_theta.
    """
    ddm_quantiles_data = {}
    ddm_acc_data = {}
    
    for ABL in ABL_range:
        for ILD in ILD_range:
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
            
            ddm_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
                delayed(run_ddm_trial_single)(i, N, r_right, r_left, original_theta, dt) 
                for i in range(n_trials)
            )
            
            ddm_array = np.array(ddm_data)
            rts = ddm_array[:, 0]
            choices = ddm_array[:, 1]
            
            valid_choices = choices[~np.isnan(rts)]
            if len(valid_choices) > 0:
                ddm_acc = np.sum(valid_choices == 1) / len(valid_choices)
            else:
                ddm_acc = 0.5
            
            quantiles = compute_quantiles_from_rt_data(rts, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])
            ddm_quantiles_data[(ABL, ILD)] = quantiles
            ddm_acc_data[(ABL, ILD)] = ddm_acc
    
    return ddm_quantiles_data, ddm_acc_data


def run_poisson_simulations_get_all(Nr0_scaled_right, Nr0_scaled_left, theta_poisson, ABL, ILD, poisson_delay, n_trials=N_sim_poisson_quantiles):
    """
    Run Poisson simulations once and compute bound offset, RT quantiles, and accuracy.
    
    Parameters:
        Nr0_scaled_right: Scaled firing rate for right channel
        Nr0_scaled_left: Scaled firing rate for left channel
        theta_poisson: Poisson threshold
        ABL: Average binaural level
        ILD: Interaural level difference
        poisson_delay: Delay to add to Poisson RTs (in seconds)
        n_trials: Number of trials to simulate
    """
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
    
    poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
        delayed(run_poisson_trial)(N, rho, r_right, r_left, theta_poisson) 
        for _ in range(n_trials)
    )
    
    poisson_array = np.array(poisson_data)
    rts = poisson_array[:, 0]
    
    # Add fitted poisson_delay to all valid Poisson reaction times
    rts = np.where(~np.isnan(rts), rts + poisson_delay, rts)
    
    choices = poisson_array[:, 1]
    bound_offsets = poisson_array[:, 2]
    
    valid_offsets = bound_offsets[~np.isnan(bound_offsets)]
    bound_offset_mean = np.mean(valid_offsets) if len(valid_offsets) > 0 else 0.0
    
    valid_choices = choices[~np.isnan(rts)]
    accuracy = np.sum(valid_choices == 1) / len(valid_choices) if len(valid_choices) > 0 else 0.5
    
    quantiles = compute_quantiles_from_rt_data(rts, quantiles=[0.1, 0.3, 0.5, 0.7, 0.9])
    
    return bound_offset_mean, quantiles, accuracy


def objective_function_rate_bound(x, original_theta, ddm_quantiles_data, ddm_acc_data, verbose=False):
    """
    Objective function for BADS optimization using RT quantiles.
    
    Parameters:
        x: [rate_scaling_right, rate_scaling_left, theta_increment, poisson_delay]
    """
    rate_scaling_right = x[0]
    rate_scaling_left = x[1]
    theta_increment_raw = x[2]
    poisson_delay = x[3]  # Already in seconds
    
    theta_increment = int(np.round(theta_increment_raw))
    
    Nr0_scaled_right = Nr0_base * rate_scaling_right
    Nr0_scaled_left = Nr0_base * rate_scaling_left
    theta_poisson = original_theta + theta_increment
    
    if verbose:
        print(f"\n  rate_scaling_right: {rate_scaling_right:.4f}")
        print(f"  rate_scaling_left: {rate_scaling_left:.4f}")
        print(f"  theta_increment (rounded): {theta_increment}")
        print(f"  theta_poisson: {theta_poisson}")
        print(f"  poisson_delay: {poisson_delay*1000:.2f} ms")
    
    poisson_quantiles_data = {}
    poisson_acc_data = {}
    
    for ABL in ABL_range:
        for ILD in ILD_range:
            _, poisson_quantiles, poisson_acc = run_poisson_simulations_get_all(
                Nr0_scaled_right, Nr0_scaled_left, theta_poisson, ABL, ILD, poisson_delay, n_trials=N_sim_poisson_quantiles
            )
            poisson_quantiles_data[(ABL, ILD)] = poisson_quantiles
            poisson_acc_data[(ABL, ILD)] = poisson_acc
    
    squared_error_quantiles = 0.0
    for ABL in ABL_range:
        for ILD in ILD_range:
            ddm_q = ddm_quantiles_data[(ABL, ILD)]
            poisson_q = poisson_quantiles_data[(ABL, ILD)]
            for i in range(5):
                if not (np.isnan(ddm_q[i]) or np.isnan(poisson_q[i])):
                    squared_error_quantiles += (poisson_q[i] - ddm_q[i])**2
    
    squared_error_acc = sum(
        (poisson_acc_data[(ABL, ILD)] - ddm_acc_data[(ABL, ILD)])**2 
        for ABL in ABL_range for ILD in ILD_range
    )
    w_acc_error = 5
    total_squared_error = squared_error_quantiles + w_acc_error * squared_error_acc
    
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
text_log_filename = f'bads_optimization_log_left_right_seperate_fit_delay_{timestamp}.txt'
pickle_filename_base = f'bads_rate_bound_optimization_results_left_right_seperate_fit_delay_{timestamp}'

# Initialize text log file
with open(text_log_filename, 'w') as log_file:
    log_file.write("="*70 + "\n")
    log_file.write("BADS OPTIMIZATION: RATE SCALING, BOUND INCREMENT, AND POISSON DELAY\n")
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
    # Parameters: [rate_scaling_right, rate_scaling_left, theta_increment, poisson_delay]
    # Note: poisson_delay is in seconds (0ms=0.0, 30ms=0.030)
    
    # Hard bounds: poisson_delay 0ms to 30ms (0.0 to 0.030 seconds)
    lower_bounds = np.array([0.01, 0.01, 0.0, 0.0])
    upper_bounds = np.array([3.0, 3.0, 7.0, 0.030])
    
    # Plausible bounds: poisson_delay 5ms to 15ms (0.005 to 0.015 seconds)
    plausible_lower_bounds = np.array([1.1, 1.1, 2.0, 0.005])
    plausible_upper_bounds = np.array([2.0, 2.0, 5.0, 0.015])
    
    # Initial guess (midpoint of plausible range)
    x0 = (plausible_lower_bounds + plausible_upper_bounds) / 2
    
    print(f"\nBADS setup:")
    print(f"  Parameter structure: [rate_scaling_right, rate_scaling_left, theta_increment, poisson_delay]")
    print(f"  Lower bounds: {lower_bounds}")
    print(f"  Upper bounds: {upper_bounds}")
    print(f"  Plausible lower bounds: {plausible_lower_bounds}")
    print(f"  Plausible upper bounds: {plausible_upper_bounds}")
    print(f"  Initial guess: {x0}")
    print(f"  Initial poisson_delay: {x0[3]*1000:.2f} ms")
    
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
    rate_scaling_right_opt = x_opt[0]
    rate_scaling_left_opt = x_opt[1]
    theta_increment_opt = int(np.round(x_opt[2]))
    poisson_delay_opt = x_opt[3]  # In seconds
    
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE FOR ORIGINAL THETA = {original_theta}")
    print(f"{'='*70}")
    print(f"Optimization time: {optimization_time:.2f} seconds")
    print(f"\nOptimized parameters:")
    print(f"  rate_scaling_right: {rate_scaling_right_opt:.6f}")
    print(f"  rate_scaling_left: {rate_scaling_left_opt:.6f}")
    print(f"  theta_increment: {theta_increment_opt}")
    print(f"  theta_poisson: {original_theta + theta_increment_opt}")
    print(f"  poisson_delay: {poisson_delay_opt*1000:.4f} ms ({poisson_delay_opt:.6f} s)")
    print(f"\nOptimization statistics:")
    print(f"  Final objective value: {optimize_result['fval']:.6f}")
    print(f"  Function evaluations: {optimize_result['func_count']}")
    print(f"  Success: {optimize_result['success']}")
    
    # Store results
    results_dict[original_theta] = {
        'original_theta': original_theta,
        'rate_scaling_right_opt': rate_scaling_right_opt,
        'rate_scaling_left_opt': rate_scaling_left_opt,
        'theta_increment_opt': theta_increment_opt,
        'theta_poisson_opt': original_theta + theta_increment_opt,
        'poisson_delay_opt': poisson_delay_opt,
        'poisson_delay_opt_ms': poisson_delay_opt * 1000,
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
        log_file.write(f"  rate_scaling_right: {rate_scaling_right_opt:.8f}\n")
        log_file.write(f"  rate_scaling_left: {rate_scaling_left_opt:.8f}\n")
        log_file.write(f"  theta_increment: {theta_increment_opt}\n")
        log_file.write(f"  theta_poisson: {original_theta + theta_increment_opt}\n")
        log_file.write(f"  poisson_delay: {poisson_delay_opt*1000:.4f} ms ({poisson_delay_opt:.8f} s)\n")
        log_file.write(f"\nOptimization statistics:\n")
        log_file.write(f"  Final objective value (MSE): {optimize_result['fval']:.8f}\n")
        log_file.write(f"  Function evaluations: {optimize_result['func_count']}\n")
        log_file.write(f"  Success: {optimize_result['success']}\n")
        log_file.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'='*70}\n")
    
    print(f"Results appended to text log: {text_log_filename}")
    
    # IMMEDIATE SAVE #2: Save intermediate pickle file for this theta
    intermediate_pickle = f"{pickle_filename_base}_theta_{original_theta}.pkl"
    intermediate_save_dict = {
        'original_theta': original_theta,
        'result': results_dict[original_theta],
        'fixed_params': {
            'lam': lam, 'l': l, 'Nr0_base': Nr0_base, 'N': N, 'rho': rho, 'dt': dt,
            'ABL_range': ABL_range, 'ILD_range': ILD_range, 'N_sim_bound_offset': N_sim_bound_offset,
        },
        'timestamp': timestamp,
    }
    
    try:
        with open(intermediate_pickle, 'wb') as f:
            pickle.dump(intermediate_save_dict, f)
        print(f"Intermediate results saved to: {intermediate_pickle}")
    except Exception as e:
        print(f"Warning: Could not save intermediate pickle: {e}")

# %%
# Save final combined pickle file with all results
final_pickle_filename = f'{pickle_filename_base}_all.pkl'

save_dict = {
    'results_dict': results_dict,
    'original_theta_values': original_theta_values,
    'fixed_params': {
        'lam': lam, 'l': l, 'Nr0_base': Nr0_base, 'N': N, 'rho': rho, 'dt': dt,
        'ABL_range': ABL_range, 'ILD_range': ILD_range, 'N_sim_bound_offset': N_sim_bound_offset,
    },
    'timestamp': timestamp,
}

print(f"\n{'='*70}")
print("ALL OPTIMIZATIONS COMPLETE")
print(f"{'='*70}")

try:
    with open(final_pickle_filename, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Final combined results saved to: {final_pickle_filename}")
except Exception as e:
    print(f"Warning: Could not save final combined pickle file: {e}")

# %%
# Print and save summary table
print(f"\n{'='*70}")
print("SUMMARY TABLE")
print(f"{'='*70}")
print(f"\n{'Orig Theta':<12} {'Rate R':<10} {'Rate L':<10} {'Th Inc':<8} {'Th P':<6} {'Delay(ms)':<12} {'MSE':<12}")
print("-" * 75)

with open(text_log_filename, 'a') as log_file:
    log_file.write(f"\n\n{'='*70}\n")
    log_file.write("FINAL SUMMARY TABLE\n")
    log_file.write(f"{'='*70}\n")
    log_file.write(f"\n{'Orig Theta':<12} {'Rate R':<10} {'Rate L':<10} {'Th Inc':<8} {'Th P':<6} {'Delay(ms)':<12} {'MSE':<12}\n")
    log_file.write("-" * 75 + "\n")

for original_theta in original_theta_values:
    result = results_dict[original_theta]
    summary_line = (f"{original_theta:<12} "
                   f"{result['rate_scaling_right_opt']:<10.4f} "
                   f"{result['rate_scaling_left_opt']:<10.4f} "
                   f"{result['theta_increment_opt']:<8} "
                   f"{result['theta_poisson_opt']:<6} "
                   f"{result['poisson_delay_opt_ms']:<12.4f} "
                   f"{result['final_objective_value']:<12.6f}")
    print(summary_line)
    with open(text_log_filename, 'a') as log_file:
        log_file.write(summary_line + "\n")

with open(text_log_filename, 'a') as log_file:
    log_file.write("-" * 75 + "\n")
    log_file.write(f"\nAll optimizations completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"\nResults saved in:\n")
    log_file.write(f"  - Text log: {text_log_filename}\n")
    log_file.write(f"  - Intermediate pickles: {pickle_filename_base}_theta_X.pkl\n")
    log_file.write(f"  - Final combined pickle: {final_pickle_filename}\n")
    log_file.write(f"\n{'='*70}\n")

print(f"\nSummary appended to text log: {text_log_filename}")

# %%
# Plot results (4 subplots now)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Rate scaling factors vs original theta
axes[0, 0].plot(original_theta_values,
    [results_dict[theta]['rate_scaling_right_opt'] for theta in original_theta_values],
    marker='o', linewidth=2, markersize=8, label='Right')
axes[0, 0].plot(original_theta_values,
    [results_dict[theta]['rate_scaling_left_opt'] for theta in original_theta_values],
    marker='o', linewidth=2, markersize=8, label='Left')
axes[0, 0].set_xlabel('Original Theta', fontsize=12)
axes[0, 0].set_ylabel('Optimal Rate Scaling Factor', fontsize=12)
axes[0, 0].set_title('Rate Scaling Factors vs Original Theta', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(frameon=False)

# Plot 2: Theta increment vs original theta
axes[0, 1].plot(original_theta_values, 
             [results_dict[theta]['theta_increment_opt'] for theta in original_theta_values],
             marker='s', linewidth=2, markersize=8, color='orange')
axes[0, 1].set_xlabel('Original Theta', fontsize=12)
axes[0, 1].set_ylabel('Optimal Theta Increment', fontsize=12)
axes[0, 1].set_title('Theta Increment vs Original Theta', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Poisson delay vs original theta
axes[1, 0].plot(original_theta_values, 
             [results_dict[theta]['poisson_delay_opt_ms'] for theta in original_theta_values],
             marker='d', linewidth=2, markersize=8, color='purple')
axes[1, 0].set_xlabel('Original Theta', fontsize=12)
axes[1, 0].set_ylabel('Optimal Poisson Delay (ms)', fontsize=12)
axes[1, 0].set_title('Poisson Delay vs Original Theta', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='10ms reference')
axes[1, 0].legend(frameon=False)

# Plot 4: Final MSE vs original theta
axes[1, 1].plot(original_theta_values, 
             [results_dict[theta]['final_objective_value'] for theta in original_theta_values],
             marker='^', linewidth=2, markersize=8, color='green')
axes[1, 1].set_xlabel('Original Theta', fontsize=12)
axes[1, 1].set_ylabel('Final MSE', fontsize=12)
axes[1, 1].set_title('Final MSE vs Original Theta', fontsize=12)
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = f'bads_optimization_summary_left_right_seperate_fit_delay_{timestamp}.png'
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
print(f"\nSummary plot saved to: {plot_filename}")
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
