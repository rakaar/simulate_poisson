# %%
"""
BADS Optimization to Find Optimal Rate Scaling Factor and Bound Increment

For each original_theta value (2, 3, 4, 5, 6, 7), this script uses BADS optimization
to find the best:
- rate_scaling_factor: How much to scale up firing rates (range: 1x to 10x)
- theta_increment: How much to increase the bound (range: +1 to +20, integer values)

The objective is to minimize squared error between DDM and Poisson predictions
for both chronometric (RT) and psychometric (accuracy) data across multiple stimuli.
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
rho = 1e-2
N = 100  # Fixed number of neurons

# Stimuli to test
ABL_range = [20, 40, 60]
ILD_range = [1, 2, 4, 8, 16]

# Number of trials for bound offset estimation
N_sim_bound_offset = int(10e3)

# Original theta values to iterate over
original_theta_values = [2,3,4,5]

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

# %%
def compute_ddm_predictions(original_theta):
    """
    Compute DDM predictions (accuracy and RT) for all stimuli with given original_theta.
    
    Returns:
        ddm_rt_data: dict with keys (ABL, ILD) and values = mean RT
        ddm_acc_data: dict with keys (ABL, ILD) and values = accuracy
    """
    ddm_rt_data = {}
    ddm_acc_data = {}
    
    for ABL in ABL_range:
        for ILD in ILD_range:
            ddm_acc, ddm_mean_rt = ddm_fc_dt(lam, l, Nr0_base, N, ABL, ILD, original_theta, dt)
            ddm_rt_data[(ABL, ILD)] = ddm_mean_rt
            ddm_acc_data[(ABL, ILD)] = ddm_acc
    
    return ddm_rt_data, ddm_acc_data


def estimate_bound_offset(Nr0_scaled, theta_poisson, ABL, ILD, n_trials=N_sim_bound_offset):
    """
    Estimate the mean bound offset for given scaled rates and theta by running simulations.
    
    Returns:
        bound_offset_mean: float
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
    
    # Run simulations to get bound offset
    poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(
        delayed(run_poisson_trial)(N, rho, r_right, r_left, theta_poisson) 
        for _ in range(n_trials)
    )
    
    bound_offsets = np.array([data[2] for data in poisson_data])
    # Filter out NaN values
    bound_offsets = bound_offsets[~np.isnan(bound_offsets)]
    
    if len(bound_offsets) == 0:
        return 0.0  # Default if no valid offsets
    
    return np.mean(bound_offsets)


def objective_function_rate_bound(x, original_theta, ddm_rt_data, ddm_acc_data, verbose=False):
    """
    Objective function for BADS optimization.
    
    Parameters:
        x: [rate_scaling_factor, theta_increment]
        original_theta: The original DDM theta value
        ddm_rt_data: dict of DDM RTs for each stimulus
        ddm_acc_data: dict of DDM accuracies for each stimulus
        verbose: Print details if True
    
    Returns:
        total_squared_error: Sum of squared errors for RT and accuracy across all stimuli
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
    
    poisson_rt_data = {}
    poisson_acc_data = {}
    
    # Compute Poisson predictions for all stimuli
    for ABL in ABL_range:
        for ILD in ILD_range:
            # Estimate bound offset for this stimulus
            bound_offset_mean = estimate_bound_offset(Nr0_scaled, theta_poisson, ABL, ILD)
            
            # Effective theta including bound offset
            theta_poisson_eff = theta_poisson + bound_offset_mean
            
            # Get Poisson predictions using analytical formulas
            poisson_acc, poisson_mean_rt = poisson_fc_dt(
                N, rho, theta_poisson_eff, lam, l, Nr0_scaled, ABL, ILD, dt
            )
            
            poisson_rt_data[(ABL, ILD)] = poisson_mean_rt
            poisson_acc_data[(ABL, ILD)] = poisson_acc
    
    # Calculate total squared error
    squared_error_rt = sum(
        (poisson_rt_data[(ABL, ILD)] - ddm_rt_data[(ABL, ILD)])**2 
        for ABL in ABL_range for ILD in ILD_range
    )
    
    squared_error_acc = sum(
        (poisson_acc_data[(ABL, ILD)] - ddm_acc_data[(ABL, ILD)])**2 
        for ABL in ABL_range for ILD in ILD_range
    )
    
    total_squared_error = squared_error_rt + squared_error_acc
    
    if verbose:
        print(f"  Squared Error (RT): {squared_error_rt:.6f}")
        print(f"  Squared Error (Acc): {squared_error_acc:.6f}")
        print(f"  Total Squared Error: {total_squared_error:.6f}")
    
    return total_squared_error


# %%
# Main optimization loop over original_theta values
results_dict = {}

for original_theta in original_theta_values:
    print("\n" + "="*70)
    print(f"OPTIMIZING FOR ORIGINAL THETA = {original_theta}")
    print("="*70)
    
    # Compute DDM predictions once for this original_theta
    print(f"\nComputing DDM predictions for original_theta = {original_theta}...")
    ddm_rt_data, ddm_acc_data = compute_ddm_predictions(original_theta)
    print(f"DDM predictions computed for {len(ddm_rt_data)} stimuli")
    
    # BADS optimization setup
    # Parameters: [rate_scaling_factor, theta_increment]
    
    # Hard bounds (actual optimization constraints)
    lower_bounds = np.array([1.0, 1.0])  # rate: [1, 10], theta_inc: [1, 20]
    upper_bounds = np.array([10.0, 20.0])
    
    # Plausible bounds (where we expect the solution to be)
    plausible_lower_bounds = np.array([1.2, 1.0])
    plausible_upper_bounds = np.array([5.0, 10.0])
    
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
        return objective_function_rate_bound(x, original_theta, ddm_rt_data, ddm_acc_data, verbose=False)
    
    # BADS options
    options = {
        'uncertainty_handling': False,  # Objective is deterministic (using analytical formulas)
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
        'bads_result': optimize_result,
        'ddm_rt_data': ddm_rt_data,
        'ddm_acc_data': ddm_acc_data,
    }

# %%
# Save results to pickle file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f'bads_rate_bound_optimization_results_{timestamp}.pkl'

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

with open(output_filename, 'wb') as f:
    pickle.dump(save_dict, f)

print(f"\n{'='*70}")
print("ALL OPTIMIZATIONS COMPLETE")
print(f"{'='*70}")
print(f"\nResults saved to: {output_filename}")

# %%
# Print summary table
print(f"\n{'='*70}")
print("SUMMARY TABLE")
print(f"{'='*70}")
print(f"\n{'Original Theta':<15} {'Rate Scaling':<15} {'Theta Inc':<12} {'Theta Poisson':<15} {'Final MSE':<12}")
print("-" * 70)

for original_theta in original_theta_values:
    result = results_dict[original_theta]
    print(f"{original_theta:<15} "
          f"{result['rate_scaling_factor_opt']:<15.4f} "
          f"{result['theta_increment_opt']:<12} "
          f"{result['theta_poisson_opt']:<15} "
          f"{result['final_objective_value']:<12.6f}")

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
plt.savefig(f'bads_optimization_summary_{timestamp}.png', dpi=150, bbox_inches='tight')
print(f"\nSummary plot saved to: bads_optimization_summary_{timestamp}.png")
plt.show()

# %%
