# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import ks_2samp
from pybads import BADS
from bads_utils import (
    lr_rates_from_ABL_ILD,
    simulate_single_ddm_trial,
    simulate_poisson_rts,
    objective_function_multistim
)

# %%
# Define DDM parameters (shared across all stimuli)
ddm_params = {
    'Nr0': 13.3,
    'lam': 1.3,
    'ell': 0.9,
    'theta': 2
}

# Define 4 stimuli
stimuli = [
    {'ABL': 20, 'ILD': 2},
    {'ABL': 20, 'ILD': 4},
    {'ABL': 20, 'ILD': 8},
    {'ABL': 20, 'ILD': 16},


    {'ABL': 60, 'ILD': 2},
    {'ABL': 60, 'ILD': 4},
    {'ABL': 60, 'ILD': 8},
    {'ABL': 60, 'ILD': 16},
]

# DDM simulation settings
dt = 1e-4
dB = 1e-2
N_sim_ddm = int(50e3)  # 50K trials per stimulus
T = 20
n_steps = int(T/dt)

print("="*70)
print("MULTI-STIMULUS DDM SIMULATION")
print("="*70)
print(f"\nDDM Parameters:")
print(f"  Nr0: {ddm_params['Nr0']}")
print(f"  lambda: {ddm_params['lam']}")
print(f"  ell: {ddm_params['ell']}")
print(f"  theta: {ddm_params['theta']}")
print(f"\nSimulating {N_sim_ddm} trials for each of {len(stimuli)} stimuli...")

# %%
# Simulate DDM data for all stimuli
ddm_data_dict = {}
ddm_rts_decided_dict = {}

for stim in stimuli:
    ABL = stim['ABL']
    ILD = stim['ILD']
    stim_key = f"ABL_{ABL}_ILD_{ILD}"
    
    print(f"\n{'='*70}")
    print(f"STIMULUS: ABL={ABL}, ILD={ILD}")
    print(f"{'='*70}")
    
    # Calculate rates
    ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
        ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
    )
    
    print(f"  DDM right rate: {ddm_right_rate:.4f}")
    print(f"  DDM left rate:  {ddm_left_rate:.4f}")
    
    # Calculate drift and diffusion
    mu_ddm = ddm_right_rate - ddm_left_rate
    sigma_sq_ddm = ddm_right_rate + ddm_left_rate
    sigma_ddm = np.sqrt(sigma_sq_ddm)
    
    print(f"  mu (drift):     {mu_ddm:.4f}")
    print(f"  sigma (diff):   {sigma_ddm:.4f}")
    
    # Run DDM simulation
    start_time_ddm = time.time()
    ddm_results = Parallel(n_jobs=-1)(
        delayed(simulate_single_ddm_trial)(i, mu_ddm, sigma_ddm, ddm_params['theta'], dt, dB, n_steps) 
        for i in tqdm(range(N_sim_ddm), desc=f'Simulating DDM {stim_key}')
    )
    ddm_data = np.array(ddm_results)
    end_time_ddm = time.time()
    
    # Process results
    ddm_decided_mask = ~np.isnan(ddm_data[:, 0])
    ddm_rts_decided = ddm_data[ddm_decided_mask, 0]
    
    # Store in dictionaries
    ddm_data_dict[stim_key] = {
        'ABL': ABL,
        'ILD': ILD,
        'mu': mu_ddm,
        'sigma': sigma_ddm,
        'right_rate': ddm_right_rate,
        'left_rate': ddm_left_rate,
        'ddm_data': ddm_data,
        'ddm_rts_decided': ddm_rts_decided
    }
    ddm_rts_decided_dict[stim_key] = ddm_rts_decided
    
    print(f"  Simulation time: {end_time_ddm - start_time_ddm:.2f} seconds")
    print(f"  Total trials: {len(ddm_data)}")
    print(f"  Decided trials: {len(ddm_rts_decided)}")
    print(f"  Mean RT: {np.mean(ddm_rts_decided):.4f} s")
    print(f"  Median RT: {np.median(ddm_rts_decided):.4f} s")

print(f"\n{'='*70}")
print("DDM SIMULATION COMPLETE")
print(f"{'='*70}\n")

# %%
# BADS optimization setup
n_trials_per_eval = int(10e3)  # 10K trials per stimulus per evaluation
seed = 42

print("="*70)
print("BADS OPTIMIZATION SETUP")
print("="*70)
print(f"\nOptimizing parameters to minimize MSE of mean RT and P(right) across all stimuli")
print(f"Objective: sum over stimuli of [(mean_RT_DDM - mean_RT_Poisson)^2 + (P_right_DDM - P_right_Poisson)^2]")
print(f"Poisson trials per evaluation per stimulus: {n_trials_per_eval}")
print(f"Total Poisson trials per evaluation: {n_trials_per_eval * len(stimuli)}\n")

# Parameter structure: [N, k, theta, rate_scaling_factor]
# - N, k, theta are shared across all stimuli
# - rate_scaling_factor scales the DDM rates to get Poisson rates
#   poisson_right_rate = ddm_right_rate * rate_scaling_factor
#   poisson_left_rate = ddm_left_rate * rate_scaling_factor

# ==== Bound set 1, where minimum N = 10 ====
# Hard bounds (actual optimization constraints)
# [N, k, theta, rate_scaling_factor]
# lower_bounds = np.array([10, 1, 2, 1])
# upper_bounds = np.array([1000, 10, 20, 10])

# # Plausible bounds (where we expect the solution to be)
# plausible_lower_bounds = np.array([50, 2, 3, 2])
# plausible_upper_bounds = np.array([500, 9, 15, 5])

# ====== Bound set 2, where minimu N = 5 =====
# # Hard bounds (actual optimization constraints)
# [N, k, theta, rate_scaling_factor]
lower_bounds = np.array([5, 1, 2, 1])
upper_bounds = np.array([1000, 5, 20, 10])

# # Plausible bounds (where we expect the solution to be)
plausible_lower_bounds = np.array([50, 2, 3, 2])
plausible_upper_bounds = np.array([500, 4, 15, 5])
 

# Initial guess (midpoint of plausible range)
x0 = (plausible_lower_bounds + plausible_upper_bounds) / 2

print("Parameter structure:")
print("  [N, k, theta, rate_scaling_factor]")
print("\nShared parameters:")
print(f"  N:                   [{lower_bounds[0]:.0f}, {upper_bounds[0]:.0f}] "
      f"(plausible: [{plausible_lower_bounds[0]:.0f}, {plausible_upper_bounds[0]:.0f}])")
print(f"  k:                   [{lower_bounds[1]:.0f}, {upper_bounds[1]:.0f}] "
      f"(plausible: [{plausible_lower_bounds[1]:.0f}, {plausible_upper_bounds[1]:.0f}])")
print(f"  theta:               [{lower_bounds[2]:.1f}, {upper_bounds[2]:.1f}] "
      f"(plausible: [{plausible_lower_bounds[2]:.1f}, {plausible_upper_bounds[2]:.1f}])")
print(f"  rate_scaling_factor: [{lower_bounds[3]:.1f}, {upper_bounds[3]:.1f}] "
      f"(plausible: [{plausible_lower_bounds[3]:.1f}, {plausible_upper_bounds[3]:.1f}])")

print(f"\nInitial guess:")
print(f"  N={x0[0]:.0f}, k={x0[1]:.1f}, theta={x0[2]:.2f}, rate_scaling_factor={x0[3]:.2f}")

print(f"\nFor each stimulus, Poisson rates will be:")
for stim in stimuli:
    stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
    ddm_data = ddm_data_dict[stim_key]
    print(f"  {stim_key}:")
    print(f"    DDM right rate: {ddm_data['right_rate']:.4f}")
    print(f"    DDM left rate:  {ddm_data['left_rate']:.4f}")
    print(f"    Poisson right rate = {ddm_data['right_rate']:.4f} × rate_scaling_factor")
    print(f"    Poisson left rate  = {ddm_data['left_rate']:.4f} × rate_scaling_factor")


# Define new objective function with rate_scaling_factor
def obj_func_with_scaling(x, ddm_rts_decided_dict, ddm_data_dict, n_trials, seed):
    """
    Objective function for BADS optimization with rate_scaling_factor.
    Uses MSE of mean RT and P(right) instead of KS-statistic.
    
    Parameters:
    x: [N, k, theta, rate_scaling_factor]
    ddm_rts_decided_dict: Dictionary of DDM RTs for each stimulus
    ddm_data_dict: Dictionary with DDM data including rates for each stimulus
    n_trials: Number of Poisson trials to simulate per stimulus
    seed: Random seed
    
    Returns:
    Total MSE (sum across all stimuli): 
        sum over stimuli of [(mean_RT_DDM - mean_RT_Poisson)^2 + (P_right_DDM - P_right_Poisson)^2]
    """
    N = int(np.round(x[0]))
    k = x[1]
    c = k / N
    theta = x[2]
    rate_scaling_factor = x[3]
    
    total_mse = 0.0
    
    # Iterate over all stimuli
    for stim_key in ddm_rts_decided_dict.keys():
        # Get DDM data for this stimulus
        ddm_data = ddm_data_dict[stim_key]
        ddm_full_data = ddm_data['ddm_data']  # Full data with [RT, choice]
        
        # DDM metrics
        ddm_decided_mask = ~np.isnan(ddm_full_data[:, 0])
        ddm_rts = ddm_full_data[ddm_decided_mask, 0]
        ddm_choices = ddm_full_data[ddm_decided_mask, 1]
        
        mean_rt_ddm = np.mean(ddm_rts)
        p_right_ddm = np.mean(ddm_choices == 1)  # P(choice == 1)
        
        # Get DDM rates and calculate Poisson rates using scaling factor
        ddm_right_rate = ddm_data['right_rate']
        ddm_left_rate = ddm_data['left_rate']
        
        poisson_right_rate = ddm_right_rate * rate_scaling_factor
        poisson_left_rate = ddm_left_rate * rate_scaling_factor
        
        # Set up Poisson model parameters
        poisson_params = {
            'N_right': N,
            'N_left': N,
            'c': c,
            'r_right': poisson_right_rate,
            'r_left': poisson_left_rate,
            'theta': theta,
            'T': 20,
            'exponential_noise_scale': 0
        }
        
        # Simulate Poisson model
        poisson_results = simulate_poisson_rts(poisson_params, n_trials=n_trials, 
                                                seed=seed, verbose=False)
        
        # Filter for decided trials
        poisson_decided_mask = ~np.isnan(poisson_results[:, 0])
        poisson_rts = poisson_results[poisson_decided_mask, 0]
        poisson_choices = poisson_results[poisson_decided_mask, 1]
        
        # Compute metrics for Poisson model
        if len(poisson_rts) > 0:
            mean_rt_poisson = np.mean(poisson_rts)
            p_right_poisson = np.mean(poisson_choices == 1)
            
            # Compute MSE components
            rt_mse = (mean_rt_ddm - mean_rt_poisson) ** 2
            choice_mse = (p_right_ddm - p_right_poisson) ** 2
            
            total_mse += rt_mse + choice_mse
        else:
            # Penalize if no trials decided
            raise ValueError("No trials decided for stimulus: " + stim_key)
    
    return total_mse


# Define objective function wrapper with fixed DDM data
def obj_func(x):
    return obj_func_with_scaling(x, ddm_rts_decided_dict, ddm_data_dict, 
                                  n_trials=n_trials_per_eval, seed=seed)

# BADS options
options = {
    'uncertainty_handling': True,   # tell BADS the objective is noisy
}

# Initialize BADS
print(f"\n{'='*70}")
print("Initializing BADS optimizer...")
bads = BADS(obj_func, x0, lower_bounds, upper_bounds, 
            plausible_lower_bounds, plausible_upper_bounds, options=options)

# Run optimization
print("Starting BADS optimization...")
print(f"{'='*70}\n")
optimize_result = bads.optimize()

# %%
# Extract and display results
print("\n" + "="*70)
print("OPTIMIZATION RESULTS")
print("="*70)
x_opt = optimize_result['x']
N_opt = int(np.round(x_opt[0]))
k_opt = x_opt[1]
c_opt = k_opt / N_opt
theta_opt = x_opt[2]
rate_scaling_factor_opt = x_opt[3]

print(f"\nShared optimized parameters:")
print(f"  N:                   {N_opt}")
print(f"  k:                   {k_opt:.4f}")
print(f"  c:                   {c_opt:.6f} (= k/N)")
print(f"  theta:               {theta_opt:.4f}")
print(f"  rate_scaling_factor: {rate_scaling_factor_opt:.4f}")

print(f"\nOptimized Poisson rates for each stimulus:")
print(f"  (Poisson rate = DDM rate × rate_scaling_factor)")
for stim in stimuli:
    stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
    ddm_data = ddm_data_dict[stim_key]
    poisson_right = ddm_data['right_rate'] * rate_scaling_factor_opt
    poisson_left = ddm_data['left_rate'] * rate_scaling_factor_opt
    print(f"\n  {stim_key}:")
    print(f"    DDM right rate:     {ddm_data['right_rate']:.4f}")
    print(f"    DDM left rate:      {ddm_data['left_rate']:.4f}")
    print(f"    Poisson right rate: {poisson_right:.4f}")
    print(f"    Poisson left rate:  {poisson_left:.4f}")
    print(f"    Poisson r_right - r_left: {poisson_right - poisson_left:.4f}")

print(f"\nOptimization Statistics:")
print(f"  Total MSE (sum across stimuli): {optimize_result['fval']:.6f}")
print(f"  Average MSE per stimulus:       {optimize_result['fval']/len(stimuli):.6f}")
print(f"  Function evaluations: {optimize_result['func_count']}")
print(f"  Optimization success: {optimize_result['success']}")
print(f"  Message: {optimize_result.get('message', 'N/A')}")
print("="*70 + "\n")

# %%
# Validate the optimized parameters with larger simulations
print("="*70)
print("VALIDATION")
print("="*70)
print(f"\nRunning larger validation simulations (50,000 trials per stimulus)...")
print(f"Using optimized parameters: N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f}, "
      f"rate_scaling_factor={rate_scaling_factor_opt:.4f}\n")

validation_n_trials = 50000
validation_results_dict = {}
mse_dict = {}

for stim in stimuli:
    stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
    
    print(f"\nValidating {stim_key}...")
    
    # Get DDM data
    ddm_data = ddm_data_dict[stim_key]
    ddm_full_data = ddm_data['ddm_data']
    ddm_decided_mask = ~np.isnan(ddm_full_data[:, 0])
    ddm_rts = ddm_full_data[ddm_decided_mask, 0]
    ddm_choices = ddm_full_data[ddm_decided_mask, 1]
    
    mean_rt_ddm = np.mean(ddm_rts)
    p_right_ddm = np.mean(ddm_choices == 1)
    
    # Get DDM rates and compute Poisson rates using scaling factor
    poisson_right = ddm_data['right_rate'] * rate_scaling_factor_opt
    poisson_left = ddm_data['left_rate'] * rate_scaling_factor_opt
    
    validation_params = {
        'N_right': N_opt,
        'N_left': N_opt,
        'c': c_opt,
        'r_right': poisson_right,
        'r_left': poisson_left,
        'theta': theta_opt,
        'T': 20,
        'exponential_noise_scale': 0
    }
    
    # Run validation simulation
    validation_results = simulate_poisson_rts(validation_params, n_trials=validation_n_trials, 
                                               seed=seed, verbose=False)
    
    # Filter for decided trials
    validation_decided_mask = ~np.isnan(validation_results[:, 0])
    poisson_rts_val = validation_results[validation_decided_mask, 0]
    poisson_choices_val = validation_results[validation_decided_mask, 1]
    
    # Compute Poisson metrics
    mean_rt_poisson = np.mean(poisson_rts_val)
    p_right_poisson = np.mean(poisson_choices_val == 1)
    
    # Compute MSE components
    rt_mse = (mean_rt_ddm - mean_rt_poisson) ** 2
    choice_mse = (p_right_ddm - p_right_poisson) ** 2
    total_mse_stim = rt_mse + choice_mse
    
    # Compute KS statistic for reference
    ks_stat_val, ks_pval = ks_2samp(ddm_rts, poisson_rts_val)
    
    # Store results
    validation_results_dict[stim_key] = {
        'validation_results': validation_results,
        'poisson_rts_val': poisson_rts_val,
        'poisson_choices_val': poisson_choices_val,
        'poisson_right_rate': poisson_right,
        'poisson_left_rate': poisson_left,
        'mean_rt_ddm': mean_rt_ddm,
        'mean_rt_poisson': mean_rt_poisson,
        'p_right_ddm': p_right_ddm,
        'p_right_poisson': p_right_poisson,
        'rt_mse': rt_mse,
        'choice_mse': choice_mse,
        'total_mse': total_mse_stim,
        'ks_stat': ks_stat_val,
        'ks_pval': ks_pval
    }
    mse_dict[stim_key] = total_mse_stim
    
    print(f"  Poisson right rate: {poisson_right:.4f}")
    print(f"  Poisson left rate:  {poisson_left:.4f}")
    print(f"  Decided trials: {len(poisson_rts_val)} / {validation_n_trials}")
    print(f"  DDM mean RT:        {mean_rt_ddm:.4f} s")
    print(f"  Poisson mean RT:    {mean_rt_poisson:.4f} s")
    print(f"  RT MSE:             {rt_mse:.6f}")
    print(f"  DDM P(right):       {p_right_ddm:.4f}")
    print(f"  Poisson P(right):   {p_right_poisson:.4f}")
    print(f"  Choice MSE:         {choice_mse:.6f}")
    print(f"  Total MSE:          {total_mse_stim:.6f}")
    print(f"  KS-statistic (ref): {ks_stat_val:.6f}")

# Summary statistics
total_mse_validation = sum(mse_dict.values())
print(f"\n{'-'*70}")
print("VALIDATION SUMMARY:")
print(f"  Total MSE (sum):     {total_mse_validation:.6f}")
print(f"  Average MSE:         {total_mse_validation/len(stimuli):.6f}")
print(f"  Individual MSE values:")
for stim_key, mse_val in mse_dict.items():
    print(f"    {stim_key}: {mse_val:.6f}")
print("="*70 + "\n")

# %%
# Save results to files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
txt_filename = f'bads_multistim_scaling_results_{timestamp}.txt'
pkl_filename = f'bads_multistim_scaling_results_{timestamp}.pkl'

print("="*70)
print("SAVING RESULTS")
print("="*70)

# Save detailed text report
with open(txt_filename, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MULTI-STIMULUS BADS OPTIMIZATION RESULTS (WITH RATE SCALING FACTOR)\n")
    f.write("="*80 + "\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # DDM Parameters
    f.write("-"*80 + "\n")
    f.write("DDM SIMULATION PARAMETERS\n")
    f.write("-"*80 + "\n")
    f.write(f"Shared DDM parameters:\n")
    f.write(f"  Nr0:    {ddm_params['Nr0']}\n")
    f.write(f"  lambda: {ddm_params['lam']}\n")
    f.write(f"  ell:    {ddm_params['ell']}\n")
    f.write(f"  theta:  {ddm_params['theta']}\n")
    f.write(f"  dt:     {dt}\n")
    f.write(f"  dB:     {dB}\n")
    f.write(f"  T:      {T}\n")
    f.write(f"  N_sim:  {N_sim_ddm} trials per stimulus\n\n")
    
    f.write("Stimuli and DDM Results:\n")
    for stim in stimuli:
        stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
        data = ddm_data_dict[stim_key]
        f.write(f"\n  {stim_key}:\n")
        f.write(f"    ABL: {data['ABL']}, ILD: {data['ILD']}\n")
        f.write(f"    Right rate: {data['right_rate']:.6f}\n")
        f.write(f"    Left rate:  {data['left_rate']:.6f}\n")
        f.write(f"    mu (drift): {data['mu']:.6f}\n")
        f.write(f"    sigma:      {data['sigma']:.6f}\n")
        f.write(f"    Total trials:   {len(data['ddm_data'])}\n")
        f.write(f"    Decided trials: {len(data['ddm_rts_decided'])}\n")
        f.write(f"    Mean RT:   {np.mean(data['ddm_rts_decided']):.6f} s\n")
        f.write(f"    Median RT: {np.median(data['ddm_rts_decided']):.6f} s\n")
        f.write(f"    Std RT:    {np.std(data['ddm_rts_decided']):.6f} s\n")
    
    # Optimization Setup
    f.write("\n" + "-"*80 + "\n")
    f.write("BADS OPTIMIZATION SETUP\n")
    f.write("-"*80 + "\n")
    f.write(f"n_trials_per_eval (per stimulus): {n_trials_per_eval}\n")
    f.write(f"Total trials per evaluation: {n_trials_per_eval * len(stimuli)}\n")
    f.write(f"seed: {seed}\n\n")
    
    f.write("Parameter structure: [N, k, theta, rate_scaling_factor]\n\n")
    f.write("Description:\n")
    f.write("  - N, k, theta are shared across all stimuli\n")
    f.write("  - rate_scaling_factor scales DDM rates to get Poisson rates:\n")
    f.write("      poisson_right_rate = ddm_right_rate × rate_scaling_factor\n")
    f.write("      poisson_left_rate = ddm_left_rate × rate_scaling_factor\n\n")
    
    f.write("Parameter Bounds:\n")
    f.write(f"  Lower bounds:       {lower_bounds}\n")
    f.write(f"  Upper bounds:       {upper_bounds}\n")
    f.write(f"  Plausible lower:    {plausible_lower_bounds}\n")
    f.write(f"  Plausible upper:    {plausible_upper_bounds}\n")
    f.write(f"  Initial guess (x0): {x0}\n\n")
    
    # Optimization Results
    f.write("-"*80 + "\n")
    f.write("OPTIMIZATION RESULTS\n")
    f.write("-"*80 + "\n")
    f.write(f"Shared optimized parameters:\n")
    f.write(f"  N:                   {N_opt}\n")
    f.write(f"  k:                   {k_opt:.6f}\n")
    f.write(f"  c:                   {c_opt:.8f} (= k/N)\n")
    f.write(f"  theta:               {theta_opt:.6f}\n")
    f.write(f"  rate_scaling_factor: {rate_scaling_factor_opt:.6f}\n\n")
    
    f.write("Optimized Poisson rates for each stimulus:\n")
    f.write("(Poisson rate = DDM rate × rate_scaling_factor)\n\n")
    for stim in stimuli:
        stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
        ddm_data = ddm_data_dict[stim_key]
        poisson_right = ddm_data['right_rate'] * rate_scaling_factor_opt
        poisson_left = ddm_data['left_rate'] * rate_scaling_factor_opt
        f.write(f"  {stim_key}:\n")
        f.write(f"    DDM right rate:     {ddm_data['right_rate']:.6f}\n")
        f.write(f"    DDM left rate:      {ddm_data['left_rate']:.6f}\n")
        f.write(f"    Poisson right rate: {poisson_right:.6f}\n")
        f.write(f"    Poisson left rate:  {poisson_left:.6f}\n")
        f.write(f"    Poisson r_right - r_left: {poisson_right - poisson_left:.6f}\n\n")
    
    f.write(f"Optimization Statistics:\n")
    f.write(f"  Objective: MSE of mean RT and P(right)\n")
    f.write(f"  Total MSE (sum across stimuli): {optimize_result['fval']:.8f}\n")
    f.write(f"  Average MSE per stimulus:       {optimize_result['fval']/len(stimuli):.8f}\n")
    f.write(f"  Function evaluations: {optimize_result['func_count']}\n")
    f.write(f"  Optimization success: {optimize_result['success']}\n")
    f.write(f"  Message: {optimize_result.get('message', 'N/A')}\n\n")
    
    # Validation Results
    f.write("-"*80 + "\n")
    f.write("VALIDATION RESULTS (50,000 trials per stimulus)\n")
    f.write("-"*80 + "\n")
    for stim_key, val_data in validation_results_dict.items():
        f.write(f"\n{stim_key}:\n")
        f.write(f"  Poisson right rate: {val_data['poisson_right_rate']:.6f}\n")
        f.write(f"  Poisson left rate:  {val_data['poisson_left_rate']:.6f}\n")
        f.write(f"  Decided trials: {len(val_data['poisson_rts_val'])} / {validation_n_trials}\n")
        f.write(f"  DDM mean RT:        {val_data['mean_rt_ddm']:.6f} s\n")
        f.write(f"  Poisson mean RT:    {val_data['mean_rt_poisson']:.6f} s\n")
        f.write(f"  RT MSE:             {val_data['rt_mse']:.8f}\n")
        f.write(f"  DDM P(right):       {val_data['p_right_ddm']:.6f}\n")
        f.write(f"  Poisson P(right):   {val_data['p_right_poisson']:.6f}\n")
        f.write(f"  Choice MSE:         {val_data['choice_mse']:.8f}\n")
        f.write(f"  Total MSE:          {val_data['total_mse']:.8f}\n")
        f.write(f"  KS-statistic (ref): {val_data['ks_stat']:.8f}\n")
    
    f.write(f"\nValidation Summary:\n")
    f.write(f"  Total MSE (sum):  {total_mse_validation:.8f}\n")
    f.write(f"  Average MSE:      {total_mse_validation/len(stimuli):.8f}\n\n")
    
    f.write("="*80 + "\n")
    f.write(f"Report saved: {txt_filename}\n")
    f.write("="*80 + "\n")

print(f"✓ Text report saved: {txt_filename}")

# Save pickle file with all results
results_dict = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'ddm_params': ddm_params,
    'stimuli': stimuli,
    'ddm_simulation': {
        'dt': dt,
        'dB': dB,
        'T': T,
        'N_sim_ddm': N_sim_ddm,
        'ddm_data_dict': ddm_data_dict,
        'ddm_rts_decided_dict': ddm_rts_decided_dict
    },
    'bads_setup': {
        'n_trials_per_eval': n_trials_per_eval,
        'seed': seed,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'plausible_lower_bounds': plausible_lower_bounds,
        'plausible_upper_bounds': plausible_upper_bounds,
        'x0': x0,
        'parameter_structure': '[N, k, theta, rate_scaling_factor]'
    },
    'bads_result': optimize_result,
    'optimized_params': {
        'N': N_opt,
        'k': k_opt,
        'c': c_opt,
        'theta': theta_opt,
        'rate_scaling_factor': rate_scaling_factor_opt,
        'x_opt': x_opt,
        'poisson_rates_per_stimulus': {
            f"ABL_{stim['ABL']}_ILD_{stim['ILD']}": {
                'ddm_right_rate': ddm_data_dict[f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"]['right_rate'],
                'ddm_left_rate': ddm_data_dict[f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"]['left_rate'],
                'poisson_right_rate': ddm_data_dict[f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"]['right_rate'] * rate_scaling_factor_opt,
                'poisson_left_rate': ddm_data_dict[f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"]['left_rate'] * rate_scaling_factor_opt
            } for stim in stimuli
        }
    },
    'validation': {
        'validation_n_trials': validation_n_trials,
        'validation_results_dict': validation_results_dict,
        'mse_dict': mse_dict,
        'total_mse_validation': total_mse_validation
    }
}

with open(pkl_filename, 'wb') as f:
    pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✓ Pickle file saved: {pkl_filename}")
print(f"\nTo recover results later, use:")
print(f"  with open('{pkl_filename}', 'rb') as f:")
print(f"      results = pickle.load(f)")
print(f"  optimize_result = results['bads_result']")
print(f"  x_opt = results['optimized_params']['x_opt']")
print(f"  rate_scaling_factor = results['optimized_params']['rate_scaling_factor']")
print(f"\n{'='*70}")
print("ALL TASKS COMPLETE")
print(f"{'='*70}\n")

