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
    {'ABL': 60, 'ILD': 2},
    {'ABL': 60, 'ILD': 4}
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
print(f"\nOptimizing parameters to minimize sum of KS-statistics across all stimuli")
print(f"Poisson trials per evaluation per stimulus: {n_trials_per_eval}")
print(f"Total Poisson trials per evaluation: {n_trials_per_eval * len(stimuli)}\n")

# Parameter structure: [N, k, theta, r1_20_2, r2_20_2, r1_20_4, r2_20_4, r1_60_2, r2_60_2, r1_60_4, r2_60_4]
# - N, k, theta are shared across all stimuli
# - Each stimulus has its own r1 (right rate) and r2 (left rate)

# Hard bounds (actual optimization constraints)
# [N, k, theta, r1_20_2, r2_20_2, r1_20_4, r2_20_4, r1_60_2, r2_60_2, r1_60_4, r2_60_4]
lower_bounds = np.array([10, 1, 2, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
upper_bounds = np.array([1000, 10, 20, 50, 50, 50, 50, 50, 50, 50, 50])

# Plausible bounds (where we expect the solution to be)
plausible_lower_bounds = np.array([50, 2, 3, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
plausible_upper_bounds = np.array([500, 9, 15, 10, 10, 10, 10, 10, 10, 10, 10])

# Initial guess (midpoint of plausible range)
x0 = (plausible_lower_bounds + plausible_upper_bounds) / 2

print("Parameter structure:")
print("  [N, k, theta, r1_20_2, r2_20_2, r1_20_4, r2_20_4, r1_60_2, r2_60_2, r1_60_4, r2_60_4]")
print("\nShared parameters:")
print(f"  N:     [{lower_bounds[0]:.0f}, {upper_bounds[0]:.0f}]")
print(f"  k:     [{lower_bounds[1]:.0f}, {upper_bounds[1]:.0f}]")
print(f"  theta: [{lower_bounds[2]:.1f}, {upper_bounds[2]:.1f}]")
print("\nRate parameters (each stimulus has r1=right, r2=left):")
for i, stim in enumerate(stimuli):
    idx_r1 = 3 + i*2
    idx_r2 = 4 + i*2
    print(f"  ABL={stim['ABL']}, ILD={stim['ILD']}: r1=[{lower_bounds[idx_r1]:.3f}, {upper_bounds[idx_r1]:.1f}], "
          f"r2=[{lower_bounds[idx_r2]:.3f}, {upper_bounds[idx_r2]:.1f}]")

print(f"\nInitial guess:")
print(f"  N={x0[0]:.0f}, k={x0[1]:.1f}, theta={x0[2]:.2f}")
for i, stim in enumerate(stimuli):
    idx_r1 = 3 + i*2
    idx_r2 = 4 + i*2
    print(f"  ABL={stim['ABL']}, ILD={stim['ILD']}: r1={x0[idx_r1]:.3f}, r2={x0[idx_r2]:.3f}")

# Define objective function with fixed DDM data
def obj_func(x):
    return objective_function_multistim(x, ddm_rts_decided_dict, 
                                        n_trials=n_trials_per_eval, seed=seed)

# Initialize BADS
print(f"\n{'='*70}")
print("Initializing BADS optimizer...")
bads = BADS(obj_func, x0, lower_bounds, upper_bounds, 
            plausible_lower_bounds, plausible_upper_bounds)

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

print(f"\nShared optimized parameters:")
print(f"  N:     {N_opt}")
print(f"  k:     {k_opt:.4f}")
print(f"  c:     {c_opt:.6f} (= k/N)")
print(f"  theta: {theta_opt:.4f}")

print(f"\nOptimized rates for each stimulus:")
for i, stim in enumerate(stimuli):
    idx_r1 = 3 + i*2
    idx_r2 = 4 + i*2
    stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
    print(f"  {stim_key}:")
    print(f"    r1 (right): {x_opt[idx_r1]:.4f}")
    print(f"    r2 (left):  {x_opt[idx_r2]:.4f}")
    print(f"    r1 - r2:    {x_opt[idx_r1] - x_opt[idx_r2]:.4f}")

print(f"\nOptimization Statistics:")
print(f"  Total KS-statistic (sum across stimuli): {optimize_result['fval']:.6f}")
print(f"  Average KS-statistic per stimulus:       {optimize_result['fval']/len(stimuli):.6f}")
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
print(f"Using optimized parameters: N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f}\n")

validation_n_trials = 50000
validation_results_dict = {}
ks_stats_dict = {}

for i, stim in enumerate(stimuli):
    idx_r1 = 3 + i*2
    idx_r2 = 4 + i*2
    stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
    
    print(f"\nValidating {stim_key}...")
    
    validation_params = {
        'N_right': N_opt,
        'N_left': N_opt,
        'c': c_opt,
        'r_right': x_opt[idx_r1],
        'r_left': x_opt[idx_r2],
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
    
    # Get corresponding DDM data
    ddm_rts = ddm_rts_decided_dict[stim_key]
    
    # Compute KS statistic
    ks_stat_val, ks_pval = ks_2samp(ddm_rts, poisson_rts_val)
    
    # Store results
    validation_results_dict[stim_key] = {
        'validation_results': validation_results,
        'poisson_rts_val': poisson_rts_val,
        'ks_stat': ks_stat_val,
        'ks_pval': ks_pval
    }
    ks_stats_dict[stim_key] = ks_stat_val
    
    print(f"  Decided trials: {len(poisson_rts_val)} / {validation_n_trials}")
    print(f"  Poisson mean RT: {np.mean(poisson_rts_val):.4f} s")
    print(f"  DDM mean RT:     {np.mean(ddm_rts):.4f} s")
    print(f"  KS-statistic:    {ks_stat_val:.6f}")
    print(f"  KS p-value:      {ks_pval:.6f}")

# Summary statistics
total_ks_validation = sum(ks_stats_dict.values())
print(f"\n{'-'*70}")
print("VALIDATION SUMMARY:")
print(f"  Total KS-statistic (sum):     {total_ks_validation:.6f}")
print(f"  Average KS-statistic:         {total_ks_validation/len(stimuli):.6f}")
print(f"  Individual KS statistics:")
for stim_key, ks_val in ks_stats_dict.items():
    print(f"    {stim_key}: {ks_val:.6f}")
print("="*70 + "\n")

# %%
# Save results to files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
txt_filename = f'bads_multistim_results_{timestamp}.txt'
pkl_filename = f'bads_multistim_results_{timestamp}.pkl'

print("="*70)
print("SAVING RESULTS")
print("="*70)

# Save detailed text report
with open(txt_filename, 'w') as f:
    f.write("="*80 + "\n")
    f.write("MULTI-STIMULUS BADS OPTIMIZATION RESULTS\n")
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
    
    f.write("Parameter structure: [N, k, theta, r1_20_2, r2_20_2, r1_20_4, r2_20_4, r1_60_2, r2_60_2, r1_60_4, r2_60_4]\n\n")
    
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
    f.write(f"  N:     {N_opt}\n")
    f.write(f"  k:     {k_opt:.6f}\n")
    f.write(f"  c:     {c_opt:.8f} (= k/N)\n")
    f.write(f"  theta: {theta_opt:.6f}\n\n")
    
    f.write("Optimized rates for each stimulus:\n")
    for i, stim in enumerate(stimuli):
        idx_r1 = 3 + i*2
        idx_r2 = 4 + i*2
        stim_key = f"ABL_{stim['ABL']}_ILD_{stim['ILD']}"
        f.write(f"  {stim_key}:\n")
        f.write(f"    r1 (right): {x_opt[idx_r1]:.6f}\n")
        f.write(f"    r2 (left):  {x_opt[idx_r2]:.6f}\n")
        f.write(f"    r1 - r2:    {x_opt[idx_r1] - x_opt[idx_r2]:.6f}\n\n")
    
    f.write(f"Optimization Statistics:\n")
    f.write(f"  Total KS-statistic (sum across stimuli): {optimize_result['fval']:.8f}\n")
    f.write(f"  Average KS-statistic per stimulus:       {optimize_result['fval']/len(stimuli):.8f}\n")
    f.write(f"  Function evaluations: {optimize_result['func_count']}\n")
    f.write(f"  Optimization success: {optimize_result['success']}\n")
    f.write(f"  Message: {optimize_result.get('message', 'N/A')}\n\n")
    
    # Validation Results
    f.write("-"*80 + "\n")
    f.write("VALIDATION RESULTS (50,000 trials per stimulus)\n")
    f.write("-"*80 + "\n")
    for stim_key, val_data in validation_results_dict.items():
        f.write(f"\n{stim_key}:\n")
        f.write(f"  Decided trials: {len(val_data['poisson_rts_val'])} / {validation_n_trials}\n")
        f.write(f"  Poisson mean RT:   {np.mean(val_data['poisson_rts_val']):.6f} s\n")
        f.write(f"  Poisson median RT: {np.median(val_data['poisson_rts_val']):.6f} s\n")
        f.write(f"  Poisson std RT:    {np.std(val_data['poisson_rts_val']):.6f} s\n")
        ddm_rts = ddm_rts_decided_dict[stim_key]
        f.write(f"  DDM mean RT:       {np.mean(ddm_rts):.6f} s\n")
        f.write(f"  KS-statistic:      {val_data['ks_stat']:.8f}\n")
        f.write(f"  KS p-value:        {val_data['ks_pval']:.8e}\n")
    
    f.write(f"\nValidation Summary:\n")
    f.write(f"  Total KS-statistic (sum):  {total_ks_validation:.8f}\n")
    f.write(f"  Average KS-statistic:      {total_ks_validation/len(stimuli):.8f}\n\n")
    
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
        'x0': x0
    },
    'bads_result': optimize_result,
    'optimized_params': {
        'N': N_opt,
        'k': k_opt,
        'c': c_opt,
        'theta': theta_opt,
        'x_opt': x_opt,
        'rates_per_stimulus': {
            f"ABL_{stim['ABL']}_ILD_{stim['ILD']}": {
                'r1_right': x_opt[3 + i*2],
                'r2_left': x_opt[4 + i*2]
            } for i, stim in enumerate(stimuli)
        }
    },
    'validation': {
        'validation_n_trials': validation_n_trials,
        'validation_results_dict': validation_results_dict,
        'ks_stats_dict': ks_stats_dict,
        'total_ks_validation': total_ks_validation
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
print(f"\n{'='*70}")
print("ALL TASKS COMPLETE")
print(f"{'='*70}\n")
