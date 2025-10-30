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
    objective_function,
    save_bads_results_txt
)

# %%
# DDM data sim
# stim
ABL = 20
ILD = 0
ddm_params = {
    'Nr0': 13.3,
    'lam': 1.3,
    'ell': 0.9,
    'theta': 2
}

ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(ABL, ILD,
     ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell'])

mu_ddm = ddm_right_rate - ddm_left_rate
sigma_sq_ddm = ddm_right_rate + ddm_left_rate
sigma_ddm = np.sqrt(sigma_sq_ddm)

dt = 1e-4
dB = 1e-2
N_sim_ddm = int(100e3)
T = 20
n_steps = int(T/dt)
# Parallel DDM simulation
print(f'\n=== DDM SIMULATION ===')
start_time_ddm = time.time()
ddm_results = Parallel(n_jobs=-1)(
    delayed(simulate_single_ddm_trial)(i, mu_ddm, sigma_ddm, ddm_params['theta'], dt, dB, n_steps) 
    for i in tqdm(range(N_sim_ddm), desc='Simulating DDM')
)
ddm_data = np.array(ddm_results)
end_time_ddm = time.time()
print(f"DDM simulation took: {end_time_ddm - start_time_ddm:.2f} seconds")
print(f"## DDM: mean RT is {np.mean(ddm_data[:, 0]):.4f}" )

# %%
# Prepare DDM data for optimization
ddm_decided_mask = ~np.isnan(ddm_data[:, 0])
ddm_rts_decided = ddm_data[ddm_decided_mask, 0]

print(f"\nDDM data prepared:")
print(f"  Total trials: {len(ddm_data)}")
print(f"  Decided trials: {len(ddm_rts_decided)}")
print(f"  Mean RT: {np.mean(ddm_rts_decided):.4f} s")
print(f"  Median RT: {np.median(ddm_rts_decided):.4f} s")

# %%
# BADS optimization setup
n_trials_per_eval = int(50e3)
seed = 42

print("\n=== BADS OPTIMIZATION ===")
print(f"Optimizing parameters to minimize KS-statistic between DDM and Poisson RTs")
print(f"DDM RT samples: {len(ddm_rts_decided)}")
print(f"Poisson trials per evaluation: {n_trials_per_eval}\n")

# Parameter bounds: [N, r1, r2, k, theta]
# Hard bounds (actual optimization constraints)
lower_bounds = np.array([10, 0.001, 0.001, 1, 2])
upper_bounds = np.array([1000, 50, 50, 10, 20])

# Plausible bounds (where we expect the solution to be)
plausible_lower_bounds = np.array([50, 0.01, 0.01, 2, 3])
plausible_upper_bounds = np.array([500, 10, 10, 9, 15])

# Initial guess (midpoint of plausible range)
x0 = (plausible_lower_bounds + plausible_upper_bounds) / 2

print(f"Parameter ranges:")
print(f"  N:     [{lower_bounds[0]:.0f}, {upper_bounds[0]:.0f}]")
print(f"  r1:    [{lower_bounds[1]:.3f}, {upper_bounds[1]:.3f}]")
print(f"  r2:    [{lower_bounds[2]:.3f}, {upper_bounds[2]:.3f}]")
print(f"  k:     [{lower_bounds[3]:.0f}, {upper_bounds[3]:.0f}]")
print(f"  theta: [{lower_bounds[4]:.1f}, {upper_bounds[4]:.1f}]")
print(f"\nInitial guess: N={x0[0]:.0f}, r1={x0[1]:.3f}, r2={x0[2]:.3f}, "
        f"k={x0[3]:.1f}, theta={x0[4]:.2f}\n")

# Define objective function with fixed DDM data
def obj_func(x):
    return objective_function(x, ddm_rts_decided, n_trials=n_trials_per_eval, seed=seed)

# Initialize BADS
bads = BADS(obj_func, x0, lower_bounds, upper_bounds, 
            plausible_lower_bounds, plausible_upper_bounds)

# Run optimization
print("Starting BADS optimization...\n")
optimize_result = bads.optimize()

# Extract and display results
print("\n=== OPTIMIZATION RESULTS ===")
x_opt = optimize_result['x']
N_opt = int(np.round(x_opt[0]))
k_opt = x_opt[3]
c_opt = k_opt / N_opt
print(f"Optimized parameters:")
print(f"  N:     {N_opt}")
print(f"  r1:    {x_opt[1]:.4f}")
print(f"  r2:    {x_opt[2]:.4f}")
print(f"  k:     {k_opt:.2f}")
print(f"  c:     {c_opt:.6f} (= k/N)")
print(f"  theta: {x_opt[4]:.4f}")
print(f"Minimum KS-statistic: {optimize_result['fval']:.6f}")
print(f"Function evaluations: {optimize_result['func_count']}")
print(f"Optimization success: {optimize_result['success']}")

# %%
# Validate the optimized parameters by running a larger simulation
print("\n=== VALIDATION ===")
x_opt = optimize_result['x']
N_opt = int(np.round(x_opt[0]))
k_opt = x_opt[3]
c_opt = k_opt / N_opt

validation_params = {
    'N_right': N_opt,
    'N_left': N_opt,
    'c': c_opt,
    'r_right': x_opt[1],
    'r_left': x_opt[2],
    'theta': x_opt[4],
    'T': 20,
    'exponential_noise_scale': 0
}

print("Running validation with optimized parameters on 50,000 trials...")
validation_results = simulate_poisson_rts(validation_params, n_trials=50000, 
                                            seed=42, verbose=True)

# Compare distributions
validation_decided_mask = ~np.isnan(validation_results[:, 0])
poisson_rts_val = validation_results[validation_decided_mask, 0]

ks_stat_val, ks_pval = ks_2samp(ddm_rts_decided, poisson_rts_val)
print(f"\nValidation KS-statistic: {ks_stat_val:.6f}")
print(f"KS p-value: {ks_pval:.6f}")

# %%
# Save results to files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
txt_filename = f'bads_optimization_results_{timestamp}.txt'
pkl_filename = f'bads_optimization_results_{timestamp}.pkl'

print(f"\n=== SAVING RESULTS ===")

# Save detailed text report using helper function
save_bads_results_txt(
    filename=txt_filename,
    ddm_params=ddm_params,
    ddm_stimulus={'ABL': ABL, 'ILD': ILD},
    ddm_simulation_params={
        'mu': mu_ddm,
        'sigma': sigma_ddm,
        'dt': dt,
        'dB': dB,
        'T': T,
        'N_sim': N_sim_ddm
    },
    ddm_data=ddm_data,
    ddm_rts_decided=ddm_rts_decided,
    bads_setup={
        'n_trials_per_eval': n_trials_per_eval,
        'seed': seed,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'plausible_lower_bounds': plausible_lower_bounds,
        'plausible_upper_bounds': plausible_upper_bounds,
        'x0': x0
    },
    optimize_result=optimize_result,
    optimized_params={
        'N': N_opt,
        'r1': x_opt[1],
        'r2': x_opt[2],
        'k': k_opt,
        'c': c_opt,
        'theta': x_opt[4],
        'x_opt': x_opt
    },
    validation_results=validation_results,
    poisson_rts_val=poisson_rts_val,
    ks_stat_val=ks_stat_val,
    ks_pval=ks_pval
)

print(f"✓ Text report saved: {txt_filename}")

# Save pickle file with all results
results_dict = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'ddm_params': ddm_params,
    'ddm_stimulus': {'ABL': ABL, 'ILD': ILD},
    'ddm_simulation': {
        'mu': mu_ddm,
        'sigma': sigma_ddm,
        'dt': dt,
        'dB': dB,
        'T': T,
        'N_sim': N_sim_ddm,
        'ddm_data': ddm_data,
        'ddm_rts_decided': ddm_rts_decided
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
        'r1': x_opt[1],
        'r2': x_opt[2],
        'k': k_opt,
        'c': c_opt,
        'theta': x_opt[4],
        'x_opt': x_opt
    },
    'validation': {
        'validation_params': validation_params,
        'validation_results': validation_results,
        'poisson_rts_val': poisson_rts_val,
        'ks_stat_val': ks_stat_val,
        'ks_pval': ks_pval
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
