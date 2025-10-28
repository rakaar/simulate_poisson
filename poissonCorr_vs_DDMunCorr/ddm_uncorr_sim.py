"""
DDM (Drift Diffusion Model) Uncorrelated Simulation
Simulates uncorrelated DDM trials and saves results
"""

import numpy as np
import time
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
from params import *
from ddm_utils import simulate_single_ddm_trial

# ===================================================================
# DDM PARAMETERS
# ===================================================================

N_neurons = N_right
mu = N_neurons * (r_right - r_left)

corr_factor_ddm = 1
sigma_sq = N_neurons * (r_right + r_left) * corr_factor_ddm
sigma = sigma_sq**0.5
theta_ddm = theta

print(f'\n=== DDM PARAMETERS ===')
print(f'mu = {mu}')
print(f'sigma = {sigma}')
print(f'theta_ddm = {theta_ddm}')

dt = 1e-4
dB = 1e-2
n_steps = int(T/dt)

# ===================================================================
# DDM SIMULATION
# ===================================================================

print(f'\n=== DDM SIMULATION ===')
start_time_ddm = time.time()
ddm_results = Parallel(n_jobs=-1)(
    delayed(simulate_single_ddm_trial)(i, mu, sigma, theta_ddm, dt, dB, n_steps) 
    for i in tqdm(range(N_sim_rtd), desc='Simulating DDM')
)
ddm_data = np.array(ddm_results)
end_time_ddm = time.time()
print(f"DDM simulation took: {end_time_ddm - start_time_ddm:.2f} seconds")

# ===================================================================
# SAVE DDM PARAMETERS AND RESULTS
# ===================================================================

ddm_output = {
    'params': {
        'N_neurons': N_neurons,
        'mu': mu,
        'sigma': sigma,
        'sigma_sq': sigma_sq,
        'theta_ddm': theta_ddm,
        'corr_factor_ddm': corr_factor_ddm,
        'dt': dt,
        'dB': dB,
        'n_steps': n_steps,
        'T': T,
        'N_sim_rtd': N_sim_rtd,
        'r_right': r_right,
        'r_left': r_left,
    },
    'results': {
        'ddm_data': ddm_data,
        'simulation_time': end_time_ddm - start_time_ddm,
    }
}

output_filename = 'ddm_uncorr_results.pkl'
with open(output_filename, 'wb') as f:
    pickle.dump(ddm_output, f)

print(f'\n=== RESULTS SAVED ===')
print(f'Saved to: {output_filename}')
print(f'DDM data shape: {ddm_data.shape}')
