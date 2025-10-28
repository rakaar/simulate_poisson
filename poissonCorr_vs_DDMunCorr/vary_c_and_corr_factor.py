"""
Vary correlation coefficient (c) and correlation factor to compare:
- Poisson correlated spiking model
- DDM uncorrelated model
- Evidence jump distributions
"""

import numpy as np
import pandas as pd
import time
import pickle
import multiprocessing
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from functools import partial

# Import parameters
from params import (
    N_sim, N_sim_rtd, lam, l, Nr0,
    abl, ild, theta, T
)

# Import utility functions
from poisson_spike_corr_with_noise_utils import (
    generate_correlated_pool,
    run_single_trial,
    get_trial_binned_spike_differences
)
from ddm_utils import simulate_single_ddm_trial

# ===================================================================
# PARAMETER ARRAYS TO VARY
# ===================================================================

# Arrays of correlation coefficients and correlation factors to test
c_array = np.array([0.01, 0.05, 0.1, 0.2])
corr_factor_array = np.array([1.1, 2, 5, 10, 20])
exponential_noise_array = np.array([0, 1e-3, 2.5e-3, 5e-3])  # 0ms, 1ms, 2.5ms, 5ms in seconds

print(f"Testing {len(c_array)} values of c: {c_array}")
print(f"Testing {len(corr_factor_array)} values of corr_factor: {corr_factor_array}")
print(f"Testing {len(exponential_noise_array)} values of exponential_noise (ms): {exponential_noise_array*1000}")
print(f"Total combinations: {len(c_array) * len(corr_factor_array) * len(exponential_noise_array)}")

# ===================================================================
# CREATE RESULTS FOLDER
# ===================================================================

results_folder = 'results'
os.makedirs(results_folder, exist_ok=True)
print(f"\nResults will be saved to folder: {results_folder}/")

# ===================================================================
# MAIN LOOP OVER PARAMETER COMBINATIONS
# ===================================================================

master_seed = 42
combination_count = 0

for c in c_array:
    for corr_factor in corr_factor_array:
        for exponential_noise_to_spk_time in exponential_noise_array:
            
            print(f"\n{'='*70}")
            print(f"Running simulations for c={c}, corr_factor={corr_factor}, noise={exponential_noise_to_spk_time*1000:.2f}ms")
            print(f"{'='*70}")
            
            # ---------------------------------------------------------------
            # Calculate derived parameters
            # ---------------------------------------------------------------
            
            N_right_and_left = round(((corr_factor - 1)/c) + 1)
            N_right = N_right_and_left  
            N_left = N_right_and_left
            
            if N_right_and_left < 1:
                print(f"Skipping: N_right_and_left < 1")
                continue
            
            theta_scaled = theta * corr_factor
            r0 = Nr0 / N_right_and_left
            r0_scaled = r0 * corr_factor
            
            # Psychometric function
            r_db = (2*abl + ild)/2
            l_db = (2*abl - ild)/2
            pr = (10 ** (r_db/20))
            pl = (10 ** (l_db/20))
            
            den = (pr ** (lam * l)) + (pl ** (lam * l))
            rr = (pr ** lam) / den
            rl = (pl ** lam) / den
            
            # Scaled firing rates (for Poisson simulation)
            r_right_scaled = r0_scaled * rr
            r_left_scaled = r0_scaled * rl
            
            # Unscaled firing rates (for DDM)
            r_right = r0 * rr
            r_left = r0 * rl
            
            print(f"\nDerived parameters:")
            print(f"  N_right_and_left = {N_right_and_left}")
            print(f"  r_right = {r_right:.4f}, r_left = {r_left:.4f}")
            print(f"  r_right_scaled = {r_right_scaled:.4f}, r_left_scaled = {r_left_scaled:.4f}")
            print(f"  theta_scaled = {theta_scaled:.4f}")
            
            # ---------------------------------------------------------------
            # 1. POISSON SPIKING MODEL SIMULATION
            # ---------------------------------------------------------------
            
            print(f'\n=== POISSON SPIKING MODEL SIMULATION ===')
            start_time_poisson = time.time()
            
            # Create partial function with fixed parameters
            run_trial_partial = partial(
                run_single_trial,
                N_right=N_right,
                c=c,
                r_right_scaled=r_right_scaled,
                T=T,
                N_left=N_left,
                r_left_scaled=r_left_scaled,
                theta_scaled=theta_scaled,
                exponential_noise_scale=exponential_noise_to_spk_time
            )
            
            tasks = [(i, master_seed) for i in range(N_sim_rtd)]
            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                poisson_results = list(tqdm(
                    pool.imap(run_trial_partial, tasks), 
                    total=N_sim_rtd, 
                    desc='Simulating Correlated Spikes'
                ))
            
            poisson_results_array = np.array(poisson_results)
            end_time_poisson = time.time()
            print(f"Poisson simulation took: {end_time_poisson - start_time_poisson:.2f} seconds")
            
            # Summary Statistics
            decision_made_mask = ~np.isnan(poisson_results_array[:, 0])
            mean_rt_poisson = np.mean(poisson_results_array[decision_made_mask, 0])
            choices_poisson = poisson_results_array[:, 1]
            prop_pos_poisson = np.sum(choices_poisson == 1) / N_sim_rtd
            prop_neg_poisson = np.sum(choices_poisson == -1) / N_sim_rtd
            prop_no_decision_poisson = np.sum(choices_poisson == 0) / N_sim_rtd
            
            print(f"Mean RT (decided trials): {mean_rt_poisson:.4f} s")
            print(f"Proportion +1 choices: {prop_pos_poisson:.2%}")
            print(f"Proportion -1 choices: {prop_neg_poisson:.2%}")
            print(f"Proportion no-decision: {prop_no_decision_poisson:.2%}")
            
            # ---------------------------------------------------------------
            # 2. DDM SIMULATION
            # ---------------------------------------------------------------
            
            # DDM parameters
            N_neurons = N_right
            mu = N_neurons * (r_right - r_left)
            corr_factor_ddm = 1
            sigma_sq = N_neurons * (r_right + r_left) * corr_factor_ddm
            sigma = sigma_sq**0.5
            theta_ddm = theta
            
            print(f'\n=== DDM SIMULATION ===')
            print(f'mu = {mu}')
            print(f'sigma = {sigma}')
            print(f'theta_ddm = {theta_ddm}')
            
            dt = 1e-4
            dB = 1e-2
            n_steps = int(T/dt)
            
            start_time_ddm = time.time()
            ddm_results = Parallel(n_jobs=-1)(
                delayed(simulate_single_ddm_trial)(i, mu, sigma, theta_ddm, dt, dB, n_steps) 
                for i in tqdm(range(N_sim_rtd), desc='Simulating DDM')
            )
            ddm_data = np.array(ddm_results)
            end_time_ddm = time.time()
            print(f"DDM simulation took: {end_time_ddm - start_time_ddm:.2f} seconds")
            
            # ---------------------------------------------------------------
            # 3. EVIDENCE JUMP DISTRIBUTION ANALYSIS
            # ---------------------------------------------------------------
            
            dt_bin = 1e-3
            
            print(f'\n=== EVIDENCE JUMP DISTRIBUTION ANALYSIS ===')
            print(f'Collecting binned spike differences from {N_sim} trials...')
            print(f'Time bin size: dt = {dt_bin*1000:.2f} ms')
            
            all_bin_differences = []
            
            for trial_idx in range(N_sim):
                bin_diffs = get_trial_binned_spike_differences(
                    trial_idx=trial_idx,
                    seed=master_seed,
                    dt_bin=dt_bin,
                    T=T,
                    N_right=N_right,
                    c=c,
                    r_right_scaled=r_right_scaled,
                    N_left=N_left,
                    r_left_scaled=r_left_scaled,
                    exponential_noise_scale=exponential_noise_to_spk_time
                )
                all_bin_differences.extend(bin_diffs)
            
            all_bin_differences = np.array(all_bin_differences)
            print(f'Total number of time bins analyzed: {len(all_bin_differences)}')
            
            # Analyze the distribution
            min_diff = int(all_bin_differences.min())
            max_diff = int(all_bin_differences.max())
            hist_bins = np.arange(min_diff - 0.5, max_diff + 1.5, 1)
            bin_diff_frequencies, bin_edges = np.histogram(all_bin_differences, bins=hist_bins)
            bin_diff_values = np.arange(min_diff, max_diff + 1)
            
            print(f'Binned spike difference distribution computed.')
            
            # ---------------------------------------------------------------
            # SAVE RESULTS FOR THIS COMBINATION
            # ---------------------------------------------------------------
            
            result_data = {
                'params': {
                    'c': c,
                    'corr_factor': corr_factor,
                    'N_right_and_left': N_right_and_left,
                    'theta': theta,
                    'theta_scaled': theta_scaled,
                    'r0': r0,
                    'r0_scaled': r0_scaled,
                    'r_right': r_right,
                    'r_left': r_left,
                    'r_right_scaled': r_right_scaled,
                    'r_left_scaled': r_left_scaled,
                    'T': T,
                    'N_sim': N_sim,
                    'N_sim_rtd': N_sim_rtd,
                    'exponential_noise_to_spk_time': exponential_noise_to_spk_time,
                },
                'poisson': {
                    'results': poisson_results_array,
                    'mean_rt': mean_rt_poisson,
                    'prop_pos': prop_pos_poisson,
                    'prop_neg': prop_neg_poisson,
                    'prop_no_decision': prop_no_decision_poisson,
                    'simulation_time': end_time_poisson - start_time_poisson,
                },
                'ddm': {
                    'params': {
                        'mu': mu,
                        'sigma': sigma,
                        'theta_ddm': theta_ddm,
                        'dt': dt,
                        'dB': dB,
                    },
                    'results': ddm_data,
                    'simulation_time': end_time_ddm - start_time_ddm,
                },
                'evidence_distribution': {
                    'dt_bin': dt_bin,
                    'all_bin_differences': all_bin_differences,
                    'bin_diff_values': bin_diff_values,
                    'bin_diff_frequencies': bin_diff_frequencies,
                    'min_diff': min_diff,
                    'max_diff': max_diff,
                }
            }
            
            # Create filename with parameter values
            filename = f"c_{c}_corrfactor_{corr_factor}_noise_{exponential_noise_to_spk_time*1000:.1f}ms.pkl"
            filepath = os.path.join(results_folder, filename)
            
            # Save to file
            with open(filepath, 'wb') as f:
                pickle.dump(result_data, f)
            
            combination_count += 1
            print(f"\n=== SAVED: {filename} ===")
            print(f"Progress: {combination_count}/{len(c_array) * len(corr_factor_array) * len(exponential_noise_array)} combinations complete")

# ===================================================================
# ALL SIMULATIONS COMPLETE
# ===================================================================

print(f"\n{'='*70}")
print(f"ALL SIMULATIONS COMPLETE")
print(f"{'='*70}")
print(f"Total parameter combinations tested: {combination_count}")
print(f"All results saved to folder: {results_folder}/")
