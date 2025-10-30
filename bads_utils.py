"""
Utility functions for BADS optimization of Poisson model parameters.

This module contains simulation functions for DDM and Poisson models,
as well as the objective function used in BADS optimization.
"""

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import ks_2samp


def lr_rates_from_ABL_ILD(ABL, ILD, Nr0, lam, ell):
    """Calculate left and right firing rates from ABL and ILD."""
    # NOTE: right, left is the order. NOT left, right
    r_db = (2*ABL + ILD)/2
    l_db = (2*ABL - ILD)/2
    pr = (10 ** (r_db/20))
    pl = (10 ** (l_db/20))
    den = (pr ** (lam * ell)) + (pl ** (lam * ell))
    rr = (pr ** lam) / den
    rl = (pl ** lam) / den
    mu1 = Nr0 * rr  # Right
    mu2 = Nr0 * rl  # Left
    return mu1, mu2


def simulate_single_ddm_trial(trial_idx, mu, sigma, theta_ddm, dt, dB, n_steps):
    """Simulate a single DDM trial."""
    DV = 0.0
    
    for step in range(n_steps):
        # Generate single evidence step
        evidence_step = mu*dt + (sigma)*np.random.normal(0, dB)
        DV += evidence_step
        
        # Check for boundary crossing
        if DV >= theta_ddm:
            return (step * dt, 1)
        elif DV <= -theta_ddm:
            return (step * dt, -1)
    
    # No decision made within time limit
    return (np.nan, 0)


def generate_correlated_pool(N, c, r, T, rng, exponential_noise_scale=0):
    """
    Generates a pool of N correlated spike trains using the thinning method.
    """
    pool_spikes = {}
    
    # Handle zero correlation case: generate independent spikes for each neuron
    if c == 0:
        for i in range(N):
            n_spikes = rng.poisson(r * T)
            neuron_spikes = np.sort(rng.random(n_spikes) * T)
            
            # Add Exponential noise to spike timings (always positive delays)
            noise = rng.exponential(scale=exponential_noise_scale, size=len(neuron_spikes))
            neuron_spikes = neuron_spikes + noise
            # Ensure spike times remain within [0, T] and are sorted
            neuron_spikes = np.clip(neuron_spikes, 0, T)
            neuron_spikes = np.sort(neuron_spikes)
            
            pool_spikes[i] = neuron_spikes
        return pool_spikes
    
    # Standard correlated case (c > 0)
    source_rate = r / c
    n_source_spikes = rng.poisson(source_rate * T)
    source_spk_timings = np.sort(rng.random(n_source_spikes) * T)
    
    for i in range(N):
        keep_spike_mask = rng.random(size=n_source_spikes) < c
        neuron_spikes = source_spk_timings[keep_spike_mask]
        
        # Add Exponential noise to spike timings (always positive delays)
        noise = rng.exponential(scale=exponential_noise_scale, size=len(neuron_spikes))
        neuron_spikes = neuron_spikes + noise
        # Ensure spike times remain within [0, T] and are sorted
        neuron_spikes = np.clip(neuron_spikes, 0, T)
        neuron_spikes = np.sort(neuron_spikes)
        
        pool_spikes[i] = neuron_spikes
    
    return pool_spikes


def simulate_single_poisson_trial(trial_idx, seed, params):
    """
    Runs a single trial of the Poisson spiking simulation.
    """
    rng_local = np.random.default_rng(seed + trial_idx)
    
    # Extract parameters
    N_right = params['N_right']
    N_left = params['N_left']
    c = params['c']
    r_right = params['r_right']
    r_left = params['r_left']
    theta = params['theta']
    T = params['T']
    exponential_noise_scale = params.get('exponential_noise_scale', 0)
    
    # Generate all spike trains for this trial
    right_pool_spikes = generate_correlated_pool(N_right, c, r_right, T, rng_local, exponential_noise_scale)
    left_pool_spikes = generate_correlated_pool(N_left, c, r_left, T, rng_local, exponential_noise_scale)
    
    # Consolidate spikes into a single stream of evidence events
    all_right_spikes = np.concatenate(list(right_pool_spikes.values()))
    all_left_spikes = np.concatenate(list(left_pool_spikes.values()))
    
    all_times = np.concatenate([all_right_spikes, all_left_spikes])
    all_evidence = np.concatenate([
        np.ones_like(all_right_spikes, dtype=int),
        -np.ones_like(all_left_spikes, dtype=int)
    ])
    
    if all_times.size == 0:
        return (np.nan, 0)
    
    # Aggregate simultaneous spikes
    events_df = pd.DataFrame({'time': all_times, 'evidence_jump': all_evidence})
    evidence_events = events_df.groupby('time')['evidence_jump'].sum().reset_index()
    
    event_times = evidence_events['time'].values
    event_jumps = evidence_events['evidence_jump'].values
    
    # Run the decision process using the cumsum method
    dv_trajectory = np.cumsum(event_jumps)
    
    # Find boundary crossings
    pos_crossings = np.where(dv_trajectory >= theta)[0]
    neg_crossings = np.where(dv_trajectory <= -theta)[0]
    
    first_pos_idx = pos_crossings[0] if pos_crossings.size > 0 else np.inf
    first_neg_idx = neg_crossings[0] if neg_crossings.size > 0 else np.inf
    
    # Determine outcome and return result
    if first_pos_idx < first_neg_idx:
        rt = event_times[first_pos_idx]
        choice = 1
        return (rt, choice)
    elif first_neg_idx < first_pos_idx:
        rt = event_times[first_neg_idx]
        choice = -1
        return (rt, choice)
    else:
        return (np.nan, 0)


def simulate_poisson_rts(params, n_trials=10000, n_jobs=-1, seed=42, verbose=True):
    """
    Simulate multiple Poisson model trials in parallel and return RTs and choices.
    """
    if verbose:
        print(f'\n=== POISSON SIMULATION ====')
        print(f'Simulating {n_trials} trials...')
        print(f'Parameters: N={params["N_right"]}, c={params["c"]:.4f}, '
              f'r_R={params["r_right"]:.4f}, r_L={params["r_left"]:.4f}, '
              f'theta={params["theta"]:.4f}')
    
    start_time = time.time()
    
    # Create task arguments
    tasks = [(i, seed, params) for i in range(n_trials)]
    
    # Run parallel simulation
    if verbose:
        results = Parallel(n_jobs=n_jobs)(
            delayed(simulate_single_poisson_trial)(trial_idx, s, p)
            for trial_idx, s, p in tqdm(tasks, desc='Simulating Poisson trials')
        )
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(simulate_single_poisson_trial)(trial_idx, s, p)
            for trial_idx, s, p in tasks
        )
    
    results_array = np.array(results)
    end_time = time.time()
    
    if verbose:
        print(f'Simulation took: {end_time - start_time:.2f} seconds')
        
        # Summary statistics
        decision_made_mask = ~np.isnan(results_array[:, 0])
        if np.sum(decision_made_mask) > 0:
            mean_rt = np.mean(results_array[decision_made_mask, 0])
            print(f'Mean RT (decided trials): {mean_rt:.4f} s')
        
        choices = results_array[:, 1]
        prop_pos = np.sum(choices == 1) / n_trials
        prop_neg = np.sum(choices == -1) / n_trials
        prop_no_decision = np.sum(choices == 0) / n_trials
        
        print(f'Proportion positive (+1): {prop_pos:.2%}')
        print(f'Proportion negative (-1): {prop_neg:.2%}')
        print(f'Proportion no decision (0): {prop_no_decision:.2%}')
    
    return results_array


def objective_function(x, ddm_rts_decided, n_trials=int(50e3), seed=42):
    """
    Objective function for BADS: compute KS-statistic between DDM and Poisson RTs.
    
    Args:
        x: Array of parameters [N, r1, r2, k, theta]
        ddm_rts_decided: DDM reaction times (decided trials only, no NaNs)
        n_trials: Number of Poisson trials to simulate
        seed: Random seed
    
    Returns:
        KS statistic (to minimize)
    """
    N, r1, r2, k, theta = x
    
    # Num of neurons is integer
    N = int(np.round(N))
    
    # Compute c from k and N
    c = k / N
    
    # theta should be int in poisson spike
    theta = int(np.ceil(theta))
    if theta < 1:
        theta = 1
    
    
    # Set up Poisson parameters
    poisson_params = {
        'N_right': N,
        'N_left': N,
        'c': c,
        'r_right': r1,
        'r_left': r2,
        'theta': theta,
        'T': 20,
        'exponential_noise_scale': 0
    }
    
    try:
        # Simulate Poisson model (no verbose output)
        poisson_results = simulate_poisson_rts(poisson_params, n_trials=n_trials, 
                                               seed=seed, verbose=False, n_jobs=-1)
        
        # Filter for decided trials
        poisson_decided_mask = ~np.isnan(poisson_results[:, 0])
        poisson_rts_decided = poisson_results[poisson_decided_mask, 0]
        

        # Compute KS statistic
        ks_stat, _ = ks_2samp(ddm_rts_decided, poisson_rts_decided)
        
        return ks_stat
        
    except Exception as e:
        raise ValueError(f"Error in objective function: {e}")


def objective_function_singlerate(x, ddm_rts_decided, n_trials=int(50e3), seed=42):
    """
    Objective function for BADS with single rate: compute KS-statistic between DDM and Poisson RTs.
    Uses the same rate for both left and right pools.
    
    Args:
        x: Array of parameters [N, r, k, theta]
        ddm_rts_decided: DDM reaction times (decided trials only, no NaNs)
        n_trials: Number of Poisson trials to simulate
        seed: Random seed
    
    Returns:
        KS statistic (to minimize)
    """
    N, r, k, theta = x
    
    # Num of neurons is integer
    N = int(np.round(N))
    
    # Compute c from k and N
    c = k / N
    
    # theta should be int in poisson spike
    theta = int(np.ceil(theta))
    if theta < 1:
        theta = 1
    
    
    # Set up Poisson parameters (same rate for both left and right)
    poisson_params = {
        'N_right': N,
        'N_left': N,
        'c': c,
        'r_right': r,
        'r_left': r,
        'theta': theta,
        'T': 20,
        'exponential_noise_scale': 0
    }
    
    try:
        # Simulate Poisson model (no verbose output)
        poisson_results = simulate_poisson_rts(poisson_params, n_trials=n_trials, 
                                               seed=seed, verbose=False, n_jobs=-1)
        
        # Filter for decided trials
        poisson_decided_mask = ~np.isnan(poisson_results[:, 0])
        poisson_rts_decided = poisson_results[poisson_decided_mask, 0]
        

        # Compute KS statistic
        ks_stat, _ = ks_2samp(ddm_rts_decided, poisson_rts_decided)
        
        return ks_stat
        
    except Exception as e:
        raise ValueError(f"Error in objective function: {e}")


def save_bads_results_txt(filename, ddm_params, ddm_stimulus, ddm_simulation_params, 
                           ddm_data, ddm_rts_decided, bads_setup, optimize_result, 
                           optimized_params, validation_results, poisson_rts_val, 
                           ks_stat_val, ks_pval):
    """
    Save BADS optimization results to a detailed text file.
    
    Args:
        filename: Output filename for the text report
        ddm_params: Dict with DDM parameters (Nr0, lam, ell, theta)
        ddm_stimulus: Dict with stimulus parameters (ABL, ILD)
        ddm_simulation_params: Dict with simulation parameters (mu, sigma, dt, dB, T, N_sim)
        ddm_data: Array of all DDM trial results
        ddm_rts_decided: Array of DDM RTs for decided trials
        bads_setup: Dict with BADS setup (n_trials_per_eval, seed, bounds, x0)
        optimize_result: BADS OptimizeResult object
        optimized_params: Dict with optimized parameters (N, r1, r2, k, c, theta, x_opt)
        validation_results: Array of validation trial results
        poisson_rts_val: Array of Poisson RTs for decided trials
        ks_stat_val: Validation KS statistic
        ks_pval: Validation KS p-value
    """
    from datetime import datetime
    
    with open(filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BADS OPTIMIZATION RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # DDM Parameters
        f.write("-"*70 + "\n")
        f.write("DDM SIMULATION PARAMETERS\n")
        f.write("-"*70 + "\n")
        f.write(f"ABL: {ddm_stimulus['ABL']}\n")
        f.write(f"ILD: {ddm_stimulus['ILD']}\n")
        f.write(f"Nr0: {ddm_params['Nr0']}\n")
        f.write(f"lambda: {ddm_params['lam']}\n")
        f.write(f"ell: {ddm_params['ell']}\n")
        f.write(f"theta: {ddm_params['theta']}\n")
        f.write(f"mu: {ddm_simulation_params['mu']:.6f}\n")
        f.write(f"sigma: {ddm_simulation_params['sigma']:.6f}\n")
        f.write(f"dt: {ddm_simulation_params['dt']}\n")
        f.write(f"dB: {ddm_simulation_params['dB']}\n")
        f.write(f"T: {ddm_simulation_params['T']}\n")
        f.write(f"N_sim_ddm: {ddm_simulation_params['N_sim']}\n\n")
        
        # DDM Results
        f.write("DDM Simulation Results:\n")
        f.write(f"  Total trials: {len(ddm_data)}\n")
        f.write(f"  Decided trials: {len(ddm_rts_decided)}\n")
        f.write(f"  Mean RT: {np.mean(ddm_rts_decided):.6f} s\n")
        f.write(f"  Median RT: {np.median(ddm_rts_decided):.6f} s\n")
        f.write(f"  Std RT: {np.std(ddm_rts_decided):.6f} s\n\n")
        
        # Optimization Setup
        f.write("-"*70 + "\n")
        f.write("BADS OPTIMIZATION SETUP\n")
        f.write("-"*70 + "\n")
        f.write(f"n_trials_per_eval: {bads_setup['n_trials_per_eval']}\n")
        f.write(f"seed: {bads_setup['seed']}\n\n")
        
        f.write("Parameter Bounds [N, r1, r2, k, theta]:\n")
        f.write(f"  Lower bounds: {bads_setup['lower_bounds']}\n")
        f.write(f"  Upper bounds: {bads_setup['upper_bounds']}\n")
        f.write(f"  Plausible lower: {bads_setup['plausible_lower_bounds']}\n")
        f.write(f"  Plausible upper: {bads_setup['plausible_upper_bounds']}\n")
        f.write(f"  Initial guess: {bads_setup['x0']}\n\n")
        
        # Optimization Results
        f.write("-"*70 + "\n")
        f.write("OPTIMIZATION RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"Optimized parameters:\n")
        f.write(f"  N:     {optimized_params['N']}\n")
        f.write(f"  r1:    {optimized_params['r1']:.6f}\n")
        f.write(f"  r2:    {optimized_params['r2']:.6f}\n")
        f.write(f"  k:     {optimized_params['k']:.6f}\n")
        f.write(f"  c:     {optimized_params['c']:.8f} (= k/N)\n")
        f.write(f"  theta: {optimized_params['theta']:.6f}\n\n")
        
        f.write(f"Optimization Statistics:\n")
        f.write(f"  Minimum KS-statistic: {optimize_result['fval']:.8f}\n")
        f.write(f"  Function evaluations: {optimize_result['func_count']}\n")
        f.write(f"  Optimization success: {optimize_result['success']}\n")
        f.write(f"  Optimization message: {optimize_result.get('message', 'N/A')}\n\n")
        
        # Validation Results
        f.write("-"*70 + "\n")
        f.write("VALIDATION RESULTS (50,000 trials)\n")
        f.write("-"*70 + "\n")
        f.write(f"Validation trials decided: {len(poisson_rts_val)} / {len(validation_results)}\n")
        f.write(f"Poisson mean RT: {np.mean(poisson_rts_val):.6f} s\n")
        f.write(f"Poisson median RT: {np.median(poisson_rts_val):.6f} s\n")
        f.write(f"Poisson std RT: {np.std(poisson_rts_val):.6f} s\n\n")
        f.write(f"Validation KS-statistic: {ks_stat_val:.8f}\n")
        f.write(f"Validation KS p-value: {ks_pval:.8e}\n\n")
        
        f.write("="*70 + "\n")
        f.write(f"Report saved: {filename}\n")
        f.write("="*70 + "\n")
