# %%
# Jitter Effect on Log Odds vs ILD
# Simulates Poisson spiking model with different jitter values and compares log odds to DDM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed
import pickle
from datetime import datetime
from mgf_helper_utils import find_h0

# ===================================================================
# 1. PARAMETERS
# ===================================================================

# Simulation parameters
N_trials = int(20e3)  # Number of trials per condition

# Poisson model parameters (fixed)
N_neurons = 1000  # Number of neurons in each pool (right and left)
rho = 1e-4 # Correlation parameter

# DDM / psychometric parameters
lam = 1.3
l = 0.9
Nr0_base = 13.3

abl = 20  # Average binaural level
theta = 20  # Decision threshold

# ILD values to test
ild_values = [1, 2, 4, 8, 16]

# Jitter values to test (in seconds) - easily modifiable
jitter_values_ms = [0, 1, 2.5, 5, 10, 25]  # in milliseconds
jitter_values = [j * 1e-3 for j in jitter_values_ms]  # convert to seconds

# Simulation duration
T = 50  # Max duration of a single trial (seconds)

# Rate scaling
rate_scalar_right = 1
rate_scalar_left = 1

print("=" * 60)
print("JITTER EFFECT ON LOG ODDS - PARAMETERS")
print("=" * 60)
print(f"N_neurons = {N_neurons}")
print(f"rho = {rho}")
print(f"theta = {theta}")
print(f"lam = {lam}, l = {l}")
print(f"Nr0_base = {Nr0_base}")
print(f"ABL = {abl}")
print(f"ILD values = {ild_values}")
print(f"Jitter values (ms) = {jitter_values_ms}")
print(f"N_trials = {N_trials}")
print("=" * 60)

# %%
# ===================================================================
# 2. HELPER FUNCTIONS
# ===================================================================

def calculate_rates(ild, abl, lam, l, Nr0_base, N_neurons, rate_scalar_right, rate_scalar_left):
    """Calculate firing rates for given ILD and parameters."""
    r0 = Nr0_base / N_neurons
    r_db = (2 * abl + ild) / 2
    l_db = (2 * abl - ild) / 2
    pr = 10 ** (r_db / 20)
    pl = 10 ** (l_db / 20)
    
    den = (pr ** (lam * l)) + (pl ** (lam * l))
    rr = (pr ** lam) / den
    rl = (pl ** lam) / den
    
    r_right = r0 * rr * rate_scalar_right
    r_left = r0 * rl * rate_scalar_left
    
    return r_right, r_left


def generate_correlated_pool(N, rho, r, T, rng, jitter_scale):
    """
    Generates a pool of N correlated spike trains using the thinning method.
    
    Parameters:
    -----------
    N : int
        Number of neurons
    rho : float
        Correlation parameter (0 to 1)
    r : float
        Firing rate per neuron
    T : float
        Trial duration
    rng : np.random.Generator
        Random number generator
    jitter_scale : float
        Scale parameter for exponential jitter (in seconds)
    
    Returns:
    --------
    dict : Dictionary where keys are neuron indices and values are spike time arrays
    """
    pool_spikes = {}
    
    # Handle zero/very small correlation case: generate independent spikes
    # if rho < 1e-10:
    #     for i in range(N):
    #         n_spikes = rng.poisson(r * T)
    #         neuron_spikes = np.sort(rng.random(n_spikes) * T)
            
    #         # Add exponential jitter to spike timings
    #         if jitter_scale > 0:
    #             noise = rng.exponential(scale=jitter_scale, size=len(neuron_spikes))
    #             neuron_spikes = neuron_spikes + noise
    #             neuron_spikes = np.clip(neuron_spikes, 0, T)
    #             neuron_spikes = np.sort(neuron_spikes)
            
    #         pool_spikes[i] = neuron_spikes
    #     return pool_spikes
    
    # Standard correlated case (rho > 0)
    source_rate = r / rho
    n_source_spikes = rng.poisson(source_rate * T)
    source_spk_timings = np.sort(rng.random(n_source_spikes) * T)
    
    for i in range(N):
        keep_spike_mask = rng.random(size=n_source_spikes) < rho
        neuron_spikes = source_spk_timings[keep_spike_mask]
        
        # Add exponential jitter to spike timings
        if jitter_scale > 0:
            noise = rng.exponential(scale=jitter_scale, size=len(neuron_spikes))
            neuron_spikes = neuron_spikes + noise
            # neuron_spikes = np.clip(neuron_spikes, 0, T)
            neuron_spikes = np.sort(neuron_spikes)
        
        pool_spikes[i] = neuron_spikes
    
    return pool_spikes


def run_single_trial(args):
    """Runs a single trial of the Poisson spiking simulation."""
    trial_idx, seed, r_right, r_left, N, rho, theta, T, jitter_scale = args
    rng_local = np.random.default_rng(seed + trial_idx)

    # Generate spike trains for this trial
    right_pool_spikes = generate_correlated_pool(N, rho, r_right, T, rng_local, jitter_scale)
    left_pool_spikes = generate_correlated_pool(N, rho, r_left, T, rng_local, jitter_scale)

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

    # Group simultaneous events
    events_df = pd.DataFrame({'time': all_times, 'evidence_jump': all_evidence})
    evidence_events = events_df.groupby('time')['evidence_jump'].sum().reset_index()

    event_times = evidence_events['time'].values
    event_jumps = evidence_events['evidence_jump'].values

    # Run the decision process using cumsum
    dv_trajectory = np.cumsum(event_jumps)

    pos_crossings = np.where(dv_trajectory >= theta)[0]
    neg_crossings = np.where(dv_trajectory <= -theta)[0]

    first_pos_idx = pos_crossings[0] if pos_crossings.size > 0 else np.inf
    first_neg_idx = neg_crossings[0] if neg_crossings.size > 0 else np.inf

    # Determine outcome
    if first_pos_idx < first_neg_idx:
        rt = event_times[first_pos_idx]
        choice = 1  # Right choice
        return (rt, choice)
    elif first_neg_idx < first_pos_idx:
        rt = event_times[first_neg_idx]
        choice = -1  # Left choice
        return (rt, choice)
    else:
        return (np.nan, 0)  # No decision


# %%
# ===================================================================
# 3. DATA GENERATION FUNCTION
# ===================================================================

def generate_simulation_data(ild_values, jitter_values, N_trials, N_neurons, rho, theta, T,
                              abl, lam, l, Nr0_base, rate_scalar_right, rate_scalar_left,
                              master_seed=42, n_jobs=-1):
    """
    Generate simulation data for all ILD and jitter combinations.
    
    Returns:
    --------
    dict : Results dictionary with structure:
           results[jitter_ms][ild] = {'n_right': int, 'n_left': int, 'n_no_decision': int,
                                       'mean_rt': float, 'rts': array, 'choices': array}
    """
    results = {}
    
    for jitter_scale in jitter_values:
        jitter_ms = jitter_scale * 1000
        results[jitter_ms] = {}
        print(f"\n>>> Processing jitter = {jitter_ms} ms")
        
        for ild in tqdm(ild_values, desc=f"Jitter {jitter_ms}ms"):
            # Calculate rates for this ILD
            r_right, r_left = calculate_rates(ild, abl, lam, l, Nr0_base, N_neurons,
                                              rate_scalar_right, rate_scalar_left)
            
            # Prepare tasks for parallel execution
            tasks = [(i, master_seed, r_right, r_left, N_neurons, rho, theta, T, jitter_scale) 
                     for i in range(N_trials)]
            
            # Run simulations in parallel
            with multiprocessing.Pool(n_jobs if n_jobs > 0 else multiprocessing.cpu_count()) as pool:
                trial_results = pool.map(run_single_trial, tasks)
            
            results_array = np.array(trial_results)
            rts = results_array[:, 0]
            choices = results_array[:, 1]
            
            # Count outcomes
            n_right = np.sum(choices == 1)
            n_left = np.sum(choices == -1)
            n_no_decision = np.sum(choices == 0)
            
            # Calculate mean RT for decided trials
            decided_mask = ~np.isnan(rts)
            mean_rt = np.mean(rts[decided_mask]) if np.any(decided_mask) else np.nan
            
            results[jitter_ms][ild] = {
                'n_right': n_right,
                'n_left': n_left,
                'n_no_decision': n_no_decision,
                'mean_rt': mean_rt,
                'rts': rts,
                'choices': choices,
                'r_right': r_right,
                'r_left': r_left
            }
            
            print(f"  ILD={ild}: n_right={n_right}, n_left={n_left}, "
                  f"P_right={n_right/(n_right+n_left+1e-10):.3f}")
    
    return results


# %%
# ===================================================================
# 4. LOG ODDS CALCULATION
# ===================================================================

def calculate_log_odds(results, ild_values, mode="n_trials_add"):
    """
    Calculate log odds from simulation results.
    
    Parameters:
    -----------
    results : dict
        Simulation results dictionary
    ild_values : list
        List of ILD values
    mode : str
        "prob_den_add" - log(P_right / (P_left + 1e-50))
        "n_trials_add" - add 1 to n_right and n_left before calculating (Laplace smoothing)
    
    Returns:
    --------
    dict : log_odds[jitter_ms] = array of log odds for each ILD
    """
    log_odds = {}
    
    for jitter_ms, ild_results in results.items():
        odds_list = []
        for ild in ild_values:
            data = ild_results[ild]
            n_right = data['n_right']
            n_left = data['n_left']
            
            if mode == "prob_den_add":
                # Add epsilon to denominator probability
                n_total = n_right + n_left
                if n_total > 0:
                    P_right = n_right / n_total
                    P_left = n_left / n_total
                    log_odd = np.log(P_right / (P_left + 1e-50))
                else:
                    log_odd = 0.0
                    
            elif mode == "n_trials_add":
                # Laplace smoothing: add 1 trial to each side
                n_right_adj = n_right + 1
                n_left_adj = n_left + 1
                n_total_adj = n_right_adj + n_left_adj
                P_right = n_right_adj / n_total_adj
                P_left = n_left_adj / n_total_adj
                log_odd = np.log(P_right / P_left)
            
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'prob_den_add' or 'n_trials_add'")
            
            odds_list.append(log_odd)
        
        log_odds[jitter_ms] = np.array(odds_list)
    
    return log_odds


def normalize_log_odds(log_odds, ild_values):
    """
    Normalize log odds by the value at the highest ILD.
    
    Returns:
    --------
    dict : normalized_log_odds[jitter_ms] = normalized array
    """
    normalized = {}
    max_ild_idx = len(ild_values) - 1
    
    for jitter_ms, odds in log_odds.items():
        max_val = odds[max_ild_idx]
        if np.abs(max_val) > 1e-10:
            normalized[jitter_ms] = odds / max_val
        else:
            normalized[jitter_ms] = odds
    
    return normalized


# %%
# ===================================================================
# 5. PLOTTING FUNCTIONS
# ===================================================================

def plot_log_odds(normalized_log_odds, ild_values, lam, jitter_values_ms,
                  N_neurons, rho, theta, abl, l, Nr0_base,
                  rate_scalar_right, rate_scalar_left, save_fig=True):
    """
    Plot normalized log odds vs ILD for different jitter values.
    Also plots DDM reference curve and theoretical Poisson log odds.
    """
    # DDM log odds (analytical)
    cont_ild = np.arange(1, max(ild_values) + 1, 0.1)
    ddm_logodds = np.tanh(lam * cont_ild / 17.37)
    ddm_logodds_norm = ddm_logodds / np.max(np.abs(ddm_logodds))
    
    # Calculate theoretical Poisson log odds (no jitter) using find_h0
    theoretical_log_odds = []
    for ild_val in ild_values:
        r_db_ild = (2 * abl + ild_val) / 2
        l_db_ild = (2 * abl - ild_val) / 2
        pr_ild = 10 ** (r_db_ild / 20)
        pl_ild = 10 ** (l_db_ild / 20)
        
        den_ild = (pr_ild ** (lam * l)) + (pl_ild ** (lam * l))
        rr_ild = (pr_ild ** lam) / den_ild
        rl_ild = (pl_ild ** lam) / den_ild
        
        r0_ild = Nr0_base / N_neurons
        r_right_ild = r0_ild * rr_ild * rate_scalar_right
        r_left_ild = r0_ild * rl_ild * rate_scalar_left
        
        h0 = find_h0(r_right_ild, r_left_ild, N_neurons, rho)
        theoretical_log_odds.append(-h0 * theta)
    
    theoretical_log_odds = np.array(theoretical_log_odds)
    theoretical_log_odds_norm = theoretical_log_odds / np.max(np.abs(theoretical_log_odds))
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Color map and markers for jitter values
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(normalized_log_odds)))
    markers = ['o', '^', 'v', 'D', 'p', 'h', '*', 'X']  # 8 different markers
    
    # Plot Poisson log odds for each jitter (simulated)
    for idx, (jitter_ms, norm_odds) in enumerate(sorted(normalized_log_odds.items())):
        marker = markers[idx % len(markers)]
        plt.plot(ild_values, norm_odds, marker=marker, markersize=10, linewidth=2.5,
                 color=colors[idx], label=f'Poisson Sim (jitter={jitter_ms}ms)')
    
    # Plot theoretical Poisson log odds (normalized)
    plt.plot(ild_values, theoretical_log_odds_norm, marker='s', markersize=10, linewidth=2.5,
             color='red', linestyle=':', label='Poisson Theory (no jitter)')
    
    # Plot DDM reference
    plt.plot(cont_ild, ddm_logodds_norm, linestyle='--', color='k', lw=2.5, 
             label='DDM (analytical)')
    
    plt.xlabel('ILD', fontsize=14, fontweight='bold')
    plt.ylabel(f'Normalized Log Odds\n(log(P_R/P_L) / max)', fontsize=14, fontweight='bold')
    plt.title(f'Normalized Log Odds vs ILD for Different Jitter Values\n'
              f'N={N_neurons}, ρ={rho}, θ={theta}, ABL={abl}, λ={lam}',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(ild_values)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    
    if save_fig:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'jitter_log_odds_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {filename}")
    
    plt.show()


def plot_raw_log_odds(log_odds, ild_values, lam, jitter_values_ms,
                      N_neurons, rho, theta, abl, l, Nr0_base, 
                      rate_scalar_right, rate_scalar_left, save_fig=True):
    """
    Plot raw (non-normalized) log odds vs ILD for different jitter values.
    Also plots theoretical Poisson log odds (no jitter) using find_h0.
    """
    # DDM log odds (analytical) - not normalized
    cont_ild = np.arange(1, max(ild_values) + 1, 0.1)
    ddm_logodds = np.tanh(lam * cont_ild / 17.37)
    
    # Calculate theoretical Poisson log odds (no jitter) using find_h0
    theoretical_log_odds = []
    for ild_val in ild_values:
        # Calculate rates for this ILD
        r_db_ild = (2 * abl + ild_val) / 2
        l_db_ild = (2 * abl - ild_val) / 2
        pr_ild = 10 ** (r_db_ild / 20)
        pl_ild = 10 ** (l_db_ild / 20)
        
        den_ild = (pr_ild ** (lam * l)) + (pl_ild ** (lam * l))
        rr_ild = (pr_ild ** lam) / den_ild
        rl_ild = (pl_ild ** lam) / den_ild
        
        r0_ild = Nr0_base / N_neurons
        r_right_ild = r0_ild * rr_ild * rate_scalar_right
        r_left_ild = r0_ild * rl_ild * rate_scalar_left
        
        h0 = find_h0(r_right_ild, r_left_ild, N_neurons, rho)
        # Theoretical log odds is -h0 * theta (h0 is negative for right-favoring)
        theoretical_log_odds.append(-h0 * theta)
    
    theoretical_log_odds = np.array(theoretical_log_odds)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Color map and markers for jitter values
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(log_odds)))
    markers = ['o', '^', 'v', 'D', 'p', 'h', '*', 'X']  # 8 different markers
    
    # Plot Poisson log odds for each jitter (simulated)
    for idx, (jitter_ms, odds) in enumerate(sorted(log_odds.items())):
        marker = markers[idx % len(markers)]
        plt.plot(ild_values, odds, marker=marker, markersize=10, linewidth=2.5,
                 color=colors[idx], label=f'Poisson Sim (jitter={jitter_ms}ms)')
    
    # Plot theoretical Poisson log odds (no jitter)
    plt.plot(ild_values, theoretical_log_odds, marker='s', markersize=10, linewidth=2.5,
             color='red', linestyle='--', label='Poisson Theory (no jitter)')
    
    # Plot DDM reference (scaled to match range)
    scale_factor = np.max([np.max(np.abs(odds)) for odds in log_odds.values()])
    ddm_scaled = ddm_logodds * scale_factor / np.max(np.abs(ddm_logodds))
    plt.plot(cont_ild, ddm_scaled, linestyle=':', color='k', lw=2.5, 
             label='DDM (scaled)')
    
    plt.xlabel('ILD', fontsize=14, fontweight='bold')
    plt.ylabel('Log Odds', fontsize=14, fontweight='bold')
    plt.title(f'Raw Log Odds vs ILD for Different Jitter Values\n'
              f'N={N_neurons}, ρ={rho}, θ={theta}, ABL={abl}, λ={lam}',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(ild_values)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    
    if save_fig:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'jitter_log_odds_raw_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {filename}")
    
    plt.show()


# %%
# ===================================================================
# 6. RUN SIMULATION
# ===================================================================
print("\n" + "=" * 60)
print("STARTING SIMULATION")
print("=" * 60)

results = generate_simulation_data(
    ild_values=ild_values,
    jitter_values=jitter_values,
    N_trials=N_trials,
    N_neurons=N_neurons,
    rho=rho,
    theta=theta,
    T=T,
    abl=abl,
    lam=lam,
    l=l,
    Nr0_base=Nr0_base,
    rate_scalar_right=rate_scalar_right,
    rate_scalar_left=rate_scalar_left,
    master_seed=42
)

# %%
# ===================================================================
# 7. CALCULATE LOG ODDS
# ===================================================================
# log_odds = calculate_log_odds(results, ild_values)
# prob_den_add or n_trials_add mode
log_odds = calculate_log_odds(results, ild_values, mode='n_trials_add')

normalized_log_odds = normalize_log_odds(log_odds, ild_values)

# Print summary
print("\nLog Odds Summary:")
print("-" * 40)
for jitter_ms in sorted(log_odds.keys()):
    print(f"\nJitter = {jitter_ms} ms:")
    print(f"  ILD:        {ild_values}")
    print(f"  Log Odds:   {[f'{x:.3f}' for x in log_odds[jitter_ms]]}")
    print(f"  Normalized: {[f'{x:.3f}' for x in normalized_log_odds[jitter_ms]]}")

# %%
# ===================================================================
# 8. PLOT NORMALIZED LOG ODDS
# ===================================================================
plot_log_odds(normalized_log_odds, ild_values, lam, jitter_values_ms,
              N_neurons, rho, theta, abl, l, Nr0_base,
              rate_scalar_right, rate_scalar_left)

# %%
# ===================================================================
# 9. PLOT RAW LOG ODDS
# ===================================================================
plot_raw_log_odds(log_odds, ild_values, lam, jitter_values_ms,
                  N_neurons, rho, theta, abl, l, Nr0_base,
                  rate_scalar_right, rate_scalar_left)

# %%
# ===================================================================
# 10. SAVE RESULTS TO PICKLE
# ===================================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_to_save = {
    'results': results,
    'log_odds': log_odds,
    'normalized_log_odds': normalized_log_odds,
    'parameters': {
        'N_neurons': N_neurons,
        'rho': rho,
        'theta': theta,
        'lam': lam,
        'l': l,
        'Nr0_base': Nr0_base,
        'abl': abl,
        'ild_values': ild_values,
        'jitter_values_ms': jitter_values_ms,
        'N_trials': N_trials,
        'rate_scalar_right': rate_scalar_right,
        'rate_scalar_left': rate_scalar_left
    }
}

pkl_filename = f'jitter_log_odds_results_{timestamp}.pkl'
with open(pkl_filename, 'wb') as f:
    pickle.dump(results_to_save, f)
print(f"\nResults saved to: {pkl_filename}")

# %%
