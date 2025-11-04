# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from pathlib import Path
from bads_utils import (
    lr_rates_from_ABL_ILD,
    simulate_poisson_rts
)

# %%
# Find and load the most recent multi-stimulus BADS results (with choice)
pkl_pattern = 'bads_multistim_scaling_results_*.pkl'
pkl_files = glob.glob(pkl_pattern)

if not pkl_files:
    raise FileNotFoundError(f"No files matching '{pkl_pattern}' found in current directory")

# Get the most recent file
latest_pkl = max(pkl_files, key=lambda x: Path(x).stat().st_mtime)
print(f"Loading results from: {latest_pkl}")

# Define dummy obj_func to allow unpickling
def obj_func(x):
    """Dummy function for unpickling - not actually used."""
    pass

def obj_func_with_scaling(x, ddm_rts_decided_dict, ddm_data_dict, n_trials, seed):
    """Dummy function for unpickling - not actually used."""
    pass

with open(latest_pkl, 'rb') as f:
    results = pickle.load(f)

# Extract data
optimized_params = results['optimized_params']
ddm_params = results['ddm_params']

# Extract shared parameters (including rate_scaling_factor)
# N_opt = optimized_params['N']
# k_opt = optimized_params['k']
# c_opt = optimized_params['c']
# theta_opt = optimized_params['theta']
# rate_scaling_factor_opt = optimized_params['rate_scaling_factor']
### TEMP HARD CODING NOTE ###
N_opt = 87
c_opt = 0.1
k_opt = c_opt * N_opt

theta_opt = 15.86
rate_scaling_factor = 1
print("\n" + "="*70)
print("LOADED POISSON PARAMETERS")
print("="*70)
print(f"  N:                   {N_opt}")
print(f"  k:                   {k_opt:.4f}")
print(f"  c:                   {c_opt:.6f} (= k/N)")
print(f"  theta:               {theta_opt:.4f}")
print(f"  rate_scaling_factor: {rate_scaling_factor:.4f}")

# %%
# Simulate Poisson for ABL=20, ILD=1
ABL = 20
ILD = 1
N_sim = int(100e3)  # 100K trials

print(f"\n{'='*70}")
print(f"SIMULATING POISSON FOR ABL={ABL}, ILD={ILD}")
print(f"{'='*70}")

# Calculate DDM rates (needed to get Poisson rates via scaling)
ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
    ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
)

# Calculate Poisson rates using scaling factor
poisson_right_rate = ddm_right_rate * rate_scaling_factor
poisson_left_rate = ddm_left_rate * rate_scaling_factor

# Poisson simulation
print(f"\n--- Poisson Simulation ---")
poisson_params = {
    'N_right': N_opt,
    'N_left': N_opt,
    'c': c_opt,
    'r_right': poisson_right_rate,
    'r_left': poisson_left_rate,
    'theta': theta_opt,
    'T': 20,
    'exponential_noise_scale': 0
}

print(f"Poisson rates: right={poisson_right_rate:.6f}, left={poisson_left_rate:.6f}")
print(f"Poisson params: N={N_opt}, c={c_opt:.6f}, theta={theta_opt:.4f}")
print(f"Simulating {N_sim} Poisson trials...")

poisson_results = simulate_poisson_rts(poisson_params, n_trials=N_sim, 
                                       seed=42, verbose=False)
poisson_decided_mask = ~np.isnan(poisson_results[:, 0])
poisson_rts_decided = poisson_results[poisson_decided_mask, 0]
poisson_choices_decided = poisson_results[poisson_decided_mask, 1]

# Poisson metrics
mean_rt_poisson = np.mean(poisson_rts_decided)
p_right_poisson = np.mean(poisson_choices_decided == 1)

print(f"Poisson: {len(poisson_rts_decided)}/{N_sim} decided")
print(f"  Mean RT:   {mean_rt_poisson:.4f}s")
print(f"  P(right):  {p_right_poisson:.4f}")

# %%
# Plot Reaction Time Distribution
plt.figure(figsize=(10, 6))
plt.hist(poisson_rts_decided, bins=np.arange(0,2,0.01), density=True, alpha=0.7, edgecolor='black', histtype='step')
plt.xlabel('Reaction Time (s)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title(f'Poisson RTD for ABL={ABL}, ILD={ILD}\n' + 
          f'N={N_opt} = 10, k={k_opt:.4f} cORR FACTOR = {1 + (N_opt-1)*c_opt:.4f}\n' +
          f'c={c_opt:.6f}, theta={theta_opt:.4f}\n',
          fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# EVIDENCE JUMP DISTRIBUTION ANALYSIS
# ===================================================================

def generate_correlated_pool(N, c, r, T, rng, exponential_noise_scale=0):
    """
    Generates a pool of N correlated spike trains using the thinning method.
    Returns a dictionary where keys are neuron indices and values are spike time arrays.
    """
    pool_spikes = {}
    
    # Handle zero correlation case: generate independent spikes for each neuron
    if c == 0:
        for i in range(N):
            n_spikes = rng.poisson(r * T)
            neuron_spikes = np.sort(rng.random(n_spikes) * T)
            
            # Add Exponential noise to spike timings (always positive delays)
            if exponential_noise_scale > 0:
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
        if exponential_noise_scale > 0:
            noise = rng.exponential(scale=exponential_noise_scale, size=len(neuron_spikes))
            neuron_spikes = neuron_spikes + noise
            # Ensure spike times remain within [0, T] and are sorted
            neuron_spikes = np.clip(neuron_spikes, 0, T)
            neuron_spikes = np.sort(neuron_spikes)
        
        pool_spikes[i] = neuron_spikes
    
    return pool_spikes


def get_trial_binned_spike_differences(trial_idx, seed, dt_bin, T, N_right, N_left, c, 
                                       r_right, r_left, exponential_noise_scale):
    """
    Runs a single trial, bins spike times into bins of size dt_bin,
    and returns the spike difference (R - L) for each time bin.
    """
    rng_local = np.random.default_rng(seed + trial_idx)
    
    # Generate all spike trains for this trial
    right_pool_spikes = generate_correlated_pool(N_right, c, r_right, T, rng_local, 
                                                 exponential_noise_scale)
    left_pool_spikes = generate_correlated_pool(N_left, c, r_left, T, rng_local, 
                                                exponential_noise_scale)
    
    # Consolidate all spikes
    all_right_spikes = np.concatenate(list(right_pool_spikes.values()))
    all_left_spikes = np.concatenate(list(left_pool_spikes.values()))
    
    # Create time bins
    n_bins = int(np.ceil(T / dt_bin))
    time_bins = np.arange(0, T + dt_bin, dt_bin)
    
    # Count spikes in each bin
    right_counts, _ = np.histogram(all_right_spikes, bins=time_bins)
    left_counts, _ = np.histogram(all_left_spikes, bins=time_bins)
    
    # Compute difference (R - L) for each bin
    spike_differences = right_counts - left_counts
    
    return spike_differences


# Time-binned analysis
N_sim_jump = 100  # Number of trials for jump distribution
dt_bin = 1e-3
T_jump = 20

print(f'\n{"="*70}')
print(f'EVIDENCE JUMP DISTRIBUTION ANALYSIS')
print(f'{"="*70}')
print(f'Collecting binned spike differences from {N_sim_jump} trials...')
print(f'Time bin size: dt = {dt_bin*1000:.2f} ms')

all_bin_differences = []

for trial_idx in range(N_sim_jump):
    bin_diffs = get_trial_binned_spike_differences(
        trial_idx, 42, dt_bin, T_jump, N_opt, N_opt, c_opt, 
        poisson_right_rate, poisson_left_rate, 
        poisson_params['exponential_noise_scale']
    )
    all_bin_differences.extend(bin_diffs)

all_bin_differences = np.array(all_bin_differences)
print(f'Total number of time bins analyzed: {len(all_bin_differences)}')

# Analyze the distribution
min_diff = int(all_bin_differences.min())
max_diff = int(all_bin_differences.max())
hist_bins = np.arange(min_diff - 0.5, max_diff + 1.5, 1)  # Bin edges around integers
bin_diff_frequencies, bin_edges = np.histogram(all_bin_differences, bins=hist_bins)
bin_diff_values = np.arange(min_diff, max_diff + 1)  # Integer values

print(f'Binned spike difference distribution computed.')

# %%
# Plot Evidence Jump Distribution
plt.figure(figsize=(10, 6))
plt.bar(bin_diff_values, bin_diff_frequencies, width=0.8, alpha=0.7, 
        edgecolor='black', color='steelblue')
plt.xlabel('Spike Difference (R - L) per Time Bin', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title(
    f'Evidence Jump Distribution for ABL={ABL}, ILD={ILD}\n'
    f'N={N_opt}, k={k_opt:.4f}, CORR FACTOR = {1 + (N_opt-1)*c_opt:.4f}\n'
    f'c={c_opt:.6f}, theta={theta_opt:.4f}\n'
    f'Time-binned at dt = {dt_bin*1000:.2f} ms, {N_sim_jump} trials, {len(all_bin_differences)} total bins',
    fontsize=12
)
plt.grid(axis='y', alpha=0.3)
plt.xticks(bin_diff_values[::max(1, len(bin_diff_values)//20)])  # Show subset of ticks
plt.yscale('log')
plt.tight_layout()
plt.show()