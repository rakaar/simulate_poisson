 # Combined Analysis: Poisson Spiking Model, DDM, and Evidence Jump Distribution
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import multiprocessing
import time
from joblib import Parallel, delayed

# ===================================================================
# 1. PARAMETERS
# ===================================================================

N_sim = 100  # Number of trials for jump distribution analysis
N_sim_rtd = int(2*50e3)  # Number of trials for RTD (reaction time distribution)

# instead of N, c is hardcoded. 
# N_right_and_left = 50
c = 0.01
corr_factor = 5
# 
N_right_and_left = round(((corr_factor - 1)/c) + 1)
N_right = N_right_and_left  
N_left = N_right_and_left   
if N_right_and_left < 1:
    raise ValueError("N_right_and_left must be greater than 1")
# tweak
theta = 2.5
# theta = 10
theta_scaled = theta * corr_factor

# get from model fits
lam = 1.3
l = 0.9
Nr0 = 13.3 
# Nr0 = 100
exponential_noise_to_spk_time = 0 # Scale parameter in seconds
# exponential_noise_to_spk_time = 1e-3 # Scale parameter in seconds

# correlation and base firing rate
# c = (corr_factor - 1) / (N_right_and_left - 1)
# if c < 0 or c >1:
#     raise ValueError("Correlation must be between 0 and 1")
r0 = Nr0/N_right_and_left
r0_scaled = r0 * corr_factor

abl = 20
ild = 0
r_db = (2*abl + ild)/2
l_db = (2*abl - ild)/2
pr = (10 ** (r_db/20))
pl = (10 ** (l_db/20))

den = (pr ** (lam * l) ) + ( pl ** (lam * l) )
rr = (pr ** lam) / den
rl = (pl ** lam) / den

# Scaled firing rates (for Poisson simulation)
r_right_scaled = r0_scaled * rr
r_left_scaled = r0_scaled * rl


# Unscaled firing rates (for DDM)
r_right = r0 * rr
r_left = r0 * rl


T = 20  # Max duration of a single trial (seconds)

# Gaussian noise parameter for spike timing jitter

print(f'Parameters:')
print(f'  corr = {c:.4f}')
print(f'  N = {N_right_and_left}')
print(f'  theta (unscaled) = {theta}')
print(f'  theta_scaled (for Poisson) = {theta_scaled}')
print(f'  corr_factor = {corr_factor}')
print(f'  r0 (unscaled) = {r0:.4f}')
print(f'  r0_scaled (for Poisson) = {r0_scaled:.4f}')
print(f'  r_right (unscaled, for DDM) = {r_right:.4f}')
print(f'  r_left (unscaled, for DDM) = {r_left:.4f}')
print(f'  r_right_scaled (for Poisson) = {r_right_scaled:.4f}')
print(f'  r_left_scaled (for Poisson) = {r_left_scaled:.4f}')
print(f'  N * r_right (unscaled) = {N_right_and_left * r_right:.4f}')
print(f'  N * r_left (unscaled) = {N_right_and_left * r_left:.4f}')
print(f'  N * r_right_scaled = {N_right_and_left * r_right_scaled:.4f}')
print(f'  N * r_left_scaled = {N_right_and_left * r_left_scaled:.4f}')
print(f'  exponential_noise_to_spk_time = {exponential_noise_to_spk_time:.4f} s')

# ===================================================================
# 2. SPIKE GENERATION FUNCTIONS
# ===================================================================

def generate_correlated_pool(N, c, r, T, rng):
    """
    Generates a pool of N correlated spike trains using the thinning method.
    Returns a dictionary where keys are neuron indices and values are spike time arrays.
    """
    pool_spikes = {}
    
    source_rate = r / c
    n_source_spikes = rng.poisson(source_rate * T)
    source_spk_timings = np.sort(rng.random(n_source_spikes) * T)
    
    for i in range(N):
        keep_spike_mask = rng.random(size=n_source_spikes) < c
        neuron_spikes = source_spk_timings[keep_spike_mask]
        
        # Add Exponential noise to spike timings (always positive delays)
        noise = rng.exponential(scale=exponential_noise_to_spk_time, size=len(neuron_spikes))
        neuron_spikes = neuron_spikes + noise
        # Ensure spike times remain within [0, T] and are sorted
        neuron_spikes = np.clip(neuron_spikes, 0, T)
        neuron_spikes = np.sort(neuron_spikes)
        
        pool_spikes[i] = neuron_spikes
    
    return pool_spikes

# ===================================================================
# 3. POISSON SPIKING MODEL
# ===================================================================

def run_single_trial(args):
    """Runs a single trial of the Poisson spiking simulation."""
    trial_idx, seed = args
    rng_local = np.random.default_rng(seed + trial_idx)

    # Generate all spike trains for this trial (using scaled rates)
    right_pool_spikes = generate_correlated_pool(N_right, c, r_right_scaled, T, rng_local)
    left_pool_spikes = generate_correlated_pool(N_left, c, r_left_scaled, T, rng_local)

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

    events_df = pd.DataFrame({'time': all_times, 'evidence_jump': all_evidence})
    evidence_events = events_df.groupby('time')['evidence_jump'].sum().reset_index()

    event_times = evidence_events['time'].values
    event_jumps = evidence_events['evidence_jump'].values

    # Run the decision process using the cumsum method
    dv_trajectory = np.cumsum(event_jumps)

    pos_crossings = np.where(dv_trajectory >= theta_scaled)[0]
    neg_crossings = np.where(dv_trajectory <= -theta_scaled)[0]

    first_pos_idx = pos_crossings[0] if pos_crossings.size > 0 else np.inf
    first_neg_idx = neg_crossings[0] if neg_crossings.size > 0 else np.inf

    # Determine outcome and store result
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

print(f'\n=== POISSON SPIKING MODEL SIMULATION ===')
start_time = time.time()
master_seed = 42
tasks = [(i, master_seed) for i in range(N_sim_rtd)]
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    results = list(tqdm(pool.imap(run_single_trial, tasks), total=N_sim_rtd, desc="Simulating Correlated Spikes"))
results_array = np.array(results)
end_time = time.time()
print(f"Correlated spikes simulation took: {end_time - start_time:.2f} seconds")

# Summary Statistics
decision_made_mask = ~np.isnan(results_array[:, 0])
mean_rt = np.mean(results_array[decision_made_mask, 0])
choices = results_array[:, 1]
prop_pos = np.sum(choices == 1) / N_sim_rtd
prop_neg = np.sum(choices == -1) / N_sim_rtd
prop_no_decision = np.sum(choices == 0) / N_sim_rtd

print(f"Mean Reaction Time (for decided trials): {mean_rt:.4f} s")
print(f"Proportion of Positive (+1) choices: {prop_pos:.2%}")
print(f"Proportion of Negative (-1) choices: {prop_neg:.2%}")
print(f"Proportion of No-Decision (0) trials: {prop_no_decision:.2%}")

# ===================================================================
# 4. CONTINUOUS DDM MODEL
# ===================================================================

N_neurons = N_right
mu = N_neurons * (r_right - r_left)
# corr_factor_ddm = 1 + ((N_neurons - 1) * c)
corr_factor_ddm = 1
sigma_sq = N_neurons * (r_right + r_left) * corr_factor_ddm
sigma = sigma_sq**0.5
theta_ddm = theta

print(f'\n=== DDM PARAMETERS ===')
print(f'mu = {mu}')
print(f'sigma = {sigma}')
print(f'theta_ddm = {theta_ddm}')

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

dt = 1e-4
dB = 1e-2
n_steps = int(T/dt)

# Parallel DDM simulation
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
# 5. EVIDENCE JUMP DISTRIBUTION ANALYSIS
# ===================================================================

def get_trial_binned_spike_differences(trial_idx, seed, dt_bin, T):
    """
    Runs a single trial, bins spike times into bins of size dt_bin,
    and returns the spike difference (R - L) for each time bin.
    """
    rng_local = np.random.default_rng(seed + trial_idx)
    
    # Generate all spike trains for this trial (using scaled rates)
    right_pool_spikes = generate_correlated_pool(N_right, c, r_right_scaled, T, rng_local)
    left_pool_spikes = generate_correlated_pool(N_left, c, r_left_scaled, T, rng_local)
    
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

# %%
# Time-binned analysis
dt_bin = 1e-3

print(f'\n=== EVIDENCE JUMP DISTRIBUTION ANALYSIS ===')
print(f'Collecting binned spike differences from {N_sim} trials...')
print(f'Time bin size: dt = {dt_bin*1000:.2f} ms')

all_bin_differences = []

for trial_idx in range(N_sim):
    bin_diffs = get_trial_binned_spike_differences(trial_idx, master_seed, dt_bin, T)
    all_bin_differences.extend(bin_diffs)

all_bin_differences = np.array(all_bin_differences)
print(f'Total number of time bins analyzed: {len(all_bin_differences)}')

# Analyze the distribution
min_diff = int(all_bin_differences.min())
max_diff = int(all_bin_differences.max())
hist_bins = np.arange(min_diff - 0.5, max_diff + 1.5, 1)  # Bin edges around integers
bin_diff_frequencies, bin_edges = np.histogram(all_bin_differences, bins=hist_bins)
bin_diff_values = np.arange(min_diff, max_diff + 1)  # Integer values

print(f'\nBinned spike difference distribution computed.')

# %%
# ===================================================================
# 6. COMBINED 2x1 PLOT
# ===================================================================

fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# --- TOP PLOT: Poisson and DDM RT Distributions ---
ax1 = axes[0]

max_T_plot = np.max(results_array[(results_array[:, 1] == 1), 0])
bins_rt = np.arange(0, max_T_plot, max_T_plot / 1000)

pos_rts_poisson = results_array[(results_array[:, 1] == 1), 0]
neg_rts_poisson = results_array[(results_array[:, 1] == -1), 0]
pos_hist_poisson, _ = np.histogram(pos_rts_poisson, bins=bins_rt, density=True)
neg_hist_poisson, _ = np.histogram(neg_rts_poisson, bins=bins_rt, density=True)

pos_rts_ddm = ddm_data[(ddm_data[:, 1] == 1), 0]
neg_rts_ddm = ddm_data[(ddm_data[:, 1] == -1), 0]
pos_hist_ddm, _ = np.histogram(pos_rts_ddm, bins=bins_rt, density=True)
neg_hist_ddm, _ = np.histogram(neg_rts_ddm, bins=bins_rt, density=True)

bin_centers = (bins_rt[:-1] + bins_rt[1:]) / 2

# Calculate fractions
poisson_up = len(pos_rts_poisson)
poisson_down = len(neg_rts_poisson)
poisson_frac_up = poisson_up / (poisson_up + poisson_down)
poisson_frac_down = poisson_down / (poisson_up + poisson_down)

ddm_up = len(pos_rts_ddm)
ddm_down = len(neg_rts_ddm)
ddm_frac_up = ddm_up / (ddm_up + ddm_down)
ddm_frac_down = ddm_down / (ddm_up + ddm_down)

ax1.plot(bin_centers, pos_hist_poisson * poisson_frac_up, label='Poisson - Positive Choice', 
         color='blue', linestyle='-', linewidth=2)
ax1.plot(bin_centers, -neg_hist_poisson * poisson_frac_down, label='Poisson - Negative Choice', 
         color='blue', linestyle='-', linewidth=2)
ax1.plot(bin_centers, pos_hist_ddm * ddm_frac_up, label='DDM - Positive Choice', 
         color='red', linestyle='-', linewidth=4, alpha=0.3)
ax1.plot(bin_centers, -neg_hist_ddm * ddm_frac_down, label='DDM - Negative Choice', 
         color='red', linestyle='-', linewidth=4, alpha=0.3)

ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax1.set_xlabel("Reaction Time (s)", fontsize=12)
ax1.set_ylabel("Density", fontsize=12)
ax1.set_title(
    f'Reaction Time Distributions: Poisson vs DDM (with Exponential spike timing jitter)\n'
    f'Poisson: θ={theta_scaled}, r_R={r_right_scaled:.4f}, r_L={r_left_scaled:.4f} (scaled) | '
    f'DDM: θ={theta}, r_R={r_right:.4f}, r_L={r_left:.4f} (unscaled)\n'
    f'N = {N_right_and_left}, corr = {c:.4f}, corr_factor = {corr_factor:.3f}, '
    f'mu = {mu:.2f}, sigma = {sigma:.2f}, λ_spike = {exponential_noise_to_spk_time:.4f}s',
    fontsize=13, fontweight='bold'
)
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)
ax1.set_xlim(0, 2)

# --- BOTTOM PLOT: Evidence Jump Distribution (Time-binned) ---
ax2 = axes[1]

ax2.bar(bin_diff_values, bin_diff_frequencies, width=0.8, alpha=0.7, 
        edgecolor='black', color='steelblue')
ax2.set_xlabel('Spike Difference (R - L) per Time Bin', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title(
    f'Evidence Jump Distribution (Time-binned at dt = {dt_bin*1000:.2f} ms)\n'
    f'{N_sim} trials, {len(all_bin_differences)} total bins',
    fontsize=13, fontweight='bold'
)
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticks(bin_diff_values[::max(1, len(bin_diff_values)//20)])  # Show subset of ticks
ax2.set_yscale('log')
plt.tight_layout()
plt.savefig('combined_poisson_ddm_jump_analysis.png', dpi=150, bbox_inches='tight')
print(f'\n=== PLOT SAVED ===')
print(f'Figure saved to: combined_poisson_ddm_jump_analysis.png')
plt.show()

# ===================================================================
# 7. SUMMARY STATISTICS
# ===================================================================

print(f'\n=== SUMMARY STATISTICS ===')
print(f'\n--- Poisson Model ---')
print(f'Mean RT: {mean_rt:.4f} s')
print(f'Positive choices: {prop_pos:.2%}')
print(f'Negative choices: {prop_neg:.2%}')
print(f'No decision: {prop_no_decision:.2%}')

print(f'\n--- Evidence Jump Distribution ---')
print(f'Mean jump value: {np.mean(all_bin_differences):.4f}')
print(f'Std dev of jumps: {np.std(all_bin_differences):.4f}')
print(f'Min jump: {np.min(all_bin_differences):.0f}')
print(f'Max jump: {np.max(all_bin_differences):.0f}')
print(f'Proportion of positive jumps: {np.sum(all_bin_differences > 0) / len(all_bin_differences):.4f}')
print(f'Proportion of negative jumps: {np.sum(all_bin_differences < 0) / len(all_bin_differences):.4f}')
print(f'Proportion of zero jumps: {np.sum(all_bin_differences == 0) / len(all_bin_differences):.4f}')

# %%
