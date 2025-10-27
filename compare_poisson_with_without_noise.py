# Compare Poisson Spiking Model: With vs Without Noise
# This script compares RTDs and evidence jump distributions between
# Poisson models with zero noise and with exponential spike timing noise
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import multiprocessing
import time
from scipy.stats import entropy
from scipy.special import rel_entr
from joblib import Parallel, delayed

# ===================================================================
# 1. PARAMETERS
# ===================================================================

N_sim = 100  # Number of trials for jump distribution analysis
N_sim_rtd = int(2*50e3)  # Number of trials for RTD (reaction time distribution)

# Correlation parameters
c = 0.01
corr_factor = 20
N_right_and_left = round(((corr_factor - 1)/c) + 1)
N_right = N_right_and_left  
N_left = N_right_and_left   

if N_right_and_left < 1:
    raise ValueError("N_right_and_left must be greater than 1")

# Threshold
theta = 2.5
theta *= corr_factor

# Model fit parameters
lam = 1.3
l = 0.9
Nr0 = 13.3 

# Exponential noise parameter (for noisy condition)
exponential_noise_scale = 50 * 1e-3  # Scale parameter in seconds (1ms)

# Correlation and base firing rate
r0 = Nr0/N_right_and_left
r0 *= (corr_factor)

# Acoustic parameters
abl = 20
ild = 0
r_db = (2*abl + ild)/2
l_db = (2*abl - ild)/2
pr = (10 ** (r_db/20))
pl = (10 ** (l_db/20))

den = (pr ** (lam * l) ) + ( pl ** (lam * l) )
rr = (pr ** lam) / den
rl = (pl ** lam) / den
rr_r0 = r0 * rr
rl_r0 = r0 * rl

# Per neuron firing rate
r_left = rl_r0
r_right = rr_r0

T = 20  # Max duration of a single trial (seconds)

print(f'Parameters:')
print(f'  corr = {c:.4f}')
print(f'  N = {N_right_and_left}')
print(f'  theta = {theta}')
print(f'  corr_factor = {corr_factor}')
print(f'  r_right = {r_right:.4f}')
print(f'  r_left = {r_left:.4f}')
print(f'  N * r_right = {N_right_and_left * r_right:.4f}')
print(f'  N * r_left = {N_right_and_left * r_left:.4f}')
print(f'  exponential_noise_scale = {exponential_noise_scale:.4f} s')

# ===================================================================
# 2. SPIKE GENERATION FUNCTIONS
# ===================================================================

def generate_correlated_pool_no_noise(N, c, r, T, rng):
    """
    Generates a pool of N correlated spike trains using the thinning method.
    NO NOISE added to spike timings.
    Returns a dictionary where keys are neuron indices and values are spike time arrays.
    """
    pool_spikes = {}
    
    source_rate = r / c
    n_source_spikes = rng.poisson(source_rate * T)
    source_spk_timings = np.sort(rng.random(n_source_spikes) * T)
    
    for i in range(N):
        keep_spike_mask = rng.random(size=n_source_spikes) < c
        neuron_spikes = source_spk_timings[keep_spike_mask]
        pool_spikes[i] = neuron_spikes
    
    return pool_spikes


def generate_correlated_pool_with_noise(N, c, r, T, rng, noise_scale):
    """
    Generates a pool of N correlated spike trains using the thinning method.
    EXPONENTIAL NOISE added to spike timings.
    Returns a dictionary where keys are neuron indices and values are spike time arrays.
    """
    pool_spikes = {}
    
    source_rate = r / c
    n_source_spikes = rng.poisson(source_rate * T)
    source_spk_timings = np.sort(rng.random(n_source_spikes) * T)
    
    for i in range(N):
        keep_spike_mask = rng.random(size=n_source_spikes) < c
        neuron_spikes = source_spk_timings[keep_spike_mask]
        
        noise = rng.exponential(scale=noise_scale, size=len(neuron_spikes))
        neuron_spikes = neuron_spikes + noise
        neuron_spikes = np.clip(neuron_spikes, 0, T)
        neuron_spikes = np.sort(neuron_spikes)
        
        pool_spikes[i] = neuron_spikes
    
    return pool_spikes

# ===================================================================
# 3. POISSON SPIKING MODEL - NO NOISE
# ===================================================================

def run_single_trial_no_noise(args):
    """Runs a single trial of the Poisson spiking simulation WITHOUT noise."""
    trial_idx, seed = args
    rng_local = np.random.default_rng(seed + trial_idx)

    # Generate all spike trains for this trial (NO NOISE)
    right_pool_spikes = generate_correlated_pool_no_noise(N_right, c, r_right, T, rng_local)
    left_pool_spikes = generate_correlated_pool_no_noise(N_left, c, r_left, T, rng_local)

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

    pos_crossings = np.where(dv_trajectory >= theta)[0]
    neg_crossings = np.where(dv_trajectory <= -theta)[0]

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


# ===================================================================
# 4. POISSON SPIKING MODEL - WITH NOISE
# ===================================================================

def run_single_trial_with_noise(args):
    """Runs a single trial of the Poisson spiking simulation WITH exponential noise."""
    trial_idx, seed, noise_scale = args
    rng_local = np.random.default_rng(seed + trial_idx)

    # Generate all spike trains for this trial (WITH NOISE)
    right_pool_spikes = generate_correlated_pool_with_noise(N_right, c, r_right, T, rng_local, noise_scale)
    left_pool_spikes = generate_correlated_pool_with_noise(N_left, c, r_left, T, rng_local, noise_scale)

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

    pos_crossings = np.where(dv_trajectory >= theta)[0]
    neg_crossings = np.where(dv_trajectory <= -theta)[0]

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


# ===================================================================
# 5. RUN SIMULATIONS
# ===================================================================

master_seed = 42

print(f'\n=== RUNNING POISSON SIMULATIONS ===')

# Simulation 1: NO NOISE
print(f'\n--- Simulation 1: NO NOISE ---')
start_time_no_noise = time.time()
tasks_no_noise = [(i, master_seed) for i in range(N_sim_rtd)]
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    results_no_noise = list(tqdm(pool.imap(run_single_trial_no_noise, tasks_no_noise), 
                                  total=N_sim_rtd, desc="Simulating (No Noise)"))
results_no_noise_array = np.array(results_no_noise)
end_time_no_noise = time.time()
print(f"No-noise simulation took: {end_time_no_noise - start_time_no_noise:.2f} seconds")

# Summary for no noise
decision_made_mask_no_noise = ~np.isnan(results_no_noise_array[:, 0])
mean_rt_no_noise = np.mean(results_no_noise_array[decision_made_mask_no_noise, 0])
choices_no_noise = results_no_noise_array[:, 1]
prop_pos_no_noise = np.sum(choices_no_noise == 1) / N_sim_rtd
prop_neg_no_noise = np.sum(choices_no_noise == -1) / N_sim_rtd

print(f"Mean RT: {mean_rt_no_noise:.4f} s")
print(f"Positive choices: {prop_pos_no_noise:.2%}")
print(f"Negative choices: {prop_neg_no_noise:.2%}")

# Simulation 2: WITH NOISE
print(f'\n--- Simulation 2: WITH NOISE (scale={exponential_noise_scale}s) ---')
start_time_with_noise = time.time()
tasks_with_noise = [(i, master_seed, exponential_noise_scale) for i in range(N_sim_rtd)]
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    results_with_noise = list(tqdm(pool.imap(run_single_trial_with_noise, tasks_with_noise), 
                                    total=N_sim_rtd, desc="Simulating (With Noise)"))
results_with_noise_array = np.array(results_with_noise)
end_time_with_noise = time.time()
print(f"With-noise simulation took: {end_time_with_noise - start_time_with_noise:.2f} seconds")

# Summary for with noise
decision_made_mask_with_noise = ~np.isnan(results_with_noise_array[:, 0])
mean_rt_with_noise = np.mean(results_with_noise_array[decision_made_mask_with_noise, 0])
choices_with_noise = results_with_noise_array[:, 1]
prop_pos_with_noise = np.sum(choices_with_noise == 1) / N_sim_rtd
prop_neg_with_noise = np.sum(choices_with_noise == -1) / N_sim_rtd

print(f"Mean RT: {mean_rt_with_noise:.4f} s")
print(f"Positive choices: {prop_pos_with_noise:.2%}")
print(f"Negative choices: {prop_neg_with_noise:.2%}")

# ===================================================================
# 5b. CONTINUOUS DDM SIMULATION
# ===================================================================

N_neurons = N_right
mu = N_neurons * (r_right - r_left)
corr_factor_ddm = 1 + ((N_neurons - 1) * c)

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

# %%
# ===================================================================
# 6. CALCULATE RTD HISTOGRAMS AND KL DIVERGENCE
# ===================================================================

print(f'\n=== CALCULATING RTD HISTOGRAMS ===')

# Get positive choice RTs for both conditions
pos_rts_no_noise = results_no_noise_array[(results_no_noise_array[:, 1] == 1), 0]
pos_rts_with_noise = results_with_noise_array[(results_with_noise_array[:, 1] == 1), 0]
pos_rts_ddm = ddm_data[(ddm_data[:, 1] == 1), 0]

# Get negative choice RTs for both conditions
neg_rts_no_noise = results_no_noise_array[(results_no_noise_array[:, 1] == -1), 0]
neg_rts_with_noise = results_with_noise_array[(results_with_noise_array[:, 1] == -1), 0]
neg_rts_ddm = ddm_data[(ddm_data[:, 1] == -1), 0]

# Create bins for histograms
max_T_plot = max(np.max(pos_rts_no_noise), np.max(pos_rts_with_noise), np.max(pos_rts_ddm))
bins_rt = np.arange(0, max_T_plot, max_T_plot / 1000)
bin_centers = (bins_rt[:-1] + bins_rt[1:]) / 2

# Calculate histograms (density=True for probability density)
pos_hist_no_noise, _ = np.histogram(pos_rts_no_noise, bins=bins_rt, density=True)
neg_hist_no_noise, _ = np.histogram(neg_rts_no_noise, bins=bins_rt, density=True)
pos_hist_with_noise, _ = np.histogram(pos_rts_with_noise, bins=bins_rt, density=True)
neg_hist_with_noise, _ = np.histogram(neg_rts_with_noise, bins=bins_rt, density=True)
pos_hist_ddm, _ = np.histogram(pos_rts_ddm, bins=bins_rt, density=True)
neg_hist_ddm, _ = np.histogram(neg_rts_ddm, bins=bins_rt, density=True)

# Calculate fractions for plotting (to combine pos and neg into one plot)
no_noise_up = len(pos_rts_no_noise)
no_noise_down = len(neg_rts_no_noise)
no_noise_frac_up = no_noise_up / (no_noise_up + no_noise_down)
no_noise_frac_down = no_noise_down / (no_noise_up + no_noise_down)

with_noise_up = len(pos_rts_with_noise)
with_noise_down = len(neg_rts_with_noise)
with_noise_frac_up = with_noise_up / (with_noise_up + with_noise_down)
with_noise_frac_down = with_noise_down / (with_noise_up + with_noise_down)

ddm_up = len(pos_rts_ddm)
ddm_down = len(neg_rts_ddm)
ddm_frac_up = ddm_up / (ddm_up + ddm_down)
ddm_frac_down = ddm_down / (ddm_up + ddm_down)

# Calculate KL divergence for RTD
# We'll compute KL divergence for the combined distribution (pos + neg)
# Create combined distributions with small epsilon to avoid log(0)
eps = 1e-10

# Combined positive distributions (normalized)
combined_no_noise_pos = pos_hist_no_noise * no_noise_frac_up
combined_with_noise_pos = pos_hist_with_noise * with_noise_frac_up

# Normalize to ensure they sum to 1
combined_no_noise_pos = combined_no_noise_pos / (combined_no_noise_pos.sum() + eps)
combined_with_noise_pos = combined_with_noise_pos / (combined_with_noise_pos.sum() + eps)

# Add epsilon to avoid division by zero
combined_no_noise_pos += eps
combined_with_noise_pos += eps

# Calculate KL divergence: D_KL(no_noise || with_noise)
kl_div_rtd_pos = np.sum(combined_no_noise_pos * np.log(combined_no_noise_pos / combined_with_noise_pos))

print(f'KL Divergence for Positive RTD (No Noise || With Noise): {kl_div_rtd_pos:.6f}')

# %%
# ===================================================================
# 7. EVIDENCE JUMP DISTRIBUTION ANALYSIS
# ===================================================================

def get_trial_binned_spike_differences_no_noise(trial_idx, seed, dt_bin, T):
    """Get binned spike differences for a trial WITHOUT noise."""
    rng_local = np.random.default_rng(seed + trial_idx)
    
    # Generate all spike trains for this trial (NO NOISE)
    right_pool_spikes = generate_correlated_pool_no_noise(N_right, c, r_right, T, rng_local)
    left_pool_spikes = generate_correlated_pool_no_noise(N_left, c, r_left, T, rng_local)
    
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


def get_trial_binned_spike_differences_with_noise(trial_idx, seed, dt_bin, T, noise_scale):
    """Get binned spike differences for a trial WITH noise."""
    rng_local = np.random.default_rng(seed + trial_idx)
    
    # Generate all spike trains for this trial (WITH NOISE)
    right_pool_spikes = generate_correlated_pool_with_noise(N_right, c, r_right, T, rng_local, noise_scale)
    left_pool_spikes = generate_correlated_pool_with_noise(N_left, c, r_left, T, rng_local, noise_scale)
    
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
dt_bin = 1e-3  # 1 ms bins

print(f'\n=== EVIDENCE JUMP DISTRIBUTION ANALYSIS ===')
print(f'Collecting binned spike differences from {N_sim} trials...')
print(f'Time bin size: dt = {dt_bin*1000:.2f} ms')

all_bin_differences_no_noise = []
all_bin_differences_with_noise = []

for trial_idx in range(N_sim):
    bin_diffs_no_noise = get_trial_binned_spike_differences_no_noise(trial_idx, master_seed, dt_bin, T)
    all_bin_differences_no_noise.extend(bin_diffs_no_noise)
    
    bin_diffs_with_noise = get_trial_binned_spike_differences_with_noise(trial_idx, master_seed, dt_bin, T, exponential_noise_scale)
    all_bin_differences_with_noise.extend(bin_diffs_with_noise)

all_bin_differences_no_noise = np.array(all_bin_differences_no_noise)
all_bin_differences_with_noise = np.array(all_bin_differences_with_noise)

print(f'Total number of time bins analyzed: {len(all_bin_differences_no_noise)}')

# Analyze the distributions
min_diff = int(min(all_bin_differences_no_noise.min(), all_bin_differences_with_noise.min()))
max_diff = int(max(all_bin_differences_no_noise.max(), all_bin_differences_with_noise.max()))
hist_bins = np.arange(min_diff - 0.5, max_diff + 1.5, 1)  # Bin edges around integers

# Get histograms for both conditions
bin_diff_freq_no_noise, _ = np.histogram(all_bin_differences_no_noise, bins=hist_bins)
bin_diff_freq_with_noise, _ = np.histogram(all_bin_differences_with_noise, bins=hist_bins)
bin_diff_values = np.arange(min_diff, max_diff + 1)  # Integer values

# Calculate KL divergence for evidence jump distributions
# Normalize to get probability distributions
prob_no_noise = bin_diff_freq_no_noise / (bin_diff_freq_no_noise.sum() + eps)
prob_with_noise = bin_diff_freq_with_noise / (bin_diff_freq_with_noise.sum() + eps)

# Add epsilon to avoid division by zero
prob_no_noise += eps
prob_with_noise += eps

# Calculate KL divergence: D_KL(no_noise || with_noise)
kl_div_jump = np.sum(prob_no_noise * np.log(prob_no_noise / prob_with_noise))

print(f'KL Divergence for Evidence Jump Distribution (No Noise || With Noise): {kl_div_jump:.6f}')

# %%
# ===================================================================
# 8. CREATE 2x1 COMPARISON PLOT
# ===================================================================

fig, axes = plt.subplots(2, 1, figsize=(15, 12))

# --- TOP PLOT: RTD Comparison ---
ax1 = axes[0]

# Poisson without noise
ax1.plot(bin_centers, pos_hist_no_noise * no_noise_frac_up, 
         label='Poisson No Noise - Positive', color='blue', linestyle='-', linewidth=2)
ax1.plot(bin_centers, -neg_hist_no_noise * no_noise_frac_down, 
         label='Poisson No Noise - Negative', color='blue', linestyle='-', linewidth=2)

# Poisson with noise
ax1.plot(bin_centers, pos_hist_with_noise * with_noise_frac_up, 
         label='Poisson With Noise - Positive', color='red', linestyle='-', linewidth=2, alpha=0.7)
ax1.plot(bin_centers, -neg_hist_with_noise * with_noise_frac_down, 
         label='Poisson With Noise - Negative', color='red', linestyle='-', linewidth=2, alpha=0.7)

# DDM (continuous)
ax1.plot(bin_centers, pos_hist_ddm * ddm_frac_up, 
         label='DDM - Positive', color='green', linestyle='--', linewidth=3, alpha=0.5)
ax1.plot(bin_centers, -neg_hist_ddm * ddm_frac_down, 
         label='DDM - Negative', color='green', linestyle='--', linewidth=3, alpha=0.5)

ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax1.set_xlabel("Reaction Time (s)", fontsize=12)
ax1.set_ylabel("Density", fontsize=12)
ax1.set_title(
    f'Reaction Time Distributions: Poisson (No Noise vs With Noise) + DDM\n'
    f'KL Divergence (No Noise || With Noise): {kl_div_rtd_pos:.6f}\n'
    f'N = {N_right_and_left}, r_R = {r_right:.4f}, r_L = {r_left:.4f}, '
    f'corr = {c:.4f}, theta = {theta}, λ_noise = {exponential_noise_scale:.4f}s',
    fontsize=13, fontweight='bold'
)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.set_xlim(0, 2)

# --- BOTTOM PLOT: Evidence Jump Distribution Comparison ---
ax2 = axes[1]

# Plot both distributions using histtype='step' to avoid overlap
ax2.hist(all_bin_differences_no_noise, bins=hist_bins, histtype='step', 
         label='No Noise', color='blue', linewidth=2)
ax2.hist(all_bin_differences_with_noise, bins=hist_bins, histtype='step', 
         label='With Noise', color='red', linewidth=2)

ax2.set_xlabel('Spike Difference (R - L) per Time Bin', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title(
    f'Evidence Jump Distribution: No Noise vs With Noise\n'
    f'KL Divergence (No Noise || With Noise): {kl_div_jump:.6f}\n'
    f'Time bins: dt = {dt_bin*1000:.2f} ms, {N_sim} trials, {len(all_bin_differences_no_noise)} total bins',
    fontsize=13, fontweight='bold'
)
ax2.legend(loc='upper right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_yscale('log')
ax2.set_xticks(bin_diff_values[::max(1, len(bin_diff_values)//20)])  # Show subset of ticks

plt.tight_layout()
plt.savefig('compare_poisson_with_without_noise.png', dpi=150, bbox_inches='tight')
print(f'\n=== PLOT SAVED ===')
print(f'Figure saved to: compare_poisson_with_without_noise.png')
plt.show()

# ===================================================================
# 9. SUMMARY
# ===================================================================

print(f'\n=== FINAL SUMMARY ===')
print(f'\n--- No Noise Condition ---')
print(f'Mean RT: {mean_rt_no_noise:.4f} s')
print(f'Positive choices: {prop_pos_no_noise:.2%}')
print(f'Negative choices: {prop_neg_no_noise:.2%}')

print(f'\n--- With Noise Condition (λ={exponential_noise_scale}s) ---')
print(f'Mean RT: {mean_rt_with_noise:.4f} s')
print(f'Positive choices: {prop_pos_with_noise:.2%}')
print(f'Negative choices: {prop_neg_with_noise:.2%}')

print(f'\n--- KL Divergences ---')
print(f'RTD KL Divergence: {kl_div_rtd_pos:.6f}')
print(f'Evidence Jump KL Divergence: {kl_div_jump:.6f}')

print(f'\n=== ANALYSIS COMPLETE ===')
