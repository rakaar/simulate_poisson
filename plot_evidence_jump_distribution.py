# Plot distribution of discrete evidence increments (jumps)
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# ===================================================================
# 1. PARAMETERS (same as corr_spikes_ddm_v2.py)
# ===================================================================

N_sim = 100  # Number of trials to analyze

N_right_and_left =100
N_right = N_right_and_left  
N_left = N_right_and_left   

# tweak
theta = 10
corr_factor = 10
theta *= corr_factor

# get from model fits
lam = 1.3
l = 0.9
# Nr0 = 13.3
Nr0 = 30

# correlation and base firing rate
c = (corr_factor - 1) / (N_right_and_left - 1)
r0 = Nr0/N_right_and_left
r0 *= (corr_factor)

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

# per neuron firing rate
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

# %%
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
        pool_spikes[i] = source_spk_timings[keep_spike_mask]
    
    return pool_spikes


def get_trial_evidence_jumps(trial_idx, seed):
    """Runs a single trial and returns the evidence jumps."""
    rng_local = np.random.default_rng(seed + trial_idx)
    
    # Generate all spike trains for this trial
    right_pool_spikes = generate_correlated_pool(N_right, c, r_right, T, rng_local)
    left_pool_spikes = generate_correlated_pool(N_left, c, r_left, T, rng_local)
    
    # Consolidate spikes into a single stream of evidence events
    all_right_spikes = np.concatenate(list(right_pool_spikes.values()))
    all_left_spikes = np.concatenate(list(left_pool_spikes.values()))
    
    all_times = np.concatenate([all_right_spikes, all_left_spikes])
    all_evidence = np.concatenate([
        np.ones_like(all_right_spikes, dtype=int),
        -np.ones_like(all_left_spikes, dtype=int)
    ])
    
    if all_times.size == 0:
        return np.array([])
    
    # Group by time and sum evidence jumps
    events_df = pd.DataFrame({'time': all_times, 'evidence_jump': all_evidence})
    evidence_events = events_df.groupby('time')['evidence_jump'].sum().reset_index()
    
    event_jumps = evidence_events['evidence_jump'].values
    
    return event_jumps

# %%
# Collect evidence jumps from all trials
print(f'\nCollecting evidence jumps from {N_sim} trials...')
all_jumps = []
master_seed = 42

for trial_idx in range(N_sim):
    jumps = get_trial_evidence_jumps(trial_idx, master_seed)
    all_jumps.extend(jumps)

all_jumps = np.array(all_jumps)
print(f'Total number of evidence jumps collected: {len(all_jumps)}')

# %%
# Analyze the distribution
jump_counts = Counter(all_jumps)
jump_values = sorted(jump_counts.keys())
jump_frequencies = [jump_counts[val] for val in jump_values]

print(f'\nEvidence jump distribution:')
for val, freq in zip(jump_values, jump_frequencies):
    pct = 100 * freq / len(all_jumps)
    # print(f'  Jump = {val:+3d}: {freq:6d} occurrences ({pct:5.2f}%)')

# %%
# Plot the distribution
fig, ax = plt.subplots(figsize=(20, 8))

# Plot 1: Bar plot of distribution
ax.bar(jump_values, jump_frequencies, width=0.8, alpha=0.7, edgecolor='black')
ax.set_xlabel('Evidence Jump Value', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(f'Distribution of Evidence Jumps\n({N_sim} trials, {len(all_jumps)} total jumps)\n'
              f'N = {N_right_and_left}, r_R = {r_right:.4f}, r_L = {r_left:.4f}, '
              f'N*r_R = {N_right_and_left * r_right:.2f}, N*r_L = {N_right_and_left * r_left:.2f}\n'
              f'corr_factor = {corr_factor:.3f}, corr = {c:.4f}, theta={theta :.3f}', 
              fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(jump_values)

# Add value labels on bars
# for val, freq in zip(jump_values, jump_frequencies):
#     ax.text(val, freq, str(freq), ha='center', va='bottom', fontsize=9)


plt.tight_layout()
plt.savefig('evidence_jump_distribution.png', dpi=150, bbox_inches='tight')
print(f'\nFigure saved to: evidence_jump_distribution.png')
plt.show()

# %%
# Additional statistics
print(f'\n--- Statistics ---')
print(f'Mean jump value: {np.mean(all_jumps):.4f}')
print(f'Std dev of jumps: {np.std(all_jumps):.4f}')
print(f'Min jump: {np.min(all_jumps):.0f}')
print(f'Max jump: {np.max(all_jumps):.0f}')
print(f'Proportion of positive jumps: {np.sum(all_jumps > 0) / len(all_jumps):.4f}')
print(f'Proportion of negative jumps: {np.sum(all_jumps < 0) / len(all_jumps):.4f}')
print(f'Proportion of zero jumps: {np.sum(all_jumps == 0) / len(all_jumps):.4f}')

# %%
# ===================================================================
# 2. TIME-BINNED ANALYSIS: Distribution of (R spikes - L spikes) per bin
# ===================================================================

dt = 0.0001  # Time bin size in seconds (0.1 ms)

def get_trial_binned_spike_differences(trial_idx, seed, dt, T):
    """
    Runs a single trial, bins spike times into bins of size dt,
    and returns the spike difference (R - L) for each time bin.
    """
    rng_local = np.random.default_rng(seed + trial_idx)
    
    # Generate all spike trains for this trial
    right_pool_spikes = generate_correlated_pool(N_right, c, r_right, T, rng_local)
    left_pool_spikes = generate_correlated_pool(N_left, c, r_left, T, rng_local)
    
    # Consolidate all spikes
    all_right_spikes = np.concatenate(list(right_pool_spikes.values()))
    all_left_spikes = np.concatenate(list(left_pool_spikes.values()))
    
    # Create time bins
    n_bins = int(np.ceil(T / dt))
    time_bins = np.arange(0, T + dt, dt)
    
    # Count spikes in each bin
    right_counts, _ = np.histogram(all_right_spikes, bins=time_bins)
    left_counts, _ = np.histogram(all_left_spikes, bins=time_bins)
    
    # Compute difference (R - L) for each bin
    spike_differences = right_counts - left_counts
    
    return spike_differences

# %%
# Collect binned spike differences from all trials
print(f'\n\n=== TIME-BINNED ANALYSIS ===')
print(f'Collecting binned spike differences from {N_sim} trials...')
print(f'Time bin size: dt = {dt*1000:.2f} ms')

all_bin_differences = []

for trial_idx in range(N_sim):
    bin_diffs = get_trial_binned_spike_differences(trial_idx, master_seed, dt, T)
    all_bin_differences.extend(bin_diffs)

all_bin_differences = np.array(all_bin_differences)
print(f'Total number of time bins analyzed: {len(all_bin_differences)}')

# %%
# Analyze the distribution
min_diff = int(all_bin_differences.min())
max_diff = int(all_bin_differences.max())
hist_bins = np.arange(min_diff - 0.5, max_diff + 1.5, 1)  # Bin edges around integers
bin_diff_frequencies, bin_edges = np.histogram(all_bin_differences, bins=hist_bins)
bin_diff_values = np.arange(min_diff, max_diff + 1)  # Integer values

print(f'\nBinned spike difference distribution:')
for val, freq in zip(bin_diff_values, bin_diff_frequencies):
    pct = 100 * freq / len(all_bin_differences)
    # print(f'  Diff = {val:+3d}: {freq:6d} bins ({pct:5.2f}%)')

# %%
# Plot the time-binned distribution
fig, ax = plt.subplots(figsize=(30, 8))

ax.bar(bin_diff_values, bin_diff_frequencies, width=0.8, alpha=0.7, 
       edgecolor='black', color='steelblue')
ax.set_xlabel('Spike Difference (R - L) per Time Bin', fontsize=12)
ax.set_ylabel('Count (log scale)', fontsize=12)
ax.set_yscale('log')
ax.set_title(f'Distribution of Binned Spike Differences\n'
              f'({N_sim} trials, {len(all_bin_differences)} total bins, dt = {dt*1000:.2f} ms)\n'
              f'N = {N_right_and_left}, r_R = {r_right:.4f}, r_L = {r_left:.4f}, '
              f'N*r_R = {N_right_and_left * r_right:.2f}, N*r_L = {N_right_and_left * r_left:.2f}\n'
              f'corr_factor = {corr_factor:.3f}, corr = {c:.4f}, theta={theta :.3f}', 
              fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(bin_diff_values)

plt.tight_layout()
plt.savefig('binned_spike_difference_distribution.png', dpi=150, bbox_inches='tight')
print(f'\nFigure saved to: binned_spike_difference_distribution.png')
plt.show()

