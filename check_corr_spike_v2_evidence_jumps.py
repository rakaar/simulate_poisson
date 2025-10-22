# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing

# %%
# Parameters
c = 0.1
N_right_and_left = int((9/c) + 1)
N_right = N_right_and_left   # Number of neurons in the "right" evidence pool
N_left = N_right_and_left   # Number of neurons in the "left" evidence pool

r_left = 0.1  # Firing rate (Hz) for each neuron in the left pool
p_right_needed = 0.9
log_odds = np.log10(p_right_needed/(1-p_right_needed))
corr_factor = 1 + ((N_right_and_left - 1) * c)
r_right = r_left * 10**( (corr_factor/40) * log_odds )  # Using theta=40 from original code
T = 1    # Max duration of a single trial (seconds)

print(f'corr = {c}')
print(f'N = {N_right_and_left}')
print(f'corr_factor = {corr_factor}')
print(f'r_right = {r_right}')
print(f'r_left = {r_left}')

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

def run_single_trial(args):
    """Runs a single trial and returns evidence increments in 1ms bins."""
    trial_idx, seed = args
    rng_local = np.random.default_rng(seed + trial_idx)

    # Generate all spike trains for this trial
    right_pool_spikes = generate_correlated_pool(N_right, c, r_right, T, rng_local)
    left_pool_spikes = generate_correlated_pool(N_left, c, r_left, T, rng_local)

    # Consolidate all spike times
    all_right_spikes = np.concatenate(list(right_pool_spikes.values()))
    all_left_spikes = np.concatenate(list(left_pool_spikes.values()))

    # Define time bins (1ms)
    dt = 1e-5  # 1 millisecond
    bins = np.arange(0, T + dt, dt)

    # Use histogram to count spikes in each bin
    right_counts, _ = np.histogram(all_right_spikes, bins=bins)
    left_counts, _ = np.histogram(all_left_spikes, bins=bins)

    # Calculate evidence increment for each bin
    evidence_increments = right_counts - left_counts

    return evidence_increments

# Simulation parameters
N_sim = 5000  # Number of trials to simulate
master_seed = 42

# Run simulation
print(f"Running {N_sim} trials...")
tasks = [(i, master_seed) for i in range(N_sim)]

# For notebook usage, we'll run sequentially (can be parallelized if needed)
all_evidence_increments = []
for task in tqdm(tasks, desc="Simulating trials"):
    increments = run_single_trial(task)
    all_evidence_increments.extend(increments)

all_evidence_increments = np.array(all_evidence_increments)

# --- Analysis of Evidence Increments ---
print(f"Total 1ms bins analyzed: {len(all_evidence_increments)}")
print(f"Unique increment values: {np.unique(all_evidence_increments)}")

# Filter out zero-increment bins for clearer statistics on actual evidence events
non_zero_increments = all_evidence_increments[all_evidence_increments != 0]
print(f"\nNumber of bins with non-zero increments: {len(non_zero_increments)}")
# %%
# --- Plotting ---
plt.figure(figsize=(12, 6))

# Determine bin edges for the histogram
min_val = all_evidence_increments.min()
max_val = all_evidence_increments.max()
# bins = np.arange(min_val - 0.0001, max_val + 0.00001, 1)
bins = np.arange(-24, 27, 0.001)
plt.hist(all_evidence_increments, bins=bins, alpha=0.7, edgecolor='black', density=True, label='All Bins (including zeros)')

# Add labels and title
plt.xlabel('Evidence Increment in 1ms Bin (#Right Spikes - #Left Spikes)')
plt.ylabel('log Density')
plt.title(f'Distribution of Net Evidence Increments in 1ms Bins\n({N_sim} trials, corr={c}, N={N_right_and_left})')
plt.grid(True, alpha=0.3)
plt.yscale('log') # Use log scale to see rare events

# Add statistics to the plot
mean_inc = np.mean(all_evidence_increments)
std_inc = np.std(all_evidence_increments)
plt.axvline(mean_inc, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_inc:.3f}')
plt.axvline(N_right_and_left*c, color='green', label=f'N * c={N_right_and_left*c:.3f}')
plt.legend()

print(f"\nOverall Evidence Increment Statistics (including zero-increment bins):")
print(f"Mean: {mean_inc:.4f}")
print(f"Std: {std_inc:.4f}")
print(f"Min: {min_val}")
print(f"Max: {max_val}")

plt.tight_layout()
plt.show()
# %%
print(max_val)
