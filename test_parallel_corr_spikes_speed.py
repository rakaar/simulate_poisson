import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import multiprocessing

# ===================================================================
# 1. PARAMETERS AND SETUP
# ===================================================================

# --- Simulation Control ---
N_sim = int(25e3)  # Number of trials to simulate

# --- Spike Train Parameters ---
N_right_and_left = int(1000)
c = 9 / N_right_and_left
N_right = N_right_and_left   # Number of neurons in the "right" eviden-ce pool
N_left = N_right_and_left    # Number of neurons in the "left" evidence pool
r_right = 4  # Firing rate (Hz) for each neuron in the right pool
r_left = 3   # Firing rate (Hz) for each neuron in the left pool
T = 1.0        # Max duration of a single trial (seconds)

# --- Decision Model Parameters ---
theta = 20    # The decision threshold (+theta for positive, -theta for negative)


def generate_correlated_pool(N, c, r, T, rng):
    """
    Generates a pool of N correlated spike trains.
    Returns a dictionary where keys are neuron indices and values are spike time arrays.
    """
    pool_spikes = {}
    lambda_common = c * r * T
    lambda_independent = (1 - c) * r * T

    n_common_spikes = rng.poisson(lambda_common)
    common_timings = np.sort(rng.random(n_common_spikes) * T)

    for i in range(N):
        n_independent_spikes = rng.poisson(lambda_independent)
        independent_timings = np.sort(rng.random(n_independent_spikes) * T)
        pool_spikes[i] = np.sort(np.concatenate([common_timings, independent_timings]))

    return pool_spikes

def run_single_trial(trial_idx, seed):
    """Runs a single trial of the simulation."""
    rng = np.random.default_rng(seed + trial_idx)

    # --- Step A: Generate all spike trains for this trial ---
    right_pool_spikes = generate_correlated_pool(N_right, c, r_right, T, rng)
    left_pool_spikes = generate_correlated_pool(N_left, c, r_left, T, rng)

    # --- Step B: Consolidate spikes into a single stream of evidence events ---
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

    # --- Step C: Run the decision process using the cumsum method ---
    dv_trajectory = np.cumsum(event_jumps)

    pos_crossings = np.where(dv_trajectory >= theta)[0]
    neg_crossings = np.where(dv_trajectory <= -theta)[0]

    first_pos_idx = pos_crossings[0] if pos_crossings.size > 0 else np.inf
    first_neg_idx = neg_crossings[0] if neg_crossings.size > 0 else np.inf

    # --- Step D: Determine outcome and store result ---
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

def run_serial_simulation(n_sim, seed):
    """Runs the simulation serially."""
    rng = np.random.default_rng(seed)
    results = []
    for _ in tqdm(range(n_sim), desc="Simulating Serially"):
        results.append(run_single_trial(_, seed)) # Pass index for compatibility
    return np.array(results)

def run_parallel_simulation(n_sim, seed):
    """Runs the simulation in parallel."""
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    tasks = [(i, seed) for i in range(n_sim)]
    results = list(tqdm(pool.starmap(run_single_trial, tasks), total=n_sim, desc="Simulating in Parallel"))
    pool.close()
    pool.join()
    return np.array(results)

if __name__ == '__main__':
    master_seed = 42

    # --- Run Serial Simulation ---
    start_time_serial = time.time()
    results_serial = run_serial_simulation(N_sim, master_seed)
    end_time_serial = time.time()
    print(f"\nSerial simulation took: {end_time_serial - start_time_serial:.2f} seconds")

    # --- Run Parallel Simulation ---
    start_time_parallel = time.time()
    results_parallel = run_parallel_simulation(N_sim, master_seed)
    end_time_parallel = time.time()
    print(f"Parallel simulation took: {end_time_parallel - start_time_parallel:.2f} seconds")

    # --- Compare Distributions ---
    plt.figure(figsize=(12, 7))
    bins = np.arange(0, T, 0.05)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Serial Results (blue)
    pos_rts_serial = results_serial[(results_serial[:, 1] == 1), 0]
    neg_rts_serial = results_serial[(results_serial[:, 1] == -1), 0]
    pos_hist_serial, _ = np.histogram(pos_rts_serial, bins=bins, density=True)
    neg_hist_serial, _ = np.histogram(neg_rts_serial, bins=bins, density=True)

    # Parallel Results (red, dashed)
    pos_rts_parallel = results_parallel[(results_parallel[:, 1] == 1), 0]
    neg_rts_parallel = results_parallel[(results_parallel[:, 1] == -1), 0]
    pos_hist_parallel, _ = np.histogram(pos_rts_parallel, bins=bins, density=True)
    neg_hist_parallel, _ = np.histogram(neg_rts_parallel, bins=bins, density=True)

    # --- Quantitative Comparison ---
    diff_pos = np.abs(pos_hist_serial - pos_hist_parallel)
    diff_neg = np.abs(neg_hist_serial - neg_hist_parallel)
    max_abs_diff = np.max(np.concatenate([diff_pos, diff_neg]))
    print(f"\nMaximum absolute difference between distributions: {max_abs_diff:.2e}")

    if max_abs_diff < 1e-9:
        print("Distributions are functionally identical.")
    else:
        print("Warning: Distributions show significant differences.")


    # --- Visual Comparison ---
    plt.plot(bin_centers, pos_hist_serial, 'b-', label='Serial - Positive')
    plt.plot(bin_centers, -neg_hist_serial, 'b--', label='Serial - Negative')
    plt.plot(bin_centers, pos_hist_parallel, 'r:', label='Parallel - Positive')
    plt.plot(bin_centers, -neg_hist_parallel, 'r-.', label='Parallel - Negative')

    plt.title('Comparison of Serial vs. Parallel Simulation RT Distributions')
    plt.xlabel('Reaction Time (s)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
