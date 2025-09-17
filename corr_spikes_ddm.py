# Correlated spiking - Skellam and DDM, checking N,c,theta
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm # Optional: for a nice progress bar

# ===================================================================
# 1. PARAMETERS AND SETUP
# ===================================================================

# --- Simulation Control ---
N_sim = int(100e3)  # Number of trials to simulate

# --- Spike Train Parameters ---
c = 0.98   # Correlation coefficient (same for both pools)

N_right_and_left = int((3/c) + 1)
N_right = N_right_and_left   # Number of neurons in the "right" evidence pool
N_left = N_right_and_left   # Number of neurons in the "left" evidence pool
r_right = 10   # Firing rate (Hz) for each neuron in the right pool
r_left = 8     # Firing rate (Hz) for each neuron in the left pool
T = 5.0        # Max duration of a single trial (seconds)

# --- Decision Model Parameters ---
theta = 40    # The decision threshold (+theta for positive, -theta for negative)

# Use a random number generator for reproducibility
rng = np.random.default_rng(seed=42)


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


# ===================================================================
# 2. THE SIMULATION LOOP
# ===================================================================

# List to store the results (RT, choice) from each trial
results = []

# Using tqdm provides a convenient progress bar
for _ in tqdm(range(N_sim), desc="Simulating Trials"):
    
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
    
    # Handle the edge case of no spikes in a trial
    if all_times.size == 0:
        results.append((np.nan, 0)) # Store as NaN RT, 0 choice
        continue

    # Use pandas for efficient consolidation
    events_df = pd.DataFrame({'time': all_times, 'evidence_jump': all_evidence})
    evidence_events = events_df.groupby('time')['evidence_jump'].sum().reset_index()
    # No need to sort again, groupby preserves order of keys

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
        # Positive decision was first
        rt = event_times[first_pos_idx]
        choice = 1
        results.append((rt, choice))
    elif first_neg_idx < first_pos_idx:
        # Negative decision was first
        rt = event_times[first_neg_idx]
        choice = -1
        results.append((rt, choice))
    else:
        # No decision was made within the trial time T
        results.append((np.nan, 0)) # Use 0 for no-choice trials

# Convert the list of results into the final N_sim x 2 NumPy array
results_array = np.array(results)


# ===================================================================
# 3. ANALYSIS AND VISUALIZATION
# ===================================================================



# --- Summary Statistics ---
# Filter out no-decision trials for RT calculations
decision_made_mask = ~np.isnan(results_array[:, 0])
mean_rt = np.mean(results_array[decision_made_mask, 0])

# Calculate choice proportions
choices = results_array[:, 1]
prop_pos = np.sum(choices == 1) / N_sim
prop_neg = np.sum(choices == -1) / N_sim
prop_no_decision = np.sum(choices == 0) / N_sim


# %%
N_neurons = N_right
mu = N_neurons * (r_right - r_left)
corr_factor = 1 + ((N_neurons - 1) * c)
sigma_sq = N_neurons * (r_right + r_left) * corr_factor
sigma = sigma_sq**0.5
ddm_data = np.zeros((N_sim, 2))
theta_ddm = theta
dt = 1e-4
dB = 1e-2
evidence_steps = mu*dt + (sigma)*np.random.normal(0, dB, size=(N_sim, int(T/dt)))

for i in tqdm(range(N_sim), desc='Simulating DDM'):
    DV = np.cumsum(evidence_steps[i])
    pos_crossings = np.where(DV >= theta_ddm)[0]
    neg_crossings = np.where(DV <= -theta_ddm)[0]
    
    # Find the first crossing indices
    first_pos_idx = pos_crossings[0] if pos_crossings.size > 0 else np.inf
    first_neg_idx = neg_crossings[0] if neg_crossings.size > 0 else np.inf
    
    # Compare which boundary is hit first
    if first_pos_idx < first_neg_idx:
        # Positive boundary hit first
        ddm_data[i, 0] = first_pos_idx * dt
        ddm_data[i, 1] = 1
    elif first_neg_idx < first_pos_idx:
        # Negative boundary hit first
        ddm_data[i, 0] = first_neg_idx * dt
        ddm_data[i, 1] = -1
    else:
        # No decision was made
        ddm_data[i, 0] = np.nan
        ddm_data[i, 1] = 0

# %%
# --- Visualization: RT Histogram ---
plt.figure(figsize=(10, 6))
bins = np.arange(0,T,0.05 )

# Poisson RTs (blue)
pos_rts_poisson = results_array[(results_array[:, 1] == 1), 0]
neg_rts_poisson = results_array[(results_array[:, 1] == -1), 0]
pos_hist_poisson, _ = np.histogram(pos_rts_poisson, bins=bins, density=True)
neg_hist_poisson, _ = np.histogram(neg_rts_poisson, bins=bins, density=True)

# DDM RTs (red)
pos_rts_ddm = ddm_data[(ddm_data[:, 1] == 1), 0]
neg_rts_ddm = ddm_data[(ddm_data[:, 1] == -1), 0]
pos_hist_ddm, _ = np.histogram(pos_rts_ddm, bins=bins, density=True)
neg_hist_ddm, _ = np.histogram(neg_rts_ddm, bins=bins, density=True)

bin_centers = (bins[:-1] + bins[1:]) / 2

# Plot Poisson RTs in blue
plt.plot(bin_centers, pos_hist_poisson, label='Poisson - Positive Choice', color='blue', linestyle='-', linewidth=2)
plt.plot(bin_centers, -neg_hist_poisson, label='Poisson - Negative Choice', color='blue', linestyle='--', linewidth=2)

# Plot DDM RTs in red
plt.plot(bin_centers, pos_hist_ddm, label='DDM - Positive Choice', color='red', linestyle='-', linewidth=2)
plt.plot(bin_centers, -neg_hist_ddm, label='DDM - Negative Choice', color='red', linestyle='--', linewidth=2)

plt.title(f'corr = {c}, N neurons = {N_neurons}, theta = {theta}')
plt.xlabel("Reaction Time (s)")
print(f"\n--- Summary Statistics ---")
print(f"Mean Reaction Time (for decided trials): {mean_rt:.4f} s")
print(f"Proportion of Positive (+1) choices: {prop_pos:.2%}")
print(f"Proportion of Negative (-1) choices: {prop_neg:.2%}")
print(f"Proportion of No-Decision (0) trials: {prop_no_decision:.2%}")

plt.ylabel("Density")
plt.legend()
plt.show()
# %%
