# Correlated spiking V2 - Skellam and DDM, checking N,c,theta
# %%
import numpy as np
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm # Optional: for a nice progress bar
import multiprocessing
import time

# ===================================================================
# 1. PARAMETERS AND SETUP
# ===================================================================


# --- Simulation Control ---
N_sim = int(1*50e3)  # Number of trials to simulate

# --- Spike Train Parameters ---
# r_left = (3000/N_right_and_left)/corr_factor# Firing rate (Hz) for each neuron in the left pool
# r_left = 0.28
# p_right_needed = 0.7
# log_odds = np.log10(p_right_needed/(1-p_right_needed))
# r_right = r_left * 10**( (corr_factor/theta) * log_odds )



N_right_and_left = 500
N_right = N_right_and_left  
N_left = N_right_and_left   

# tweak
theta = 45
corr_factor = 10

# get from model fits
lam = 2.13
l = 0.9
Nr0 = 40000

# correlation and base firing rate
c = (corr_factor - 1) / (N_right_and_left - 1)
r0 = Nr0/N_right_and_left
r0 /= (corr_factor**2)


abl = 60; ild = 0
p0 = 20e-6
r_db = (2*abl + ild)/2
l_db = (2*abl - ild)/2
pr = p0 * (10 ** (r_db/20))
pl = p0 * (10 ** (l_db/20))

den = (pr ** (lam * l) ) + ( pl ** (lam * l) )
rr = (pr ** lam) / den
rl = (pl ** lam) / den
rr_r0 = r0 * rr
rl_r0 = r0 * rl

# per neuron firing rate
r_left = rl_r0
r_right = rr_r0

p_right_needed = np.nan
log_odds = np.nan

T = 5    # Max duration of a single trial (seconds)

# example firing rates in normalization model:
# ABL = 40, ILD = 4
# single neuron
# r_R = 0.197
# r_L = 0.074
# base line firing rate:
# R0 = 7.7
# Nr_R, N r_L = 1.52, 0.57

# ABL = 20, ILD = 1
# (0.7050033372660773, 0.5516849960767949)
# ABL = 60, ILD = 16
# (3.9562292156607732, 0.07821345048684913)

# print the params
print(f'corr = {c}')
print(f'N = {N_right_and_left}')
print(f'theta = {theta}')
print(f'corr_factor = {corr_factor}')
print(f'r_right = {r_right}')
print(f'r_left = {r_left}')
print(f' N * r_right = {N_right_and_left * r_right}')
print(f' N * r_left = {N_right_and_left * r_left}')
theta_prime = theta/corr_factor
print(f'theta prime = {theta/corr_factor}')

if theta_prime < 2:
    raise ValueError("Theta prime must be greater than 2")
# Use a random number generator for reproducibility
rng = np.random.default_rng(seed=42)


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
    """Runs a single trial of the simulation."""
    trial_idx, seed = args
    rng_local = np.random.default_rng(seed + trial_idx)

    # --- Step A: Generate all spike trains for this trial ---
    right_pool_spikes = generate_correlated_pool(N_right, c, r_right, T, rng_local)
    left_pool_spikes = generate_correlated_pool(N_left, c, r_left, T, rng_local)

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

# ===================================================================
# 2. THE SIMULATION LOOP (PARALLEL)
# ===================================================================
start_time = time.time()
master_seed = 42
tasks = [(i, master_seed) for i in range(N_sim)]
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    results = list(tqdm(pool.imap(run_single_trial, tasks), total=N_sim, desc="Simulating Correlated Spikes"))
results_array = np.array(results)
end_time = time.time()
print(f"Correlated spikes simulation took: {end_time - start_time:.2f} seconds")

# --- Summary Statistics ---
decision_made_mask = ~np.isnan(results_array[:, 0])
mean_rt = np.mean(results_array[decision_made_mask, 0])
choices = results_array[:, 1]
prop_pos = np.sum(choices == 1) / N_sim
prop_neg = np.sum(choices == -1) / N_sim
prop_no_decision = np.sum(choices == 0) / N_sim
# %%
# --- DDM Simulation ---
N_neurons = N_right
mu = N_neurons * (r_right - r_left)
corr_factor = 1 + ((N_neurons - 1) * c)

sigma_sq = N_neurons * (r_right + r_left) * corr_factor
sigma = sigma_sq**0.5
theta_ddm = theta

print(f'mu = {mu}')
print(f'sigma = {sigma}')
print(f'theta = {theta}')

# sigma_sq = N_neurons * (r_right + r_left)
# sigma = sigma_sq**0.5
# theta_ddm = theta / corr_factor

ddm_data = np.zeros((N_sim, 2))
dt = 1e-4
dB = 1e-2
evidence_steps = mu*dt + (sigma)*np.random.normal(0, dB, size=(N_sim, int(T/dt)))

for i in tqdm(range(N_sim), desc='Simulating DDM'):
    DV = np.cumsum(evidence_steps[i])
    pos_crossings = np.where(DV >= theta_ddm)[0]
    neg_crossings = np.where(DV <= -theta_ddm)[0]
    first_pos_idx = pos_crossings[0] if pos_crossings.size > 0 else np.inf
    first_neg_idx = neg_crossings[0] if neg_crossings.size > 0 else np.inf
    if first_pos_idx < first_neg_idx:
        ddm_data[i, 0] = first_pos_idx * dt
        ddm_data[i, 1] = 1
    elif first_neg_idx < first_pos_idx:
        ddm_data[i, 0] = first_neg_idx * dt
        ddm_data[i, 1] = -1
    else:
        ddm_data[i, 0] = np.nan
        ddm_data[i, 1] = 0
# %%
# --- Visualization: RT Histogram ---
plt.figure(figsize=(12, 6))
bins = np.arange(0,T,0.05 )
pos_rts_poisson = results_array[(results_array[:, 1] == 1), 0]
neg_rts_poisson = results_array[(results_array[:, 1] == -1), 0]
pos_hist_poisson, _ = np.histogram(pos_rts_poisson, bins=bins, density=True)
neg_hist_poisson, _ = np.histogram(neg_rts_poisson, bins=bins, density=True)
pos_rts_ddm = ddm_data[(ddm_data[:, 1] == 1), 0]
neg_rts_ddm = ddm_data[(ddm_data[:, 1] == -1), 0]
pos_hist_ddm, _ = np.histogram(pos_rts_ddm, bins=bins, density=True)
neg_hist_ddm, _ = np.histogram(neg_rts_ddm, bins=bins, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# poisson
poisson_up = len(pos_rts_poisson)
poisson_down = len(neg_rts_poisson)
print(f'Poisson - Up: {poisson_up}, Down: {poisson_down}')
poisson_frac_up = poisson_up / (poisson_up + poisson_down)
poisson_frac_down = poisson_down / (poisson_up + poisson_down)

# ddm
ddm_up = len(pos_rts_ddm)
ddm_down = len(neg_rts_ddm)
ddm_frac_up = ddm_up / (ddm_up + ddm_down)
ddm_frac_down = ddm_down / (ddm_up + ddm_down)

plt.plot(bin_centers, pos_hist_poisson * poisson_frac_up, label='Poisson - Positive Choice', color='blue', linestyle='-', linewidth=2)
plt.plot(bin_centers, -neg_hist_poisson * poisson_frac_down, label='Poisson - Negative Choice', color='blue', linestyle='-', linewidth=2)
plt.plot(bin_centers, pos_hist_ddm * ddm_frac_up, label='DDM - Positive Choice', color='red', linestyle='-', linewidth=4, alpha=0.3)
plt.plot(bin_centers, -neg_hist_ddm * ddm_frac_down, label='DDM - Negative Choice', color='red', linestyle='-', linewidth=4, alpha=0.3)
plt.title(
    f'corr = {c:.3f}, N = {N_right_and_left}, r_right = {r_right:.2f}, r_left = {r_left:.2f}, T = {T:.2f}\n'
    f'corr_factor = {corr_factor:.2f}, N*r_right = {N_right*r_right:.2f}, N*r_left = {N_left*r_left:.2f}\n'
    f'p_right = {prop_pos:.3f}, p_left = {prop_neg:.3f}\n'
    f'Poisson: frac_up = {poisson_frac_up:.3f}, frac_down = {poisson_frac_down:.3f} | '
    f'DDM: frac_up = {ddm_frac_up:.3f}, frac_down = {ddm_frac_down:.3f} \n'
    f'theta = {theta}, theta_prime = {theta}/{corr_factor:.2f} = {theta/corr_factor:.2f}'
)
plt.axhline(0)
plt.xlabel("Reaction Time (s)")
plt.ylabel("Density")
plt.legend()

print(f"\n--- Summary Statistics ---")
print(f"Mean Reaction Time (for decided trials): {mean_rt:.4f} s")
print(f"Proportion of Positive (+1) choices: {prop_pos:.2%}")
print(f"Proportion of Negative (-1) choices: {prop_neg:.2%}")
print(f"Proportion of No-Decision (0) trials: {prop_no_decision:.2%}")

# Create directory if it doesn't exist
output_dir = "corr_N_RTDs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Generate timestamped filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"rt_distribution_{timestamp}.png"
filepath = os.path.join(output_dir, filename)

# Save the figure
plt.savefig(filepath)
print(f"Figure saved to {filepath}")
plt.show()
# %%
# --- Other Calculations ---
p0 = 20 * 1e-6 # Pa
ABL = 20 # dB
ILD = 1 # dB
slr = ABL + (ILD/2) # db
sll = ABL - (ILD/2) # db
pr = p0 * (10**(slr/20)) # pa
pl = p0 * (10**(sll/20)) # pa

## norm model
Nr0 = 1000/283 
lam = 4.6
ell = 0.9
rr = Nr0 * ((pr/p0)**lam)
rl = Nr0 * ((pl/p0)**lam)
norm_term = rr**(ell ) + rl**(ell )
print('\nnorm')
print(f' R: {rr/norm_term}')
print(f' L: {rl/norm_term}')

# vanilla
Nr0 = 1000/0.43
lam = 0.13
rr = Nr0 * ((pr/p0)**lam)
rl = Nr0 * ((pl/p0)**lam)
norm_term = 1
print('\nvanilla')
print(f' R: {rr/norm_term}')
print(f' L: {rl/norm_term}')

# %%
N_right_and_left * c