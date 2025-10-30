# Separated Poisson and DDM Simulation with RT Histogram Comparison
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
import time
from joblib import Parallel, delayed

# ===================================================================
# 1. SHARED PARAMETERS
# ===================================================================

N_sim_rtd = int(2 * 50e3)  # Number of trials for RTD (reaction time distribution)

N_right_and_left = 100 + 1
c = 0.01

corr_factor = 1 + (N_right_and_left - 1)*c
N_right = N_right_and_left
N_left = N_right_and_left   
if N_right_and_left < 1:
    raise ValueError("N_right_and_left must be greater than 1")

theta = 6
theta_scaled = theta * corr_factor

# Random animal's params
lam = 1.3
l = 0.9
Nr0 = 13.3 * 3
exponential_noise_to_spk_time = 0  # Scale parameter in seconds

r0 = Nr0/N_right_and_left
r0_scaled = r0 * corr_factor

abl = 20
ild = 0
r_db = (2*abl + ild)/2
l_db = (2*abl - ild)/2
pr = (10 ** (r_db/20))
pl = (10 ** (l_db/20))

den = (pr ** (lam * l)) + (pl ** (lam * l))
rr = (pr ** lam) / den
rl = (pl ** lam) / den

# Scaled firing rates (for Poisson simulation)
r_right_scaled = r0_scaled * rr
r_left_scaled = r0_scaled * rl

# Unscaled firing rates (for DDM)
r_right = r0 * rr
r_left = r0 * rl

T = 20  # Max duration of a single trial (seconds)

print(f'=== SHARED PARAMETERS ===')
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
print(f'  exponential_noise_to_spk_time = {exponential_noise_to_spk_time:.4f} s')

# ===================================================================
# 2. POISSON SIMULATION CODE
# ===================================================================

def generate_correlated_pool(N, c, r, T, rng):
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
            noise = rng.exponential(scale=exponential_noise_to_spk_time, size=len(neuron_spikes))
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
        noise = rng.exponential(scale=exponential_noise_to_spk_time, size=len(neuron_spikes))
        neuron_spikes = neuron_spikes + noise
        # Ensure spike times remain within [0, T] and are sorted
        neuron_spikes = np.clip(neuron_spikes, 0, T)
        neuron_spikes = np.sort(neuron_spikes)
        
        pool_spikes[i] = neuron_spikes
    
    return pool_spikes


def run_single_poisson_trial(args):
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
start_time_poisson = time.time()
master_seed = 42
tasks = [(i, master_seed) for i in range(N_sim_rtd)]
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    poisson_results = list(tqdm(pool.imap(run_single_poisson_trial, tasks), 
                                 total=N_sim_rtd, desc="Simulating Poisson"))
poisson_results_array = np.array(poisson_results)
end_time_poisson = time.time()

print(f"Poisson simulation took: {end_time_poisson - start_time_poisson:.2f} seconds")

# Summary Statistics for Poisson
decision_made_mask = ~np.isnan(poisson_results_array[:, 0])
poisson_mean_rt = np.mean(poisson_results_array[decision_made_mask, 0])
poisson_choices = poisson_results_array[:, 1]
poisson_prop_pos = np.sum(poisson_choices == 1) / N_sim_rtd
poisson_prop_neg = np.sum(poisson_choices == -1) / N_sim_rtd
poisson_prop_no_decision = np.sum(poisson_choices == 0) / N_sim_rtd

print(f"Mean Reaction Time: {poisson_mean_rt:.4f} s")
print(f"Proportion of Positive choices: {poisson_prop_pos:.2%}")
print(f"Proportion of Negative choices: {poisson_prop_neg:.2%}")
print(f"Proportion of No-Decision trials: {poisson_prop_no_decision:.2%}")

# ===================================================================
# 3. DDM SIMULATION CODE
# ===================================================================

N_neurons = N_right
mu = N_neurons * (r_right - r_left)
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

print(f'\n=== DDM SIMULATION ===')
start_time_ddm = time.time()
ddm_results = Parallel(n_jobs=-1)(
    delayed(simulate_single_ddm_trial)(i, mu, sigma, theta_ddm, dt, dB, n_steps) 
    for i in tqdm(range(N_sim_rtd), desc='Simulating DDM')
)
ddm_data = np.array(ddm_results)
end_time_ddm = time.time()

print(f"DDM simulation took: {end_time_ddm - start_time_ddm:.2f} seconds")

# Summary Statistics for DDM
ddm_decision_made_mask = ~np.isnan(ddm_data[:, 0])
ddm_mean_rt = np.mean(ddm_data[ddm_decision_made_mask, 0])
ddm_choices = ddm_data[:, 1]
ddm_prop_pos = np.sum(ddm_choices == 1) / N_sim_rtd
ddm_prop_neg = np.sum(ddm_choices == -1) / N_sim_rtd
ddm_prop_no_decision = np.sum(ddm_choices == 0) / N_sim_rtd

print(f"Mean Reaction Time: {ddm_mean_rt:.4f} s")
print(f"Proportion of Positive choices: {ddm_prop_pos:.2%}")
print(f"Proportion of Negative choices: {ddm_prop_neg:.2%}")
print(f"Proportion of No-Decision trials: {ddm_prop_no_decision:.2%}")
# %%
# ===================================================================
# 4. PLOT HISTOGRAMS OF REACTION TIMES (NOT SEPARATED BY CHOICE)
# ===================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Extract RTs (excluding no-decision trials)
poisson_rts = poisson_results_array[decision_made_mask, 0]
ddm_rts = ddm_data[ddm_decision_made_mask, 0]

# Determine common x-axis limits
max_rt = max(np.max(poisson_rts), np.max(ddm_rts))
bins = np.linspace(0, max_rt, 50)

# --- LEFT PLOT: Poisson RT Histogram ---
ax1 = axes[0]
ax1.hist(poisson_rts, bins=bins, alpha=0.7, color='blue', edgecolor='black', density=True, histtype='step')
ax1.set_xlabel('Reaction Time (s)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title(
    f'Poisson Model - Reaction Time Distribution\n'
    f'N={N_right_and_left}, θ={theta_scaled:.2f}, corr={c:.4f}\n'
    f'Mean RT: {poisson_mean_rt:.4f} s',
    fontsize=12, fontweight='bold'
)
ax1.grid(axis='y', alpha=0.3)
ax1.set_xlim(0, max_rt)

# --- RIGHT PLOT: DDM RT Histogram ---
ax2 = axes[1]
ax2.hist(ddm_rts, bins=bins, alpha=0.7, color='red', edgecolor='black', density=True, histtype='step')
ax2.set_xlabel('Reaction Time (s)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title(
    f'DDM - Reaction Time Distribution\n'
    f'μ={mu:.2f}, σ={sigma:.2f}, θ={theta_ddm:.2f}\n'
    f'Mean RT: {ddm_mean_rt:.4f} s',
    fontsize=12, fontweight='bold'
)
ax2.grid(axis='y', alpha=0.3)
ax2.set_xlim(0, max_rt)

plt.tight_layout()
plt.savefig('poisson_vs_ddm_rt_histograms.png', dpi=150, bbox_inches='tight')
print(f'\n=== PLOT SAVED ===')
print(f'Figure saved to: poisson_vs_ddm_rt_histograms.png')
plt.show()
# %%
# ===================================================================
# 5. OVERLAY PLOT (Optional: both distributions on same axes)
# ===================================================================

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
bins = np.arange(0,5,0.01)
ax.hist(poisson_rts, bins=bins, alpha=0.5, color='blue', edgecolor='black', 
        density=True, label=f'Poisson (Mean RT: {poisson_mean_rt:.4f}s)', histtype='step')
ax.hist(ddm_rts, bins=bins, alpha=0.5, color='red', edgecolor='black', 
        density=True, label=f'DDM (Mean RT: {ddm_mean_rt:.4f}s)', histtype='step')

ax.set_xlabel('Reaction Time (s)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(
    f'Poisson vs DDM - Reaction Time Distribution Comparison\n'
    f'N={N_right_and_left}, corr={c:.4f}, θ={theta:.2f}',
    fontsize=13, fontweight='bold'
)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_xlim(0, max_rt)

plt.tight_layout()
plt.savefig('poisson_vs_ddm_rt_overlay.png', dpi=150, bbox_inches='tight')
print(f'Overlay figure saved to: poisson_vs_ddm_rt_overlay.png')
plt.show()

# ===================================================================
# 6. SUMMARY STATISTICS
# ===================================================================

print(f'\n=== FINAL SUMMARY ===')
print(f'\n--- Poisson Model ---')
print(f'Total trials: {N_sim_rtd}')
print(f'Decided trials: {np.sum(decision_made_mask)}')
print(f'Mean RT: {poisson_mean_rt:.4f} s')
print(f'Positive choices: {poisson_prop_pos:.2%}')
print(f'Negative choices: {poisson_prop_neg:.2%}')
print(f'No decision: {poisson_prop_no_decision:.2%}')

print(f'\n--- DDM ---')
print(f'Total trials: {N_sim_rtd}')
print(f'Decided trials: {np.sum(ddm_decision_made_mask)}')
print(f'Mean RT: {ddm_mean_rt:.4f} s')
print(f'Positive choices: {ddm_prop_pos:.2%}')
print(f'Negative choices: {ddm_prop_neg:.2%}')
print(f'No decision: {ddm_prop_no_decision:.2%}')

# %%
# ===================================================================
# 6. PLOT SEPARATED BY CHOICE (like original plot)
# ===================================================================

fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Determine bins for RT histograms
max_T_plot = max(np.max(poisson_rts), np.max(ddm_rts))
bins_rt = np.arange(0, max_T_plot, max_T_plot / 1000)

# Separate by choice for Poisson
pos_rts_poisson = poisson_results_array[(poisson_results_array[:, 1] == 1), 0]
neg_rts_poisson = poisson_results_array[(poisson_results_array[:, 1] == -1), 0]
pos_hist_poisson, _ = np.histogram(pos_rts_poisson, bins=bins_rt, density=True)
neg_hist_poisson, _ = np.histogram(neg_rts_poisson, bins=bins_rt, density=True)

# Separate by choice for DDM
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

# Plot Poisson results
ax.plot(bin_centers, pos_hist_poisson * poisson_frac_up, 
        label='Poisson - Positive Choice', color='blue', linestyle='-', linewidth=2)
ax.plot(bin_centers, -neg_hist_poisson * poisson_frac_down, 
        label='Poisson - Negative Choice', color='blue', linestyle='-', linewidth=2)

# Plot DDM results
ax.plot(bin_centers, pos_hist_ddm * ddm_frac_up, 
        label='DDM - Positive Choice', color='red', linestyle='-', linewidth=4, alpha=0.3)
ax.plot(bin_centers, -neg_hist_ddm * ddm_frac_down, 
        label='DDM - Negative Choice', color='red', linestyle='-', linewidth=4, alpha=0.3)

ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel("Reaction Time (s)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title(
    f'Reaction Time Distributions Separated by Choice: Poisson vs DDM\n'
    f'Poisson: θ={theta_scaled:.2f}, r_R={r_right_scaled:.4f}, r_L={r_left_scaled:.4f} (scaled) | '
    f'DDM: θ={theta:.2f}, r_R={r_right:.4f}, r_L={r_left:.4f} (unscaled)\n'
    f'N = {N_right_and_left}, corr = {c:.4f}, corr_factor = {corr_factor:.3f}, '
    f'mu = {mu:.2f}, sigma = {sigma:.2f}',
    fontsize=13, fontweight='bold'
)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.set_xlim(0, 2)

plt.tight_layout()
plt.savefig('poisson_vs_ddm_rt_separated_by_choice.png', dpi=150, bbox_inches='tight')
print(f'\n=== CHOICE-SEPARATED PLOT SAVED ===')
print(f'Figure saved to: poisson_vs_ddm_rt_separated_by_choice.png')
plt.show()