# Compare Mean Reaction Times: Correlated vs Uncorrelated Poisson Models
# %%
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from collections import defaultdict

# ===================================================================
# PARAMETERS
# ===================================================================

# Fixed parameters
ABL = 20
ILD_values = [0, 2, 4, 8, 10, 12]
c_values = [ 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]  # Correlation values
theta = 2
N = 101
T = 20  # Max trial duration
N_sim = int(100e3)  # Number of trials per condition
exponential_noise_to_spk_time = 0  # Zero noise as requested

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def lr_rates_from_ABL_ILD(ABL, ILD, r0=13.3, lam=1.3, ell=0.9):
    """Calculate left and right firing rates from ABL and ILD."""
    r_db = (2*ABL + ILD)/2
    l_db = (2*ABL - ILD)/2
    pr = (10 ** (r_db/20))
    pl = (10 ** (l_db/20))
    den = (pr ** (lam * ell)) + (pl ** (lam * ell))
    rr = (pr ** lam) / den
    rl = (pl ** lam) / den
    mu1 = r0 * rr  # Right
    mu2 = r0 * rl  # Left
    return mu1, mu2

# ===================================================================
# UNCORRELATED POISSON MODEL
# ===================================================================

def simulate_fpt_uncorr(mu1, mu2, theta, rng=None):
    """
    Simulates one trial of the uncorrelated Skellam process dX = dN1 - dN2
    with absorbing boundaries at +/- theta. Returns the first passage time.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = 0
    time = 0.0
    total_rate = mu1 + mu2
    prob_up = mu1 / total_rate

    while abs(x) < theta:
        # exponential waiting time for next jump
        dt = rng.exponential(1.0 / total_rate)
        time += dt

        # decide jump direction
        if rng.random() < prob_up:
            x += 1
        else:
            x -= 1

    return time

def run_single_trial_uncorr(args):
    """Run single uncorrelated Poisson trial."""
    trial_idx, seed, mu1, mu2, theta = args
    rng = np.random.default_rng(seed + trial_idx)
    rt = simulate_fpt_uncorr(mu1, mu2, theta, rng=rng)
    return rt

# ===================================================================
# CORRELATED POISSON MODEL
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

def run_single_trial_corr(args):
    """Run single correlated Poisson trial."""
    trial_idx, seed, mu1, mu2, c, N_right, N_left, theta, T = args
    rng = np.random.default_rng(seed + trial_idx)
    
    # Calculate per-neuron rates (no scaling)
    r_right = mu1 / N_right
    r_left = mu2 / N_left
    
    # Generate spike trains
    right_pool_spikes = generate_correlated_pool(N_right, c, r_right, T, rng)
    left_pool_spikes = generate_correlated_pool(N_left, c, r_left, T, rng)
    
    # Consolidate spikes
    all_right_spikes = np.concatenate(list(right_pool_spikes.values()))
    all_left_spikes = np.concatenate(list(left_pool_spikes.values()))
    
    all_times = np.concatenate([all_right_spikes, all_left_spikes])
    all_evidence = np.concatenate([
        np.ones_like(all_right_spikes, dtype=int),
        -np.ones_like(all_left_spikes, dtype=int)
    ])
    
    if all_times.size == 0:
        return np.nan
    
    # Sort by time
    sort_idx = np.argsort(all_times)
    all_times = all_times[sort_idx]
    all_evidence = all_evidence[sort_idx]
    
    # Aggregate evidence at same time points
    import pandas as pd
    events_df = pd.DataFrame({'time': all_times, 'evidence_jump': all_evidence})
    evidence_events = events_df.groupby('time')['evidence_jump'].sum().reset_index()
    
    event_times = evidence_events['time'].values
    event_jumps = evidence_events['evidence_jump'].values
    
    # Run decision process (use theta directly, no scaling)
    dv_trajectory = np.cumsum(event_jumps)
    
    pos_crossings = np.where(dv_trajectory >= theta - 1e-10)[0]
    neg_crossings = np.where(dv_trajectory <= -theta + 1e-10 )[0]
    
    first_pos_idx = pos_crossings[0] if pos_crossings.size > 0 else np.inf
    first_neg_idx = neg_crossings[0] if neg_crossings.size > 0 else np.inf
    
    # Determine outcome
    if first_pos_idx < first_neg_idx:
        return event_times[first_pos_idx]
    elif first_neg_idx < first_pos_idx:
        return event_times[first_neg_idx]
    else:
        return np.nan

# ===================================================================
# MAIN SIMULATION
# ===================================================================
# %%
print("Starting simulations...")
print(f"Parameters: ABL={ABL}, theta={theta}, N={N}, N_sim={N_sim}")
print(f"ILD values: {ILD_values}")
print(f"Correlation values: {c_values}")
print()

# Storage for results
mean_rts_uncorr = []
mean_rts_corr = {c: [] for c in c_values}
# Store full RT distributions
all_rts_uncorr = {}
all_rts_corr = {c: {} for c in c_values}

# %%
# Loop over ILD values
for ILD in ILD_values:
    print(f"\n=== Processing ILD = {ILD} ===")
    
    # Get firing rates
    mu1, mu2 = lr_rates_from_ABL_ILD(ABL, ILD)
    print(f"mu1 (right) = {mu1:.4f}, mu2 (left) = {mu2:.4f}")
    
    # 1. Uncorrelated Poisson
    print(f"Simulating uncorrelated Poisson...")
    tasks = [(i, 42, mu1, mu2, theta) for i in range(N_sim)]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = list(pool.imap(run_single_trial_uncorr, tasks))
    rts_uncorr = np.array(results)
    mean_rt_uncorr = np.mean(rts_uncorr[~np.isnan(rts_uncorr)])
    mean_rts_uncorr.append(mean_rt_uncorr)
    all_rts_uncorr[ILD] = rts_uncorr
    print(f"  Uncorrelated mean RT = {mean_rt_uncorr:.4f}")
    
    # 2. Correlated Poisson for each c value
    for c in c_values:
        print(f"Simulating correlated Poisson with c={c}...")
        N_right = N
        N_left = N
        
        tasks = [(i, 42, mu1, mu2, c, N_right, N_left, theta, T) for i in range(N_sim)]
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = list(pool.imap(run_single_trial_corr, tasks))
        rts_corr = np.array(results)
        mean_rt_corr = np.mean(rts_corr[~np.isnan(rts_corr)])
        mean_rts_corr[c].append(mean_rt_corr)
        all_rts_corr[c][ILD] = rts_corr
        print(f"  c={c}, mean RT={mean_rt_corr:.4f}")

# %%
# Plot results
plt.figure(figsize=(10, 6))
plt.plot(ILD_values, mean_rts_uncorr, label='Uncorrelated', marker='o', linewidth=2, markersize=8)
for c in c_values:
    plt.plot(ILD_values, mean_rts_corr[c], label=f'Correlated c={c}', marker='o', linewidth=2, markersize=8)

plt.xlabel('ILD (dB)', fontsize=12)
plt.ylabel('Mean RT (s)', fontsize=12)
plt.title(f'Mean Reaction Time vs ILD\nABL={ABL}, θ={theta}, N={N}', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mean_rt_vs_ild_comparison.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'mean_rt_vs_ild_comparison.png'")
plt.show()

# %%
# Print summary table
print("\n=== SUMMARY TABLE ===")
print(f"{'ILD':>6} | {'Uncorr':>10} | " + " | ".join([f"c={c:4.2f}" for c in c_values]))
print("-" * (6 + 10 + 3 + len(c_values) * 13))
for i, ILD in enumerate(ILD_values):
    row = f"{ILD:>6} | {mean_rts_uncorr[i]:>10.4f} | "
    row += " | ".join([f"{mean_rts_corr[c][i]:>8.4f}" for c in c_values])
    print(row)

# %%
# Plot RT Distributions for each ILD value
bins = np.arange(0, 2 + 0.01, 0.01)

# Calculate optimal subplot layout
n_ilds = len(ILD_values)
n_cols = min(4, n_ilds)  # Max 4 columns
n_rows = int(np.ceil(n_ilds / n_cols))

# Create subplots for each ILD
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
if n_ilds > 1:
    axes = axes.flatten()
else:
    axes = [axes]  # Make it iterable for single subplot

# Generate colors for correlation values
n_correlations = len(c_values)
colors = plt.cm.tab10(np.linspace(0, 1, n_correlations))

for idx, ILD in enumerate(ILD_values):
    ax = axes[idx]
    
    # Plot uncorrelated
    rts_unc = all_rts_uncorr[ILD]
    rts_unc_clean = rts_unc[~np.isnan(rts_unc)]
    ax.hist(rts_unc_clean, bins=bins, alpha=0.5, label='Uncorrelated', density=True, histtype='step', linewidth=2)
    
    # Plot correlated for each c value
    for c_idx, c in enumerate(c_values):
        rts_cor = all_rts_corr[c][ILD]
        rts_cor_clean = rts_cor[~np.isnan(rts_cor)]
        ax.hist(rts_cor_clean, bins=bins, alpha=0.5, label=f'c={c}', density=True, 
                histtype='step', linewidth=2, color=colors[c_idx])
    
    # Get firing rates for title
    mu1, mu2 = lr_rates_from_ABL_ILD(ABL, ILD)
    ax.set_xlabel('RT (s)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'ILD={ILD} dB\nμ₁={mu1:.2f}, μ₂={mu2:.2f}', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.2)

# Hide unused subplots
for idx in range(n_ilds, n_rows * n_cols):
    axes[idx].set_visible(False)

plt.suptitle(f'RT Distributions: Uncorrelated vs Correlated (ABL={ABL}, θ={theta}, N={N})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rt_distributions_by_ILD.png', dpi=150, bbox_inches='tight')
print("\nRT distributions plot saved as 'rt_distributions_by_ILD.png'")
plt.show()

# %%
# Plot RTDs for one specific ILD to see details more clearly
# Choose middle ILD value, or first if only one
ILD_focus = ILD_values[len(ILD_values)//2] if len(ILD_values) > 1 else ILD_values[0]
print(f"\n=== Detailed view for ILD = {ILD_focus} ===")

plt.figure(figsize=(10, 6))
rts_unc = all_rts_uncorr[ILD_focus]
rts_unc_clean = rts_unc[~np.isnan(rts_unc)]
plt.hist(rts_unc_clean, bins=bins, alpha=0.6, label='Uncorrelated', density=True, histtype='step', linewidth=2)

# Use same colors as before for consistency
n_correlations = len(c_values)
colors = plt.cm.tab10(np.linspace(0, 1, n_correlations))

for c_idx, c in enumerate(c_values):
    rts_cor = all_rts_corr[c][ILD_focus]
    rts_cor_clean = rts_cor[~np.isnan(rts_cor)]
    plt.hist(rts_cor_clean, bins=bins, alpha=0.4, label=f'Correlated c={c}', 
             density=True, histtype='step', linewidth=2, color=colors[c_idx])
    print(f"c={c}: mean={np.mean(rts_cor_clean):.4f}, median={np.median(rts_cor_clean):.4f}, std={np.std(rts_cor_clean):.4f}")

mu1, mu2 = lr_rates_from_ABL_ILD(ABL, ILD_focus)
print(f"Uncorrelated: mean={np.mean(rts_unc_clean):.4f}, median={np.median(rts_unc_clean):.4f}, std={np.std(rts_unc_clean):.4f}")

plt.xlabel('RT (s)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title(f'RT Distributions for ILD={ILD_focus} dB (μ₁={mu1:.2f}, μ₂={mu2:.2f})\nABL={ABL}, θ={theta}, N={N}', 
          fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 2)
plt.tight_layout()
plt.savefig(f'rt_distribution_ILD_{ILD_focus}.png', dpi=150, bbox_inches='tight')
print(f"\nDetailed RT distribution plot saved as 'rt_distribution_ILD_{ILD_focus}.png'")
plt.show()
# %%
