# %%
# Test: Compare Empirical vs Theoretical Poisson Log Odds (no jitter)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from mgf_helper_utils import find_h0

# %%
# ===================================================================
# 1. PARAMETERS (same as jitter_log_odds.py)
# ===================================================================

N_trials = int(50e3)
N_neurons = 1000
rho = 1e-3

lam = 1.3
l = 0.9
Nr0_base = 13.3

abl = 20
theta = 20

ild_values = [1, 2, 4, 8, 16]

T = 50
rate_scalar_right = 1
rate_scalar_left = 1

# NO JITTER
jitter_scale = 0

print("=" * 60)
print("TEST: EMPIRICAL vs THEORETICAL POISSON LOG ODDS")
print("=" * 60)
print(f"N_neurons = {N_neurons}")
print(f"rho = {rho}")
print(f"theta = {theta}")
print(f"lam = {lam}, l = {l}")
print(f"Nr0_base = {Nr0_base}")
print(f"ABL = {abl}")
print(f"ILD values = {ild_values}")
print(f"N_trials = {N_trials}")
print(f"jitter_scale = {jitter_scale} (NO JITTER)")
print("=" * 60)

# %%
# ===================================================================
# 2. HELPER FUNCTIONS
# ===================================================================

def calculate_rates(ild):
    """Calculate firing rates for given ILD."""
    r0 = Nr0_base / N_neurons
    r_db = (2 * abl + ild) / 2
    l_db = (2 * abl - ild) / 2
    pr = 10 ** (r_db / 20)
    pl = 10 ** (l_db / 20)
    
    den = (pr ** (lam * l)) + (pl ** (lam * l))
    rr = (pr ** lam) / den
    rl = (pl ** lam) / den
    
    r_right = r0 * rr * rate_scalar_right
    r_left = r0 * rl * rate_scalar_left
    
    return r_right, r_left


def generate_correlated_pool(N, rho, r, T, rng):
    """
    Generates a pool of N correlated spike trains using the thinning method.
    NO JITTER version.
    """
    pool_spikes = {}
    
    # Standard correlated case (rho > 0)
    source_rate = r / rho
    n_source_spikes = rng.poisson(source_rate * T)
    source_spk_timings = np.sort(rng.random(n_source_spikes) * T)
    
    for i in range(N):
        keep_spike_mask = rng.random(size=n_source_spikes) < rho
        neuron_spikes = source_spk_timings[keep_spike_mask]
        pool_spikes[i] = neuron_spikes
    
    return pool_spikes


def run_single_trial(args):
    """Runs a single trial of the Poisson spiking simulation."""
    trial_idx, seed, r_right, r_left = args
    rng_local = np.random.default_rng(seed + trial_idx)

    # Generate spike trains for this trial
    right_pool_spikes = generate_correlated_pool(N_neurons, rho, r_right, T, rng_local)
    left_pool_spikes = generate_correlated_pool(N_neurons, rho, r_left, T, rng_local)

    # Consolidate spikes
    all_right_spikes = np.concatenate(list(right_pool_spikes.values()))
    all_left_spikes = np.concatenate(list(left_pool_spikes.values()))

    all_times = np.concatenate([all_right_spikes, all_left_spikes])
    all_evidence = np.concatenate([
        np.ones_like(all_right_spikes, dtype=int),
        -np.ones_like(all_left_spikes, dtype=int)
    ])

    if all_times.size == 0:
        return (np.nan, 0, np.nan)

    # Group simultaneous events
    events_df = pd.DataFrame({'time': all_times, 'evidence_jump': all_evidence})
    evidence_events = events_df.groupby('time')['evidence_jump'].sum().reset_index()

    event_times = evidence_events['time'].values
    event_jumps = evidence_events['evidence_jump'].values

    # Run decision process
    dv_trajectory = np.cumsum(event_jumps)

    pos_crossings = np.where(dv_trajectory >= theta)[0]
    neg_crossings = np.where(dv_trajectory <= -theta)[0]

    first_pos_idx = pos_crossings[0] if pos_crossings.size > 0 else np.inf
    first_neg_idx = neg_crossings[0] if neg_crossings.size > 0 else np.inf

    if first_pos_idx < first_neg_idx:
        bound_offset = dv_trajectory[first_pos_idx] - theta  # overshoot beyond +theta
        return (event_times[first_pos_idx], 1, bound_offset)
    elif first_neg_idx < first_pos_idx:
        bound_offset = (-dv_trajectory[first_neg_idx]) - theta  # overshoot beyond -theta
        return (event_times[first_neg_idx], -1, bound_offset)
    else:
        return (np.nan, 0, np.nan)


# %%
# ===================================================================
# 3. RUN SIMULATION
# ===================================================================
print("\nRunning simulations...")

results = {}
master_seed = 42

for ild in tqdm(ild_values, desc="Simulating ILDs"):
    r_right, r_left = calculate_rates(ild)
    
    tasks = [(i, master_seed, r_right, r_left) for i in range(N_trials)]
    
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        trial_results = pool.map(run_single_trial, tasks)
    
    results_array = np.array(trial_results)
    choices = results_array[:, 1]
    bound_offsets = results_array[:, 2]
    
    n_right = np.sum(choices == 1)
    n_left = np.sum(choices == -1)
    
    results[ild] = {
        'n_right': n_right,
        'n_left': n_left,
        'r_right': r_right,
        'r_left': r_left,
        'bound_offset_mean': np.nanmean(bound_offsets)
    }
    
    print(f"  ILD={ild}: n_right={n_right}, n_left={n_left}, P_right={n_right/(n_right+n_left):.4f}")

# %%
# ===================================================================
# 4. CALCULATE LOG ODDS
# ===================================================================

# Empirical log odds (Laplace smoothing)
empirical_log_odds = []
for ild in ild_values:
    extra_N_trials = 100
    n_right = results[ild]['n_right'] + extra_N_trials
    n_left = results[ild]['n_left'] + extra_N_trials
    P_right = n_right / (n_right + n_left)
    P_left = n_left / (n_right + n_left)
    log_odd = np.log(P_right / P_left)
    empirical_log_odds.append(log_odd)

empirical_log_odds = np.array(empirical_log_odds)

# Theoretical log odds using find_h0
theoretical_log_odds = []
for ild in ild_values:
    r_right, r_left = calculate_rates(ild)
    h0 = find_h0(r_right, r_left, N_neurons, rho)
    # Theoretical log odds is -h0 * theta
    theoretical_log_odds.append(-h0 * theta)

theoretical_log_odds = np.array(theoretical_log_odds)

print("\n" + "=" * 60)
print("LOG ODDS COMPARISON")
print("=" * 60)
print(f"{'ILD':<6} {'Empirical':<12} {'Theoretical':<12} {'Ratio':<10}")
print("-" * 40)
for i, ild in enumerate(ild_values):
    ratio = empirical_log_odds[i] / theoretical_log_odds[i] if theoretical_log_odds[i] != 0 else np.nan
    print(f"{ild:<6} {empirical_log_odds[i]:<12.4f} {theoretical_log_odds[i]:<12.4f} {ratio:<10.4f}")

# %%
# ===================================================================
# 5. PLOT COMPARISON
# ===================================================================

plt.figure(figsize=(10, 7))

plt.plot(ild_values, empirical_log_odds, 'o-', markersize=12, linewidth=2.5,
         color='blue', label='Empirical (simulated)')
plt.plot(ild_values, theoretical_log_odds, 's--', markersize=12, linewidth=2.5,
         color='red', label='Theoretical (-h₀×θ)')

plt.xlabel('ILD', fontsize=14, fontweight='bold')
plt.ylabel('Log Odds', fontsize=14, fontweight='bold')
plt.title(f'Empirical vs Theoretical Poisson Log Odds\n'
          f'N={N_neurons}, ρ={rho}, θ={theta}, ABL={abl}, N_trials={N_trials}',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(ild_values)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# %%
# ===================================================================
# 6. DEBUG: Check P_right values
# ===================================================================
print("\n" + "=" * 60)
print("DEBUG: CHOICE PROBABILITIES")
print("=" * 60)

print(f"{'ILD':<6} {'P_right (sim)':<15} {'P_right (theory)':<15}")
print("-" * 40)
for i, ild in enumerate(ild_values):
    # Simulated P_right
    n_right = results[ild]['n_right']
    n_left = results[ild]['n_left']
    P_right_sim = n_right / (n_right + n_left)
    
    # Theoretical P_right from h0
    r_right, r_left = calculate_rates(ild)
    h0 = find_h0(r_right, r_left, N_neurons, rho)
    # P_right = 1 / (1 + exp(h0 * theta))
    P_right_theory = 1 / (1 + np.exp(h0 * theta))
    
    print(f"{ild:<6} {P_right_sim:<15.4f} {P_right_theory:<15.4f}")

# %%
# ===================================================================
# 7. NORMALIZED COMPARISON (by highest ILD) - removes theta dependence
# ===================================================================

# Normalize by value at highest ILD
empirical_normalized = empirical_log_odds / empirical_log_odds[-1]
theoretical_normalized = theoretical_log_odds / theoretical_log_odds[-1]

print("\n" + "=" * 60)
print("NORMALIZED LOG ODDS (by ILD=16)")
print("=" * 60)
print(f"{'ILD':<6} {'Emp (norm)':<12} {'Theory (norm)':<12} {'Ratio':<10}")
print("-" * 40)
for i, ild in enumerate(ild_values):
    ratio = empirical_normalized[i] / theoretical_normalized[i] if theoretical_normalized[i] != 0 else np.nan
    print(f"{ild:<6} {empirical_normalized[i]:<12.4f} {theoretical_normalized[i]:<12.4f} {ratio:<10.4f}")

# Plot normalized comparison
plt.figure(figsize=(10, 7))

plt.plot(ild_values, empirical_normalized, 'o-', markersize=12, linewidth=2.5,
         color='blue', label='Empirical (simulated)')
plt.plot(ild_values, theoretical_normalized, 's--', markersize=12, linewidth=2.5,
         color='red', label='Theoretical (-h₀×θ)')

plt.xlabel('ILD', fontsize=14, fontweight='bold')
plt.ylabel('Normalized Log Odds (by ILD=16)', fontsize=14, fontweight='bold')
plt.title(f'Normalized Log Odds Comparison\n'
          f'N={N_neurons}, ρ={rho}, θ={theta}, ABL={abl}',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(ild_values)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# %%
# ===================================================================
# 8. P_RIGHT vs ILD (Empirical vs Theoretical)
# ===================================================================

# Empirical P_right from simulation
empirical_P_right = []
for ild in ild_values:
    n_right = results[ild]['n_right']
    n_left = results[ild]['n_left']
    P_right = n_right / (n_right + n_left)
    empirical_P_right.append(P_right)

empirical_P_right = np.array(empirical_P_right)

# Theoretical P_right from h0 with bound correction:
# theta_eff = theta + mean overshoot (bound_offset_mean) per ILD
# P_right = 1 / (1 + exp(h0 * theta_eff))
theoretical_P_right = []
for ild in ild_values:
    r_right, r_left = calculate_rates(ild)
    h0 = find_h0(r_right, r_left, N_neurons, rho)
    bound_offset_mean = results[ild].get('bound_offset_mean', 0.0)
    if np.isnan(bound_offset_mean):
        bound_offset_mean = 0.0
    theta_eff = theta + bound_offset_mean
    P_right = 1 / (1 + np.exp(h0 * theta_eff))
    theoretical_P_right.append(P_right)

theoretical_P_right = np.array(theoretical_P_right)

print("\n" + "=" * 60)
print("P_RIGHT vs ILD COMPARISON")
print("=" * 60)
print(f"{'ILD':<6} {'Emp P_right':<14} {'Theory P_right':<14} {'Diff':<10}")
print("-" * 44)
for i, ild in enumerate(ild_values):
    diff = empirical_P_right[i] - theoretical_P_right[i]
    print(f"{ild:<6} {empirical_P_right[i]:<14.4f} {theoretical_P_right[i]:<14.4f} {diff:<10.4f}")

# Plot P_right comparison
plt.figure(figsize=(10, 7))

plt.plot(ild_values, empirical_P_right, 'o-', markersize=12, linewidth=2.5,
         color='blue', label='Empirical (simulated)')
plt.plot(ild_values, theoretical_P_right, 's--', markersize=12, linewidth=2.5,
         color='red', label='Theoretical (1/(1+exp(h₀×θ)))')

plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (0.5)')

plt.xlabel('ILD', fontsize=14, fontweight='bold')
plt.ylabel('P(Right)', fontsize=14, fontweight='bold')
plt.title(f'P(Right) vs ILD: Empirical vs Theoretical\n'
          f'N={N_neurons}, ρ={rho}, θ={theta}, ABL={abl}, N_trials={N_trials}',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(ild_values)
plt.ylim([0.45, 1.02])
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# %%
print(f'normalied model: ')
for ild in [1,2,4]:
    rr,rl = calculate_rates(ild)
    print(f'ILD={ild}, rates={N_neurons*rr : .2f}, {N_neurons*rl :.2f} ')
    print(f'mu = {N_neurons * (rr - rl) :.2f}, sigma^2={N_neurons * (rr + rl) :.2f}')
    # print(f'theta/sigma^2 = {4/(N_neurons*(rr + rl))}')
print('--------------------')
print(f'non-normalized models')
non_norm_nr0 = 1/ ( 0.4 * 1e-3)
non_norm_lam = 0.1
for ild in [1,2,4]: 
    r0 = non_norm_nr0 / N_neurons
    r_db = (2 * abl + ild) / 2
    l_db = (2 * abl - ild) / 2
    pr = 10 ** (r_db / 20)
    pl = 10 ** (l_db / 20)
    
    # den = (pr ** (lam * l)) + (pl ** (lam * l)
    den = 1
    rr = (pr ** non_norm_lam) / den
    rl = (pl ** non_norm_lam) / den
    
    r_right = r0 * rr
    r_left = r0 * rl
    
    print(f'ILD={ild}, rates={N_neurons*r_right : .2f}, {N_neurons*r_left :.2f} ')
    print(f'mu = {N_neurons*(r_right - r_left) :.2f}, sigma^2={N_neurons*(r_right + r_left) :.2f}')
    # print(f'theta^2/sigma^2 = {1600/(N_neurons*(r_right + r_left))}')
    # %%
    x = np.arange(-16,16,0.1)
    plt.plot(x, np.tanh(x/17.37))
    plt.plot(x, np.tanh(2.13*x/17.37))
    plt.plot(x, np.tanh(0.13*x/17.37))
    plt.ylim(-2,2)

# %%
ild_r = np.arange(-16,16, 0.1)
th_ = 2
lam_ = 2.13
ratio = 10 ** (lam * ild_r/20)

sk_lo = 0.5 * ((ratio ** th_) - 1)
dm_lo = th_ * ((ratio - 1) / (ratio + 1))

# plt.plot(ild_r, sk_lo)
plt.plot(ild_r, dm_lo, label='small theta')
# plt.plot(ild_r, 30 * ((ratio - 1) / (ratio + 1)) , label='large theta')
plt.title
plt.legend()
plt.title('logodds')
plt.show()

# %%
r0 = 20
A = 1.676
k = 0.4
theta = 30
abl = 40
sig = lambda z: 1/(1 + np.exp(-z))
r_rates = r0 + A*sig(k * ild_r)
l_rates = r0 + A*sig(-k * ild_r)
r_rates *= abl
l_rates *= abl


plt.plot(ild_r, theta * np.log(r_rates / l_rates), ls = '--', color='r', label='skellam', lw=3)

mu = r_rates  - l_rates
sigma_sq = r_rates + l_rates
plt.plot(ild_r , 2*theta * (mu/sigma_sq), label='ddm')
plt.title(f'logodds, theta={theta}')
plt.legend()
plt.show()
# %%
# plot omega
def omega_abl_fn(abl):
    r_rates =(r0 + A*sig(k * ild_r)) * abl
    l_rates =(r0 + A*sig(-k * ild_r)) * abl
    sigma_sq = r_rates + l_rates

    return (sigma_sq) / (theta ** 2)

plt.plot(ild_r, omega_abl_fn(20), label='ABL = 20')
plt.plot(ild_r, omega_abl_fn(40), label='ABL = 40')
plt.plot(ild_r, omega_abl_fn(60), label='ABL = 60')
plt.title('omega NEW')
plt.legend()
plt.show()

# %%
# OMEGA ddm
def omega_ddm(abl):
    lam = 0.076
    T0 =  0.2 * 1e-3
    theta = 50
    t_avg = T0 * (theta**2) * (10 ** (-lam * abl/ 20)) / ( 2 * np.cosh(lam * ild_r/17.37))
    return 1/t_avg

plt.plot(ild_r, omega_ddm(20), label='ABL = 20')
plt.plot(ild_r, omega_ddm(40), label='ABL = 40')
plt.plot(ild_r, omega_ddm(60), label='ABL = 60')
plt.title('omega ddm')
plt.legend()
plt.show()
# %%
