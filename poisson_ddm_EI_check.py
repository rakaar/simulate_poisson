    # Poisson process spiking simulation with E-I populations
# %%
import numpy as np


def generate_spike_timings(rate, T, rng=None):
    """
    Generate spike timings from a homogeneous Poisson process.
    
    Parameters
    ----------
    rate : float
        Firing rate (spikes per second).
    T : float
        Duration of the trial (seconds).
    rng : np.random.Generator or None
        Random number generator.
    
    Returns
    -------
    spike_times : np.ndarray
        Array of spike times in [0, T].
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_spikes = rng.poisson(rate * T)
    spike_times = np.sort(rng.uniform(0, T, size=n_spikes))
    
    return spike_times


def simulate_ei_trial(rR_E, rR_I, rL_E, rL_I, alpha, theta, T, rng=None):
    """
    Simulate one trial of E-I decision process.
    
    Evidence contributions:
        Right Excitatory spike: +1
        Right Inhibitory spike: -alpha
        Left Excitatory spike:  -1
        Left Inhibitory spike:  +alpha
    
    DV = (R_exc - R_inh) - (L_exc - L_inh)
    
    Parameters
    ----------
    rR_E, rR_I : float
        Right side excitatory and inhibitory rates.
    rL_E, rL_I : float
        Left side excitatory and inhibitory rates.
    alpha : float
        Weight for inhibitory spikes.
    theta : float
        Decision boundary (symmetric at ±theta).
    T : float
        Max trial duration.
    rng : np.random.Generator or None
        Random number generator.
    
    Returns
    -------
    rt : float
        Reaction time (or np.nan if no decision).
    choice : int
        +1 (right) or -1 (left), or 0 if no decision.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate spike trains for all 4 populations
    spk_R_E = generate_spike_timings(rR_E, T, rng)
    spk_R_I = generate_spike_timings(rR_I, T, rng)
    spk_L_E = generate_spike_timings(rL_E, T, rng)
    spk_L_I = generate_spike_timings(rL_I, T, rng)
    
    # Create evidence array: (time, evidence_jump)
    times = np.concatenate([spk_R_E, spk_R_I, spk_L_E, spk_L_I])
    evidence = np.concatenate([
        np.ones(len(spk_R_E)) * 1,       # R_E: +1
        np.ones(len(spk_R_I)) * (-alpha), # R_I: -alpha
        np.ones(len(spk_L_E)) * (-1),     # L_E: -1
        np.ones(len(spk_L_I)) * alpha     # L_I: +alpha
    ])
    
    if len(times) == 0:
        return np.nan, 0
    
    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    evidence = evidence[sort_idx]
    
    # Accumulate evidence and check for boundary crossing
    dv = 0.0
    for i in range(len(times)):
        dv += evidence[i]
        if dv >= theta:
            return times[i], 1
        elif dv <= -theta:
            return times[i], -1
    
    return np.nan, 0


# %%
# Parameters
beta = 1/5
rR_E = 50
rR_I = rR_E*beta
rL_E = 45
rL_I = rL_E*beta

alpha = 0.2  # weight for inhibitory spikes
theta = 10   # decision boundary
T = 20.0     # max trial duration

# Run simulation
N_trials = int(100e3)
rng = np.random.default_rng(42)

rts = []
choices = []

for i in range(N_trials):
    rt, choice = simulate_ei_trial(rR_E, rR_I, rL_E, rL_I, alpha, theta, T, rng)
    rts.append(rt)
    choices.append(choice)

rts = np.array(rts)
choices = np.array(choices)

# Summary
decided_mask = ~np.isnan(rts)
n_decided = np.sum(decided_mask)
mean_rt = np.mean(rts[decided_mask]) if n_decided > 0 else np.nan
p_right = np.mean(choices[decided_mask] == 1) if n_decided > 0 else np.nan

print(f"=== E-I Model Results ===")
print(f"Rates: rR_E={rR_E}, rR_I={rR_I}, rL_E={rL_E}, rL_I={rL_I}")
print(f"Alpha (inh weight): {alpha}")
print(f"Theta (boundary): {theta}")
print(f"Trials: {N_trials}")
print(f"Decided: {n_decided}/{N_trials} ({100*n_decided/N_trials:.1f}%)")
print(f"Mean RT: {mean_rt:.4f} s")
print(f"P(right): {p_right:.4f}")


# %%
# Compare with DDM 
import matplotlib.pyplot as plt

def simulate_ddm_vectorized(mu, sigma, theta, T, N_trials, dt=1e-3, rng=None):
    """
    Vectorized DDM simulation: dX = mu*dt + sigma*dW
    with absorbing boundaries at ±theta.
    
    Returns (rts, choices) arrays.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_steps = int(T / dt)
    sqrt_dt = np.sqrt(dt)
    
    # Initialize
    x = np.zeros(N_trials)
    rts = np.full(N_trials, np.nan)
    choices = np.zeros(N_trials, dtype=int)
    alive = np.ones(N_trials, dtype=bool)
    
    for step in range(n_steps):
        if not np.any(alive):
            break
        
        # Update only alive trials
        idx = np.where(alive)[0]
        x[idx] += mu * dt + sigma * rng.normal(0, sqrt_dt, size=len(idx))
        
        # Check boundaries
        hit_upper = (x >= theta) & alive
        hit_lower = (x <= -theta) & alive
        
        if np.any(hit_upper):
            rts[hit_upper] = (step + 1) * dt
            choices[hit_upper] = 1
            alive[hit_upper] = False
        
        if np.any(hit_lower):
            rts[hit_lower] = (step + 1) * dt
            choices[hit_lower] = -1
            alive[hit_lower] = False
    
    return rts, choices


# DDM parameters from E-I rates
mu_ddm = (rR_E - alpha * rR_I) - (rL_E - alpha * rL_I)
sigma_sq_ddm = rR_E + (alpha**2) * rR_I + rL_E + (alpha**2) * rL_I
sigma_ddm = np.sqrt(sigma_sq_ddm)

print(f"\n=== DDM Parameters ===")
print(f"mu = {mu_ddm:.4f}")
print(f"sigma^2 = {sigma_sq_ddm:.4f}")
print(f"sigma = {sigma_ddm:.4f}")
print(f"theta = {theta}")

# Run DDM simulation (vectorized)
rng_ddm = np.random.default_rng(42)
rts_ddm, choices_ddm = simulate_ddm_vectorized(mu_ddm, sigma_ddm, theta, T, N_trials, rng=rng_ddm)

# DDM Summary
decided_mask_ddm = ~np.isnan(rts_ddm)
n_decided_ddm = np.sum(decided_mask_ddm)
mean_rt_ddm = np.mean(rts_ddm[decided_mask_ddm]) if n_decided_ddm > 0 else np.nan
p_right_ddm = np.mean(choices_ddm[decided_mask_ddm] == 1) if n_decided_ddm > 0 else np.nan

print(f"\n=== DDM Results ===")
print(f"Decided: {n_decided_ddm}/{N_trials} ({100*n_decided_ddm/N_trials:.1f}%)")
print(f"Mean RT: {mean_rt_ddm:.4f} s")
print(f"P(right): {p_right_ddm:.4f}")

# %%
# Plot RTD separated by choice
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
bins = np.arange(0, T, 0.05)


# Separate by choice - Poisson E-I
rts_ei_right = rts[choices == 1]
rts_ei_left = rts[choices == -1]
frac_right_ei = len(rts_ei_right) / n_decided
frac_left_ei = len(rts_ei_left) / n_decided

# Separate by choice - DDM
rts_ddm_right = rts_ddm[choices_ddm == 1]
rts_ddm_left = rts_ddm[choices_ddm == -1]
frac_right_ddm = len(rts_ddm_right) / n_decided_ddm
frac_left_ddm = len(rts_ddm_left) / n_decided_ddm

# Left plot: Right choices (area = fraction of right choices)
ax = axes[0]
ax.hist(rts_ei_right, bins=bins, density=True, weights=np.ones(len(rts_ei_right)) * frac_right_ei,
        alpha=0.5, label=f'Poisson E-I (frac={frac_right_ei:.3f})', histtype='step', linewidth=2)
ax.hist(rts_ddm_right, bins=bins, density=True, weights=np.ones(len(rts_ddm_right)) * frac_right_ddm,
        alpha=0.5, label=f'DDM (frac={frac_right_ddm:.3f})', histtype='step', linewidth=2)
ax.set_xlabel('RT (s)')
ax.set_ylabel('Density × P(choice)')
ax.set_title(f'Right Choices (+1)')
ax.legend()
ax.set_xlim(0,5)

# Right plot: Left choices (area = fraction of left choices)
ax = axes[1]
ax.hist(rts_ei_left, bins=bins, density=True, weights=np.ones(len(rts_ei_left)) * frac_left_ei,
        alpha=0.5, label=f'Poisson E-I (frac={frac_left_ei:.3f})', histtype='step', linewidth=2)
ax.hist(rts_ddm_left, bins=bins, density=True, weights=np.ones(len(rts_ddm_left)) * frac_left_ddm,
        alpha=0.5, label=f'DDM (frac={frac_left_ddm:.3f})', histtype='step', linewidth=2)
ax.set_xlabel('RT (s)')
ax.set_ylabel('Density × P(choice)')
ax.set_title(f'Left Choices (-1)')
ax.legend()
ax.set_xlim(0,5)

plt.suptitle(f'E-I Poisson vs DDM: RTD by Choice\n'
             f'rR_E={rR_E}, rR_I={rR_I}, rL_E={rL_E}, rL_I={rL_I}, α={alpha}, θ={theta} beta={beta}\n'
             f'DDM: μ={mu_ddm:.2f}, σ={sigma_ddm:.2f}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('ei_poisson_vs_ddm_rtd.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: ei_poisson_vs_ddm_rtd.png")
plt.show()

# %%