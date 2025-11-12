#!/usr/bin/env python
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import multiprocessing
import mgf_helper_utils as utils
from scipy import stats
import time

# %%
# Parameters from mgfs.py
N = 100
theta = 2
rho = 0.01  # In combined_poisson_ddm, this is referred to as 'c'

# ddm rates parameters
lam = 1.3
l = 0.9
Nr0 = 13.3 * 1
abl = 20
ild = 1
dt = 1e-4  # Time step for continuous DDM simulation
dB = 1e-2
# Calculate rates
r0 = Nr0/N
r_db = (2*abl + ild)/2
l_db = (2*abl - ild)/2
pr = (10 ** (r_db/20))
pl = (10 ** (l_db/20))

den = (pr ** (lam * l)) + (pl ** (lam * l))
rr = (pr ** lam) / den
rl = (pl ** lam) / den
r_right = r0 * rr
r_left = r0 * rl

print(f'Parameters:')
print(f'  N = {N}')
print(f'  theta = {theta}')
print(f'  rho/c = {rho}')
print(f'  r0 = {r0:.4f}')
print(f'  r_right = {r_right:.4f}')
print(f'  r_left = {r_left:.4f}')
print(f' mu = {N * (r_right - r_left)}')
print(f'sigma = {np.sqrt(N*(r_right + r_left))}')
# %% 
# Calculate theoretical predictions using utility functions
# Define a range of theta values to test
theta_range = np.arange(1, 11, 3)  # Larger step size for faster execution

# Store results
theoretical_results = {
    'theta': theta_range,
    'poisson_fc': [],
    'poisson_dt': [],
    'ddm_fc': [],
    'ddm_dt': []
}

# Calculate theoretical predictions for each theta
for theta_val in theta_range:
    # Use the high-level utility functions directly
    
    # Calculate FC and DT for Poisson using the utility function
    fc_poisson, dt_poisson = utils.poisson_fc_dt(
        N=N, 
        rho=rho, 
        theta=theta_val, 
        lam=lam, 
        l=l, 
        Nr0=Nr0, 
        abl=abl, 
        ild=ild, 
        dt=dt
    )
    
    theoretical_results['poisson_fc'].append(fc_poisson)
    theoretical_results['poisson_dt'].append(dt_poisson)
    
    # Calculate FC and DT for DDM using the utility function
    fc_ddm, dt_ddm = utils.ddm_fc_dt(
        lam=lam, 
        l=l, 
        Nr0=Nr0, 
        N=N, 
        abl=abl, 
        ild=ild, 
        theta=theta_val, 
        dt=dt
    )
    
    theoretical_results['ddm_fc'].append(fc_ddm)
    theoretical_results['ddm_dt'].append(dt_ddm)

# %%
# Simulation Functions

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
            pool_spikes[i] = neuron_spikes
        return pool_spikes
    
    # Standard correlated case (c > 0)
    source_rate = r / c
    n_source_spikes = rng.poisson(source_rate * T)
    source_spk_timings = np.sort(rng.random(n_source_spikes) * T)
    
    for i in range(N):
        keep_spike_mask = rng.random(size=n_source_spikes) < c
        neuron_spikes = source_spk_timings[keep_spike_mask]
        pool_spikes[i] = neuron_spikes
    
    return pool_spikes

def run_poisson_trial(trial_theta, N, r_right, r_left):
    """Runs a single trial of the Poisson spiking simulation."""
    rng_local = np.random.default_rng(int(np.random.normal()*1000))
    
    # Maximum simulation time
    T = 20  # seconds
    
    # Generate all spike trains for this trial
    right_pool_spikes = generate_correlated_pool(N, rho, r_right, T, rng_local)
    left_pool_spikes = generate_correlated_pool(N, rho, r_left, T, rng_local)

    # Consolidate spikes into a single stream of evidence events
    all_right_spikes = np.concatenate(list(right_pool_spikes.values()))
    all_left_spikes = np.concatenate(list(left_pool_spikes.values()))

    all_times = np.concatenate([all_right_spikes, all_left_spikes])
    all_evidence = np.concatenate([
        np.ones_like(all_right_spikes, dtype=int),
        -np.ones_like(all_left_spikes, dtype=int)
    ])

    if all_times.size == 0:
        return (np.nan, 0)  # No spikes generated

    # Sort by time
    sort_indices = np.argsort(all_times)
    all_times = all_times[sort_indices]
    all_evidence = all_evidence[sort_indices]
    
    # Group by time (sum evidence jumps that occur at the same time)
    events_df = pd.DataFrame({'time': all_times, 'evidence_jump': all_evidence})
    evidence_events = events_df.groupby('time')['evidence_jump'].sum().reset_index()
    
    event_times = evidence_events['time'].values
    event_jumps = evidence_events['evidence_jump'].values

    # Run the decision process using the cumsum method
    dv_trajectory = np.cumsum(event_jumps)
    
    # Check for threshold crossings
    pos_crossings = np.where(dv_trajectory >= trial_theta)[0]
    neg_crossings = np.where(dv_trajectory <= -trial_theta)[0]
    
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
        return (np.nan, 0)  # No decision made within time limit

def run_ddm_trial(trial_theta, N, r_right, r_left):
    
    # DDM parameters
    mu = N * (r_right - r_left)  # Drift rate
    sigma = np.sqrt(N * (r_right + r_left))  # Noise standard deviation
    
    # Simulation parameters
    max_steps = int(20 / dt)  # 20 seconds max
    
    # Initialize
    position = 0
    time = 0
    
    # Run the diffusion process
    for step in range(max_steps):
        # Update position
        position += mu * dt + sigma * np.random.normal(0, dB)
        time += dt
        
        # Check for threshold crossing
        if position >= trial_theta:
            return (time, 1)  # Hit upper bound
        elif position <= -trial_theta:
            return (time, -1)  # Hit lower bound
    
    # Time limit reached without decision
    raise ValueError("Time limit reached without decision")
    return (np.nan, 0)  

def simulate_for_theta(theta_value, model_type, n_trials=10000):
    """Simulate trials for a given theta value and model type."""
    master_seed = 42
    
    if model_type == 'poisson':
        trial_func = run_poisson_trial
    else:  # DDM
        trial_func = run_ddm_trial
    
    # Run simulations in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        tasks = [(theta_value, N, r_right, r_left) for i in range(n_trials)]
        results = list(tqdm(
            pool.starmap(trial_func, tasks), 
            total=n_trials, 
            desc=f"Simulating {model_type.upper()} with θ={theta_value}"
        ))
    
    # Convert to numpy array for analysis
    results_array = np.array(results)
    
    # Calculate metrics
    decision_made_mask = ~np.isnan(results_array[:, 0])
    if np.sum(decision_made_mask) == 0:
        return 0, np.nan  # No decisions made
        
    # FC = proportion of +1 choices among all decisions made
    choices = results_array[:, 1]
    fc = np.sum(choices == 1) / np.sum(decision_made_mask)
    
    # DT = mean reaction time for all trials where a decision was made
    dt = np.mean(results_array[decision_made_mask, 0])
    
    return fc, dt

# %%
# Run simulations for multiple theta values
n_trials_per_theta = int(1e3)  # Reduced for faster execution

# Dictionary to store simulation results
simulation_results = {
    'theta': theta_range,
    'poisson_fc_sim': [],
    'poisson_dt_sim': [],
    'ddm_fc_sim': [],
    'ddm_dt_sim': []
}

print("\n=== Running Simulations ===")

# Run Poisson simulations
print("\nRunning Poisson Model Simulations:")
for theta_val in theta_range:
    fc, dt = simulate_for_theta(theta_val, 'poisson', n_trials_per_theta)
    simulation_results['poisson_fc_sim'].append(fc)
    simulation_results['poisson_dt_sim'].append(dt)
    print(f"θ={theta_val}: FC={fc:.4f}, DT={dt:.4f}")

# Run DDM simulations  
print("\nRunning DDM Model Simulations:")
for theta_val in theta_range:
    fc, dt = simulate_for_theta(theta_val, 'ddm', n_trials_per_theta)
    simulation_results['ddm_fc_sim'].append(fc)
    simulation_results['ddm_dt_sim'].append(dt)
    print(f"θ={theta_val}: FC={fc:.4f}, DT={dt:.4f}")

# %%
# Create plots comparing theoretical and simulated results
plt.figure(figsize=(12, 10))

# Plot FC comparison - Poisson
plt.subplot(2, 2, 1)
plt.plot(theta_range, theoretical_results['poisson_fc'], 'b-', label='Theoretical')
plt.plot(theta_range, simulation_results['poisson_fc_sim'], 'bo', label='Simulated')
plt.xlabel('Threshold (θ)')
plt.ylabel('Accuracy (FC)')
plt.title('Poisson Model: Accuracy vs Threshold')
plt.grid(True)
plt.legend()

# Plot DT comparison - Poisson
plt.subplot(2, 2, 2)
plt.plot(theta_range, theoretical_results['poisson_dt'], 'r-', label='Theoretical')
plt.plot(theta_range, simulation_results['poisson_dt_sim'], 'ro', label='Simulated')
plt.xlabel('Threshold (θ)')
plt.ylabel('Mean RT (s)')
plt.title('Poisson Model: Mean RT vs Threshold')
plt.grid(True)
plt.legend()

# Plot FC comparison - DDM
plt.subplot(2, 2, 3)
plt.plot(theta_range, theoretical_results['ddm_fc'], 'b-', label='Theoretical')
plt.plot(theta_range, simulation_results['ddm_fc_sim'], 'bs', label='Simulated')
plt.xlabel('Threshold (θ)')
plt.ylabel('Accuracy (FC)')
plt.title('DDM: Accuracy vs Threshold')
plt.grid(True)
plt.legend()

# Plot DT comparison - DDM
plt.subplot(2, 2, 4)
plt.plot(theta_range, theoretical_results['ddm_dt'], 'r-', label='Theoretical')
plt.plot(theta_range, simulation_results['ddm_dt_sim'], 'rs', label='Simulated')
plt.xlabel('Threshold (θ)')
plt.ylabel('Mean RT (s)')
plt.title('DDM: Mean RT vs Threshold')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('theory_vs_simulation_comparison.png')
plt.show()

# %%
# Calculate error metrics between theoretical and simulated results
poisson_fc_error = np.abs(np.array(theoretical_results['poisson_fc']) - np.array(simulation_results['poisson_fc_sim']))
poisson_dt_error = np.abs(np.array(theoretical_results['poisson_dt']) - np.array(simulation_results['poisson_dt_sim']))
ddm_fc_error = np.abs(np.array(theoretical_results['ddm_fc']) - np.array(simulation_results['ddm_fc_sim']))
ddm_dt_error = np.abs(np.array(theoretical_results['ddm_dt']) - np.array(simulation_results['ddm_dt_sim']))

print("\n=== Error Analysis ===")
print(f"Poisson FC Mean Absolute Error: {np.mean(poisson_fc_error):.6f}")
print(f"Poisson DT Mean Absolute Error: {np.mean(poisson_dt_error):.6f}")
print(f"DDM FC Mean Absolute Error: {np.mean(ddm_fc_error):.6f}")
print(f"DDM DT Mean Absolute Error: {np.mean(ddm_dt_error):.6f}")

# Create a dataframe for all results
results_df = pd.DataFrame({
    'Theta': theta_range,
    'Poisson_FC_Theory': theoretical_results['poisson_fc'],
    'Poisson_FC_Sim': simulation_results['poisson_fc_sim'],
    'Poisson_DT_Theory': theoretical_results['poisson_dt'],
    'Poisson_DT_Sim': simulation_results['poisson_dt_sim'],
    'DDM_FC_Theory': theoretical_results['ddm_fc'],
    'DDM_FC_Sim': simulation_results['ddm_fc_sim'],
    'DDM_DT_Theory': theoretical_results['ddm_dt'],
    'DDM_DT_Sim': simulation_results['ddm_dt_sim']
})

print("\n=== Results Table ===")
print(results_df.round(4).to_string(index=False))

# %%
