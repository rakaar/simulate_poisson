import numpy as np
import pandas as pd

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

def run_poisson_trial(N, c, r_right_scaled, r_left_scaled, theta_scaled):
    N_right = N
    N_left = N
    T = 50

    """Runs a single trial of the Poisson spiking simulation."""
    # Use absolute value to ensure the seed is non-negative
    rng_local = np.random.default_rng(abs(int(np.random.normal()*int(1e5))))

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
        print('No spikes generated')
        return (np.nan, 0, np.nan)

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
        deviation = dv_trajectory[first_pos_idx] - theta_scaled
        return (rt, choice, deviation)
    elif first_neg_idx < first_pos_idx:
        rt = event_times[first_neg_idx]
        choice = -1
        deviation = abs(dv_trajectory[first_neg_idx] - (-theta_scaled))
        return (rt, choice, deviation)
    else:
        print('No crossings')
        return (np.nan, 0, np.nan)
