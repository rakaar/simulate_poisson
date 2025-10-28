"""
Utility functions for Poisson correlated spike generation with noise
"""

import numpy as np
import pandas as pd


def generate_correlated_pool(N, c, r, T, rng, exponential_noise_scale=0):
    """
    Generates a pool of N correlated spike trains using the thinning method.
    
    Parameters:
    -----------
    N : int
        Number of neurons in the pool
    c : float
        Correlation coefficient (0 to 1)
    r : float
        Firing rate (Hz)
    T : float
        Total time duration (seconds)
    rng : np.random.Generator
        Random number generator
    exponential_noise_scale : float
        Scale parameter for exponential noise added to spike times (seconds)
        Default is 0 (no noise)
    
    Returns:
    --------
    dict
        Dictionary where keys are neuron indices and values are spike time arrays
    """
    pool_spikes = {}
    
    source_rate = r / c
    n_source_spikes = rng.poisson(source_rate * T)
    source_spk_timings = np.sort(rng.random(n_source_spikes) * T)
    
    for i in range(N):
        keep_spike_mask = rng.random(size=n_source_spikes) < c
        neuron_spikes = source_spk_timings[keep_spike_mask]
        
        # Add Exponential noise to spike timings (always positive delays)
        noise = rng.exponential(scale=exponential_noise_scale, size=len(neuron_spikes))
        neuron_spikes = neuron_spikes + noise
        # Ensure spike times remain within [0, T] and are sorted
        neuron_spikes = np.clip(neuron_spikes, 0, T)
        neuron_spikes = np.sort(neuron_spikes)
        
        pool_spikes[i] = neuron_spikes
    
    return pool_spikes


def run_single_trial(args, N_right, c, r_right_scaled, T, N_left, r_left_scaled, 
                     theta_scaled, exponential_noise_scale=0):
    """
    Runs a single trial of the Poisson spiking simulation.
    
    Parameters:
    -----------
    args : tuple
        (trial_idx, seed) for the trial
    N_right : int
        Number of neurons in right pool
    c : float
        Correlation coefficient
    r_right_scaled : float
        Scaled firing rate for right pool
    T : float
        Maximum trial duration
    N_left : int
        Number of neurons in left pool
    r_left_scaled : float
        Scaled firing rate for left pool
    theta_scaled : float
        Decision threshold (scaled)
    exponential_noise_scale : float
        Scale parameter for exponential noise (seconds)
    
    Returns:
    --------
    tuple
        (reaction_time, choice) where choice is 1 (right), -1 (left), or 0 (no decision)
    """
    trial_idx, seed = args
    rng_local = np.random.default_rng(seed + trial_idx)

    # Generate all spike trains for this trial (using scaled rates)
    right_pool_spikes = generate_correlated_pool(N_right, c, r_right_scaled, T, 
                                                  rng_local, exponential_noise_scale)
    left_pool_spikes = generate_correlated_pool(N_left, c, r_left_scaled, T, 
                                                 rng_local, exponential_noise_scale)

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


def get_trial_binned_spike_differences(trial_idx, seed, dt_bin, T, N_right, c, 
                                       r_right_scaled, N_left, r_left_scaled, 
                                       exponential_noise_scale=0):
    """
    Runs a single trial, bins spike times into bins of size dt_bin,
    and returns the spike difference (R - L) for each time bin.
    
    Parameters:
    -----------
    trial_idx : int
        Trial index
    seed : int
        Random seed
    dt_bin : float
        Time bin size (seconds)
    T : float
        Total time duration (seconds)
    N_right : int
        Number of neurons in right pool
    c : float
        Correlation coefficient
    r_right_scaled : float
        Scaled firing rate for right pool
    N_left : int
        Number of neurons in left pool
    r_left_scaled : float
        Scaled firing rate for left pool
    exponential_noise_scale : float
        Scale parameter for exponential noise (seconds)
    
    Returns:
    --------
    np.ndarray
        Spike differences (R - L) for each time bin
    """
    rng_local = np.random.default_rng(seed + trial_idx)
    
    # Generate all spike trains for this trial (using scaled rates)
    right_pool_spikes = generate_correlated_pool(N_right, c, r_right_scaled, T, 
                                                  rng_local, exponential_noise_scale)
    left_pool_spikes = generate_correlated_pool(N_left, c, r_left_scaled, T, 
                                                 rng_local, exponential_noise_scale)
    
    # Consolidate all spikes
    all_right_spikes = np.concatenate(list(right_pool_spikes.values()))
    all_left_spikes = np.concatenate(list(left_pool_spikes.values()))
    
    # Create time bins
    n_bins = int(np.ceil(T / dt_bin))
    time_bins = np.arange(0, T + dt_bin, dt_bin)
    
    # Count spikes in each bin
    right_counts, _ = np.histogram(all_right_spikes, bins=time_bins)
    left_counts, _ = np.histogram(all_left_spikes, bins=time_bins)
    
    # Compute difference (R - L) for each bin
    spike_differences = right_counts - left_counts
    
    return spike_differences
