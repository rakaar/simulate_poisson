"""
Utility functions for DDM (Drift Diffusion Model) simulations
"""

import numpy as np


def simulate_single_ddm_trial(trial_idx, mu, sigma, theta_ddm, dt, dB, n_steps):
    """
    Simulate a single DDM trial.
    
    Parameters:
    -----------
    trial_idx : int
        Trial index (not used in computation, for compatibility with parallel processing)
    mu : float
        Drift rate
    sigma : float
        Diffusion coefficient (standard deviation)
    theta_ddm : float
        Decision threshold
    dt : float
        Time step size
    dB : float
        Brownian motion increment scale
    n_steps : int
        Maximum number of steps
    
    Returns:
    --------
    tuple
        (reaction_time, choice) where choice is 1 (upper boundary), -1 (lower boundary), 
        or 0 (no decision within time limit)
    """
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
