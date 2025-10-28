# Poisson Correlated vs DDM Uncorrelated Comparison Framework

This directory contains a systematic framework for comparing correlated Poisson spiking models with uncorrelated Drift Diffusion Models (DDM) across multiple parameter combinations.

## Overview

The simulations run two types of models in parallel:
1. **Poisson Model**: Correlated spike trains with configurable noise
2. **DDM Model**: Uncorrelated continuous diffusion process

Both models use shared base parameters from `params.py`, but vary three key parameters across a grid to study their effects on decision-making dynamics.

## Base Parameters (params.py)

All simulations use these shared parameters:

### Simulation Settings
- `N_sim = 100` - Number of trials for evidence jump distribution analysis
- `N_sim_rtd = 100,000` - Number of trials for reaction time distribution
- `T = 20` - Maximum trial duration (seconds)

### Model Parameters
- `theta = 2.5` - Decision threshold (unscaled, used for DDM)
- `lam = 1.3` - Psychometric curve parameter
- `l = 0.9` - Psychometric curve parameter
- `Nr0 = 13.3` - Base population firing rate
- `abl = 20` - Average binaural level (dB)
- `ild = 0` - Interaural level difference (dB)

### DDM-Specific
- `corr_factor_ddm = 1` - DDM uses uncorrelated noise

## Varied Parameters

The main analysis script `vary_c_and_corr_factor.py` systematically varies three parameters:

### 1. Correlation Coefficient (c)
**Values**: `[0.01, 0.05, 0.1, 0.2]`

Controls the correlation between neurons in the Poisson model. Higher values mean more correlation between spike trains.

### 2. Correlation Factor
**Values**: `[1.1, 2, 5, 10, 20]`

Scaling factor that affects:
- `theta_scaled = theta × corr_factor` (Poisson threshold)
- `r0_scaled = r0 × corr_factor` (Poisson firing rates)
- `N_right_and_left = round(((corr_factor - 1)/c) + 1)` (Number of neurons)

### 3. Exponential Noise (spike timing jitter)
**Values**: `[0, 1e-3, 2.5e-3, 5e-3]` seconds (i.e., 0ms, 1ms, 2.5ms, 5ms)

Adds exponential noise to spike timings in the Poisson model, simulating temporal jitter in neural responses.

### Total Combinations
**4 × 5 × 4 = 80 parameter combinations**

## Output Structure

### File Organization
Results are saved in the `results/` folder with individual pickle files for each parameter combination:

```
results/
├── c_0.01_corrfactor_1.1_noise_0.0ms.pkl
├── c_0.01_corrfactor_1.1_noise_1.0ms.pkl
├── c_0.01_corrfactor_1.1_noise_2.5ms.pkl
├── ...
└── c_0.2_corrfactor_20_noise_5.0ms.pkl
```

### Pickle File Structure

Each `.pkl` file contains a dictionary with four main keys:

```python
{
    'params': {
        # Parameter values for this combination
        'c': float,                              # Correlation coefficient
        'corr_factor': float,                    # Correlation factor
        'exponential_noise_to_spk_time': float,  # Noise scale (seconds)
        'N_right_and_left': int,                 # Number of neurons per pool
        'theta': float,                          # Unscaled threshold (DDM)
        'theta_scaled': float,                   # Scaled threshold (Poisson)
        'r0': float,                             # Unscaled base firing rate
        'r0_scaled': float,                      # Scaled base firing rate
        'r_right': float,                        # Right pool firing rate (unscaled)
        'r_left': float,                         # Left pool firing rate (unscaled)
        'r_right_scaled': float,                 # Right pool firing rate (scaled)
        'r_left_scaled': float,                  # Left pool firing rate (scaled)
        'T': float,                              # Max trial duration
        'N_sim': int,                            # Trials for evidence distribution
        'N_sim_rtd': int,                        # Trials for RT distribution
    },
    
    'poisson': {
        # Poisson model simulation results
        'results': np.ndarray,                   # Shape: (N_sim_rtd, 2)
                                                  # Column 0: reaction times
                                                  # Column 1: choices (1, -1, or 0)
        'mean_rt': float,                        # Mean RT for decided trials
        'prop_pos': float,                       # Proportion of +1 choices
        'prop_neg': float,                       # Proportion of -1 choices
        'prop_no_decision': float,               # Proportion of no-decision trials
        'simulation_time': float,                # Time taken (seconds)
    },
    
    'ddm': {
        # DDM simulation results
        'params': {
            'mu': float,                         # Drift rate
            'sigma': float,                      # Diffusion coefficient
            'theta_ddm': float,                  # Decision threshold
            'dt': float,                         # Time step (1e-4)
            'dB': float,                         # Brownian increment (1e-2)
        },
        'results': np.ndarray,                   # Shape: (N_sim_rtd, 2)
                                                  # Column 0: reaction times
                                                  # Column 1: choices (1, -1, or 0)
        'simulation_time': float,                # Time taken (seconds)
    },
    
    'evidence_distribution': {
        # Evidence jump distribution analysis
        'dt_bin': float,                         # Time bin size (1e-3 seconds)
        'all_bin_differences': np.ndarray,       # All binned spike differences (R-L)
        'bin_diff_values': np.ndarray,           # Unique difference values
        'bin_diff_frequencies': np.ndarray,      # Frequency of each difference
        'min_diff': int,                         # Minimum spike difference
        'max_diff': int,                         # Maximum spike difference
    }
}
```

## Usage Example

### Loading and Analyzing Results

```python
import pickle
import numpy as np

# Load a specific parameter combination
with open('results/c_0.01_corrfactor_5_noise_1.0ms.pkl', 'rb') as f:
    data = pickle.load(f)

# Access parameters
c = data['params']['c']
corr_factor = data['params']['corr_factor']
noise = data['params']['exponential_noise_to_spk_time']

# Access Poisson results
poisson_rts = data['poisson']['results'][:, 0]  # Reaction times
poisson_choices = data['poisson']['results'][:, 1]  # Choices

# Access DDM results
ddm_rts = data['ddm']['results'][:, 0]
ddm_choices = data['ddm']['results'][:, 1]

# Access evidence distribution
spike_diffs = data['evidence_distribution']['all_bin_differences']
```

### Plotting RT Distributions

```python
import matplotlib.pyplot as plt

# Filter decided trials
poisson_decided = poisson_rts[~np.isnan(poisson_rts)]
ddm_decided = ddm_rts[~np.isnan(ddm_rts)]

# Plot
plt.figure(figsize=(10, 6))
plt.hist(poisson_decided, bins=50, alpha=0.5, label='Poisson', density=True)
plt.hist(ddm_decided, bins=50, alpha=0.5, label='DDM', density=True)
plt.xlabel('Reaction Time (s)')
plt.ylabel('Density')
plt.legend()
plt.title(f'c={c}, corr_factor={corr_factor}, noise={noise*1000:.1f}ms')
plt.show()
```

## Running Simulations

### Quick Test Run
```bash
cd /home/ragha/code/simulate_poisson/poissonCorr_vs_DDMunCorr
../.venv/bin/python vary_c_and_corr_factor.py
```

### Production Run (in tmux)
```bash
tmux new -s poisson_sim
cd /home/ragha/code/simulate_poisson/poissonCorr_vs_DDMunCorr
../.venv/bin/python vary_c_and_corr_factor.py
# Detach: Ctrl+b then d
```

**Note**: Full simulation with 80 combinations and 100,000 trials each will take several hours.

## File Descriptions

### Core Utility Files
- **`params.py`** - Shared base parameters for all simulations
- **`ddm_utils.py`** - DDM simulation functions (`simulate_single_ddm_trial`)
- **`poisson_spike_corr_with_noise_utils.py`** - Poisson spike generation utilities:
  - `generate_correlated_pool()` - Generate correlated spike trains
  - `run_single_trial()` - Run single Poisson trial with decision boundary
  - `get_trial_binned_spike_differences()` - Bin spikes for evidence distribution

### Simulation Scripts
- **`ddm_uncorr_sim.py`** - Standalone DDM simulation (single parameter set)
- **`vary_c_and_corr_factor.py`** - Main script that varies all three parameters and saves results

## Key Differences: Poisson vs DDM

| Aspect | Poisson Model | DDM Model |
|--------|---------------|-----------|
| **Noise** | Correlated (via shared source) | Uncorrelated |
| **Threshold** | `theta_scaled` (scaled by corr_factor) | `theta` (unscaled) |
| **Firing Rates** | `r_right_scaled`, `r_left_scaled` | `r_right`, `r_left` (unscaled) |
| **Evidence** | Discrete spike events | Continuous drift + diffusion |
| **Neurons** | N_right_and_left per pool | Combined into drift rate μ |
| **Spike Timing** | Can add exponential jitter | N/A |

## Analysis Goals

This framework enables studying:
1. How correlation affects decision-making dynamics
2. The relationship between correlated Poisson models and equivalent DDMs
3. Effects of spike timing jitter on decision accuracy and speed
4. Scaling relationships between model parameters
5. Comparison of RT distributions across parameter regimes
