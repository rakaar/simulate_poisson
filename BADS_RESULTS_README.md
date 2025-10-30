# BADS Optimization Results - Save and Recovery Guide

## Overview

The BADS optimization script now automatically saves results in two formats:
1. **Text file (.txt)**: Human-readable report with all parameters and results
2. **Pickle file (.pkl)**: Complete Python object for programmatic recovery

## Saved Files

After running `bads_find_optimum_params.py`, you'll get:
- `bads_optimization_results_YYYYMMDD_HHMMSS.txt` - Detailed text report
- `bads_optimization_results_YYYYMMDD_HHMMSS.pkl` - Pickle file with all data

## What's Saved in the Pickle File

The pickle file contains a dictionary with the following structure:

```python
{
    'timestamp': str,                    # When the optimization was run
    'ddm_params': dict,                  # DDM simulation parameters
    'ddm_stimulus': dict,                # ABL and ILD values
    'ddm_simulation': {                  # DDM simulation data
        'mu': float,
        'sigma': float,
        'dt': float,
        'dB': float,
        'T': float,
        'N_sim': int,
        'ddm_data': ndarray,            # All DDM trial results
        'ddm_rts_decided': ndarray      # DDM RTs (decided trials only)
    },
    'bads_setup': {                      # BADS configuration
        'n_trials_per_eval': int,
        'seed': int,
        'lower_bounds': ndarray,
        'upper_bounds': ndarray,
        'plausible_lower_bounds': ndarray,
        'plausible_upper_bounds': ndarray,
        'x0': ndarray                   # Initial guess
    },
    'bads_result': OptimizeResult,      # Complete BADS optimization result
    'optimized_params': {                # Extracted optimal parameters
        'N': int,
        'r1': float,
        'r2': float,
        'k': float,
        'c': float,
        'theta': float,
        'x_opt': ndarray               # Full parameter vector
    },
    'validation': {                      # Validation run results
        'validation_params': dict,
        'validation_results': ndarray,
        'poisson_rts_val': ndarray,
        'ks_stat_val': float,
        'ks_pval': float
    }
}
```

## How to Load and Use Saved Results

### Method 1: Using the Helper Script

```bash
python load_bads_results.py bads_optimization_results_20250429_143022.pkl
```

This will print a summary of the results and show you available keys.

### Method 2: In Your Own Script

```python
import pickle

# Load the results
with open('bads_optimization_results_YYYYMMDD_HHMMSS.pkl', 'rb') as f:
    results = pickle.load(f)

# Access optimized parameters
x_opt = results['optimized_params']['x_opt']
N_opt = results['optimized_params']['N']
c_opt = results['optimized_params']['c']

# Access BADS result object
optimize_result = results['bads_result']
ks_statistic = optimize_result['fval']
n_evaluations = optimize_result['func_count']

# Access DDM and Poisson data
ddm_rts = results['ddm_simulation']['ddm_rts_decided']
poisson_rts = results['validation']['poisson_rts_val']

# Access any other saved component
ddm_params = results['ddm_params']
validation_params = results['validation']['validation_params']
```

### Method 3: Using the Example Script

See `example_load_bads.py` for a complete working example with visualization.

## Quick Recovery Examples

### Get optimized parameters:
```python
import pickle

with open('your_file.pkl', 'rb') as f:
    results = pickle.load(f)

N = results['optimized_params']['N']
c = results['optimized_params']['c']
theta = results['optimized_params']['theta']
r1 = results['optimized_params']['r1']
r2 = results['optimized_params']['r2']
```

### Rerun validation with different parameters:
```python
from bads_utils import simulate_poisson_rts

# Load saved results
with open('your_file.pkl', 'rb') as f:
    results = pickle.load(f)

# Get optimized parameters
opt_params = results['optimized_params']

# Create new validation parameters (e.g., different number of trials)
new_params = {
    'N_right': opt_params['N'],
    'N_left': opt_params['N'],
    'c': opt_params['c'],
    'r_right': opt_params['r1'],
    'r_left': opt_params['r2'],
    'theta': opt_params['theta'],
    'T': 20,
    'exponential_noise_scale': 0
}

# Run new validation
new_validation = simulate_poisson_rts(new_params, n_trials=100000, seed=42)
```

### Compare multiple optimization runs:
```python
import pickle
import numpy as np

# Load multiple results
files = ['result1.pkl', 'result2.pkl', 'result3.pkl']
all_results = []

for f in files:
    with open(f, 'rb') as file:
        all_results.append(pickle.load(file))

# Compare KS statistics
for i, res in enumerate(all_results):
    print(f"Run {i+1}: KS = {res['bads_result']['fval']:.6f}")
    print(f"  N={res['optimized_params']['N']}, c={res['optimized_params']['c']:.6f}")
```

## Important Notes

1. **Python Version**: Pickle files should be loaded with the same Python version used to save them
2. **Dependencies**: Make sure you have all required packages (numpy, scipy, etc.) installed
3. **File Size**: Pickle files can be large (~10-100 MB) as they contain full trial data
4. **Security**: Only load pickle files from trusted sources (pickle can execute arbitrary code)

## Files Created by the Refactoring

- `bads_utils.py` - Helper functions for simulations and optimization
- `bads_tests.py` - Test code for validating DDM/Poisson alignment
- `bads_find_optimum_params.py` - Main optimization script (now with save functionality)
- `load_bads_results.py` - Helper script to load and display saved results
- `example_load_bads.py` - Complete example showing how to use loaded results
- `BADS_RESULTS_README.md` - This documentation file

## Troubleshooting

**Problem**: Can't load pickle file
```python
# Try specifying encoding
with open('file.pkl', 'rb') as f:
    results = pickle.load(f, encoding='latin1')
```

**Problem**: Need to extract just specific data
```python
# Load only what you need
with open('file.pkl', 'rb') as f:
    results = pickle.load(f)
    x_opt = results['optimized_params']['x_opt']
    # Don't keep the full results in memory if not needed
    del results
```

## Contact

For questions or issues, refer to the main project documentation.
