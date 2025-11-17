# Rate Scaling and Bound Increment Optimization Using BADS

This documentation describes the BADS-based optimization workflow for finding optimal rate scaling factors and bound increments to match DDM and Poisson model predictions.

## Overview

The optimization addresses the following problem:
- **Goal**: For different original theta values, find the rate scaling factor and bound increment that minimize the difference between DDM and Poisson model predictions
- **Metrics**: Minimize squared error for both chronometric (RT) and psychometric (accuracy) data across multiple stimuli
- **Method**: Bayesian Adaptive Direct Search (BADS) optimization

## Files

### 1. `find_bound_incr_rate_scale_bads.py`
Main optimization script that:
- Iterates over original_theta values: [2, 3, 4, 5, 6, 7]
- For each theta, optimizes two parameters:
  - **rate_scaling_factor**: Multiplier for firing rates (range: 1x to 10x)
  - **theta_increment**: Integer increment for bound (range: +1 to +20)
- Uses BADS to minimize total MSE across stimuli (ABL: [20, 40, 60], ILD: [1, 2, 4, 8, 16])
- Saves results to timestamped pickle file

### 2. `analyse_bads_rate_bound_optimization.py`
Analysis script that:
- Loads optimization results from pickle file
- Displays summary table of optimized parameters
- Creates comprehensive visualizations
- Performs detailed validation for specific theta values
- Analyzes trends and correlations

## Usage

### Running the Optimization

```bash
# Make sure you're in the project directory with the virtual environment activated
cd /home/ragha/code/simulate_poisson
source .venv/bin/activate  # or: ./.venv/bin/python

# Run the optimization (this may take several hours)
./.venv/bin/python find_bound_incr_rate_scale_bads.py
```

**Expected runtime**: 
- ~30-60 minutes per original_theta value (depends on convergence)
- Total: ~3-6 hours for all 6 theta values

**Output files**:
- `bads_rate_bound_optimization_results_YYYYMMDD_HHMMSS.pkl` - Full optimization results
- `bads_optimization_summary_YYYYMMDD_HHMMSS.png` - Summary plots

### Analyzing Results

```bash
# Run the analysis script
./.venv/bin/python analyse_bads_rate_bound_optimization.py
```

**Output**:
- Summary table printed to console
- Comprehensive analysis plots
- Detailed comparison plots for selected theta values
- Correlation analysis

## Key Parameters

### Fixed Parameters (shared across all optimizations)
```python
lam = 1.3              # Lambda parameter
l = 0.9                # Ell parameter
Nr0_base = 13.3        # Base firing rate (before scaling)
N = 100                # Number of neurons
rho = 1e-2             # Correlation parameter
dt = 1e-6              # Time step
```

### Optimization Parameters
```python
# Parameter 1: rate_scaling_factor
Lower bound: 1.0
Upper bound: 10.0
Plausible range: [1.2, 5.0]

# Parameter 2: theta_increment (integer)
Lower bound: 1
Upper bound: 20
Plausible range: [1, 10]
```

### Stimuli
```python
ABL_range = [20, 40, 60]         # 3 levels
ILD_range = [1, 2, 4, 8, 16]     # 5 levels
Total stimuli = 15
```

## Objective Function

The optimization minimizes:

```
Total MSE = Σ [(RT_DDM - RT_Poisson)² + (Acc_DDM - Acc_Poisson)²]
```

where the sum is over all 15 stimuli (3 ABL × 5 ILD).

### Computation Flow

For each evaluation:
1. Extract parameters: `rate_scaling_factor`, `theta_increment` (rounded to integer)
2. Calculate scaled rates: `Nr0_scaled = Nr0_base × rate_scaling_factor`
3. Calculate Poisson theta: `theta_poisson = original_theta + theta_increment`
4. For each stimulus:
   - Estimate bound offset using simulations (`run_poisson_trial`)
   - Calculate effective theta: `theta_eff = theta_poisson + bound_offset_mean`
   - Get DDM predictions: `ddm_fc_dt(original_theta)`
   - Get Poisson predictions: `poisson_fc_dt(theta_eff, scaled_rates)`
5. Compute MSE for RT and accuracy across all stimuli
6. Return total MSE

## Results Structure

The saved pickle file contains:

```python
{
    'results_dict': {
        original_theta: {
            'rate_scaling_factor_opt': float,
            'theta_increment_opt': int,
            'theta_poisson_opt': int,
            'final_objective_value': float,
            'func_count': int,
            'success': bool,
            'optimization_time': float,
            'bads_result': dict,  # Full BADS result
            'ddm_rt_data': dict,  # DDM RTs for all stimuli
            'ddm_acc_data': dict, # DDM accuracies for all stimuli
        },
        # ... for each original_theta value
    },
    'original_theta_values': [2, 3, 4, 5, 6, 7],
    'fixed_params': {...},
    'timestamp': str,
}
```

## Key Differences from Manual Tuning

| Aspect | Manual Approach | BADS Approach |
|--------|----------------|---------------|
| **Search strategy** | Grid search or trial-and-error | Adaptive Bayesian optimization |
| **Efficiency** | Many manual iterations | Converges in ~50-100 function evals |
| **Optimality** | May miss optimal values | Finds local optimum reliably |
| **Documentation** | Parameters scattered in code | All results saved systematically |
| **Reproducibility** | Hard to reproduce exact values | Fully reproducible with saved results |

## Constraints and Design Choices

### Why theta_increment must be integer?
- In the Poisson spike-based model, theta represents a discrete count threshold
- Integer values ensure biologically plausible spike count thresholds

### Why estimate bound offset?
- The effective threshold in the Poisson model differs from the nominal threshold due to:
  - Discrete spike events
  - Correlation structure
  - Boundary overshoot
- Bound offset is estimated empirically via simulations for accuracy

### Why use analytical formulas (`ddm_fc_dt`, `poisson_fc_dt`)?
- Fast computation (no simulations needed)
- Deterministic objective function (no noise from simulation stochasticity)
- BADS can use `uncertainty_handling=False` for faster convergence

## Troubleshooting

### Issue: Optimization takes too long
**Solution**: Reduce `N_sim_bound_offset` (default: 10,000) to 5,000 or fewer trials

### Issue: BADS not converging
**Possible causes**:
- Initial guess too far from optimum
- Bounds too restrictive
- Objective function has multiple local minima

**Solutions**:
- Adjust plausible bounds based on manual exploration
- Try multiple initial guesses
- Increase BADS max iterations

### Issue: High MSE even after optimization
**Possible causes**:
- Parameter ranges may not include optimal solution
- DDM and Poisson models fundamentally differ for this regime
- Need to verify analytical formulas match simulations

**Solutions**:
- Expand search bounds
- Validate predictions with high-fidelity simulations
- Check if analytical formulas are accurate for this parameter regime

## Example Output

```
SUMMARY TABLE
======================================================================

Original Theta   Rate Scaling    Theta Inc    Theta Poisson   Final MSE   
------------------------------------------------------------------------------
2                1.7234          2            4               0.000234
3                1.6891          2            5               0.000189
4                1.6523          2            6               0.000156
5                1.6198          3            8               0.000201
6                1.5876          3            9               0.000178
7                1.5543          3            10              0.000165
```

## References

- BADS paper: Acerbi & Ma (2017), "Practical Bayesian Optimization for Model Fitting with Bayesian Adaptive Direct Search"
- PyBADS documentation: https://acerbilab.github.io/pybads/

## Contact

For questions or issues, refer to the main project README or contact the project maintainer.
