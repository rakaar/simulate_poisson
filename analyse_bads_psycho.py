# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from tqdm import tqdm
from bads_utils import (
    lr_rates_from_ABL_ILD,
    simulate_poisson_rts
)

# %%
# Load BADS optimization results
pkl_filename = 'bads_optimization_results_singlerate_20251029_195526.pkl'

# Define dummy obj_func to allow unpickling (BADS result contains reference to it)
def obj_func(x):
    """Dummy function for unpickling - not actually used."""
    pass

print(f"Loading results from: {pkl_filename}")
with open(pkl_filename, 'rb') as f:
    results = pickle.load(f)

# Extract optimized parameters
optimize_result = results['bads_result']
x_opt = results['optimized_params']['x_opt']
N_opt = results['optimized_params']['N']
r_opt = results['optimized_params']['r']
k_opt = results['optimized_params']['k']
c_opt = results['optimized_params']['c']
theta_opt = results['optimized_params']['theta']

# Extract DDM parameters
ddm_params = results['ddm_params']
ddm_stimulus = results['ddm_stimulus']

print("\n=== LOADED BADS OPTIMIZATION RESULTS ===")
print(f"Optimized parameters:")
print(f"  N:     {N_opt}")
print(f"  r:     {r_opt:.4f} (same for left and right)")
print(f"  k:     {k_opt:.2f}")
print(f"  c:     {c_opt:.6f} (= k/N)")
print(f"  theta: {theta_opt:.4f}")

# %%
# Define stimulus conditions for psychometric analysis
ILD_values = np.arange(-16, 17, 2)  # Sparse: -16 to 16 in steps of 2
ABL_values = [20, 40, 60]

print("\n=== PSYCHOMETRIC ANALYSIS SETUP ===")
print(f"ILD values: {ILD_values}")
print(f"ABL values: {ABL_values}")
print(f"Total conditions: {len(ABL_values) * len(ILD_values)}")

# %%
# Simulate Poisson model for all ABL/ILD combinations
# This will save TWO pickle files:
#   1. psycho_simulation_data_<timestamp>.pkl - Raw RT/choice data for all conditions
#   2. psycho_curves_<timestamp>.pkl - Calculated psychometric curves (Poisson & DDM)
N_trials_per_condition = int(50e3)  # 50K trials per condition

# Calculate rate scaling factor (constant, based on optimization result)
# Use the stimulus from the original optimization
ABL_opt = ddm_stimulus['ABL']
ILD_opt = ddm_stimulus['ILD']

# Calculate DDM rates at the optimization stimulus
ddm_right_rate_opt, ddm_left_rate_opt = lr_rates_from_ABL_ILD(
    ABL_opt, ILD_opt, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
)

# Verify rates are equal (they should be for ILD=0)
print(f"\nDDM rates at optimization stimulus (ABL={ABL_opt}, ILD={ILD_opt}):")
print(f"  Right rate: {ddm_right_rate_opt:.6f}")
print(f"  Left rate:  {ddm_left_rate_opt:.6f}")
print(f"  Rates equal: {np.isclose(ddm_right_rate_opt, ddm_left_rate_opt)}")

# %%
# Calculate effective rate per neuron
r_ddm_effective = ddm_right_rate_opt / N_opt
ratio_r = r_opt / r_ddm_effective
print(f'\nrate_r_ddm_effective = {r_ddm_effective:.6f}')
print(f'ratio_r = {ratio_r:.6f}')
# %%
print("\n=== SIMULATING POISSON MODEL ===")
print(f"Trials per condition: {N_trials_per_condition}")
print(f"Using optimized parameters: N={N_opt}, r={r_opt:.4f}, c={c_opt:.6f}, theta={theta_opt:.4f}")
print(f"Rate scaling factor (ratio_r): {ratio_r:.6f}")

# Store results for each condition
poisson_psychometric_data = {}

# Start timing
start_time = time.time()
total_conditions = len(ABL_values) * len(ILD_values)

# Create nested progress bars
for ABL in ABL_values:
# for ABL in [40]:
    poisson_psychometric_data[ABL] = {}
    
    for ILD in tqdm(ILD_values, desc=f'ABL={ABL} dB', ncols=100):
    # for ILD in [0]:
        # Calculate left and right rates for this stimulus
        r_right, r_left = lr_rates_from_ABL_ILD(
            ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
        )
        # Scale rates using the constant ratio_r
        # NOTE: lr_rates_from_ABL_ILD returns TOTAL population rates
        # We need to divide by N to get per-neuron rates for the simulation
        r_right_scaled = (r_right * ratio_r) / N_opt
        r_left_scaled = (r_left * ratio_r) / N_opt

        # Set T_max = 2 for all conditions to speed up simulation
        T_max = 2
        
        # Debug: Check for extremely high rates that will cause slow simulations
        total_rate = (r_right_scaled + r_left_scaled) * N_opt
        expected_spikes_per_trial = total_rate * T_max  # Use adjusted T_max
        if expected_spikes_per_trial > 500000:  # More than 500K spikes per trial
            print(f'\n  WARNING: High spike count for ILD={ILD} (T={T_max}s):')
            print(f'    r_right={r_right_scaled:.2f}, r_left={r_left_scaled:.2f}')
            print(f'    Expected ~{expected_spikes_per_trial/1e6:.2f}M spikes/trial')
            print(f'    This may take a while...')
        
        poisson_params = {
            'N_right': N_opt,
            'N_left': N_opt,
            'c': c_opt,
            'r_right': r_right_scaled,
            'r_left': r_left_scaled,
            'theta': theta_opt,
            'T': T_max,
            'exponential_noise_scale': 0
        }
        
        # Simulate trials
        results = simulate_poisson_rts(
            poisson_params, 
            n_trials=N_trials_per_condition,
            seed=42 + int(ABL) + int(abs(ILD*10)),  # Unique seed for each condition
            verbose=False,
            n_jobs=30
        )
        
        # Store results (RT, choice) for this condition
        poisson_psychometric_data[ABL][ILD] = results

# End timing
end_time = time.time()
total_time = end_time - start_time

print("\n✓ Poisson simulations complete!")
print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"Average time per condition: {total_time/total_conditions:.2f}s")

# Save raw simulation data to pickle file
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
pkl_filename_sim = f'psycho_simulation_data_{timestamp}.pkl'

print(f"\nSaving raw simulation data to: {pkl_filename_sim}")
simulation_data = {
    'poisson_psychometric_data': poisson_psychometric_data,
    'parameters': {
        'N_opt': N_opt,
        'r_opt': r_opt,
        'k_opt': k_opt,
        'c_opt': c_opt,
        'theta_opt': theta_opt,
        'ratio_r': ratio_r,
        'T_max': T_max,
        'N_trials_per_condition': N_trials_per_condition
    },
    'stimulus': {
        'ILD_values': ILD_values,
        'ABL_values': ABL_values
    },
    'ddm_params': ddm_params,
    'metadata': {
        'timestamp': timestamp,
        'total_simulation_time_seconds': total_time
    }
}

with open(pkl_filename_sim, 'wb') as f:
    pickle.dump(simulation_data, f)
print(f"✓ Raw simulation data saved!")

# %%
# Calculate psychometric curves from Poisson simulations
# Psychometric = P(choice == 1) = number of trials with choice == 1 / total trials

print("\n=== CALCULATING POISSON PSYCHOMETRIC CURVES ===")

poisson_psychometric_curves = {}

for ABL in ABL_values:
    poisson_psychometric_curves[ABL] = []
    
    for ILD in ILD_values:
        results = poisson_psychometric_data[ABL][ILD]
        
        # Extract choices (column 1: 1 = right, -1 = left, 0 = no decision)
        choices = results[:, 1]
        
        # Calculate P(right) = P(choice == 1) / total trials
        n_right_choices = np.sum(choices == 1)
        total_trials = len(choices)
        p_right = n_right_choices / total_trials
        
        poisson_psychometric_curves[ABL].append(p_right)
    
    # Convert to numpy array for easier plotting
    poisson_psychometric_curves[ABL] = np.array(poisson_psychometric_curves[ABL])
    
    print(f"ABL = {ABL} dB: P(right) ranges from {poisson_psychometric_curves[ABL].min():.4f} to {poisson_psychometric_curves[ABL].max():.4f}")

print("\n✓ Poisson psychometric curves calculated!")

# %%
# Calculate DDM psychometric curves using analytical formula
def psyc_ddm(ABL, ILD, ddm_params):
    """Calculate DDM psychometric function (analytical)."""
    ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
        ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
    )
    mu = ddm_right_rate - ddm_left_rate
    sigma_sq = ddm_right_rate + ddm_left_rate
    gamma = (ddm_params['theta']) * (mu / sigma_sq)
    return 1 / (1 + np.exp(-2 * gamma))

print("\n=== CALCULATING DDM PSYCHOMETRIC CURVES ===")

ddm_psychometric_curves = {}

for ABL in ABL_values:
    ddm_psychometric_curves[ABL] = []
    
    for ILD in ILD_values:
        p_right_ddm = psyc_ddm(ABL, ILD, ddm_params)
        ddm_psychometric_curves[ABL].append(p_right_ddm)
    
    # Convert to numpy array
    ddm_psychometric_curves[ABL] = np.array(ddm_psychometric_curves[ABL])
    
    print(f"ABL = {ABL} dB: P(right) ranges from {ddm_psychometric_curves[ABL].min():.4f} to {ddm_psychometric_curves[ABL].max():.4f}")

print("\n✓ DDM psychometric curves calculated!")

# Save psychometric curves to pickle file
pkl_filename_psycho = f'psycho_curves_{timestamp}.pkl'
print(f"\nSaving psychometric curves to: {pkl_filename_psycho}")
psychometric_data = {
    'poisson_psychometric_curves': poisson_psychometric_curves,
    'ddm_psychometric_curves': ddm_psychometric_curves,
    'ILD_values': ILD_values,
    'ABL_values': ABL_values,
    'parameters': {
        'N_opt': N_opt,
        'r_opt': r_opt,
        'c_opt': c_opt,
        'theta_opt': theta_opt,
        'ratio_r': ratio_r
    },
    'metadata': {
        'timestamp': timestamp,
        'N_trials_per_condition': N_trials_per_condition
    }
}

with open(pkl_filename_psycho, 'wb') as f:
    pickle.dump(psychometric_data, f)
print(f"✓ Psychometric curves saved!")

# %%
# Plot psychometric curves: P(right) vs ILD for both models across ABL values
print("\n=== PLOTTING PSYCHOMETRIC CURVES ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, ABL in enumerate(ABL_values):
    ax = axes[idx]
    
    # Plot DDM psychometric (analytical)
    ax.plot(ILD_values, ddm_psychometric_curves[ABL], 'b-', 
            linewidth=2, label='DDM (analytical)', alpha=0.7)
    
    # Plot Poisson psychometric (from simulations)
    ax.plot(ILD_values, poisson_psychometric_curves[ABL], 'ro', 
            markersize=6, label='Poisson (simulated)', alpha=0.7)
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('ILD (dB)', fontsize=12)
    ax.set_ylabel('P(Right)', fontsize=12)
    ax.set_title(f'ABL = {ABL} dB', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-17, 17)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Psychometric Curves: DDM vs Poisson Model\n(Poisson: {N_trials_per_condition} trials per condition)', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('psychometric_comparison_ddm_vs_poisson_simulated.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: psychometric_comparison_ddm_vs_poisson_simulated.png")
plt.show()

# %%
# Calculate and display absolute differences between models
print("\n=== PSYCHOMETRIC CURVE DIFFERENCES ===")

for ABL in ABL_values:
    abs_diff = np.abs(ddm_psychometric_curves[ABL] - poisson_psychometric_curves[ABL])
    max_diff = np.max(abs_diff)
    max_diff_ild = ILD_values[np.argmax(abs_diff)]
    mean_diff = np.mean(abs_diff)
    
    print(f"\nABL = {ABL} dB:")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Max absolute difference:  {max_diff:.6f} (at ILD = {max_diff_ild} dB)")
    print(f"  RMS difference:           {np.sqrt(np.mean(abs_diff**2)):.6f}")

print("\n✓ Analysis complete!")

