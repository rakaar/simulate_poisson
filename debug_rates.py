# %%
# Quick diagnostic to check firing rates
import numpy as np
import pickle
from bads_utils import lr_rates_from_ABL_ILD

# Load BADS optimization results
pkl_filename = 'bads_optimization_results_singlerate_20251029_195526.pkl'

def obj_func(x):
    """Dummy function for unpickling."""
    pass

with open(pkl_filename, 'rb') as f:
    results = pickle.load(f)

# Extract parameters
N_opt = results['optimized_params']['N']
r_opt = results['optimized_params']['r']
theta_opt = results['optimized_params']['theta']
ddm_params = results['ddm_params']

# Calculate rate scaling
r_ddm_effective = ddm_params['Nr0'] / N_opt
ratio_r = r_opt / r_ddm_effective

print("=== PARAMETER VALUES ===")
print(f"N_opt = {N_opt}")
print(f"r_opt = {r_opt:.6f} (per neuron rate)")
print(f"theta_opt = {theta_opt:.4f}")
print(f"ddm_params['Nr0'] = {ddm_params['Nr0']:.2f}")
print(f"r_ddm_effective = {r_ddm_effective:.6f}")
print(f"ratio_r = {ratio_r:.6f}")

print("\n=== CHECKING RATES FOR ABL=20, ILD=0 ===")
ABL, ILD = 20, 0

# Get rates from lr_rates_from_ABL_ILD
r_right, r_left = lr_rates_from_ABL_ILD(
    ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
)

print(f"\nRates from lr_rates_from_ABL_ILD (TOTAL population rates?):")
print(f"  r_right = {r_right:.2f} spikes/s")
print(f"  r_left = {r_left:.2f} spikes/s")

# After scaling (OLD - WRONG)
r_right_scaled_old = r_right * ratio_r
r_left_scaled_old = r_left * ratio_r

print(f"\nOLD (WRONG) - Scaled rates without dividing by N:")
print(f"  r_right_scaled = {r_right_scaled_old:.4f} spikes/s per neuron")
print(f"  r_left_scaled = {r_left_scaled_old:.4f} spikes/s per neuron")

# After scaling (FIXED)
r_right_scaled = (r_right * ratio_r) / N_opt
r_left_scaled = (r_left * ratio_r) / N_opt

print(f"\nFIXED - Scaled rates divided by N (per-neuron rates):")
print(f"  r_right_scaled = {r_right_scaled:.6f} spikes/s per neuron")
print(f"  r_left_scaled = {r_left_scaled:.6f} spikes/s per neuron")

# Total firing rate in simulation
total_rate = (r_right_scaled + r_left_scaled) * N_opt
print(f"\nTotal population firing rate:")
print(f"  (r_right_scaled + r_left_scaled) * N = {total_rate:.2f} spikes/s")

# Expected time to threshold
print(f"\n=== TIME TO THRESHOLD ESTIMATE ===")
print(f"Threshold: theta = {theta_opt:.4f} spikes")
expected_rate_diff = (r_right_scaled - r_left_scaled) * N_opt
print(f"Expected net evidence rate: (r_right - r_left) * N = {expected_rate_diff:.4f} spikes/s")
if abs(expected_rate_diff) > 0:
    time_to_threshold = theta_opt / abs(expected_rate_diff)
    print(f"Expected time to threshold: theta / net_rate = {time_to_threshold:.6f} s = {time_to_threshold*1000:.3f} ms")

print("\n=== CHECKING ANOTHER CONDITION: ABL=60, ILD=16 ===")
ABL, ILD = 60, 16

r_right, r_left = lr_rates_from_ABL_ILD(
    ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
)
print(f"Total population rates from lr_rates_from_ABL_ILD:")
print(f"  r_right = {r_right:.2f}, r_left = {r_left:.2f} spikes/s")

# FIXED version
r_right_scaled = (r_right * ratio_r) / N_opt
r_left_scaled = (r_left * ratio_r) / N_opt
print(f"Per-neuron rates (FIXED):")
print(f"  r_right_scaled = {r_right_scaled:.6f}, r_left_scaled = {r_left_scaled:.6f} spikes/s/neuron")

total_rate = (r_right_scaled + r_left_scaled) * N_opt
expected_rate_diff = (r_right_scaled - r_left_scaled) * N_opt

print(f"Total firing rate: {total_rate:.2f} spikes/s")
print(f"Net evidence rate: {expected_rate_diff:.4f} spikes/s")
if abs(expected_rate_diff) > 0:
    time_to_threshold = theta_opt / abs(expected_rate_diff)
    print(f"Expected time to threshold: {time_to_threshold:.6f} s = {time_to_threshold*1000:.2f} ms")
