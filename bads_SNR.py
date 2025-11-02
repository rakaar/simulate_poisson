# %%
# imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
from bads_utils import lr_rates_from_ABL_ILD

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
# Define stimulus conditions
ILD_values = np.arange(-16, 17, 2)  # -16 to 16 in steps of 2
ABL_values = [20, 40, 60]

print("\n=== STIMULUS CONDITIONS ===")
print(f"ILD values: {ILD_values}")
print(f"ABL values: {ABL_values}")
print(f"Total conditions: {len(ABL_values) * len(ILD_values)}")

# %%
# Calculate rate scaling factor (constant, based on optimization result)
# Use the stimulus from the original optimization
ABL_opt = ddm_stimulus['ABL']
ILD_opt = ddm_stimulus['ILD']

# Calculate DDM rates at the optimization stimulus
ddm_right_rate_opt, ddm_left_rate_opt = lr_rates_from_ABL_ILD(
    ABL_opt, ILD_opt, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
)

print(f"\nDDM rates at optimization stimulus (ABL={ABL_opt}, ILD={ILD_opt}):")
print(f"  Right rate: {ddm_right_rate_opt:.6f}")
print(f"  Left rate:  {ddm_left_rate_opt:.6f}")
print(f"  Rates equal: {np.isclose(ddm_right_rate_opt, ddm_left_rate_opt)}")

# Calculate effective rate per neuron
r_ddm_effective = ddm_right_rate_opt / N_opt
ratio_r = r_opt / r_ddm_effective
print(f'\nrate_r_ddm_effective = {r_ddm_effective:.6f}')
print(f'ratio_r = {ratio_r:.6f}')

# %%
# Calculate SNR for both DDM and Poisson models for each ABL and ILD combination
print("\n" + "="*80)
print("CALCULATING SNR FOR DDM AND POISSON MODELS")
print("="*80)

T_max = 2  # Time window for simulations

# Get DDM theta
theta_ddm = ddm_params['theta']
print(f"\nDDM theta: {theta_ddm:.4f}")
print(f"Poisson theta_opt: {theta_opt:.4f}")
print(f"Poisson N_opt: {N_opt}")
print(f"Poisson c_opt: {c_opt:.6f}")

# Store SNR values for plotting
snr_ddm = {}
snr_poisson = {}

for ABL in ABL_values:
    snr_ddm[ABL] = []
    snr_poisson[ABL] = []
    
    print(f"\n{'='*80}")
    print(f"ABL = {ABL} dB")
    print(f"{'='*80}")
    
    for ILD in ILD_values:
        # Calculate left and right rates for this stimulus (DDM rates)
        r_right, r_left = lr_rates_from_ABL_ILD(
            ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
        )
        
        # Calculate DDM SNR
        # SNR_DDM = theta_ddm * (r_right - r_left) / (r_right + r_left)
        if (r_right + r_left) > 0:
            snr_ddm_val = theta_ddm * (r_right - r_left) / (r_right + r_left)
        else:
            snr_ddm_val = 0.0
        
        # Scale rates for Poisson model
        # NOTE: lr_rates_from_ABL_ILD returns TOTAL population rates
        # We need to divide by N to get per-neuron rates for the simulation
        r_right_scaled = (r_right * ratio_r) / N_opt
        r_left_scaled = (r_left * ratio_r) / N_opt
        
        # Calculate Poisson SNR
        # SNR_Poisson = theta_opt * (r_right_scaled - r_left_scaled) / ((r_right_scaled + r_left_scaled) * (1 + (N-1)*c_opt))
        if (r_right_scaled + r_left_scaled) > 0:
            denominator = (r_right_scaled + r_left_scaled) * (1 + (N_opt - 1) * c_opt)
            snr_poisson_val = theta_opt * (r_right_scaled - r_left_scaled) / denominator
        else:
            snr_poisson_val = 0.0
        
        # Store SNR values
        snr_ddm[ABL].append(snr_ddm_val)
        snr_poisson[ABL].append(snr_poisson_val)
        
        print(f"  ILD = {ILD:+3d} dB:  SNR_DDM = {snr_ddm_val:+8.4f},  SNR_Poisson = {snr_poisson_val:+8.4f}")
    
    # Convert to numpy arrays
    snr_ddm[ABL] = np.array(snr_ddm[ABL])
    snr_poisson[ABL] = np.array(snr_poisson[ABL])

print("\n" + "="*80)
print("SNR CALCULATION COMPLETE")
print("="*80)

# %%
# Plot SNR vs ILD for both models across ABL values
print("\n=== PLOTTING SNR COMPARISON ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, ABL in enumerate(ABL_values):
    ax = axes[idx]
    
    # Plot DDM SNR
    ax.plot(ILD_values, snr_ddm[ABL], 'b-', 
            linewidth=2, label='DDM', alpha=0.7, marker='o', markersize=5)
    
    # Plot Poisson SNR
    ax.plot(ILD_values, snr_poisson[ABL], 'r-', 
            linewidth=2, label='Poisson', alpha=0.7, marker='s', markersize=5)
    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('ILD (dB)', fontsize=12)
    ax.set_ylabel('SNR', fontsize=12)
    ax.set_title(f'ABL = {ABL} dB', fontsize=13, fontweight='bold')
    ax.set_xlim(-17, 17)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=-2, alpha=0.5)
    ax.axvline(x=2, alpha=0.5)


plt.suptitle(f'SNR Comparison: DDM vs Poisson Model\n(theta_DDM={theta_ddm:.4f}, theta_Poisson={theta_opt:.4f})', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('SNR_comparison_ddm_vs_poisson.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: SNR_comparison_ddm_vs_poisson.png")
plt.show()

# %%
# Print summary statistics
print("\n=== SNR STATISTICS ===")
for ABL in ABL_values:
    print(f"\nABL = {ABL} dB:")
    print(f"  DDM SNR range:     [{snr_ddm[ABL].min():+.4f}, {snr_ddm[ABL].max():+.4f}]")
    print(f"  Poisson SNR range: [{snr_poisson[ABL].min():+.4f}, {snr_poisson[ABL].max():+.4f}]")
    print(f"  Ratio (Poisson/DDM) at ILD=0: {snr_poisson[ABL][len(ILD_values)//2] / snr_ddm[ABL][len(ILD_values)//2] if snr_ddm[ABL][len(ILD_values)//2] != 0 else np.nan:.4f}")

print("\n✓ Analysis complete!")
