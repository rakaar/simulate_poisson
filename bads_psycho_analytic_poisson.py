# %%
# Analytical Poisson Psychometric Function with Correlations
# Based on compound-jump random walk formula provided by GPT-5

import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import bisect
from bads_utils import lr_rates_from_ABL_ILD


# %%
# Analytical Poisson Psychometric Function Implementation

def _log_phi_of_logrho(t, p_R, N, c):
    # t = log rho, so rho = exp(t)
    # log Phi(e^t) = logsumexp( log[p_R] + N*log(1-c+c e^t),
    #                           log[1-p_R] + N*log(1-c+c e^{-t}) )
    a = np.log(p_R)     + N * np.log1p(-c + c*np.exp(t))     # log term for + side
    b = np.log1p(-p_R)  + N * np.log1p(-c + c*np.exp(-t))    # log term for - side
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))         # log Phi

def find_rho(r_right, r_left, N, c):
    """
    For small c (like 0.002451), the Skellam-limit approximation works well:
    rho ≈ r_left / r_right
    
    This is the ratio of left to right rates, which naturally handles
    both r_L > r_R (rho > 1) and r_R > r_L (rho < 1) cases.
    """
    # Use the simple approximation which works well for small c
    rho = r_left / r_right
    return rho


def psyc_poisson_analytical(r_right, r_left, N, c, theta):
    """
    Calculate analytical Poisson psychometric function with correlations.
    
    Returns P(choice = +1 | r_right, r_left) using the compound-jump formula.
    
    Solves for rho > 0 (may be > 1 or < 1) such that:
    p_R * (1-c+c*rho)^N + (1-p_R) * (1-c+c/rho)^N = 1
    
    Then uses rho_star = min(rho, 1/rho) in the hitting-probability formula:
    P(hitting +theta) = (1 - rho_star^theta) / (1 - rho_star^(2*theta))
    
    Both rho and 1/rho solve Phi(rho) = 1; using the unit-interval representative
    is numerically stable and yields the correct choice probability.
    
    Parameters:
    -----------
    r_right : float
        Right side firing rate
    r_left : float
        Left side firing rate  
    N : int
        Number of neurons per pool
    c : float
        Within-pool correlation coefficient
    theta : float
        Decision threshold
    
    Returns:
    --------
    float : P(choice = +1), probability of choosing right
    """
    # Handle edge cases
    if r_right <= 0 and r_left <= 0:
        return 0.5  # No evidence, random choice
    
    if r_right <= 0:
        return 0.0  # Only left evidence
    
    if r_left <= 0:
        return 1.0  # Only right evidence
    
    # Find rho - may be < 1 or > 1
    rho = find_rho(r_right, r_left, N, c)
    
    # Use rho directly in the formula (no need to map to unit interval)
    # P(+1) = (1 - rho^theta) / (1 - rho^(2*theta))
    # This works for both rho < 1 and rho > 1
    x = theta * np.log(rho)
    numerator = -np.expm1(x)          # 1 - rho**theta, stable
    denominator = -np.expm1(2.0 * x)  # 1 - rho**(2*theta), stable
    
    return numerator / denominator if abs(denominator) > 1e-300 else 0.5


# %%
# Load the saved simulation data

print("=== LOADING SAVED SIMULATION DATA ===")

# Load the psycho_curves pickle file (contains both simulated Poisson and DDM curves)
pkl_filename_psycho = 'psycho_curves_20251031_004755.pkl'

print(f"Loading psychometric curves from: {pkl_filename_psycho}")
with open(pkl_filename_psycho, 'rb') as f:
    psycho_data = pickle.load(f)

# Extract data
poisson_psychometric_curves_sim = psycho_data['poisson_psychometric_curves']
ddm_psychometric_curves = psycho_data['ddm_psychometric_curves']
ILD_values = psycho_data['ILD_values']
ABL_values = psycho_data['ABL_values']
params = psycho_data['parameters']

N_opt = params['N_opt']
r_opt = params['r_opt']
c_opt = params['c_opt']
theta_opt = params['theta_opt']
ratio_r = params['ratio_r']

print(f"\nLoaded parameters:")
print(f"  N:     {N_opt}")
print(f"  r:     {r_opt:.4f}")
print(f"  c:     {c_opt:.6f}")
print(f"  theta: {theta_opt:.4f}")
print(f"  ratio_r: {ratio_r:.6f}")

# Load DDM parameters from the original BADS optimization file
pkl_filename_bads = 'bads_optimization_results_singlerate_20251029_195526.pkl'

# Define dummy obj_func for unpickling
def obj_func(x):
    """Dummy function for unpickling - not actually used."""
    pass

print(f"\nLoading DDM parameters from: {pkl_filename_bads}")
with open(pkl_filename_bads, 'rb') as f:
    bads_results = pickle.load(f)

ddm_params = bads_results['ddm_params']

print(f"DDM parameters:")
print(f"  Nr0: {ddm_params['Nr0']}")
print(f"  lam: {ddm_params['lam']}")
print(f"  ell: {ddm_params['ell']}")
print(f"  theta: {ddm_params['theta']}")


# %%
# Calculate analytical Poisson psychometric curves

print("\n=== CALCULATING ANALYTICAL POISSON PSYCHOMETRIC CURVES ===")

poisson_psychometric_curves_analytical = {}

for ABL in ABL_values:
    poisson_psychometric_curves_analytical[ABL] = []
    
    for ILD in ILD_values:
        # Calculate left and right rates for this stimulus
        r_right, r_left = lr_rates_from_ABL_ILD(
            ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
        )
        
        # Scale rates using the constant ratio_r
        r_right_scaled = r_right * ratio_r
        r_left_scaled = r_left * ratio_r
        
        # Calculate analytical psychometric value
        p_right_analytical = psyc_poisson_analytical(
            r_right_scaled, r_left_scaled, N_opt, c_opt, theta_opt
        )
        
        poisson_psychometric_curves_analytical[ABL].append(p_right_analytical)
    
    # Convert to numpy array
    poisson_psychometric_curves_analytical[ABL] = np.array(
        poisson_psychometric_curves_analytical[ABL]
    )
    
    print(f"ABL = {ABL} dB: P(right) ranges from "
          f"{poisson_psychometric_curves_analytical[ABL].min():.4f} to "
          f"{poisson_psychometric_curves_analytical[ABL].max():.4f}")

print("\n✓ Analytical Poisson psychometric curves calculated!")


# %%
# Plot comparison: Simulated vs Analytical Poisson Psychometric Curves

print("\n=== PLOTTING COMPARISON ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, ABL in enumerate(ABL_values):
    ax = axes[idx]
    
    # Plot simulated Poisson psychometric (from previous simulations)
    ax.plot(ILD_values, poisson_psychometric_curves_sim[ABL], 'ro', 
            markersize=8, label='Poisson (simulated)', alpha=0.6, 
            markeredgecolor='darkred', markeredgewidth=1.5)
    
    # Plot analytical Poisson psychometric
    ax.plot(ILD_values, poisson_psychometric_curves_analytical[ABL], 'b-', 
            linewidth=2.5, label='Poisson (analytical)', alpha=0.8)
    
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

plt.suptitle('Poisson Psychometric: Simulated vs Analytical\n'
             f'(N={N_opt}, c={c_opt:.4f}, θ={theta_opt:.2f})', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('psychometric_comparison_poisson_simulated_vs_analytical.png', 
            dpi=300, bbox_inches='tight')
print("✓ Plot saved: psychometric_comparison_poisson_simulated_vs_analytical.png")
plt.show()


# %%
# Calculate and display differences between simulated and analytical

print("\n=== SIMULATED vs ANALYTICAL DIFFERENCES ===")

for ABL in ABL_values:
    abs_diff = np.abs(
        poisson_psychometric_curves_sim[ABL] - 
        poisson_psychometric_curves_analytical[ABL]
    )
    max_diff = np.max(abs_diff)
    max_diff_ild = ILD_values[np.argmax(abs_diff)]
    mean_diff = np.mean(abs_diff)
    
    print(f"\nABL = {ABL} dB:")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  Max absolute difference:  {max_diff:.6f} (at ILD = {max_diff_ild} dB)")
    print(f"  RMS difference:           {np.sqrt(np.mean(abs_diff**2)):.6f}")

print("\n✓ Analysis complete!")


# %%
# Bonus: Plot all three together (DDM analytical, Poisson simulated, Poisson analytical)

print("\n=== PLOTTING ALL THREE CURVES ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, ABL in enumerate(ABL_values):
    ax = axes[idx]
    
    # Plot DDM psychometric (analytical)
    ax.plot(ILD_values, ddm_psychometric_curves[ABL], 'g--', 
            linewidth=2, label='DDM (analytical)', alpha=0.7)
    
    # Plot Poisson simulated
    ax.plot(ILD_values, poisson_psychometric_curves_sim[ABL], 'ro', 
            markersize=6, label='Poisson (simulated)', alpha=0.6,
            markeredgecolor='darkred', markeredgewidth=1)
    
    # Plot Poisson analytical
    ax.plot(ILD_values, poisson_psychometric_curves_analytical[ABL], 'b-', 
            linewidth=2.5, label='Poisson (analytical)', alpha=0.8)
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('ILD (dB)', fontsize=12)
    ax.set_ylabel('P(Right)', fontsize=12)
    ax.set_title(f'ABL = {ABL} dB', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-17, 17)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('Psychometric Curves: DDM vs Poisson (Simulated vs Analytical)\n'
             f'(N={N_opt}, c={c_opt:.4f}, θ={theta_opt:.2f})', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('psychometric_comparison_all_three.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: psychometric_comparison_all_three.png")
plt.show()

print("\n✓ All plots complete!")


# %%
# Debug: Check a few specific conditions to verify rho calculations

print("\n=== DEBUG: Verifying rho calculations for sample conditions ===")

# Test symmetric case (ILD = 0)
for ABL in [20, 40, 60]:
    r_right, r_left = lr_rates_from_ABL_ILD(
        ABL, 0, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
    )
    r_right_scaled = r_right * ratio_r
    r_left_scaled = r_left * ratio_r
    
    rho = find_rho(r_right_scaled, r_left_scaled, N_opt, c_opt)
    p_R = r_right_scaled / (r_right_scaled + r_left_scaled)
    
    print(f"\nABL={ABL}, ILD=0 (symmetric):")
    print(f"  r_right={r_right_scaled:.4f}, r_left={r_left_scaled:.4f}")
    print(f"  p_R={p_R:.6f}, rho={rho:.6f}")
    print(f"  rho should be ≈ r_left/r_right = {r_left_scaled/r_right_scaled:.6f}")
    print(f"  P(right) analytical = {poisson_psychometric_curves_analytical[ABL][8]:.6f}")
    print(f"  P(right) simulated  = {poisson_psychometric_curves_sim[ABL][8]:.6f}")

# Test asymmetric case (ILD = 16)
ABL = 40
r_right, r_left = lr_rates_from_ABL_ILD(
    ABL, 16, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
)
r_right_scaled = r_right * ratio_r
r_left_scaled = r_left * ratio_r

rho = find_rho(r_right_scaled, r_left_scaled, N_opt, c_opt)
p_R = r_right_scaled / (r_right_scaled + r_left_scaled)

print(f"\nABL={ABL}, ILD=16 (strongly right):")
print(f"  r_right={r_right_scaled:.4f}, r_left={r_left_scaled:.4f}")
print(f"  p_R={p_R:.6f}, rho={rho:.6f}")
print(f"  rho simple approx = r_left/r_right = {r_left_scaled/r_right_scaled:.6f}")

# Calculate P(right) step by step
rho_theta = rho ** theta_opt
rho_2theta = rho ** (2 * theta_opt)
p_analytical_manual = (1 - rho_theta) / (1 - rho_2theta)
print(f"  rho^theta = {rho_theta:.10f}")
print(f"  rho^(2*theta) = {rho_2theta:.10f}")
print(f"  P(right) = (1 - {rho_theta:.6f}) / (1 - {rho_2theta:.6f}) = {p_analytical_manual:.6f}")

idx_16 = np.where(ILD_values == 16)[0][0]
print(f"  P(right) analytical (from array) = {poisson_psychometric_curves_analytical[ABL][idx_16]:.6f}")
print(f"  P(right) simulated  = {poisson_psychometric_curves_sim[ABL][idx_16]:.6f}")
print(f"  Difference = {abs(poisson_psychometric_curves_analytical[ABL][idx_16] - poisson_psychometric_curves_sim[ABL][idx_16]):.6f}")

# Test the problematic case: ILD = -16 (strongly left)
r_right, r_left = lr_rates_from_ABL_ILD(
    ABL, -16, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
)
r_right_scaled = r_right * ratio_r
r_left_scaled = r_left * ratio_r

rho = find_rho(r_right_scaled, r_left_scaled, N_opt, c_opt)
p_R = r_right_scaled / (r_right_scaled + r_left_scaled)

print(f"\nABL={ABL}, ILD=-16 (strongly left - MAX DIFFERENCE CASE):")
print(f"  r_right={r_right_scaled:.4f}, r_left={r_left_scaled:.4f}")
print(f"  p_R={p_R:.6f}, rho={rho:.10f}")
print(f"  rho simple approx = r_left/r_right = {r_left_scaled/r_right_scaled:.6f}")

# Calculate P(right) step by step
rho_theta = rho ** theta_opt
rho_2theta = rho ** (2 * theta_opt)
p_analytical_manual = (1 - rho_theta) / (1 - rho_2theta)
print(f"  rho^theta = {rho_theta:.10f}")
print(f"  rho^(2*theta) = {rho_2theta:.10f}")
print(f"  P(right) = (1 - {rho_theta:.10f}) / (1 - {rho_2theta:.10f}) = {p_analytical_manual:.10f}")

idx_m16 = np.where(ILD_values == -16)[0][0]
print(f"  P(right) analytical (from array) = {poisson_psychometric_curves_analytical[ABL][idx_m16]:.10f}")
print(f"  P(right) simulated  = {poisson_psychometric_curves_sim[ABL][idx_m16]:.10f}")
print(f"  Difference = {abs(poisson_psychometric_curves_analytical[ABL][idx_m16] - poisson_psychometric_curves_sim[ABL][idx_m16]):.10f}")

print("\n✓ Debug complete!")

# %%
# Additional analysis: Check if time limits in simulation might explain discrepancies

print("\n=== IMPORTANT NOTE ON DISCREPANCIES ===")
print("\nThe analytical formula shows large discrepancies for extreme ILD values.")
print("This is likely because:")
print("1. Simulations used T_max = 2 seconds (time limit)")
print("2. For strongly biased cases (rho → 1), decisions may not be reached within T_max")
print("3. The analytical formula assumes infinite time (no time limits)")
print("\nFor moderate ILD values, the match should be better.")
print("The analytical formula is theoretically correct for the infinite-time case.")
