# %%
# Plot p_R vs c for different ILD values
# For each c, find rho that solves Phi(rho) = 1, then compute p_R from stimulus rates

import numpy as np
import matplotlib.pyplot as plt
import pickle
from bads_utils import lr_rates_from_ABL_ILD

# %%
# Load BADS optimization results to get DDM parameters and rate scaling

pkl_filename = 'bads_optimization_results_singlerate_20251029_195526.pkl'

# Define dummy obj_func to allow unpickling
def obj_func(x):
    """Dummy function for unpickling - not actually used."""
    pass

print(f"Loading results from: {pkl_filename}")
with open(pkl_filename, 'rb') as f:
    results = pickle.load(f)

# Extract parameters
N_opt = results['optimized_params']['N']
r_opt = results['optimized_params']['r']
theta_opt = results['optimized_params']['theta']
ddm_params = results['ddm_params']

# Calculate rate scaling factor
r_ddm_effective = ddm_params['Nr0'] / N_opt
ratio_r = r_opt / r_ddm_effective

print("\n=== PARAMETERS ===")
print(f"N: {N_opt}")
print(f"theta: {theta_opt:.4f}")
print(f"Rate scaling factor: {ratio_r:.6f}")
print(f"DDM params: Nr0={ddm_params['Nr0']}, lam={ddm_params['lam']}, ell={ddm_params['ell']}")

# %%
# Define Phi function

def compute_phi(rho, p_R, N, c):
    """
    Compute Phi(rho) = p_R*(1-c+c*rho)^N + (1-p_R)*(1-c+c/rho)^N
    
    Parameters:
    -----------
    rho : float
        The martingale root (NOT the simple rate ratio; they coincide only as c→0)
    p_R : float
        Probability of right event = r_right / (r_right + r_left)
    N : int
        Number of neurons per pool
    c : float
        Within-pool correlation coefficient
    
    Returns:
    --------
    float : Phi(rho)
    """
    term1 = p_R * ((1 - c) + c * rho) ** N
    term2 = (1 - p_R) * ((1 - c) + c / rho) ** N
    return term1 + term2


# %%
# Setup parameters

ABL = 20  # Fixed ABL
ILD_values = [-16, -8, -4, -2, 0, 2, 4, 8, 16]  # Include both signs to see >0.5 for ILD>0
N = 615  # Fixed N

# Range of c values to explore
c_values = np.linspace(0.0, 0.01, 50)[1:]  # (0, 0.2], skip c=0 to avoid trivial root handling

print(f"\n=== SETUP ===")
print(f"ABL: {ABL} dB")
print(f"ILD values: {ILD_values}")
print(f"N: {N}")
print(f"c range: [{c_values[0]:.6f}, {c_values[-1]:.6f}] (n={len(c_values)})")

# %%
# For each ILD and c, find rho(c) such that Phi(rho, p_R, N, c) = 1
# where p_R is FIXED from the stimulus rates
# NOTE: ρ is the martingale root, NOT the simple rate ratio (they coincide only as c→0)

from scipy.optimize import brentq

def _log_phi_of_logrho(t, p_R, N, c):
    """
    Compute log(Phi(exp(t))) in stable form.
    Returns >0 if Phi>1, <0 if Phi<1, =0 if Phi=1.
    """
    rho = np.exp(t)
    a = np.log(p_R)     + N * np.log((1 - c) + c * rho)
    b = np.log1p(-p_R)  + N * np.log((1 - c) + c / rho)
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))  # log(Phi)

def find_rho_for_phi_equals_1(p_R, N, c, eps=1e-6, T0=2.0, Tstep=2.0, Tmax=80.0):
    """
    Find the non-trivial root rho such that Phi(rho, p_R, N, c) = 1.
    
    rho=1 (t=0) is always a trivial root. For biased stimuli:
    - If p_R > 0.5 (right-biased): non-trivial root at t < 0 (rho < 1)
    - If p_R < 0.5 (left-biased): non-trivial root at t > 0 (rho > 1)
    
    In the small-c limit, rho → r_L/r_R = (1-p_R)/p_R.
    """
    # Symmetric case: only root is rho=1
    if abs(p_R - 0.5) < 1e-12:
        return 1.0

    # Choose the side of the non-trivial root
    # t_sign > 0 means search for rho>1; t_sign < 0 for rho<1
    t_sign = +1.0 if p_R < 0.5 else -1.0
    left  = t_sign * eps
    right = t_sign * T0

    fL = _log_phi_of_logrho(left,  p_R, N, c)   # should be < 0 just off zero
    fR = _log_phi_of_logrho(right, p_R, N, c)

    # If we didn't straddle zero, expand outward until we do (or we hit Tmax)
    T = T0
    while np.sign(fL) == np.sign(fR) and T < Tmax:
        T *= Tstep
        right = t_sign * T
        fR = _log_phi_of_logrho(right, p_R, N, c)

        # If still no sign change, try nudging closer to zero on the inner side
        if np.sign(fL) == np.sign(fR):
            left = t_sign * (abs(left) / 2.0)
            if abs(left) < 1e-15:  # avoid exactly t=0
                left = t_sign * 1e-15
            fL = _log_phi_of_logrho(left, p_R, N, c)

    # If still no bracket, fall back to small-c limit
    if np.sign(fL) == np.sign(fR):
        return (1 - p_R) / p_R

    t_root = brentq(lambda t: _log_phi_of_logrho(t, p_R, N, c), left, right, xtol=1e-12, maxiter=1000)
    return float(np.exp(t_root))


def P_right_from_rates_c(r_right, r_left, N, theta, c):
    """
    Compute P(choose Right) given rates and correlation c.
    
    Steps:
    1. Fix p_R from rates
    2. Solve for rho(c) from Phi(rho)=1
    3. Use rho directly in the formula (works for both rho < 1 and rho > 1)
    4. Compute P(Right) = (1 - rho^theta) / (1 - rho^(2*theta))
    
    For rho < 1 (right-biased): P(Right) > 0.5
    For rho > 1 (left-biased): P(Right) < 0.5
    For rho = 1 (symmetric): P(Right) = 0.5
    """
    # Symmetric case (ILD=0): P=0.5 for any c
    if np.isclose(r_right, r_left):
        return 0.5
    
    p_R = r_right / (r_right + r_left)
    rho = find_rho_for_phi_equals_1(p_R, N, c)
    
    # Use rho directly (no min mapping needed)
    x = theta * np.log(rho)
    num = -np.expm1(x)            # 1 - rho**theta (stable)
    den = -np.expm1(2.0 * x)      # 1 - rho**(2*theta)
    return float(num / den) if abs(den) > 1e-300 else 0.5


print("\n=== COMPUTING P(Right) VS C FOR EACH ILD ===")

results = {}

for ILD in ILD_values:
    print(f"\nILD = {ILD} dB:")
    
    # Get DDM rates for this stimulus
    ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
        ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
    )
    
    # Scale to Poisson rates
    poisson_right_rate = ddm_right_rate * ratio_r
    poisson_left_rate = ddm_left_rate * ratio_r
    
    print(f"  r_right = {poisson_right_rate:.4f}, r_left = {poisson_left_rate:.4f}")
    
    # Compute P(Right) for each c value
    P_right_values = []
    for c in c_values:
        P_right = P_right_from_rates_c(poisson_right_rate, poisson_left_rate, N_opt, theta_opt, c)
        P_right_values.append(P_right)
    
    P_right_values = np.array(P_right_values)
    
    results[ILD] = {
        'P_right_values': P_right_values,
        'r_right': poisson_right_rate,
        'r_left': poisson_left_rate
    }
    
    print(f"  P(Right) range: [{P_right_values.min():.6f}, {P_right_values.max():.6f}]")

print("\n✓ Computations complete!")

# %%
# Create plot: P(Right) vs c for all ILDs

print("\n=== CREATING PLOT ===")

fig, ax = plt.subplots(figsize=(10, 6))

for ILD in ILD_values:
    data = results[ILD]
    P_right_values = data['P_right_values']
    
    # Plot P(Right) vs c
    ax.plot(c_values, P_right_values, linewidth=2, marker='o', markersize=4, label=f'ILD = {ILD} dB')

# Add horizontal line at P=0.5
ax.axhline(0.5, ls=':', color='gray', lw=1.5, alpha=0.7, label='P = 0.5')

# Add vertical line at c=0.0025 (your fitted value)
ax.axvline(x=0.0025, color='red')

# Formatting
ax.set_xlabel('Correlation (c)', fontsize=12)
ax.set_ylabel('P(choose Right)', fontsize=12)
ax.set_title(f'P(Right) vs Correlation (c) for Different ILD Values\n'
             f'ABL = {ABL} dB, N = {N_opt}, θ = {theta_opt:.2f}', 
             fontsize=14, fontweight='bold')
ax.set_xlim(c_values.min(), c_values.max())
ax.set_ylim(0, 1)
# ax.grid(alpha=0.3)
ax.legend(title='Stimulus', ncol=2, fontsize=9, loc='best')

plt.tight_layout()

# Save plot
plot_filename = f'P_right_vs_c_ABL{ABL}_N{N_opt}.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved: {plot_filename}")

plt.show()

# %%
# Additional analysis: Check P(Right) behavior

print("\n=== ANALYSIS: P(Right) AT FITTED c ===")

c_fitted = 0.0025
print(f"\nAt fitted c = {c_fitted}:")

for ILD in ILD_values:
    # Get the P(Right) at the fitted c value (find closest c)
    idx_closest = np.argmin(np.abs(c_values - c_fitted))
    c_closest = c_values[idx_closest]
    P_at_fitted = results[ILD]['P_right_values'][idx_closest]
    
    # Also compute at smallest c for comparison
    P_at_min_c = results[ILD]['P_right_values'][0]
    
    print(f"\n  ILD = {ILD:3d} dB:")
    print(f"    P(Right) at c={c_closest:.5f}: {P_at_fitted:.6f}")
    print(f"    P(Right) at c={c_values[0]:.5f} (min): {P_at_min_c:.6f}")
    print(f"    Change: {P_at_fitted - P_at_min_c:+.6f}")

print("\n✓ Analysis complete!")
