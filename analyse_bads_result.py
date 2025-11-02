# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import ks_2samp
from joblib import Parallel, delayed
from tqdm import tqdm
from bads_utils import (
    lr_rates_from_ABL_ILD,
    simulate_single_ddm_trial,
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
print(f"\nOriginal KS-statistic: {optimize_result['fval']:.6f}")
print(f"Function evaluations: {optimize_result['func_count']}")

# %%
# Simulate DDM data with same parameters as original optimization
print("\n=== DDM SIMULATION ===")
ABL = ddm_stimulus['ABL']
ILD = ddm_stimulus['ILD']

ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
    ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
)

mu_ddm = ddm_right_rate - ddm_left_rate
sigma_sq_ddm = ddm_right_rate + ddm_left_rate
sigma_ddm = np.sqrt(sigma_sq_ddm)

# Simulation parameters
dt = 1e-4
dB = 1e-2
N_sim_ddm = int(250e3)
T = 20
n_steps = int(T/dt)

print(f"Simulating {N_sim_ddm} DDM trials...")
print(f"Parameters: mu={mu_ddm:.4f}, sigma={sigma_ddm:.4f}, theta={ddm_params['theta']:.4f}")

# Parallel DDM simulation
ddm_results = Parallel(n_jobs=-1)(
    delayed(simulate_single_ddm_trial)(i, mu_ddm, sigma_ddm, ddm_params['theta'], dt, dB, n_steps) 
    for i in tqdm(range(N_sim_ddm), desc='Simulating DDM')
)
ddm_data = np.array(ddm_results)

# Filter decided trials
ddm_decided_mask = ~np.isnan(ddm_data[:, 0])
ddm_rts_decided = ddm_data[ddm_decided_mask, 0]

print(f"DDM simulation complete:")
print(f"  Total trials: {len(ddm_data)}")
print(f"  Decided trials: {len(ddm_rts_decided)}")
print(f"  Mean RT: {np.mean(ddm_rts_decided):.4f} s")
print(f"  Median RT: {np.median(ddm_rts_decided):.4f} s")

# %%
# Simulate Poisson model with optimized parameters
print("\n=== POISSON SIMULATION ===")

poisson_params = {
    'N_right': N_opt,
    'N_left': N_opt,
    'c': c_opt,
    'r_right': r_opt,
    'r_left': r_opt,
    'theta': theta_opt,
    'T': 20,
    'exponential_noise_scale': 0
}

print(f"Simulating {N_sim_ddm} Poisson trials...")
poisson_results = simulate_poisson_rts(poisson_params, n_trials=N_sim_ddm, 
                                       seed=42, verbose=True)

# Filter decided trials
poisson_decided_mask = ~np.isnan(poisson_results[:, 0])
poisson_rts_decided = poisson_results[poisson_decided_mask, 0]

print(f"\nPoisson simulation complete:")
print(f"  Total trials: {len(poisson_results)}")
print(f"  Decided trials: {len(poisson_rts_decided)}")
print(f"  Mean RT: {np.mean(poisson_rts_decided):.4f} s")
print(f"  Median RT: {np.median(poisson_rts_decided):.4f} s")

# %%
# Compute KS-statistic
ks_stat, ks_pval = ks_2samp(ddm_rts_decided, poisson_rts_decided)

print("\n=== KS-STATISTIC COMPARISON ===")
print(f"KS-statistic: {ks_stat:.6f}")
print(f"KS p-value: {ks_pval:.6f}")

# %%
# Plot RTD histograms, CDFs, and CDF difference
print("\n=== PLOTTING RTD COMPARISON ===")

bins = np.arange(0, 2.01, 0.01)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Create 1x3 subplot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: RTD histograms
ax1 = axes[0]
ddm_hist, _ = np.histogram(ddm_rts_decided, bins=bins, density=True)
ax1.hist(ddm_rts_decided, bins=bins, density=True, color='b', alpha=0.7, label='DDM', histtype='step', linewidth=2)
poisson_hist, _ = np.histogram(poisson_rts_decided, bins=bins, density=True)
ax1.hist(poisson_rts_decided, bins=bins, density=True, color='r', alpha=0.7, label='Poisson (optimized)', histtype='step', linewidth=2)

ax1.set_xlabel('Reaction Time (s)', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title('RTD Comparison', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: CDFs
ax2 = axes[1]
# Sort data for CDF
ddm_sorted = np.sort(ddm_rts_decided)
poisson_sorted = np.sort(poisson_rts_decided)

# Compute CDFs
ddm_cdf = np.arange(1, len(ddm_sorted) + 1) / len(ddm_sorted)
poisson_cdf = np.arange(1, len(poisson_sorted) + 1) / len(poisson_sorted)

ax2.plot(ddm_sorted, ddm_cdf, 'b-', linewidth=2, label='DDM', alpha=0.7)
ax2.plot(poisson_sorted, poisson_cdf, 'r--', linewidth=2, label='Poisson (optimized)', alpha=0.7)

ax2.set_xlabel('Reaction Time (s)', fontsize=12)
ax2.set_ylabel('Cumulative Probability', fontsize=12)
ax2.set_title('CDF Comparison', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 2)

# Plot 3: Absolute difference of CDFs
ax3 = axes[2]
# Interpolate to common time points for difference calculation
common_times = np.linspace(0, 2, 1000)
ddm_cdf_interp = np.interp(common_times, ddm_sorted, ddm_cdf, left=0, right=1)
poisson_cdf_interp = np.interp(common_times, poisson_sorted, poisson_cdf, left=0, right=1)
cdf_diff = np.abs(ddm_cdf_interp - poisson_cdf_interp)

ax3.plot(common_times, cdf_diff, 'g-', linewidth=2)
ax3.axhline(y=ks_stat, color='k', linestyle='--', linewidth=1.5, 
            label=f'Max diff (KS = {ks_stat:.6f})')

ax3.set_xlabel('Reaction Time (s)', fontsize=12)
ax3.set_ylabel('|CDF difference|', fontsize=12)
ax3.set_title('Absolute CDF Difference', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 2)

plt.suptitle(f'DDM vs Poisson Model Comparison (KS-statistic = {ks_stat:.6f})', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('rtd_comparison_ddm_vs_poisson.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: rtd_comparison_ddm_vs_poisson.png")
plt.show()

# %%
# Parameter comparison table
print("\n=== PARAMETER COMPARISON TABLE ===")

# Verify DDM rates are equal (they should be for ILD=0)
print(f"DDM right rate: {ddm_right_rate:.6f}")
print(f"DDM left rate: {ddm_left_rate:.6f}")
print(f"Rates equal: {np.isclose(ddm_right_rate, ddm_left_rate)}")

# Calculate DDM's effective r (using actual DDM rate / N_poisson)
r_ddm_effective = ddm_right_rate / N_opt

# Calculate percentage differences and ratios
pct_diff_r = 100 * (r_opt - r_ddm_effective) / r_ddm_effective
pct_diff_theta = 100 * (theta_opt - ddm_params['theta']) / ddm_params['theta']
ratio_r = r_opt / r_ddm_effective
ratio_theta = theta_opt / ddm_params['theta']

print("\n{:<10} {:<15} {:<20} {:<12} {:<12}".format(
    "Parameter", "Poisson", "DDM", "Ratio", "% Diff"
))
print("-" * 70)

# N row
print("{:<10} {:<15} {:<20} {:<12} {:<12}".format(
    "N", f"{N_opt}", "NA", "NA", "NA"
))

# r row (with note about DDM calculation)
print("{:<10} {:<15} {:<20} {:<12} {:<12}".format(
    "r", f"{r_opt:.4f}", f"{r_ddm_effective:.6f} (rate/N)", f"{ratio_r:.4f}", f"{pct_diff_r:.2f}%"
))

# theta row
print("{:<10} {:<15} {:<20} {:<12} {:<12}".format(
    "theta", f"{theta_opt:.4f}", f"{ddm_params['theta']:.4f}", f"{ratio_theta:.4f}", f"{pct_diff_theta:.2f}%"
))

# Additional Poisson-specific parameters
print("\n{:<10} {:<20}".format("Parameter", "Poisson Value"))
print("-" * 35)
print("{:<10} {:<20}".format("k", f"{k_opt:.2f}"))
print("{:<10} {:<20}".format("c", f"{c_opt:.6f}"))

print("\n\nNote: DDM's effective r = ddm_rate/N_poisson = {:.6f}/{} = {:.6f}".format(
    ddm_right_rate, N_opt, r_ddm_effective
))

# RT Statistics
print("\n\n=== REACTION TIME STATISTICS ===")
print("\n{:<20} {:<15} {:<15}".format("Metric", "DDM", "Poisson"))
print("-" * 50)
print("{:<20} {:<15} {:<15}".format("N trials", f"{N_sim_ddm}", f"{N_sim_ddm}"))
print("{:<20} {:<15} {:<15}".format("N decided", f"{len(ddm_rts_decided)}", f"{len(poisson_rts_decided)}"))
print("{:<20} {:<15.4f} {:<15.4f}".format("Mean RT (s)", np.mean(ddm_rts_decided), np.mean(poisson_rts_decided)))
print("{:<20} {:<15.4f} {:<15.4f}".format("Std RT (s)", np.std(ddm_rts_decided), np.std(poisson_rts_decided)))
print("{:<20} {:<15.4f} {:<15.4f}".format("Median RT (s)", np.median(ddm_rts_decided), np.median(poisson_rts_decided)))

# KS Test Results
print("\n\n=== KOLMOGOROV-SMIRNOV TEST ===")
print(f"KS-statistic: {ks_stat:.6f}")
print(f"p-value: {ks_pval:.6f}")

# %%
# psychometric
ILD_values  = np.arange(-16,16,0.01)
ABL_values = [20, 40, 60]

def psyc_ddm(ABL, ILD, ddm_params):
    ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
        ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
    )
    mu = ddm_right_rate - ddm_left_rate
    sigma_sq = ddm_right_rate + ddm_left_rate
    gamma =  (ddm_params['theta'])*(mu/sigma_sq)
    return 1/ ( 1 + np.exp(- 2 * gamma))


def psyc_poisson(ABL, ILD, ddm_params, theta_poisson, rate_scaling_factor):
    # pass
    ddm_right_rate, ddm_left_rate = lr_rates_from_ABL_ILD(
        ABL, ILD, ddm_params['Nr0'], ddm_params['lam'], ddm_params['ell']
    )
    if ddm_right_rate == ddm_left_rate:
        return 0.5
    poisson_right_rate = ddm_right_rate * rate_scaling_factor
    poisson_left_rate = ddm_left_rate * rate_scaling_factor
    ratio = poisson_left_rate / poisson_right_rate
    return (1 - (ratio ** theta_poisson)) / (1 - (ratio ** (2*theta_poisson)))


    

# %%
# theta_opt
# ratio_r
print(f'ratio poisson/ddm rates at ILD 0 = {ratio_r}')
print(f'poisson theta = {theta_opt}')

# %%
# %%
# Plot psychometric curves: P_right vs ILD for both models across ABL values
print("\n=== PLOTTING PSYCHOMETRIC CURVES ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, ABL in enumerate(ABL_values):
    ax = axes[idx]
    
    # Calculate P_right for DDM across all ILD values
    p_right_ddm = np.array([psyc_ddm(ABL, ILD, ddm_params) for ILD in ILD_values])
    
    # Calculate P_right for Poisson across all ILD values
    p_right_poisson = np.array([psyc_poisson(ABL, ILD, ddm_params, theta_opt, ratio_r) 
                                 for ILD in ILD_values])
    
    # Plot both curves
    ax.plot(ILD_values, p_right_ddm, 'b-', linewidth=2, label='DDM', alpha=0.7)
    ax.plot(ILD_values, p_right_poisson, 'r--', linewidth=2, label='Poisson', alpha=0.7)
    
    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('ILD (dB)', fontsize=12)
    ax.set_ylabel('P(Right)', fontsize=12)
    ax.set_title(f'ABL = {ABL} dB', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-16, 16)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

plt.suptitle('Psychometric Curves: DDM vs Poisson Model', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('psychometric_comparison_ddm_vs_poisson.png', dpi=300, bbox_inches='tight')
print("✓ Plot saved: psychometric_comparison_ddm_vs_poisson.png")
plt.show()
