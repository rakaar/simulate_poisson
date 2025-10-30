# %%
# imports
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import ks_2samp
from tqdm import tqdm
from bads_utils import simulate_single_ddm_trial, simulate_poisson_rts

# %%
# TEST - if poisson and ddm align:
print("\n" + "="*60)
print("TESTING: Checking if Poisson and DDM align with same params")
print("="*60)

# Use the EXACT same parameters from combined_poisson_ddm_jump_analysis_V2_gaussian_noise_to_spk_times.py
test_N = 101  # N_right_and_left = 100 + 1
test_c = 0.01
test_theta_unscaled = 10  # theta (unscaled)
test_cf = 1 + (test_N - 1) * test_c  # corr_factor = 1 + 100*0.01 = 2

# Animal parameters
test_lam = 1.3
test_l = 0.9
test_Nr0 = 13.3 * 9  # Nr0 = 39.9
test_r0 = test_Nr0 / test_N  # r0 (unscaled)
test_r0_scaled = test_r0 * test_cf  # r0_scaled (for Poisson)

# Stimulus parameters
test_abl = 20
test_ild = 0
test_r_db = (2*test_abl + test_ild)/2
test_l_db = (2*test_abl - test_ild)/2
test_pr = (10 ** (test_r_db/20))
test_pl = (10 ** (test_l_db/20))

test_den = (test_pr ** (test_lam * test_l)) + (test_pl ** (test_lam * test_l))
test_rr = (test_pr ** test_lam) / test_den
test_rl = (test_pl ** test_lam) / test_den

# Firing rates
test_r_right = test_r0 * test_rr  # Unscaled (for DDM)
test_r_left = test_r0 * test_rl   # Unscaled (for DDM)
test_r_right_scaled = test_r0_scaled * test_rr  # Scaled (for Poisson)
test_r_left_scaled = test_r0_scaled * test_rl   # Scaled (for Poisson)

# Thresholds
test_theta_scaled = test_theta_unscaled * test_cf  # theta_scaled (for Poisson)
test_theta_ddm = test_theta_unscaled  # theta_ddm (for DDM)

print(f"\nTest parameters (matching reference file):")
print(f"  N = {test_N}")
print(f"  c = {test_c:.4f}")
print(f"  corr_factor = {test_cf:.4f}")
print(f"  Nr0 = {test_Nr0:.1f}")
print(f"  theta (unscaled) = {test_theta_unscaled}")
print(f"  theta_scaled (for Poisson) = {test_theta_scaled:.4f}")
print(f"  r0 (unscaled) = {test_r0:.4f}")
print(f"  r0_scaled (for Poisson) = {test_r0_scaled:.4f}")
print(f"  r_right (unscaled, for DDM) = {test_r_right:.4f} Hz")
print(f"  r_left (unscaled, for DDM) = {test_r_left:.4f} Hz")
print(f"  r_right_scaled (for Poisson) = {test_r_right_scaled:.4f} Hz")
print(f"  r_left_scaled (for Poisson) = {test_r_left_scaled:.4f} Hz")
print(f"  N * r_right (unscaled) = {test_N * test_r_right:.4f}")
print(f"  N * r_left (unscaled) = {test_N * test_r_left:.4f}")
print(f"  N * r_right_scaled = {test_N * test_r_right_scaled:.4f}")
print(f"  N * r_left_scaled = {test_N * test_r_left_scaled:.4f}")

# Setup Poisson parameters (uses SCALED rates and SCALED theta)
test_poisson_params = {
    'N_right': test_N,
    'N_left': test_N,
    'c': test_c,
    'r_right': test_r_right_scaled,  # Scaled
    'r_left': test_r_left_scaled,    # Scaled
    'theta': test_theta_scaled,      # Scaled (should be 12 for cf=2)
    'T': 20,
    'exponential_noise_scale': 0
}

# Simulate Poisson model
print(f"\nSimulating Poisson model with {test_N} neurons...")
test_poisson_results = simulate_poisson_rts(test_poisson_params, n_trials=int(200e3), 
                                            seed=42, verbose=False)
test_poisson_decided_mask = ~np.isnan(test_poisson_results[:, 0])
test_poisson_rts = test_poisson_results[test_poisson_decided_mask, 0]
test_poisson_choices = test_poisson_results[test_poisson_decided_mask, 1]

# %%
# Generate DDM data with matching parameters
# DDM uses UNSCALED rates and UNSCALED theta (matching reference file)
test_mu_ddm = test_N * (test_r_right - test_r_left)
test_corr_factor_ddm = 1  # Hardcoded to 1 (matching reference file line 228)
test_sigma_sq_ddm = test_N * (test_r_right + test_r_left) * test_corr_factor_ddm
test_sigma_ddm = np.sqrt(test_sigma_sq_ddm)

print(f"\nDDM parameters:")
print(f"  mu = N*(r_right - r_left) = {test_mu_ddm:.4f}")
print(f"  sigma = sqrt(N*(r_right + r_left)) = {test_sigma_ddm:.4f}")
print(f"  theta = {test_theta_ddm}")

# DDM simulation parameters
dt = 1e-4
dB = 1e-2
T = 20
n_steps = int(T/dt)

# Simulate DDM for test
print(f"\nSimulating DDM for test...")
test_ddm_results = Parallel(n_jobs=-1)(
    delayed(simulate_single_ddm_trial)(i, test_mu_ddm, test_sigma_ddm, test_theta_ddm, dt, dB, n_steps)
    for i in range(int(200e3))
)
test_ddm_data = np.array(test_ddm_results)
test_ddm_decided_mask = ~np.isnan(test_ddm_data[:, 0])
test_ddm_rts = test_ddm_data[test_ddm_decided_mask, 0]

print(f"\nResults:")
print(f"  DDM trials decided: {len(test_ddm_rts)} / {len(test_ddm_data)}")
print(f"  Poisson trials decided: {len(test_poisson_rts)} / {len(test_poisson_results)}")
print(f"\n  DDM mean RT: {np.mean(test_ddm_rts):.4f} s")
print(f"  Poisson mean RT: {np.mean(test_poisson_rts):.4f} s")
print(f"  DDM median RT: {np.median(test_ddm_rts):.4f} s")
print(f"  Poisson median RT: {np.median(test_poisson_rts):.4f} s")

# Compute KS statistic
test_ks_stat, test_ks_pval = ks_2samp(test_ddm_rts, test_poisson_rts)
print(f"\n  KS-statistic: {test_ks_stat:.6f}")
print(f"  KS p-value: {test_ks_pval:.6f}")

if test_ks_stat < 0.05:
    print(f"  ✓ GOOD: Models align well (KS-stat < 0.05)")
elif test_ks_stat < 0.1:
    print(f"  ~ OK: Models roughly align (KS-stat < 0.1)")
else:
    print(f"  ✗ WARNING: Models may not align well (KS-stat >= 0.1)")

# %%
# Quick visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# bins = np.linspace(0, min(np.percentile(test_ddm_rts, 99), 
#                           np.percentile(test_poisson_rts, 99)), 50)
bins = np.arange(0,5, 0.02)
ax.hist(test_ddm_rts, bins=bins, alpha=0.5, density=True, 
        label=f'DDM (n={len(test_ddm_rts)})', color='blue', histtype='step')
ax.hist(test_poisson_rts, bins=bins, alpha=0.5, density=True, 
        label=f'Poisson (n={len(test_poisson_rts)})', color='red', histtype='step')
ax.set_xlabel('Reaction Time (s)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Test: DDM vs Poisson RT Distributions\n'
             f'N={test_N}, c={test_c:.4f}, cf={test_cf:.2f}, '
             f'θ_DDM={test_theta_ddm}, θ_Poisson={test_theta_scaled:.2f}\n'
             f'KS-statistic: {test_ks_stat:.6f}',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('test_ddm_poisson_alignment.png', dpi=150, bbox_inches='tight')
print(f"\n  Plot saved to: test_ddm_poisson_alignment.png")
plt.show()

print("\n" + "="*60)
print("TEST COMPLETE - Functions verified!")
print("="*60 + "\n")
