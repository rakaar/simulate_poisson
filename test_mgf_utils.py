# %%
import numpy as np
import matplotlib.pyplot as plt
import mgf_helper_utils as utils
# %%
# Parameters from original mgfs.py
N = 100
theta = 2
rho = 0.01

# ddm rates parameters
lam = 1.3
l = 0.9
Nr0 = 13.3 * 1
abl = 20
ild = 1
dt = 1e-4

# %%
# Original code approach
r0 = Nr0/N
r_db = (2*abl + ild)/2
l_db = (2*abl - ild)/2
pr = (10 ** (r_db/20))
pl = (10 ** (l_db/20))

den = (pr ** (lam * l) ) + ( pl ** (lam * l) )
rr = (pr ** lam) / den
rl = (pl ** lam) / den
r_right = r0 * rr
r_left = r0 * rl

# Calculate h0 using existing approach
def f(t):
    lambda_p = r_right
    lambda_n = r_left
    
    term1 = (((1 + rho * (np.exp(t) - 1)) ** N) - 1) * lambda_p
    term2 = (((1 + rho * (np.exp(-t) - 1)) ** N) - 1) * lambda_n
    
    return term1 + term2

# Find h0 from original code
from scipy import optimize
a = -10.0
b = -0.001
h0 = optimize.brentq(f, a, b)
print(f"Original code h0 = {h0}")

# Calculate E_W from original code
E_W = dt * N * (r_right - r_left)

# Calculate DDM parameters from original code
mu_ddm = N * (r_right - r_left)
sigma_sq_ddm = N * (r_right + r_left)
h0_ddm = -2 * mu_ddm/sigma_sq_ddm
E_W_ddm = mu_ddm * dt

# %%
# Calculate using helper utils
r_right_utils, r_left_utils = utils.ddm_rates(lam, l, Nr0, N, abl, ild)
print(f"\nOriginal: r_right = {r_right}, r_left = {r_left}")
print(f"Utils: r_right = {r_right_utils}, r_left = {r_left_utils}")

# Calculate h0 using utility function
h0_utils = utils.find_h0(r_right_utils, r_left_utils, N, rho)

# %%
# Define a range of theta values
theta_range = np.arange(1, 11, 1)  # From 1 to 10 in steps of 1

# Original approach - calculate FC and DT arrays
fc_values_orig = []
dt_values_orig = []
fc_ddm_values_orig = []
dt_ddm_values_orig = []

for theta_val in theta_range:
    # Original calculations for Poisson
    fc = 1 / (1 + np.exp(h0 * theta_val))
    fc_values_orig.append(fc)
    
    decision_time = (theta_val * dt / E_W) * np.tanh(-h0 * theta_val / 2)
    dt_values_orig.append(decision_time)
    
    # Original calculations for DDM
    fc_ddm = 1 / (1 + np.exp(h0_ddm * theta_val))
    fc_ddm_values_orig.append(fc_ddm)
    
    dt_ddm_value = (theta_val * dt / E_W_ddm) * np.tanh(-h0_ddm * theta_val / 2)
    dt_ddm_values_orig.append(dt_ddm_value)

# %%
# Utils approach - calculate FC and DT arrays
fc_values_utils = []
dt_values_utils = []
fc_ddm_values_utils = []
dt_ddm_values_utils = []

for theta_val in theta_range:  # Renamed t to theta_val for clarity
    # Using high-level utility function for Poisson
    fc_utils, dt_utils = utils.poisson_fc_dt(
        N=N,
        rho=rho,
        theta=theta_val,
        lam=lam,
        l=l,
        Nr0=Nr0,
        abl=abl,
        ild=ild,
        dt=dt
    )
    fc_values_utils.append(fc_utils)
    dt_values_utils.append(dt_utils)
    
    # Using high-level utility function for DDM
    fc_ddm_utils, dt_ddm_utils = utils.ddm_fc_dt(
        lam=lam,
        l=l,
        Nr0=Nr0,
        N=N,
        abl=abl,
        ild=ild,
        theta=theta_val,
        dt=dt
    )
    fc_ddm_values_utils.append(fc_ddm_utils)
    dt_ddm_values_utils.append(dt_ddm_utils)

# %%
# Create subplots to compare original and utils results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot FC for Poisson model - original vs utils
ax1.plot(theta_range, fc_values_orig, 'o-', color='blue', label='Original Code')
ax1.plot(theta_range, fc_values_utils, 'x--', color='cyan', label='Utils Functions')
ax1.set_xlabel('Theta (θ)')
ax1.set_ylabel('Accuracy (FC)')
ax1.set_title('Poisson Model: Accuracy vs Theta')
ax1.grid(True)
ax1.legend()

# Plot DT for Poisson model - original vs utils
ax2.plot(theta_range, dt_values_orig, 'o-', color='red', label='Original Code')
ax2.plot(theta_range, dt_values_utils, 'x--', color='orange', label='Utils Functions')
ax2.set_xlabel('Theta (θ)')
ax2.set_ylabel('Mean RT (DT)')
ax2.set_title('Poisson Model: Mean RT vs Theta')
ax2.grid(True)
ax2.legend()

# Plot FC for DDM model - original vs utils
ax3.plot(theta_range, fc_ddm_values_orig, 's-', color='darkblue', label='Original Code')
ax3.plot(theta_range, fc_ddm_values_utils, 'x--', color='lightblue', label='Utils Functions')
ax3.set_xlabel('Theta (θ)')
ax3.set_ylabel('Accuracy (FC)')
ax3.set_title('DDM Model: Accuracy vs Theta')
ax3.grid(True)
ax3.legend()

# Plot DT for DDM model - original vs utils
ax4.plot(theta_range, dt_ddm_values_orig, 's-', color='darkred', label='Original Code')
ax4.plot(theta_range, dt_ddm_values_utils, 'x--', color='salmon', label='Utils Functions')
ax4.set_xlabel('Theta (θ)')
ax4.set_ylabel('Mean RT (DT)')
ax4.set_title('DDM Model: Mean RT vs Theta')
ax4.grid(True)
ax4.legend()

# plt.tight_layout()
# plt.savefig('utils_validation_plots.png')

plt.show()  # Use plt.show() instead of fig.show() for notebook display


# %%
# Print the maximum difference between original and utils calculations
print("\nMaximum differences between original and utility calculations:")
print(f"Poisson FC: {max(abs(np.array(fc_values_orig) - np.array(fc_values_utils))):.10f}")
print(f"Poisson DT: {max(abs(np.array(dt_values_orig) - np.array(dt_values_utils))):.10f}")
print(f"DDM FC: {max(abs(np.array(fc_ddm_values_orig) - np.array(fc_ddm_values_utils))):.10f}")
print(f"DDM DT: {max(abs(np.array(dt_ddm_values_orig) - np.array(dt_ddm_values_utils))):.10f}")

# %%
