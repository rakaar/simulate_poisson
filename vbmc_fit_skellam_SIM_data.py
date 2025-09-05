# VBMC single animal
# %% 
import numpy as np
import pandas as pd
from pyvbmc import VBMC
import os
import matplotlib.pyplot as plt
from vbmc_skellam_utils import fpt_density_skellam, fpt_choice_skellam
# %%
# Simulate Skellam FPT data (no truncation) and build a DataFrame with RT and Choice

# True simulation parameters
R_true = 15.0
L_true = 12.0
theta_true = 3
N_TRIALS = 5_000

def simulate_trial(mu1, mu2, theta, rng=None):
    """
    Simulate one Skellam trial dX = dN1 - dN2 with absorbing boundaries at +/- theta.
    Returns (first_passage_time, choice) where choice is +1 for +theta, -1 for -theta.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = 0
    time = 0.0
    total_rate = mu1 + mu2
    prob_up = mu1 / total_rate
    while abs(x) < theta:
        dt = rng.exponential(1.0 / total_rate)
        time += dt
        if rng.random() < prob_up:
            x += 1
        else:
            x -= 1
    choice = 1 if x >= theta else -1
    return time, choice

# Generate simulated dataset
rng = np.random.default_rng(0)
sim_times = np.empty(N_TRIALS, dtype=float)
sim_choices = np.empty(N_TRIALS, dtype=int)
for i in range(N_TRIALS):
    t, ch = simulate_trial(R_true, L_true, theta_true, rng=rng)
    sim_times[i] = t
    sim_choices[i] = ch

# Create DataFrame with required columns and save to CSV
df_sim = pd.DataFrame({
    'RTwrtStim': sim_times,
    'Choice': sim_choices,
})
df_sim.to_csv('simulated_skellam_data.csv', index=False)
print(f"Simulated {len(df_sim)} trials with R={R_true}, L={L_true}, theta={theta_true} -> saved to simulated_skellam_data.csv")

# %%
# funcs
def loglike_fn(params):
    logR, logL, theta = params

    # Transform params
    R = 10**logR
    L = 10**logL
    theta_i = int(round(theta))
    if theta_i < 1:
        theta_i = 1

    # Vectorize across simulated trials
    rts = df_sim['RTwrtStim'].to_numpy()
    choices = df_sim['Choice'].to_numpy().astype(int)

    # Probability of hitting + boundary (scalar)
    p_pos = fpt_choice_skellam(R, L, theta_i, 1)
    p_choice = np.where(choices == 1, p_pos, 1 - p_pos)

    # FPT density at each RT
    rho_t = fpt_density_skellam(rts, R, L, theta_i)

    # No truncation: per-trial probability = density * choice-probability
    p = rho_t * p_choice
    p = np.where(p <= 0, 1e-50, p)

    loglike = float(np.sum(np.log(p)))
    return loglike

def prior(params):
    logR, logL, theta = params
    
    # Proper normalized uniform log-density inside bounds
    logR_width = logR_bounds[1] - logR_bounds[0]
    logL_width = logL_bounds[1] - logL_bounds[0]
    theta_width = theta_bounds[1] - theta_bounds[0]
    return - (np.log(logR_width) + np.log(logL_width) + np.log(theta_width))

def joint_likelihood(params):
    return loglike_fn(params) + prior(params)

# %%
# Declare lb, ub, plaus_lb, plaus_ub
# %%
print('Using simulated data; no ABL/ILD grouping.')

# Define bounds for logR and logL (using the existing bounds)
# For theta, we'll use a separate set of bounds
logR_bounds = [0.5, 4]
logL_bounds = [0.5, 4]
theta_bounds = [0.5, 60.5]

logR_plausible_bounds = [1, 3.8]
logL_plausible_bounds = [1, 3.8]
theta_plausible_bounds = [1, 45]

# Define the joint likelihood function for VBMC
# This function should take the parameters as a single array
# and return the joint log-likelihood
    
# Define bounds for VBMC
lb = np.array([logR_bounds[0], logL_bounds[0], theta_bounds[0]])
ub = np.array([logR_bounds[1], logL_bounds[1], theta_bounds[1]])
plb = np.array([logR_plausible_bounds[0], logL_plausible_bounds[0], theta_plausible_bounds[0]])
pub = np.array([logR_plausible_bounds[1], logL_plausible_bounds[1], theta_plausible_bounds[1]])

# Initialize with random values within plausible bounds
np.random.seed(42)
logR_0 = np.random.uniform(logR_plausible_bounds[0], logR_plausible_bounds[1])
logL_0 = np.random.uniform(logL_plausible_bounds[0], logL_plausible_bounds[1])
theta_0 = np.random.uniform(theta_plausible_bounds[0], theta_plausible_bounds[1])
# logR_0 = np.log10(R_true + 0.1)
# logL_0 = np.log10(L_true + 0.1)
# theta_0 = theta_true

x_0 = np.array([logR_0, logL_0, theta_0])

# Run VBMC on simulated data
print('Initializing VBMC on simulated data...')
try:
    vbmc = VBMC(joint_likelihood, x_0, lb, ub, plb, pub, options={'display': 'on', 'max_fun_evals': 200 * (2 + 3)})
    vp, results = vbmc.optimize()
    print('VBMC optimization completed successfully')
    os.makedirs('poisson_fit_pkls', exist_ok=True)
    vbmc.save('poisson_fit_pkls/vbmc_poisson_fit_SIM.pkl', overwrite=True)
except Exception as e:
    raise ValueError(f'VBMC optimization failed: {e}')

# %%
vp_samples = vp.sample(int(1e6))

logR, logL, theta = vp_samples[0][:, 0].mean(), vp_samples[0][:, 1].mean(), vp_samples[0][:, 2].mean()
print(f"logR: {logR}, logL: {logL}, theta: {theta}")
R = 10**logR
L = 10**logL
theta = int(round(theta))
print(f"R: {R}, L: {L}, theta: {theta}")

# %%
vp_samples = vp.sample(int(1e6))
logR_samples = vp_samples[0][:, 0]
R_samples = 10**logR_samples

logL_samples = vp_samples[0][:, 1]
L_samples = 10**logL_samples

theta_samples = vp_samples[0][:, 2]
theta_samples_jittered = theta_samples + np.random.uniform(-0.5, 0.5, size=theta_samples.shape)
theta_samples_rounded = np.round(theta_samples_jittered).astype(int)

fig, axs = plt.subplots(1, 3, figsize=(14, 4))

axs[0].hist(R_samples, bins=75, color='C0', alpha=0.8)
axs[0].set_title('R samples')
axs[0].set_xlabel('R')
axs[0].set_ylabel('Frequency')

axs[1].hist(L_samples, bins=75, color='C1', alpha=0.8)
axs[1].set_title('L samples')
axs[1].set_xlabel('L')
axs[1].set_ylabel('Frequency')

theta_min, theta_max = theta_samples_rounded.min(), theta_samples_rounded.max()
axs[2].hist(theta_samples_rounded, bins=np.arange(theta_min - 0.5, theta_max + 1.5), color='C2', alpha=0.8)
axs[2].set_title('Theta samples (rounded)')
axs[2].set_xlabel('Theta (integer)')
axs[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()


# %%
R_mean = R_samples.mean()
L_mean = L_samples.mean()
theta_mean = round(theta_samples_rounded.mean())
print(f"R_mean: {R_mean}, L_mean: {L_mean}, theta_mean: {theta_mean}")

# %%
# Compare distributions: true vs posterior means (simulation-based)
def simulate_many(mu1, mu2, theta, n_trials, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    times = np.empty(n_trials, dtype=float)
    choices = np.empty(n_trials, dtype=int)
    for i in range(n_trials):
        t, ch = simulate_trial(mu1, mu2, theta, rng=rng)
        times[i] = t
        choices[i] = ch
    return times, choices
 
 # Visual comparison only: histograms and ECDFs

# Number of simulations for comparison
N_CMP = 500_000
rng_cmp = np.random.default_rng(123)

# Simulate from true parameters
rt_true, ch_true = simulate_many(R_true, L_true, theta_true, N_CMP, rng=rng_cmp)

# Simulate from posterior mean parameters
rt_mean, ch_mean = simulate_many(R_mean, L_mean, theta_mean, N_CMP, rng=rng_cmp)
 
 # No numeric tests or summaries per request; proceed to plotting only.

# Visualization: upper vs lower bound RTs using mirrored step histograms
# t_max = float(np.percentile(np.concatenate([rt_true, rt_mean]), 99.0))
# bins = np.linspace(0.0, t_max, 100)
t_max = 2
bins = np.arange(0, 2, 0.01)

# Split RTs by bound hit
rt_true_up = rt_true[ch_true == 1]
rt_true_lo = rt_true[ch_true == -1]
rt_mean_up = rt_mean[ch_mean == 1]
rt_mean_lo = rt_mean[ch_mean == -1]

# Compute density histograms for step plotting
h_tu, edges = np.histogram(rt_true_up, bins=bins, density=True)
h_mu, _     = np.histogram(rt_mean_up, bins=bins, density=True)
h_tl, _     = np.histogram(rt_true_lo, bins=bins, density=True)
h_ml, _     = np.histogram(rt_mean_lo, bins=bins, density=True)

# Replace NaNs with zeros (can happen if a class is empty)
# h_tu = np.nan_to_num(h_tu, nan=0.0)
# h_mu = np.nan_to_num(h_mu, nan=0.0)
# h_tl = np.nan_to_num(h_tl, nan=0.0)
# h_ml = np.nan_to_num(h_ml, nan=0.0)

plt.figure(figsize=(9, 5))
plt.axhline(0.0, color='k', linewidth=1)

# Upper bound (positive y)
plt.step(edges[:-1], h_tu, where='post', label='TRUE upper', color='C0')
plt.step(edges[:-1], h_mu, where='post', label='MEAN upper', color='C1')

# Lower bound (negative y)
plt.step(edges[:-1], -h_tl, where='post', label='TRUE lower', color='C0', linestyle='--')
plt.step(edges[:-1], -h_ml, where='post', label='MEAN lower', color='C1', linestyle='--')

ymax = 1.1 * float(max(h_tu.max(), h_mu.max(), h_tl.max(), h_ml.max(), 1e-12))
plt.ylim(-ymax, ymax)
plt.xlabel('RT')
plt.ylabel('Density (upper positive, lower negative)')
plt.title('RT distributions by bound: TRUE vs MEAN')
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

# Probability-weighted mirrored step histograms (areas equal choice probabilities)
# Use the same bin edges as above for direct visual comparison
widths = np.diff(edges)

# Empirical choice probabilities
p_true_up = rt_true_up.size / max(rt_true.size, 1)
p_true_lo = rt_true_lo.size / max(rt_true.size, 1)
p_mean_up = rt_mean_up.size / max(rt_mean.size, 1)
p_mean_lo = rt_mean_lo.size / max(rt_mean.size, 1)

# Current areas under density histograms (may be <1 if tails are cut by t_max)
area_tu = float(np.sum(h_tu * widths))
area_tl = float(np.sum(h_tl * widths))
area_mu = float(np.sum(h_mu * widths))
area_ml = float(np.sum(h_ml * widths))

# Scale densities so that areas match the corresponding probabilities
eps = 1e-12
h_tu_w = h_tu * (p_true_up / max(area_tu, eps))
h_tl_w = h_tl * (p_true_lo / max(area_tl, eps))
h_mu_w = h_mu * (p_mean_up / max(area_mu, eps))
h_ml_w = h_ml * (p_mean_lo / max(area_ml, eps))

plt.figure(figsize=(9, 5))
plt.axhline(0.0, color='k', linewidth=1)

# Upper bound (positive y)
plt.step(edges[:-1], h_tu_w, where='post', label='TRUE upper (area=P(+1))', color='C0')
plt.step(edges[:-1], h_mu_w, where='post', label='MEAN upper (area=P(+1))', color='C1')

# Lower bound (negative y)
plt.step(edges[:-1], -h_tl_w, where='post', label='TRUE lower (area=P(-1))', color='C0', linestyle='--')
plt.step(edges[:-1], -h_ml_w, where='post', label='MEAN lower (area=P(-1))', color='C1', linestyle='--')

ymax_w = 1.1 * float(max(h_tu_w.max(), h_mu_w.max(), h_tl_w.max(), h_ml_w.max(), 1e-12))
plt.ylim(-ymax_w, ymax_w)
plt.xlabel('RT')
plt.ylabel('Probability density (areas reflect choice probabilities)')
plt.title('RT distributions by bound: TRUE vs MEAN (area = frac)')
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

# Optional: overlay ECDFs for RTs
def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys

x_true, y_true = ecdf(rt_true)
x_mean, y_mean = ecdf(rt_mean)

plt.figure(figsize=(8, 4))
plt.plot(x_true, y_true, label='TRUE ECDF')
plt.plot(x_mean, y_mean, label='MEAN ECDF')
plt.xlabel('RT')
plt.ylabel('ECDF')
plt.title('RT ECDFs: TRUE vs MEAN')
plt.legend()
plt.tight_layout()
plt.show()

# Choice probability comparison (bar plot)
p_pos_true = float(np.mean(ch_true == 1))
p_pos_mean = float(np.mean(ch_mean == 1))
plt.figure(figsize=(4, 4))
plt.bar(['TRUE', 'MEAN'], [p_pos_true, p_pos_mean], color=['C0', 'C1'], alpha=0.8)
plt.ylim(0, 1)
plt.ylabel('P(choice = +1)')
plt.title('Choice probability: TRUE vs MEAN')
plt.tight_layout()
plt.show()

# %%
# Validate fpt_choice_skellam orientation via simulation at TRUE params
# We simulate fresh data with (R_true, L_true, theta_true) and compare
# empirical P(+1) against the two orderings:
#   1) fpt_choice_skellam(L, R, theta, +1)  [mu2=L, mu1=R]
#   2) fpt_choice_skellam(R, L, theta, +1)  [mu2=R, mu1=L]

N_VALIDATE = 100000
rng_val = np.random.default_rng(777)

choices_val = np.empty(N_VALIDATE, dtype=int)
for i in range(N_VALIDATE):
    _, ch = simulate_trial(R_true, L_true, theta_true, rng=rng_val)
    choices_val[i] = ch

p_empirical = float(np.mean(choices_val == 1))

# Theoretical probabilities using both orderings
p_theory_LR = float(fpt_choice_skellam(L_true, R_true, int(theta_true), 1))
p_theory_RL = float(fpt_choice_skellam(R_true, L_true, int(theta_true), 1))

print("==== Validate fpt_choice_skellam orientation ====")
print(f"TRUE params: R={R_true}, L={L_true}, theta={theta_true}, N={N_VALIDATE}")
print(f"Theory using (L,R): P(+1) = {p_theory_LR:.6f}")
print(f"Theory using (R,L): P(+1) = {p_theory_RL:.6f}")
print(f"Empirical from simulation: P(+1) = {p_empirical:.6f}")

# %%
