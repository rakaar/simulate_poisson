"""
VBMC fit of Skellam FPT model with theta fixed on a grid (1..50),
fitting only logR and logL on simulated Skellam FPT data.

- Simulates a single dataset once using TRUE parameters (R_true, L_true, theta_true).
- For each fixed theta in {1, ..., 50}, builds a 2-parameter posterior (logR, logL)
  and runs VBMC.
- Saves a separate pickle per theta into `poisson_fit_pkls_theta_fixed_logR_logL/`
  with the fixed theta embedded in the filename.

This file mirrors the structure of `vbmc_fit_skellam_SIM_data.py` but now fixes
integer theta and only fits logR and logL.
"""

# %% Imports
import os
import numpy as np
import pandas as pd
from pyvbmc import VBMC
import contextlib
import traceback

from vbmc_skellam_utils import fpt_density_skellam, fpt_choice_skellam

# %%
# Simulate Skellam FPT data (no truncation) and build a DataFrame with RT and Choice
# True simulation parameters (unchanged)
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

# Generate simulated dataset once
rng = np.random.default_rng(0)
sim_times = np.empty(N_TRIALS, dtype=float)
sim_choices = np.empty(N_TRIALS, dtype=int)
for i in range(N_TRIALS):
    t, ch = simulate_trial(R_true, L_true, theta_true, rng=rng)
    sim_times[i] = t
    sim_choices[i] = ch

# Create DataFrame with required columns (not saving to CSV here)
df_sim = pd.DataFrame({
    'RTwrtStim': sim_times,
    'Choice': sim_choices,
})
print(f"Simulated {len(df_sim)} trials with R={R_true}, L={L_true}, theta={theta_true}")

# %% Bounds and plausible bounds for the two parameters (unchanged from 3-parameter version)
logR_bounds = [0.5, 4]
logL_bounds = [0.5, 4]

logR_plausible_bounds = [1, 3.8]
logL_plausible_bounds = [1, 3.8]

# Build arrays for VBMC (2 parameters only)
lb_2d = np.array([logR_bounds[0], logL_bounds[0]])
ub_2d = np.array([logR_bounds[1], logL_bounds[1]])
plb_2d = np.array([logR_plausible_bounds[0], logL_plausible_bounds[0]])
pub_2d = np.array([logR_plausible_bounds[1], logL_plausible_bounds[1]])

# %% Output directory for runs with theta fixed
out_dir = 'poisson_fit_pkls_theta_fixed_logR_logL'
os.makedirs(out_dir, exist_ok=True)
print(f"Saving VBMC pickles to: {out_dir}")

# Separate directory for verbose logs printed during VBMC optimize
logs_dir = 'poisson_fit_logs_theta_fixed_logR_logL'
os.makedirs(logs_dir, exist_ok=True)
print(f"Saving VBMC logs to: {logs_dir}")

# %% Helper to make a 2-parameter log-likelihood with fixed theta
rts = df_sim['RTwrtStim'].to_numpy()
choices = df_sim['Choice'].to_numpy().astype(int)

logR_width = logR_bounds[1] - logR_bounds[0]
logL_width = logL_bounds[1] - logL_bounds[0]


def make_loglike_and_prior(theta_fixed: int):
    """Factory returning (loglike_fn, prior_fn) for fixed integer theta."""
    theta_i = int(round(theta_fixed))
    if theta_i < 1:
        theta_i = 1

    def loglike_fn_2(params):
        # params: [logR, logL]
        logR, logL = params
        R = 10 ** logR
        L = 10 ** logL

        # Probability of + boundary (scalar)
        p_pos = fpt_choice_skellam(R, L, theta_i, 1)
        p_choice = np.where(choices == 1, p_pos, 1 - p_pos)

        # FPT density at each RT
        rho_t = fpt_density_skellam(rts, R, L, theta_i)

        # Per-trial probability (no truncation)
        p = rho_t * p_choice
        p = np.where(p <= 0, 1e-50, p)

        loglike = float(np.sum(np.log(p)))
        return loglike

    def prior_fn_2(params):
        # Uniform log-prior over (logR, logL) bounds
        return - (np.log(logR_width) + np.log(logL_width))

    return loglike_fn_2, prior_fn_2


def joint_logpost_factory(theta_fixed: int):
    ll_fn, pr_fn = make_loglike_and_prior(theta_fixed)

    def joint_logpost(params):
        return ll_fn(params) + pr_fn(params)

    return joint_logpost

# %% Loop over fixed theta values and fit (logR, logL) with VBMC
print('Initializing VBMC for fixed-theta grid (theta=1..50)...')

summary = []
for theta_fixed in range(1, 51):
    print(f"\n=== Fitting (logR, logL) with theta fixed to {theta_fixed} ===")

    # Random initialization within plausible bounds (seeded per-theta for reproducibility)
    rng_init = np.random.default_rng(42 + theta_fixed)
    logR_0 = rng_init.uniform(logR_plausible_bounds[0], logR_plausible_bounds[1])
    logL_0 = rng_init.uniform(logL_plausible_bounds[0], logL_plausible_bounds[1])
    x0_2d = np.array([logR_0, logL_0])

    # Build joint log-posterior for this fixed theta
    joint_logpost = joint_logpost_factory(theta_fixed)

    log_file = os.path.join(logs_dir, f"vbmc_poisson_fit_SIM_theta{theta_fixed:02d}.txt")

    try:
        with open(log_file, 'w') as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
            vbmc = VBMC(
                joint_logpost,
                x0_2d,
                lb_2d,
                ub_2d,
                plb_2d,
                pub_2d,
                options={'display': 'on', 'max_fun_evals': 200 * (2 + 2)}
            )
            vp, results = vbmc.optimize()
        out_pkl = os.path.join(out_dir, f"vbmc_poisson_fit_SIM_theta{theta_fixed:02d}.pkl")
        vbmc.save(out_pkl, overwrite=True)
        print(f"Saved: {out_pkl}")
        summary.append((theta_fixed, out_pkl, 'ok'))
    except Exception as e:
        print(f"VBMC optimization failed for theta={theta_fixed}: {e}")
        try:
            with open(log_file, 'a') as lf:
                lf.write(f"\nException during VBMC run for theta={theta_fixed}: {e}\n")
                traceback.print_exc(file=lf)
        except Exception:
            pass
        summary.append((theta_fixed, None, f"error: {e}"))

print("\n=== Done. Summary (theta, file, status) ===")
for row in summary:
    print(row)
