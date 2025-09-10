# %%
import numpy as np
import matplotlib.pyplot as plt

# %%


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


V_A = 1.2
theta = 1.8
t_A_aff = -0.2

N_sim = int(100e3)
T = 2
dt = 1e-4
dB = 1e-2
def proactive_proces(V_A, theta_A, t_A_aff):
    dv = 0
    t = t_A_aff
    while dv < theta_A:
        dv += V_A * dt + np.random.normal(0, dB)
        t += dt
    return t

# Simulate N_sim proactive and N_sim poisson trials
R_true = 15
L_true = 12
theta_true = 7
proactive_times = np.empty(N_sim, dtype=float)
poisson_times = np.empty(N_sim, dtype=float)
poisson_choices = np.empty(N_sim, dtype=int)
for i in range(N_sim):
    proactive_times[i] = proactive_proces(V_A, theta, t_A_aff)
    poisson_times[i], poisson_choices[i] = simulate_trial(R_true, L_true, theta_true)

# %%
from vbmc_skellam_utils import fpt_density_skellam, fpt_cdf_skellam
R_mean = 382.835
L_mean = 515.598
theta = 11

t_pts = np.arange(0,1,0.001)
up_theory = fpt_density_skellam(t_pts, R_mean, L_mean, theta)
plt.plot(t_pts, up_theory)
plt.axhline(0)
plt.axvline(0)
plt.show()
print(f'Area under up_theory: {np.trapezoid(up_theory, t_pts):.8f}')
# %%
from scipy.integrate import cumulative_trapezoid
cdf = fpt_cdf_skellam(t_pts, R_mean, L_mean, theta)
plt.plot(t_pts, cdf)
plt.plot(t_pts, cumulative_trapezoid(up_theory, t_pts, initial=0), ls='--', alpha=0.6, lw=4)
plt.axhline(0)
plt.axvline(0)
plt.show()



# %%
