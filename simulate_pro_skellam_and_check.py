# %% 
import numpy as np
import pandas as pd
rng = np.random.default_rng(0)

N_TRIALS = int(50e3)
R_true=87
L_true=107
theta_true=7
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
skellam_times = np.empty(N_TRIALS, dtype=float)
skellam_choices = np.empty(N_TRIALS, dtype=int)
for i in range(N_TRIALS):
    t, ch = simulate_trial(R_true, L_true, theta_true, rng=rng)
    skellam_times[i] = t
    skellam_choices[i] = ch
# %%
V_A=1.1774684269283593
theta_A=1
t_A_aff=0.04163924630404911
dt = 1e-4
dB = 1e-2

t_E_aff = 0.073
del_go = 1.90
# %%
stim_times = rng.exponential(scale=0.4, size=N_TRIALS) + 0.2
pro_and_skellam = np.zeros((N_TRIALS, 2))
T_max = 5
proactive_win_count = 0
reactive_win_count = 0
for i in range(N_TRIALS):
    # proactive trial
    pro_dv = 0
    t_pro = t_A_aff
    while True:
        pro_dv += V_A*dt + np.random.normal(0, dB)
        t_pro += dt
        if pro_dv >= theta_A:
            break

    # reactive hit time
    t_skellam = skellam_times[i]
    t_skellam_with_delay = t_skellam - t_E_aff - stim_times[i]
    if t_skellam_with_delay < 0:
        t_skellam_with_delay = np.inf
    # case 1
    # reactive wins
    if (t_skellam_with_delay < t_pro) :
        pro_and_skellam[i, 0] = t_skellam_with_delay
        pro_and_skellam[i, 1] = skellam_choices[i]
        # print(f'REACTIVE wins : {pro_and_skellam[i, 0] :.2f}, t_pro = {t_pro :.2f}, sk time delay = {t_skellam_with_delay :.2f}')
        reactive_win_count += 1
    # case 2: proactive wins
    elif (t_pro < t_skellam_with_delay):
        pro_and_skellam[i, 0] = t_pro
        # print(f'pro wins = {pro_and_skellam[i, 0]}')
        proactive_win_count += 1

        # case 2 a
        # is there a hit between t_pro_with_delay + del_go
        if t_pro < t_skellam_with_delay < t_pro + del_go - t_E_aff: 
            pro_and_skellam[i, 1] = skellam_choices[i]
        # case 2 b
        # no hit between t_pro_with_delay + del_go, coin flip
        else:
            pro_and_skellam[i, 1] = 1 if rng.random() < 0.5 else -1
    else:
        print(f'skelam with delay = {t_skellam_with_delay}, t_pro = {t_pro}')
        raise ValueError(f'unknown case')


print(f'pro win count = {proactive_win_count}')
print(f'reactive win count = {reactive_win_count}')
# %%
import matplotlib.pyplot as plt
plt.hist(pro_and_skellam[:, 0] - stim_times, bins=np.arange(-5,5,0.05));
# plt.xlim(0,1)


# %%
