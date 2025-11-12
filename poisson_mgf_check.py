# %%
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import multiprocessing
import mgf_helper_utils as utils
from scipy import stats
import time
from corr_poisson_utils_subtractive import run_poisson_trial
from mgf_helper_utils import poisson_fc_dt
# %%
N = 100
theta = 2
rho = 0.001  # In combined_poisson_ddm, this is referred to as 'c'

# ddm rates parameters
lam = 1.3
l = 0.9
Nr0 = 13.3 * 1
abl = 20
ild = 1
dt = 1e-6  # Time step for continuous DDM simulation
dB = 1e-3
# Calculate rates
r0 = Nr0/N
r_db = (2*abl + ild)/2
l_db = (2*abl - ild)/2
pr = (10 ** (r_db/20))
pl = (10 ** (l_db/20))

den = (pr ** (lam * l)) + (pl ** (lam * l))
rr = (pr ** lam) / den
rl = (pl ** lam) / den
r_right = r0 * rr
r_left = r0 * rl


# %%
# run_poisson_trial(trial_theta, N, r_right, r_left)
N_sim = int(25e3)
# %%
poisson_data_theta_wise = {}
theta_range = [2,4,6,8]
for theta in theta_range:
    print(f'Running for theta = {theta}')
    poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(delayed(run_poisson_trial)(N, rho, r_right, r_left, theta) for _ in tqdm(range(N_sim)))
    poisson_data_theta_wise[theta] = poisson_data

# %%
data_mean_rt_theta_wise = {}
data_acc_theta_wise = {}

theory_mean_rt_theta_wise = {}
theory_acc_theta_wise = {}

bound_offset_theta_wise = {}

for theta in theta_range:
    rt_choice_data = poisson_data_theta_wise[theta]
    rts = np.array([data[0] for data in rt_choice_data])
    choices = np.array([data[1] for data in rt_choice_data])
    bound_offsets = np.array([data[2] for data in rt_choice_data])
    bound_offset_mean = np.mean(bound_offsets)
    print(f'theta = {theta}, bound offset mean = {bound_offset_mean}')

    data_mean_rt_theta_wise[theta] = np.mean(rts)
    data_acc_theta_wise[theta] = np.mean(choices == 1)
    bound_offset_theta_wise[theta] = np.nanmean(bound_offsets)

    # poisson_fc_dt(N, rho, theta, lam, l, Nr0, abl, ild, dt)
    # NOTE: to show that offset zero causes mismatch
    # bound_offset_mean = 0
    theta_eff = theta + bound_offset_mean
    theory_acc, theory_mean_rt = poisson_fc_dt(N, rho, theta_eff, lam, l, Nr0, abl, ild, dt)
    theory_mean_rt_theta_wise[theta] = theory_mean_rt
    theory_acc_theta_wise[theta] = theory_acc
    
    



# %%
plt.scatter(data_mean_rt_theta_wise.keys(), data_mean_rt_theta_wise.values(), color='r')
plt.plot(theory_mean_rt_theta_wise.keys(), theory_mean_rt_theta_wise.values(), color='b')
plt.xlabel('Theta')
plt.ylabel('Mean RT')
plt.title(f'N={N},rho={rho},abl={abl},ild={ild}')
plt.show()

# %%
plt.scatter(data_acc_theta_wise.keys(), data_acc_theta_wise.values(), color='r')
plt.plot(theory_acc_theta_wise.keys(), theory_acc_theta_wise.values(), color='b')
plt.xlabel('Theta')
plt.ylabel('Accuracy')
plt.title(f'N={N},rho={rho},abl={abl},ild={ild}')

plt.show()

# %%
