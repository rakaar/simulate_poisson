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
from mgf_helper_utils import poisson_fc_dt, ddm_fc_dt
# %%
# TIED
lam = 1.3
l = 0.9
Nr0 = 13.3  # Fixed Nr0 (no scaling)

abl = 20
ild = 1
dt = 1e-6  # Time step for continuous DDM simulation
dB = 1e-3



# %%
N_sim = int(10e3)

rho = 1e-2
N = 500  # Fixed number of neurons
theta_range = [2,4,6]
poisson_bound_increment_arr = [0, 3, 4, 5]  # Array of bound increments for Poisson

ddm_acc_theory = {}
ddm_rt_theory = {}

poisson_acc_theory = {}
poisson_rt_theory = {}

for poisson_bound_increment in poisson_bound_increment_arr:
    for theta in theta_range:
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
        print('-----------------------------')
        print(f'Running for poisson_bound_increment={poisson_bound_increment}, theta={theta}')
        poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(delayed(run_poisson_trial)(N, rho, r_right, r_left, theta + poisson_bound_increment) for _ in tqdm(range(N_sim)))
        bound_offsets = np.array([data[2] for data in poisson_data])
        bound_offset_mean = np.mean(bound_offsets)
        print(f'bound offset mean = {bound_offset_mean}')

        # DDM uses standard theta (no bound increment)
        ddm_acc, ddm_mean_rt  = ddm_fc_dt(lam, l, Nr0, N, abl, ild, theta, dt)

        # Poisson uses theta with bound increment applied
        theta_eff = theta + bound_offset_mean + poisson_bound_increment
        poisson_acc, poisson_mean_rt = poisson_fc_dt(N, rho, theta_eff, lam, l, Nr0, abl, ild, dt)

        ddm_acc_theory[(poisson_bound_increment,theta)] = ddm_acc
        ddm_rt_theory[(poisson_bound_increment,theta)] = ddm_mean_rt

        poisson_acc_theory[(poisson_bound_increment,theta)] = poisson_acc
        poisson_rt_theory[(poisson_bound_increment,theta)] = poisson_mean_rt
        

# %%
# Accuracy plot
plt.figure(figsize=(10,6))

for poisson_bound_increment in poisson_bound_increment_arr:
    # Filter out theta values that don't have entries in the dictionaries
    valid_theta_ddm = [theta for theta in theta_range if (poisson_bound_increment,theta) in ddm_acc_theory]
    valid_theta_poisson = [theta for theta in theta_range if (poisson_bound_increment,theta) in poisson_acc_theory]
    
    # Only plot if we have valid data
    if valid_theta_ddm:
        plt.scatter(valid_theta_ddm, [ddm_acc_theory[(poisson_bound_increment,theta)] for theta in valid_theta_ddm], label=f'DDM', s=100)
    if valid_theta_poisson:
        plt.plot(valid_theta_poisson, [poisson_acc_theory[(poisson_bound_increment,theta)] for theta in valid_theta_poisson], label=f'Poisson: bound +{poisson_bound_increment}')
plt.xlabel('Theta')
plt.ylabel('Accuracy')
plt.title(f'Accuracy vs Theta (N={N}, rho={rho}, abl={abl}, ild={ild})')
plt.legend()

# %%
# Mean RT plot
plt.figure(figsize=(10,6))

for poisson_bound_increment in poisson_bound_increment_arr:
    # Filter out theta values that don't have entries in the dictionaries
    valid_theta_ddm = [theta for theta in theta_range if (poisson_bound_increment,theta) in ddm_rt_theory]
    valid_theta_poisson = [theta for theta in theta_range if (poisson_bound_increment,theta) in poisson_rt_theory]
    
    # Only plot if we have valid data
    if valid_theta_ddm:
        plt.scatter(valid_theta_ddm, [ddm_rt_theory[(poisson_bound_increment,theta)] for theta in valid_theta_ddm], label=f'DDM', s=100)
    if valid_theta_poisson:
        plt.plot(valid_theta_poisson, [poisson_rt_theory[(poisson_bound_increment,theta)] for theta in valid_theta_poisson], label=f'Poisson: bound +{poisson_bound_increment}')
plt.xlabel('Theta')
plt.ylabel('Mean RT')
plt.title(f'Mean RT vs Theta (N={N}, rho={rho}, abl={abl}, ild={ild})')
plt.legend()

# %%
