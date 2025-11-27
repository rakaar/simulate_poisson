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
from mgf_helper_utils import ddm_fc_dt, find_h0, FC, DT
# %%
# Local version of poisson_fc_dt that takes r_right, r_left directly
def poisson_fc_dt_scaled(N, rho, theta, r_right, r_left, dt):
    E_W = dt * N * (r_right - r_left)
    h0 = find_h0(r_right, r_left, N, rho)
    fc = FC(h0, theta)
    mean_dt = DT(E_W, h0, theta, dt)
    return fc, mean_dt

# %%
# TIED
lam = 1.3
l = 0.9
Nr0_base = 13.3  # Base Nr0 before multiplication

abl = 20
ild = 1
dt = 1e-6  # Time step for continuous DDM simulation
dB = 1e-3



# %%
N_sim = int(10e3)

rho = 1e-3
N = 100  # Fixed number of neurons
theta_range = [2, 4]
rate_multiplication_factor_range = [(1, 1), (1.2, 1.1), (1.8, 1.9)]  # Array of (right_factor, left_factor) tuples
theta_increment_arr = [0]

ddm_acc_theory = {}
ddm_rt_theory = {}

poisson_acc_theory = {}
poisson_rt_theory = {}
for theta_p_inc in theta_increment_arr:
    for rate_factor in rate_multiplication_factor_range:
        rate_factor_right, rate_factor_left = rate_factor  # Unpack tuple
        for theta in theta_range:
            # Calculate base rates (unscaled)
            r0 = Nr0_base/N
            r_db = (2*abl + ild)/2
            l_db = (2*abl - ild)/2
            pr = (10 ** (r_db/20))
            pl = (10 ** (l_db/20))

            den = (pr ** (lam * l)) + (pl ** (lam * l))
            rr = (pr ** lam) / den
            rl = (pl ** lam) / den
            # Apply separate scaling factors to right and left rates
            r_right = r0 * rr * rate_factor_right
            r_left = r0 * rl * rate_factor_left
            print('-----------------------------')
            print(f'Running for rate_factor=(R:{rate_factor_right}, L:{rate_factor_left}), theta={theta}')
            theta_poisson_increased = theta + theta_p_inc
            print(f'to match acc: theta is increased to {theta_poisson_increased}')
            poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(delayed(run_poisson_trial)(N, rho, r_right, r_left, theta_poisson_increased) for _ in tqdm(range(N_sim)))
            bound_offsets = np.array([data[2] for data in poisson_data])
            bound_offset_mean = np.mean(bound_offsets)
            print(f'bound offset mean = {bound_offset_mean}')

            # DDM uses unscaled Nr0_base (not affected by rate_factor)
            ddm_acc, ddm_mean_rt  = ddm_fc_dt(lam, l, Nr0_base, N, abl, ild, theta, dt)

            # Poisson uses separately scaled rates
            theta_eff = theta_poisson_increased + bound_offset_mean
            poisson_acc, poisson_mean_rt = poisson_fc_dt_scaled(N, rho, theta_eff, r_right, r_left, dt)

            ddm_acc_theory[(rate_factor,theta, theta_p_inc)] = ddm_acc
            ddm_rt_theory[(rate_factor,theta, theta_p_inc)] = ddm_mean_rt

            poisson_acc_theory[(rate_factor,theta, theta_p_inc)] = poisson_acc
            poisson_rt_theory[(rate_factor,theta, theta_p_inc)] = poisson_mean_rt

# %%
plt.figure(figsize=(20,5))
for t_idx, theta_p_inc in enumerate(theta_increment_arr):
    plt.subplot(1, len(theta_increment_arr), t_idx + 1)
    for rate_factor in rate_multiplication_factor_range:
        # Filter out theta values that don't have entries in the dictionaries
        valid_theta_ddm = [theta for theta in theta_range if (rate_factor,theta, theta_p_inc) in ddm_acc_theory]
        valid_theta_poisson = [theta for theta in theta_range if (rate_factor,theta, theta_p_inc) in poisson_acc_theory]
        
        # Only plot if we have valid data
        if valid_theta_ddm:
            plt.scatter(valid_theta_ddm, [ddm_acc_theory[(rate_factor,theta, theta_p_inc)] for theta in valid_theta_ddm], label=f'DDM', s=100)
            
        if valid_theta_poisson:
            plt.plot(valid_theta_poisson, [poisson_acc_theory[(rate_factor,theta, theta_p_inc)] for theta in valid_theta_poisson], label=f'ratex=(R:{rate_factor[0]},L:{rate_factor[1]})')
    
    plt.axhline(0.58, color='k', ls='--')
    plt.title(f'theta_p_inc = {theta_p_inc}')
    plt.legend()
    plt.xlabel('theta range')
    plt.ylabel('accuracy')
# %%
plt.figure(figsize=(20,5))
for t_idx, theta_p_inc in enumerate(theta_increment_arr):
    plt.subplot(1, len(theta_increment_arr), t_idx + 1)
    for rate_factor in rate_multiplication_factor_range:
        # Filter out theta values that don't have entries in the dictionaries
        valid_theta_ddm = [theta for theta in theta_range if (rate_factor,theta, theta_p_inc) in ddm_acc_theory]
        valid_theta_poisson = [theta for theta in theta_range if (rate_factor,theta, theta_p_inc) in poisson_acc_theory]
        
        # Only plot if we have valid data
        if valid_theta_ddm:
            plt.scatter(valid_theta_ddm, [ddm_rt_theory[(rate_factor,theta, theta_p_inc)] for theta in valid_theta_ddm], label=f'DDM', s=100)
        if valid_theta_poisson:
            plt.plot(valid_theta_poisson, [poisson_rt_theory[(rate_factor,theta, theta_p_inc)] for theta in valid_theta_poisson], label=f'ratex=(R:{rate_factor[0]},L:{rate_factor[1]})')
    
    plt.axhline(0.2, color='k', ls='--')
    plt.title(f'theta_p_inc = {theta_p_inc}')
    plt.legend()
    plt.xlabel('theta range')
    plt.ylabel('mean RT')
# %%
