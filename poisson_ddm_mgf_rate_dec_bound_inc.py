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
rate_multiplication_factor_range = [1, 1.3]  # Array of rate multiplication factors
theta_increment_arr = [0, 1]

ddm_acc_theory = {}
ddm_rt_theory = {}

poisson_acc_theory = {}
poisson_rt_theory = {}
for theta_p_inc in theta_increment_arr:
    for rate_factor in rate_multiplication_factor_range:
        for theta in theta_range:
            # Calculate Nr0 for this rate factor
            Nr0 = Nr0_base * rate_factor
            
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
            print(f'Running for rate_factor={rate_factor}, theta={theta}')
            theta_poisson_increased = theta + theta_p_inc
            print(f'to match acc: theta is increased to {theta_poisson_increased}')
            poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(delayed(run_poisson_trial)(N, rho, r_right, r_left, theta_poisson_increased) for _ in tqdm(range(N_sim)))
            bound_offsets = np.array([data[2] for data in poisson_data])
            bound_offset_mean = np.mean(bound_offsets)
            print(f'bound offset mean = {bound_offset_mean}')

            # DDM uses unscaled Nr0_base (not affected by rate_factor)
            ddm_acc, ddm_mean_rt  = ddm_fc_dt(lam, l, Nr0_base, N, abl, ild, theta, dt)

            # Poisson uses scaled Nr0 (affected by rate_factor)
            theta_eff = theta_poisson_increased + bound_offset_mean
            poisson_acc, poisson_mean_rt = poisson_fc_dt(N, rho, theta_eff, lam, l, Nr0, abl, ild, dt)

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
            plt.plot(valid_theta_poisson, [poisson_acc_theory[(rate_factor,theta, theta_p_inc)] for theta in valid_theta_poisson], label=f'ratex={rate_factor}')
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
            plt.plot(valid_theta_poisson, [poisson_rt_theory[(rate_factor,theta, theta_p_inc)] for theta in valid_theta_poisson], label=f'ratex={rate_factor}')
    plt.title(f'theta_p_inc = {theta_p_inc}')
    plt.legend()
    plt.xlabel('theta range')
    plt.ylabel('mean RT')

# %%
# sim with increased 
ABL_range = [20, 40, 60]
ILD_range = [1,2,4,8,16]

best_rate_factor = 1.3
best_theta_inc = 0

original_theta = 2  

theta_og_ddm_rt_data = {}
theta_og_ddm_acc_data = {}

theta_og_poisson_rt_data = {}
theta_og_poisson_acc_data = {}

for ABL in ABL_range:
    for ILD in ILD_range:
        print(f'running ABL ={ABL}, ILD = {ILD}')
        # DDM
        ddm_acc, ddm_mean_rt  = ddm_fc_dt(lam, l, Nr0_base, N, ABL, ILD, original_theta, dt)
        theta_og_ddm_rt_data[(ABL, ILD)] = ddm_mean_rt
        theta_og_ddm_acc_data[(ABL, ILD)] = ddm_acc
        
        # poisson
        Nr0 = Nr0_base * best_rate_factor
        r0 = Nr0/N
        r_db = (2*ABL + ILD)/2
        l_db = (2*ABL - ILD)/2
        pr = (10 ** (r_db/20))
        pl = (10 ** (l_db/20))

        den = (pr ** (lam * l)) + (pl ** (lam * l))
        rr = (pr ** lam) / den
        rl = (pl ** lam) / den
        r_right = r0 * rr
        r_left = r0 * rl

        theta_poisson = original_theta + best_theta_inc
        poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(delayed(run_poisson_trial)(N, rho, r_right, r_left, theta_poisson) for _ in tqdm(range(N_sim)))
        bound_offsets = np.array([data[2] for data in poisson_data])
        bound_offset_mean = np.mean(bound_offsets)

        theta_poisson_eff = theta_poisson + bound_offset_mean
        poisson_acc, poisson_mean_rt = poisson_fc_dt(N, rho, theta_poisson_eff, lam, l, Nr0, ABL, ILD, dt)
        theta_og_poisson_rt_data[(ABL, ILD)] = poisson_mean_rt
        theta_og_poisson_acc_data[(ABL, ILD)] = poisson_acc
        
# %%
# chronometric
plt.figure(figsize=(10,5))
ABL_color_map = {20: 'blue', 40: 'orange', 60: 'green'}
# Calculate squared error
squared_error = sum((theta_og_poisson_rt_data[(ABL, ILD)] - theta_og_ddm_rt_data[(ABL, ILD)])**2 
                    for ABL in ABL_range for ILD in ILD_range)
for ABL in ABL_range:
    plt.plot([ILD for ILD in ILD_range], [theta_og_ddm_rt_data[(ABL, ILD)] for ILD in ILD_range], label=f'DDM ABL={ABL}', marker='o', color=ABL_color_map[ABL])
    plt.plot([ILD for ILD in ILD_range], [theta_og_poisson_rt_data[(ABL, ILD)] for ILD in ILD_range], label=f'Poisson ABL={ABL}', marker='x', color=ABL_color_map[ABL])
plt.xlabel('ILD')
plt.ylabel('Mean RT')
plt.title(f'rate x = {best_rate_factor}, bound inc = {best_theta_inc}, original theta = {original_theta}, Sq.Err = {squared_error:.6f}')
plt.legend()
    # %%
# psychometric: FC vs ILD
plt.figure(figsize=(10,5))
# Calculate squared error for accuracy
squared_error_acc = sum((theta_og_poisson_acc_data[(ABL, ILD)] - theta_og_ddm_acc_data[(ABL, ILD)])**2 
                        for ABL in ABL_range for ILD in ILD_range)
for a_idx, ABL in enumerate(ABL_range):
    plt.subplot(1, len(ABL_range), a_idx+1)
    plt.plot([ILD for ILD in ILD_range], [theta_og_ddm_acc_data[(ABL, ILD)] for ILD in ILD_range], label=f'DDM ABL={ABL}', marker='o', color=ABL_color_map[ABL])
    plt.plot([ILD for ILD in ILD_range], [theta_og_poisson_acc_data[(ABL, ILD)] for ILD in ILD_range], label=f'Poisson ABL={ABL}', marker='x', color=ABL_color_map[ABL])
    plt.xlabel('ILD')
    if a_idx == 0:
        plt.ylabel('Accuracy')
    plt.title(f'ABL={ABL}')
    # plt.legend()
plt.suptitle(f'rate x = {best_rate_factor}, bound inc = {best_theta_inc}, original theta = {original_theta}, Sq.Err = {squared_error_acc:.6f}')
# %%
best_theta_inc