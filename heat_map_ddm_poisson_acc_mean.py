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
def poisson_fc_dt_scaled(N, rho, theta, r_right, r_left, dt):
    E_W = dt * N * (r_right - r_left)
    h0 = find_h0(r_right, r_left, N, rho)
    fc = FC(h0, theta)
    mean_dt = DT(E_W, h0, theta, dt)
    return fc, mean_dt

# %%
# poisson params
N = 10
rho = 1e-3  
# %%
# DDM data
# TIED
lam = 1.3
l = 0.9
Nr0_base = 13.3  # Base Nr0 before multiplication

abl = 20
ild_range = [1,2,4]
dt = 1e-6  # Time step for continuous DDM simulation
dB = 1e-3

theta_ddm = 2
ddm_data_dict = {}
for ild in ild_range:
    ddm_acc, ddm_mean_rt  = ddm_fc_dt(lam, l, Nr0_base, N, abl, ild, theta_ddm, dt)
    ddm_data_dict[ild] = (ddm_acc, ddm_mean_rt)

# %%
theta_poisson = 3
rate_scalars_left = [1, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]
rate_scalars_right = [1, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]
poisson_data_dict = {}
N_sim_poisson = int(10e3)
for rate_scalar_left in rate_scalars_left:
    for rate_scalar_right in rate_scalars_right:
        for ild in ild_range:
            print(f'ild={ild}, rate_scalar_left={rate_scalar_left}, rate_scalar_right={rate_scalar_right}')
            # scale rates
            r0 = Nr0_base/N
            r_db = (2*abl + ild)/2
            l_db = (2*abl - ild)/2
            pr = (10 ** (r_db/20))
            pl = (10 ** (l_db/20))

            den = (pr ** (lam * l)) + (pl ** (lam * l))
            rr = (pr ** lam) / den
            rl = (pl ** lam) / den

            r_right = r0 * rr * rate_scalar_right
            r_left = r0 * rl * rate_scalar_left

            # effective theta poisson
            poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(delayed(run_poisson_trial)(N, rho, r_right, r_left, theta_poisson) for _ in tqdm(range(N_sim_poisson)))
            bound_offsets = np.array([data[2] for data in poisson_data])
            bound_offset_mean = np.mean(bound_offsets)
            theta_poisson_eff = theta_poisson + bound_offset_mean

            # acc,rt
            try:
                poisson_acc, poisson_mean_rt = poisson_fc_dt_scaled(N, rho, theta_poisson_eff, r_right, r_left, dt)
            except:
                poisson_acc, poisson_mean_rt = np.nan, np.nan
            poisson_data_dict[(ild, rate_scalar_right, rate_scalar_left)] = (poisson_acc, poisson_mean_rt)


# %%
# diff matrices
acc_ddm_minus_poisson = np.zeros((len(rate_scalars_right), len(rate_scalars_left)))
rt_ddm_minus_poisson = np.zeros((len(rate_scalars_right), len(rate_scalars_left)))

for r_idx, right_scalar in enumerate(rate_scalars_right):
    for l_idx, left_scalar in enumerate(rate_scalars_left):
        for ild_idx, ild in enumerate(ild_range):
            acc_ddm_minus_poisson[r_idx, l_idx] += abs(ddm_data_dict[ild][0] - poisson_data_dict[(ild, right_scalar, left_scalar)][0])
            rt_ddm_minus_poisson[r_idx, l_idx] += abs(ddm_data_dict[ild][1] - poisson_data_dict[(ild, right_scalar, left_scalar)][1])
# %%
# Find min indices (ignoring NaN)
min_idx_acc = np.unravel_index(np.nanargmin(acc_ddm_minus_poisson), acc_ddm_minus_poisson.shape)
min_idx_rt = np.unravel_index(np.nanargmin(rt_ddm_minus_poisson), rt_ddm_minus_poisson.shape)

# Get corresponding scalar values and min values
acc_min_right = rate_scalars_right[min_idx_acc[0]]
acc_min_left = rate_scalars_left[min_idx_acc[1]]
acc_min_val = acc_ddm_minus_poisson[min_idx_acc]
rt_min_right = rate_scalars_right[min_idx_rt[0]]
rt_min_left = rate_scalars_left[min_idx_rt[1]]
rt_min_val = rt_ddm_minus_poisson[min_idx_rt]

plt.figure(figsize=(20,10))

# Suptitle with parameters
suptitle_str = (f'N={N}, rho={rho}, lam={lam}, l={l}, Nr0_base={Nr0_base}, '
                f'abl={abl}, ild_range={ild_range}, dt={dt}, dB={dB}, '
                f'theta_ddm={theta_ddm}, theta_poisson={theta_poisson}\n'
                f'Acc min: {acc_min_val:.4f} (right={acc_min_right}, left={acc_min_left}) | '
                f'RT min: {rt_min_val:.4f} (right={rt_min_right}, left={rt_min_left})')
plt.suptitle(suptitle_str, fontsize=18)

# Accuracy heatmap
plt.subplot(1,2,1)
plt.imshow(acc_ddm_minus_poisson, cmap='viridis', aspect='auto')
plt.title('Absolute Difference in Accuracy')
plt.xlabel('Rate Scalar Left')
plt.ylabel('Rate Scalar Right')
plt.xticks(range(len(rate_scalars_left)), rate_scalars_left)
plt.yticks(range(len(rate_scalars_right)), rate_scalars_right)
plt.colorbar()
# Mark minimum with "x"
plt.plot(min_idx_acc[1], min_idx_acc[0], 'rx', markersize=15, markeredgewidth=3)

# RT heatmap
plt.subplot(1,2,2)
plt.imshow(rt_ddm_minus_poisson, cmap='viridis', aspect='auto')
plt.title('Absolute Difference in RT')
plt.xlabel('Rate Scalar Left')
plt.ylabel('Rate Scalar Right')
plt.xticks(range(len(rate_scalars_left)), rate_scalars_left)
plt.yticks(range(len(rate_scalars_right)), rate_scalars_right)
plt.colorbar()
# Mark minimum with "x"
plt.plot(min_idx_rt[1], min_idx_rt[0], 'rx', markersize=15, markeredgewidth=3)

plt.tight_layout()
plt.show()


