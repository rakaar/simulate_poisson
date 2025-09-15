# %%
import matplotlib.pyplot as plt
import numpy as np
from simulators import simulate_pro_skellam_trial, cont_ddm_simulate, simulate_skellam_trial
from joblib import Parallel, delayed
from vbmc_skellam_utils import up_or_down_hit_fn
from vbmc_skellam_utils import cum_pro_and_reactive_trunc_fn
from vbmc_skellam_utils import fpt_choice_skellam, fpt_cdf_skellam, fpt_density_skellam
from vbmc_skellam_utils import truncated_cum_A_t_fn, truncated_rho_A_t_fn
from vbmc_skellam_utils import up_or_down_hit_truncated_proactive_V2_fn
# Small picklable wrappers that return the index along with results to ensure alignment
def _run_skellam_indexed(i, mu1, mu2, theta_E):
    rt, choice = simulate_skellam_trial(mu1, mu2, theta_E)
    return i, rt, choice

def _run_pro_skellam_indexed(i, V_A, theta_A, t_A_aff, dt, dB, t_E_aff, del_go, t_stim, t_skellam, choice_skellam):
    rt, choice = simulate_pro_skellam_trial(
        V_A, theta_A, t_A_aff, dt, dB, t_E_aff, del_go, t_stim, t_skellam, choice_skellam
    )
    return i, rt, choice

def _run_cont_ddm_indexed(i, V_A, theta_A, t_A_aff, dt, dB, t_E_aff, del_go, t_stim, mu1, mu2, theta_E):
    rt, choice = cont_ddm_simulate(
        V_A, theta_A, t_A_aff, dt, dB, t_E_aff, del_go, t_stim, mu1, mu2, theta_E
    )
    return i, rt, choice

# %%
# PARAMS
V_A=1.1
theta_A=1
t_A_aff=0.1
dt = 1e-4
dB = 1e-2

t_E_aff = 0.073
del_go = 0.13

mu1=160
mu2=140
theta_E=11
# %%
N_sim = int(100e3)
t_stim_for_sim = np.random.exponential(scale=0.4, size=N_sim) + 0.2
# t_stim_for_sim = np.zeros(N_sim)
# t_stim_for_sim = np.ones(N_sim) * 0.25

skellam_RT_choice = np.zeros((N_sim, 2))
pro_skellam_RT_choice = np.zeros((N_sim, 2))
cont_DDM_RT_choice = np.zeros((N_sim, 2))

# Run simulations in parallel and keep outputs aligned with t_stim_for_sim by index
# 1) Reactive Skellam-only trials (return index to guarantee alignment)
print(f'Running skellam trials...')
skellam_indexed = Parallel(n_jobs=-1, backend="loky")(
    delayed(_run_skellam_indexed)(i, mu1, mu2, theta_E) for i in range(N_sim)
)
for i, rt, choice in skellam_indexed:
    skellam_RT_choice[i, 0] = rt
    skellam_RT_choice[i, 1] = choice

# 2) Proactive + Skellam trials (requires Skellam results and t_stim)
print(f'Running proactive + skellam trials...')
pro_indexed = Parallel(n_jobs=-1, backend="loky")(
    delayed(_run_pro_skellam_indexed)(
        i,
        V_A, theta_A, t_A_aff, dt, dB, t_E_aff, del_go,
        float(t_stim_for_sim[i]),
        float(skellam_RT_choice[i, 0]),  # t_skellam
        int(skellam_RT_choice[i, 1])     # choice_skellam
    )
    for i in range(N_sim)
)
for i, rt, choice in pro_indexed:
    pro_skellam_RT_choice[i, 0] = rt
    pro_skellam_RT_choice[i, 1] = choice

# 3) Continuous DDM trials (uses t_stim)
print(f'Running cont DDM trials...')
cont_indexed = Parallel(n_jobs=-1, backend="loky")(
    delayed(_run_cont_ddm_indexed)(
        i,
        V_A, theta_A, t_A_aff, dt, dB, t_E_aff, del_go,
        float(t_stim_for_sim[i]),
        mu1, mu2, theta_E
    )
    for i in range(N_sim)
)
for i, rt, choice in cont_indexed:
    cont_DDM_RT_choice[i, 0] = rt
    cont_DDM_RT_choice[i, 1] = choice
# %%
# CONT DDM & pro + skellam should match
bins = np.arange(-5,5,0.01)
cont_DDM_RT_wrt_stim = cont_DDM_RT_choice[:, 0] - t_stim_for_sim
pro_skellam_RT_wrt_stim = pro_skellam_RT_choice[:, 0] - t_stim_for_sim

plt.hist(pro_skellam_RT_wrt_stim, bins=bins, label='Pro Skellam', density=True, histtype='step', color='r')
plt.hist(cont_DDM_RT_wrt_stim, bins=bins, alpha=0.5, label='Cont DDM', density=True, histtype='step', ls='--', lw=3)
plt.legend()
plt.show()

# %%
# theoretical likelihood
N_theory = int(10e3)

# %%
### TRUNCATION TEST
c_A_trunc_time = 0.3
t_pts_wrt_stim = np.arange(-2, 2, 0.001)
t_stim_for_theory = np.random.choice(t_stim_for_sim, size=N_theory, replace=True)

actual_truncated_theory_pro_skellam_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))
for j, t_stim in enumerate(t_stim_for_theory):
    t_pts_wrt_fix = t_pts_wrt_stim + t_stim
    actual_truncated_theory_pro_skellam_samples[j, :] = \
        up_or_down_hit_truncated_proactive_V2_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, 1) \
        + up_or_down_hit_truncated_proactive_V2_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, -1)
        
        # up_or_down_hit_truncated_proactive_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, 1) \
        # + up_or_down_hit_truncated_proactive_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, -1)
        
        
        
        
        
# %%
bins = np.arange(-5,5,0.001)
actual_truncated_mean = np.mean(actual_truncated_theory_pro_skellam_samples, axis=0)
area_theory = np.trapezoid(actual_truncated_mean, t_pts_wrt_stim)
print(f'theory area truncated = {area_theory}')
actual_truncated_mean_norm = actual_truncated_mean / area_theory

rt_values = pro_skellam_RT_choice[:, 0]
# TEST: NOTE
remove_mask = (rt_values < t_stim_for_sim) & (rt_values < c_A_trunc_time)
# remove_mask = rt_values < c_A_trunc_time
pro_skellam_truncated_RT_wrt_stim = rt_values[~remove_mask] - t_stim_for_sim[~remove_mask]

# Same for cont DDM
rt_values_cont = cont_DDM_RT_choice[:, 0]
remove_mask_cont = (rt_values_cont < t_stim_for_sim) & (rt_values_cont < c_A_trunc_time)
cont_DDM_truncated_RT_wrt_stim = rt_values_cont[~remove_mask_cont] - t_stim_for_sim[~remove_mask_cont]

bins = np.arange(-5,5,0.01)
plt.figure(figsize=(8, 4))
plt.hist(pro_skellam_truncated_RT_wrt_stim, bins=bins, label='Pro Skellam Truncated', density=True, histtype='step', color='C0')
plt.hist(cont_DDM_truncated_RT_wrt_stim, bins=bins, label='Cont DDM Truncated', density=True, histtype='step', color='C1', lw=2, linestyle='dashed')
# plt.plot(t_pts_wrt_stim, actual_truncated_mean, color='C3', lw=3, linestyle='-.', alpha=0.8, label='Theory Truncated')
plt.scatter(t_pts_wrt_stim, actual_truncated_mean_norm, color='C3', alpha=0.9, label='Theory Truncated', s=7)
plt.xlim(-1,1)
plt.xlabel('RT - t_stim')
plt.axvline(0, color='C1', lw=2, linestyle='dashed', alpha=0.5, label='0')
plt.axvline(t_E_aff, color='C4', lw=2, linestyle='dashed', alpha=0.5, label='t_E_aff')
plt.axvline(c_A_trunc_time, color='C5', lw=2, linestyle='dashed', alpha=0.5, label='c_A_trunc_time')

plt.axvline(t_A_aff, color='k', lw=2, linestyle='dashed', alpha=0.5, label='t_A_aff')
plt.legend()

plt.ylabel('Density')
plt.title(f'trunc time {c_A_trunc_time}')
plt.tight_layout()
plt.show()

# %%
# up and down seperately
# skellam
rt_values = pro_skellam_RT_choice[:, 0]
rt_choices = pro_skellam_RT_choice[:, 1]
remove_mask = (rt_values < t_stim_for_sim) & (rt_values < c_A_trunc_time)
pro_skellam_truncated_RT_wrt_stim = rt_values[~remove_mask] - t_stim_for_sim[~remove_mask]
pro_skellam_choices = rt_choices[~remove_mask]
# up and down seperately
up_mask = pro_skellam_choices == 1
down_mask = pro_skellam_choices == -1
up_rt_values = pro_skellam_truncated_RT_wrt_stim[up_mask]
down_rt_values = pro_skellam_truncated_RT_wrt_stim[down_mask]
N_up = len(up_rt_values)
N_down = len(down_rt_values)
N_total = N_up + N_down
bins = np.arange(-2,2,0.01)
up_hist_skellam, _  = np.histogram(up_rt_values, bins=bins, density=True)
down_hist_skellam, _ = np.histogram(down_rt_values, bins=bins, density=True)

plt.plot(bins[:-1], up_hist_skellam * (N_up / N_total), 'r', label='up skellam')
plt.plot(bins[:-1], -down_hist_skellam * (N_down / N_total), 'r', label='down skellam')



# Same for cont DDM
rt_values_cont = cont_DDM_RT_choice[:, 0]
rt_choices_cont = cont_DDM_RT_choice[:, 1]
remove_mask_cont = (rt_values_cont < t_stim_for_sim) & (rt_values_cont < c_A_trunc_time)
cont_DDM_truncated_RT_wrt_stim = rt_values_cont[~remove_mask_cont] - t_stim_for_sim[~remove_mask_cont]
cont_DDM_choices = rt_choices_cont[~remove_mask_cont]
# up and down seperately
up_mask = cont_DDM_choices == 1
down_mask = cont_DDM_choices == -1
up_rt_values = cont_DDM_truncated_RT_wrt_stim[up_mask]
down_rt_values = cont_DDM_truncated_RT_wrt_stim[down_mask]
N_up = len(up_rt_values)
N_down = len(down_rt_values)
N_total = N_up + N_down
bins = np.arange(-2,2,0.01)
up_hist_cont, _  = np.histogram(up_rt_values, bins=bins, density=True)
down_hist_cont, _ = np.histogram(down_rt_values, bins=bins, density=True)

plt.plot(bins[:-1], up_hist_cont * (N_up / N_total), 'b', label='up cont DDM')
plt.plot(bins[:-1], -down_hist_cont * (N_down / N_total), 'b', label='down cont DDM')
plt.legend()

# theory
t_pts_wrt_stim = np.arange(-2,2,0.001)
t_stim_for_theory = np.random.choice(t_stim_for_sim, size=N_theory, replace=True)
up_theory_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))
down_theory_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))
for j, t_stim in enumerate(t_stim_for_theory):
    t_pts_wrt_fix = t_pts_wrt_stim + t_stim
    up_theory_samples[j, :] = \
        up_or_down_hit_truncated_proactive_V2_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, 1)
    down_theory_samples[j, :] = \
        up_or_down_hit_truncated_proactive_V2_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, -1)
    
up_samples_mean = up_theory_samples.mean(axis=0)
down_samples_mean = down_theory_samples.mean(axis=0)
plt.plot(t_pts_wrt_stim, up_samples_mean, 'k', label='up theory', ls='--', lw=3)
plt.plot(t_pts_wrt_stim, -down_samples_mean, 'k', label='down theory', ls='--', lw=3)
plt.show()

# %%
# up and down seperately
# skellam
rt_values = pro_skellam_RT_choice[:, 0]
rt_choices = pro_skellam_RT_choice[:, 1]
rt_wrt_stim = rt_values - t_stim_for_sim
# 0 and 1
mask_0_1 = (rt_wrt_stim > 0) & (rt_wrt_stim < 1)
rt_wrt_stim_01 = rt_wrt_stim[mask_0_1]
choices_01 = rt_choices[mask_0_1]

# up and down seperately
up_mask = choices_01 == 1
down_mask = choices_01 == -1
up_rt_values = rt_wrt_stim_01[up_mask]
down_rt_values = rt_wrt_stim_01[down_mask]
N_up = len(up_rt_values)
N_down = len(down_rt_values)
N_total = N_up + N_down
bins = np.arange(0,1,0.01)
up_hist_skellam, _  = np.histogram(up_rt_values, bins=bins, density=True)
down_hist_skellam, _ = np.histogram(down_rt_values, bins=bins, density=True)

plt.plot(bins[:-1], up_hist_skellam * (N_up / N_total), 'r', label='up skellam')
plt.plot(bins[:-1], -down_hist_skellam * (N_down / N_total), 'r', label='down skellam')



# Same for cont DDM
rt_values_cont = cont_DDM_RT_choice[:, 0]
rt_choices_cont = cont_DDM_RT_choice[:, 1]
rt_wrt_stim_cont = rt_values_cont - t_stim_for_sim
# 0 and 1
mask_0_1_cont = (rt_wrt_stim_cont > 0) & (rt_wrt_stim_cont < 1)
rt_wrt_stim_01_cont = rt_wrt_stim_cont[mask_0_1_cont]
choices_01_cont = rt_choices_cont[mask_0_1_cont]

# up and down seperately
up_mask = choices_01_cont == 1
down_mask = choices_01_cont == -1
up_rt_values = rt_wrt_stim_01_cont[up_mask]
down_rt_values = rt_wrt_stim_01_cont[down_mask]
N_up = len(up_rt_values)
N_down = len(down_rt_values)
N_total = N_up + N_down
bins = np.arange(0,1,0.01)
up_hist_cont, _  = np.histogram(up_rt_values, bins=bins, density=True)
down_hist_cont, _ = np.histogram(down_rt_values, bins=bins, density=True)

plt.plot(bins[:-1], up_hist_cont * (N_up / N_total), 'b', label='up cont DDM')
plt.plot(bins[:-1], -down_hist_cont * (N_down / N_total), 'b', label='down cont DDM')
plt.legend()

# theory
t_pts_wrt_stim = np.arange(0,1,0.001)
t_stim_for_theory = np.random.choice(t_stim_for_sim, size=N_theory, replace=True)
up_theory_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))
down_theory_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))
for j, t_stim in enumerate(t_stim_for_theory):
############ CORRECT WAY #######################
    trunc_factor_p_joint = cum_pro_and_reactive_trunc_fn(
                                t_stim + 1, c_A_trunc_time,
                                V_A, theta_A, t_A_aff,
                                t_stim, t_E_aff, mu1, mu2, theta_E) - \
                            cum_pro_and_reactive_trunc_fn(
                                t_stim, c_A_trunc_time,
                                V_A, theta_A, t_A_aff,
                                t_stim, t_E_aff, mu1, mu2, theta_E)
    
    t_pts_wrt_fix = t_pts_wrt_stim + t_stim
    up_theory_samples[j, :] = \
        up_or_down_hit_truncated_proactive_V2_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, 1)
    down_theory_samples[j, :] = \
        up_or_down_hit_truncated_proactive_V2_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, -1)
        
    # past method
    # up_theory_samples[j, :] = \
    #     up_or_down_hit_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, 1)
    # down_theory_samples[j, :] = \
    #     up_or_down_hit_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, -1)
    up_theory_samples[j, :] /= trunc_factor_p_joint
    down_theory_samples[j, :] /= trunc_factor_p_joint

###########################################################################
    print(f'area under up theory: {np.trapezoid(up_theory_samples[j, :], t_pts_wrt_stim)}')
    print(f'area under down theory: {np.trapezoid(down_theory_samples[j, :], t_pts_wrt_stim)}')
up_samples_mean = up_theory_samples.mean(axis=0)
down_samples_mean = down_theory_samples.mean(axis=0)
plt.plot(t_pts_wrt_stim, up_samples_mean, 'k', label='up theory', ls='--', lw=3)
plt.plot(t_pts_wrt_stim, -down_samples_mean, 'k', label='down theory', ls='--', lw=3)
plt.show()

# %%
t_pts_wrt_stim = np.arange(-5, 5, 0.001)

t_stim_for_theory = np.random.choice(t_stim_for_sim, size=N_theory, replace=True)
theory_pro_skellam_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))

for idx, t_stim in enumerate(t_stim_for_theory):
    t_pts_wrt_fix = t_pts_wrt_stim + t_stim
    up = up_or_down_hit_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, 1) 
    down = up_or_down_hit_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, -1)
    theory_pro_skellam_samples[idx, :] = up + down

# %%
theory_pro_skellam_mean = np.mean(theory_pro_skellam_samples, axis=0)
plt.plot(t_pts_wrt_stim, theory_pro_skellam_mean, 'r', alpha=0.5, lw=3, ls='--')
plt.hist(pro_skellam_RT_wrt_stim, bins=bins, label='Pro Skellam', density=True, histtype='step', color='b')
plt.show()

# %%
# Truncated likelihood, but truncation time = 0
c_A_trunc_time = 0
truncated_theory_pro_skellam_samples = np.zeros((N_theory, len(t_pts_wrt_stim)))
for j, t_stim in enumerate(t_stim_for_theory):
    t_pts_wrt_fix = t_pts_wrt_stim + t_stim
    truncated_theory_pro_skellam_samples[j, :] = \
        up_or_down_hit_truncated_proactive_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, 1) \
        + up_or_down_hit_truncated_proactive_fn(t_pts_wrt_fix, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, -1)
# %%
plt.scatter(t_pts_wrt_stim, truncated_theory_pro_skellam_samples.mean(axis=0), alpha=0.6, color='g', label='Truncated Theory (mean)')
plt.plot(t_pts_wrt_stim, theory_pro_skellam_mean, 'r', alpha=0.5, lw=3, ls='--', label='Theory')
plt.hist(pro_skellam_RT_wrt_stim, bins=bins, label='Pro Skellam', density=True, histtype='step', color='b')
plt.title('Truncation Time = 0: Theory vs Simulated', fontsize=14)
plt.legend()
plt.show()

# %%
# mask test
rt_values_1 = pro_skellam_RT_choice[:, 0]
remove_mask_1 = (rt_values_1 < t_stim_for_sim) & (rt_values_1 < c_A_trunc_time)
pro_skellam_truncated_RT_wrt_stim_1 = rt_values_1[~remove_mask_1] - t_stim_for_sim[~remove_mask_1]

rt_values_2 = pro_skellam_RT_choice[:, 0]
remove_mask_2 = rt_values_2 < c_A_trunc_time
pro_skellam_truncated_RT_wrt_stim_2 = rt_values_2[~remove_mask_2] - t_stim_for_sim[~remove_mask_2]


plt.hist(pro_skellam_truncated_RT_wrt_stim_1, bins=np.arange(-5,5, 0.02), label='2 mask terms', histtype='step', color='b', density=True)
plt.hist(pro_skellam_truncated_RT_wrt_stim_2, bins=np.arange(-5,5,0.02), label='1 mask term', color='r', histtype='step', ls='--', density=True)
plt.scatter(t_pts_wrt_stim, actual_truncated_mean_norm, alpha=0.5, label='Theory Truncated', s=7, color='g')
# plt.xlim(-1,1)
plt.legend()
plt.show()

# %%
c_A_trunc_time = 0.3
t_pts_wrt_stim_01 = np.arange(0, 1, 0.001)
truncated_theory_01 = np.zeros((N_theory, len(t_pts_wrt_stim_01)))
abs_diff_theory_trunc_and_area = np.zeros(N_theory) 
for j, t_stim in enumerate(t_stim_for_theory):
    t_pts_wrt_fix_01 = t_pts_wrt_stim_01 + t_stim
    truncated_theory_01[j, :] = \
        up_or_down_hit_truncated_proactive_V2_fn(t_pts_wrt_fix_01, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, 1) \
        + up_or_down_hit_truncated_proactive_V2_fn(t_pts_wrt_fix_01, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, -1)
    
    # truncation factor theoretical
    trunc_factor_theoretical = cum_pro_and_reactive_trunc_fn(
                                    t_stim + 1, c_A_trunc_time,
                                    V_A, theta_A, t_A_aff,
                                    t_stim, t_E_aff, mu1, mu2, theta_E) - \
                                cum_pro_and_reactive_trunc_fn(
                                    t_stim, c_A_trunc_time,
                                    V_A, theta_A, t_A_aff,
                                    t_stim, t_E_aff, mu1, mu2, theta_E)
    # area under cuve should match trunc_factor_theoretical
    area = np.trapezoid(truncated_theory_01[j, :], t_pts_wrt_stim_01)
    abs_diff_theory_trunc_and_area[j] = np.abs(trunc_factor_theoretical - area)

# %%
plt.hist(abs_diff_theory_trunc_and_area, bins=500, density=True)
plt.title('Absolute difference between theoretical truncation factor and area under curve')
plt.show()
# %%
# truncated 0 to 1
truncated_theory_01_mean = np.mean(truncated_theory_01, axis=0)
area = np.trapezoid(truncated_theory_01_mean, t_pts_wrt_stim_01)
print(f'theory area = {area}')
# skellam
pro_skellam_truncated_RT_wrt_stim_01 = pro_skellam_truncated_RT_wrt_stim[
    (pro_skellam_truncated_RT_wrt_stim >= 0) & (pro_skellam_truncated_RT_wrt_stim <= 1)
]

plt.figure(figsize=(8, 4))
plt.hist(
    pro_skellam_truncated_RT_wrt_stim_01,
    bins=np.arange(0, 1.01, 0.01),
    label='Pro Skellam Truncated',
    density=True,
    histtype='step',
    color='C0',
    lw=2,
    alpha=0.85
)
plt.plot(
    t_pts_wrt_stim_01,
    truncated_theory_01_mean / area,
    color='C3',
    lw=3,
    linestyle='--',
    alpha=0.85,
    label='Theory Truncated'
)
plt.legend(frameon=True, fontsize=12)
plt.xlabel('RT - t_stim', fontsize=13)
plt.ylabel('Density', fontsize=13)
plt.title('Truncated RT Distribution (0 to 1 s)\nTheory vs Simulated', fontsize=14)
plt.tight_layout()
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()

# %%
t_pts_wrt_stim = np.arange(-5, 5, 0.001)
t_stim = 0.4
t_pts_wrt_fix = t_pts_wrt_stim + t_stim
t2 = t_pts_wrt_fix - t_stim - t_E_aff + del_go
t1 = t_pts_wrt_fix - t_stim - t_E_aff
bound = 1
proactive_trunc_time = 0.3
P_A = truncated_rho_A_t_fn(t_pts_wrt_fix - t_A_aff, V_A, theta_A, proactive_trunc_time)
prob_EA_hits_either_bound = fpt_cdf_skellam(t_pts_wrt_fix - t_stim - t_E_aff + del_go, mu1, mu2, theta_E)

prob_EA_survives = 1 - prob_EA_hits_either_bound
random_readout_if_EA_surives = 0.5 * prob_EA_survives
p_choice = fpt_choice_skellam(mu1, mu2, theta_E, bound)

# Safe CDF values with negative times treated as 0 - vectorized version
c2 = np.zeros_like(t2, dtype=float)
c1 = np.zeros_like(t1, dtype=float)
mask2 = t2 > 0
mask1 = t1 > 0
if np.any(mask2):
    c2[mask2] = fpt_cdf_skellam(t2[mask2], mu1, mu2, theta_E)
if np.any(mask1):
    c1[mask1] = fpt_cdf_skellam(t1[mask1], mu1, mu2, theta_E)
P_E_plus_or_minus_cum = p_choice * (c2 - c1)

dt_pdf = t_pts_wrt_fix - t_E_aff - t_stim
P_E_plus_or_minus = np.zeros_like(dt_pdf, dtype=float)
mask_pdf = dt_pdf > 0
if np.any(mask_pdf):
    P_E_plus_or_minus[mask_pdf] = fpt_density_skellam(dt_pdf[mask_pdf], mu1, mu2, theta_E)
P_E_plus_or_minus *= p_choice

# C_A = cum_A_t_fn(t - t_A_aff, V_A, theta_A)
C_A = truncated_cum_A_t_fn(t_pts_wrt_fix - t_A_aff, V_A, theta_A, proactive_trunc_time)
# (P_A*(random_readout_if_EA_surives + P_E_plus_or_minus_cum) + P_E_plus_or_minus*(1-C_A))

proactive_coin_flip_term = P_A * random_readout_if_EA_surives
proactive_and_reactive_hit_term = P_A * P_E_plus_or_minus_cum
reactive_win_term = P_E_plus_or_minus * (1 - C_A)

plt.plot(t_pts_wrt_stim, proactive_coin_flip_term, label='Proactive Coin Flip Term')
plt.plot(t_pts_wrt_stim, proactive_and_reactive_hit_term, label='Proactive and Reactive Hit Term')
plt.plot(t_pts_wrt_stim, reactive_win_term, label='Reactive Win Term')
plt.legend()
plt.axvline(0, color='k', linestyle='--', alpha=0.5)
plt.axvline(t_E_aff, color='k', linestyle='--', alpha=0.5)
plt.xlim(-1,1)
plt.title(f't-stim = {t_stim}')
plt.show() 

# %%
