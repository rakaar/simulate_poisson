# %%
import numpy as np
from vbmc_skellam_utils import up_or_down_hit_truncated_proactive_fn
# %%
N_theory = int(5e3)
t_stim = 0.3
# t_stim = 0

V_A = 1.1774684269283593
theta_A = 8
t_A_aff = 0.04163924630404911
# t_A_aff = 0.3
t_E_aff = 0.07315391552330006
del_go = 0.19088319539826992
mu1 = 383.2133783138636
mu2 = 515.9030272556699
theta_E = 11
c_A_trunc_time = 0

# %%
bound = 1
from vbmc_skellam_utils import fpt_cdf_skellam, fpt_density_skellam, fpt_choice_skellam

t_pts_wrt_stim = np.arange(0,1,0.001)
t_pts_wrt_fix = t_pts_wrt_stim + t_stim
proactive_trunc_time = 0
t2 = t_pts_wrt_fix - t_stim - t_E_aff + del_go
t1 = t_pts_wrt_fix - t_stim - t_E_aff
# PA hits
P_A = truncated_rho_A_t_fn(t_pts_wrt_fix - t_A_aff, V_A, theta_A, proactive_trunc_time)

# EA survies and random choice
prob_EA_hits_either_bound = fpt_cdf_skellam(t_pts_wrt_fix - t_stim - t_E_aff + del_go, mu1, mu2, theta_E)
prob_EA_survives = 1 - prob_EA_hits_either_bound
random_readout_if_EA_surives = 0.5 * prob_EA_survives

# EA hits
p_choice = fpt_choice_skellam(mu1, mu2, theta_E, bound)
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

C_A = truncated_cum_A_t_fn(t_pts_wrt_fix - t_A_aff, V_A, theta_A, proactive_trunc_time)

# %%
plt.plot(t_pts_wrt_stim, (P_A*(random_readout_if_EA_surives + P_E_plus_or_minus_cum) + P_E_plus_or_minus*(1-C_A)), label='all terms')
plt.plot(t_pts_wrt_stim, P_A*(random_readout_if_EA_surives + P_E_plus_or_minus_cum), label='proactive term', ls='--', lw=3, alpha=0.6)
plt.scatter(t_pts_wrt_stim, P_A*(random_readout_if_EA_surives), label='proactive coin term', marker='o')
plt.scatter(t_pts_wrt_stim, P_A * P_E_plus_or_minus_cum, label='proactive EA term', marker='x', color='r')
plt.title(f'V_A  = {V_A :.2f}, theta_A = {theta_A}')
plt.legend()
plt.show()
# %%
t_pts_wrt_stim = np.arange(-5, 5, 0.01)
up1 = up_or_down_hit_truncated_proactive_fn(t_pts_wrt_stim + t_stim, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, 1)
down1 = up_or_down_hit_truncated_proactive_fn(t_pts_wrt_stim + t_stim, V_A, theta_A, t_A_aff, t_stim, t_E_aff, del_go, mu1, mu2, theta_E, c_A_trunc_time, -1)

area_up = np.trapezoid(up1, t_pts_wrt_stim)
area_down = np.trapezoid(down1, t_pts_wrt_stim)

total_area = area_up + area_down
print(f'total area = {total_area}')


# %%
import matplotlib.pyplot as plt

plt.plot(t_pts_wrt_stim, up1)
plt.plot(t_pts_wrt_stim, down1)
plt.xlim(0,3)
plt.show()

# %%
from vbmc_skellam_utils import truncated_cum_A_t_fn, truncated_rho_A_t_fn

c = truncated_cum_A_t_fn(t_pts_wrt_stim + t_stim, V_A, theta_A, c_A_trunc_time)

p = truncated_rho_A_t_fn(t_pts_wrt_stim + t_stim, V_A, theta_A, c_A_trunc_time)

plt.plot(t_pts_wrt_stim, c)
plt.show()
plt.plot(t_pts_wrt_stim, p)
plt.show()
# %%
from vbmc_skellam_utils import fpt_cdf_skellam, fpt_density_skellam

c1 = fpt_density_skellam(t_pts_wrt_stim - t_stim, mu1, mu2, theta_E)
plt.plot(t_pts_wrt_stim, c1)
plt.show()

from scipy.integrate import cumulative_trapezoid as cumtrapz
# cumtrapz of c1
plt.plot(t_pts_wrt_stim, cumtrapz(c1, t_pts_wrt_stim, initial=0))
# plt.show()

c2 = fpt_cdf_skellam(t_pts_wrt_stim - t_stim, mu1, mu2, theta_E)
plt.plot(t_pts_wrt_stim, c2, alpha=0.4, lw=3,ls='--')
plt.show()
# %%
tpts = np.arange(-1, 1, 0.001)
c3 = fpt_cdf_skellam(tpts - t_E_aff, mu1, mu2, theta_E)
plt.plot(tpts, c3)
plt.axvline(x=0, color='r', linestyle='--')
plt.show()


# %%




# %%
plt.plot(t_pts_wrt_stim, P_A*(random_readout_if_EA_surives + P_E_plus_or_minus_cum))
plt.show()

# %%
plt.plot(t_pts_wrt_stim, P_E_plus_or_minus*(1-C_A))
plt.show()

# %%

plt.plot(t_pts_wrt_stim, P_A*(random_readout_if_EA_surives))
plt.show()

# %%

plt.plot(t_pts_wrt_stim, P_A*(P_E_plus_or_minus_cum))
plt.show()  

# %%
plt.plot(t_pts_wrt_stim, P_A)