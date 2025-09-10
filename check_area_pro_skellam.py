# %%
import numpy as np
from vbmc_skellam_utils import up_or_down_hit_truncated_proactive_fn
# %%
N_theory = int(5e3)
t_stim = 0.3
# t_stim = 0

V_A = 1
theta_A = 2
t_A_aff = -0.3
# t_A_aff = 0.3
t_E_aff = 0.08
del_go = 0.13
mu1 = 150
mu2 = 130
theta_E = 11
c_A_trunc_time = 0

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
