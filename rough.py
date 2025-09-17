# %%
import numpy as np
import matplotlib.pyplot as plt


# %%

# %%

# r = np.arange(0.95, 1.05, 0.01)
r = np.arange(0.001, 80, 0.1)

# theta = 40
theta_vals = [1,2,3, 5, 20, 40]

fig, axes = plt.subplots(1, len(theta_vals), figsize=(5 * len(theta_vals), 4), sharex=True, sharey=True)
if len(theta_vals) == 1:
    axes = [axes]
for ax, theta in zip(axes, theta_vals):
    y1 = 1 / (1 + np.exp(-2 * theta * (r - 1) / (r + 1)))
    r1 = 1 / r
    y2 = (1 - r1**theta) / (1 - r1**(2 * theta))
    ax.plot((r-1)/(r+1), y1, label=r"$\frac{1}{1+\exp\left[-2\theta\frac{r-1}{r+1}\right]}$", color='blue', ls='--')
    ax.plot((r-1)/(r+1), y2, label=r"$\frac{1 - (1/r)^{\theta}}{1 - (1/r)^{2\theta}}$", color='red', ls='--', lw=3, alpha=0.5)
    ax.set_title(f"$\\theta$ = {theta}")
    ax.legend(fontsize=11)
    ax.set_xlabel('r-1/(r+1)')
    ax.set_ylabel('prob of choosing right')
plt.tight_layout()
plt.show()
# %%
fig, axes = plt.subplots(1, len(theta_vals), figsize=(5 * len(theta_vals), 4), sharex=True, sharey=True)
if len(theta_vals) == 1:
    axes = [axes]
for ax, theta in zip(axes, theta_vals):
    y10 = 1 / (1 + np.exp(-2 * theta * (r - 1) / (r + 1)))
    y1 = np.log(y10 / (1-y10))
    r1 = 1 / r
    y20 = (1 - r1**theta) / (1 - r1**(2 * theta))
    y2 = np.log(y20 / (1-y20))
    ax.plot((r-1)/(r+1), y1, label=r"$\frac{1}{1+\exp\left[-2\theta\frac{r-1}{r+1}\right]}$", color='blue', ls='--')
    ax.plot((r-1)/(r+1), y2, label=r"$\frac{1 - (1/r)^{\theta}}{1 - (1/r)^{2\theta}}$", color='red', ls='--', lw=3, alpha=0.5)
    ax.set_title(f"$\\theta$ = {theta}")
    ax.legend(fontsize=11)
    ax.set_xlabel('r-1/(r+1)')
    ax.set_ylabel('log (p / 1-p)')
plt.tight_layout()
plt.show()

# %%
r = np.arange(0.01, 2, 0.1)

plt.plot(r, (r-1)/(r+1))
plt.title('r-1/(r+1)')
plt.xlabel('r')
plt.show()

# %%
fname = '/home/ragha/code/simulate_poisson/real_data_cond_wise_theta_fixed/ABL_20_ILD_1/ABL_20_ILD_1_theta05.pkl'

import pickle
from pyvbmc import VBMC

vp = VBMC.load(fname)
vp_samples =vp.vp.sample(int(100e3))
logr = vp_samples[0][:, 0]
logl = vp_samples[0][:, 1]
r_samp = 10**logr
l_samp = 10**logl

plt.hist(r_samp, bins=100, color='C0', alpha=0.8, histtype='step')
plt.hist(l_samp, bins=100, color='C1', alpha=0.8, histtype='step')
plt.axvline(np.mean(r_samp), color='C2', ls='--')
plt.axvline(np.mean(l_samp), color='C3', ls='--')

print(f'mean = {np.mean(r_samp)}, {np.mean(l_samp)}')
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt
max_ild = 100
x = np.arange(-max_ild, max_ild, 0.1)
plt.plot(x, np.tanh(2.13* x/17.37))
# %%
ild_range = np.arange(-16,16,0.1) # sound level difference between left and right ear  (dB)
abl = 20 # average sound level in left and right year  - dB
r_db = abl + (ild_range/2) # sound level in righ ear - dB
l_db = abl - (ild_range/2) # sound level in left ear - dB
p0 = 20 * 1e-6 # reference pressure level - Pa
p_r = p0 * (10**(r_db/20)) # pressure in right ear - Pa
p_l = p0 * (10**(l_db/20)) # pressure in left ear - Pa

R0 = 10 # base firing rate
lam = 2.13 # power law from stimulus to rate mapping: rate = base_rate * (pressure / reference pressure level) ^ lambda
theta = 3 # bound for accumulation stopping

r_R = R0 * ((p_r/p0)**lam) # firing rate in right side of brain
r_L = R0 * ((p_l/p0)**lam) # firing rate in left side of brain

x_ax = 20 * np.log10(p_r/p_l) # sound pressure difference
y_ax =  theta * np.log10(r_R/r_L) # log odds in poisson model

plt.plot(x_ax, y_ax)
plt.xlabel('sound pressure difference')
plt.ylabel('log odds')
plt.title('poisson model')
plt.show()


