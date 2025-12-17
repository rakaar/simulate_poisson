# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
# TIED params 
lam = 2.13
R0 = 13.3
theta = 2.13
ell = 0.9 

# print params
print(f'lam = {lam}')
print(f'R0 = {R0}')
print(f'theta = {theta}')
print(f'ell = {ell}')
# %%
# new params 
chi = 17.37

K = 2 * lam / chi
lam_prime = lam*(1-ell)/20
theta_prime = 6 # this is free

# calculate r0 and A from the constraint equations
x = theta_prime / theta
A = R0 * x
r0 = (A/2) * (x-1)

print(f'x = theta_prime/theta = {x:.4f}')
print(f'A = {A:.4f}')
print(f'r0 = {r0:.4f}')

# %%
def tied_gamma(ild):
    return theta * (np.tanh(lam * ild/chi))

def tied_omega(abl,ild):
    return (R0/(theta**2)) * (10 ** ( lam * (1 - ell) * abl / 20 )) * (np.cosh(lam*ild/chi) / np.cosh(lam*ell*ild/chi))
# %%
ild_range = np.arange(-16,16,0.1)
extra_ratio  = np.cosh(lam * ild_range/chi) / np.cosh(lam * ell * ild_range/chi)
plt.plot(ild_range, extra_ratio)
plt.xlabel('ILD')
plt.ylabel(r'$\frac{\cosh(\lambda \cdot \text{ILD} / \chi)}{\cosh(\lambda \cdot \ell \cdot \text{ILD} / \chi)}$')
plt.title(r'if $\lambda $ is large, then the ratio matters')

# %%
def new_gamma(ild):
    return ( theta_prime / ( (2*r0/A) + 1) ) *  np.tanh(K * ild/2)

def new_omega(abl,ild):
    return np.ones_like(ild) * ( ( 2*r0 + A ) / ( theta_prime**2) ) * (10 ** (lam_prime * abl))
# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gamma plot
ax1.plot(ild_range, new_gamma(ild_range), color='r', ls='--', lw=3, alpha=0.4, label='New')
ax1.plot(ild_range, tied_gamma(ild_range), color='b', label='Tied')
ax1.set_xlabel('ILD')
ax1.set_ylabel(r'$\Gamma$')
ax1.legend()
ax1.set_title('Gamma')

# Omega plot
abl_range = [20,40,60]
for i, abl in enumerate(abl_range):
    ax2.plot(ild_range, new_omega(abl,ild_range), ls='--', color='r', lw=3, alpha=0.4, label='New' if i == 0 else None)
    ax2.plot(ild_range, tied_omega(abl,ild_range), color='b', lw=3, label='Tied' if i == 0 else None)
ax2.set_xlabel('ILD')
ax2.set_ylabel(r'$\Omega$')
ax2.legend()
ax2.set_title('Omega')

fig.suptitle(f'New vs TIED\nTIED: λ={lam}, R0={R0}, θ={theta}, ℓ={ell}\nNEW: χ={chi}, K={K:.4f}, λ\'={lam_prime:.4f}, θ\'={theta_prime}, A={A:.4f}, r0={r0:.4f}')
plt.tight_layout()
# %%
ild_range = np.arange(-32,32,0.1)
abl = 40
abl_gain = 10**(lam_prime * abl)
right_rates = (r0 + A /(1 + np.exp(-K*ild_range))) * abl_gain
plt.plot(ild_range, right_rates)
plt.xlabel('ILD')
plt.ylabel('right rate')
plt.ylim(bottom=r0 - 10)
plt.axhline(y=r0, color='r', linestyle='--')
plt.title(f'theta prime = {theta_prime}, abl={abl}, r0={r0 :.2f}, A={A :.2f},r0/A={r0/A :.2f}')

# %%



