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
theta_prime = 40 # this is free
B = theta / theta_prime
r0 = R0/(B**2)

# %%
def tied_gamma(ild):
    return theta * (np.tanh(lam * ild/chi))

def tied_omega(abl,ild):
    return (R0/(theta**2)) * (10 ** ( lam * (1 - ell) * abl / 20 )) * (np.cosh(lam*ild/chi) / np.cosh(lam*ell*ild/chi))


# %%
def new_gamma(ild):
    return ( theta_prime * B) *  np.tanh(lam * ild/chi)

def new_omega(abl,ild):
    return (r0/(theta_prime**2)) * (10 ** ( lam * (1 - ell) * abl / 20 )) * (np.cosh(lam*ild/chi) / np.cosh(lam*ell*ild/chi))

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
    ax2.plot(ild_range, tied_omega(abl,ild_range), color='b', lw=1, label='Tied' if i == 0 else None)
    ax2.plot(ild_range, new_omega(abl,ild_range), ls='--', color='r', lw=3, alpha=0.4, label='New' if i == 0 else None)
ax2.set_xlabel('ILD')
ax2.set_ylabel(r'$\Omega$')
ax2.legend()
ax2.set_title('Omega')

fig.suptitle(f'New vs TIED\nTIED: λ={lam}, R0={R0}, θ={theta}, ℓ={ell}\nNEW: χ={chi}, θ\'={theta_prime}, B={B:.4f}, r0={r0:.4f}')
plt.tight_layout()
# %%


import numpy as np
import matplotlib.pyplot as plt

B_target = B

def B_of(alpha, beta):
    return (1 - alpha*beta) / np.sqrt(1 + alpha*alpha*beta)


beta_grid = np.linspace(0.01, 10.0, 2000)
alpha_grid = np.linspace(0.01, 10.0, 2000)

# --- Solve for alpha given beta (B fixed) ---
A = beta_grid * (beta_grid - B_target**2)          # b(b - B^2)
Bcoef = -2 * beta_grid                              # -2b
C = 1 - B_target**2

disc = np.maximum(Bcoef**2 - 4*A*C, 0.0)
sqrt_disc = np.sqrt(disc)

alpha_root1 = (-Bcoef + sqrt_disc) / (2*A)
alpha_root2 = (-Bcoef - sqrt_disc) / (2*A)

alpha_sol = np.full_like(beta_grid, np.nan)
for i, b in enumerate(beta_grid):
    for a in (alpha_root1[i], alpha_root2[i]):
        if np.isfinite(a) and a > 0 and a*b < 1 and abs(B_of(a, b) - B_target) < 1e-6:
            alpha_sol[i] = a
            break

mask_ab = np.isfinite(alpha_sol)
beta_v1, alpha_v1 = beta_grid[mask_ab], alpha_sol[mask_ab]

# --- Solve for beta given alpha (B fixed) ---
A2 = alpha_grid**2
Bcoef2 = -(2*alpha_grid + (B_target**2)*(alpha_grid**2))
C2 = 1 - B_target**2

disc2 = np.maximum(Bcoef2**2 - 4*A2*C2, 0.0)
sqrt_disc2 = np.sqrt(disc2)

beta_root1 = (-Bcoef2 + sqrt_disc2) / (2*A2)
beta_root2 = (-Bcoef2 - sqrt_disc2) / (2*A2)

beta_sol = np.full_like(alpha_grid, np.nan)
for i, a in enumerate(alpha_grid):
    for b in (beta_root1[i], beta_root2[i]):
        if np.isfinite(b) and b > 0 and a*b < 1 and abs(B_of(a, b) - B_target) < 1e-6:
            beta_sol[i] = b
            break

mask_ba = np.isfinite(beta_sol)
alpha_v2, beta_v2 = alpha_grid[mask_ba], beta_sol[mask_ba]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1) Linear: alpha vs beta
axes[0, 0].scatter(beta_v1, alpha_v1, s=10)
axes[0, 0].set_xlabel('beta')
axes[0, 0].set_ylabel('alpha')
axes[0, 0].set_title(f'Linear scale, B={B_target}: alpha vs beta')

# 2) Linear: beta vs alpha
axes[0, 1].scatter(alpha_v2, beta_v2, s=10)
axes[0, 1].set_xlabel('alpha')
axes[0, 1].set_ylabel('beta')
axes[0, 1].set_title(f'Linear scale, B={B_target}: beta vs alpha')

# 3) Log-log: alpha vs beta
axes[1, 0].scatter(beta_v1, alpha_v1, s=10)
axes[1, 0].set_xscale('log')
axes[1, 0].set_yscale('log')
axes[1, 0].set_xlabel('beta')
axes[1, 0].set_ylabel('alpha')
axes[1, 0].set_title(f'Log-log scale, B={B_target}: alpha vs beta')

# 4) Log-log: beta vs alpha
axes[1, 1].scatter(alpha_v2, beta_v2, s=10)
axes[1, 1].set_xscale('log')
axes[1, 1].set_yscale('log')
axes[1, 1].set_xlabel('alpha')
axes[1, 1].set_ylabel('beta')
axes[1, 1].set_title(f'Log-log scale, B={B_target}: beta vs alpha')

plt.tight_layout()
plt.show()
