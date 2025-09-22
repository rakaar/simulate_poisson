# see how skelalm and ddm mean RT vs ILD
# %%
import numpy as np
import matplotlib.pyplot as plt

def skellam_mean_rt(theta, muR, muL):
    """
    Compute mean RT for Skellam model.
    
    Parameters:
        theta : float
        muR   : float
        muL   : float
    
    Returns:
        mean RT
    """
    numerator = theta * np.tanh(theta * np.log(muR / muL))
    denominator = muR - muL + 1e-10
    return numerator / denominator


def ddm_mean_rt(theta, muR, muL):
    """
    Compute mean RT for DDM model.
    
    Parameters:
        theta : float
        muR   : float
        muL   : float
    
    Returns:
        mean RT
    """
    numerator = theta * np.tanh(theta * (muR - muL) / (muR + muL))
    denominator = muR - muL + 1e-10
    return numerator / denominator

Nr0 = 1000/0.43
lam = 0.13
def calc_mu_r_l(ABL, ILD):
    slr = ABL + (ILD/2) # db
    sll = ABL - (ILD/2) # db
    
    # Nr0 = 1000/283 
    # lam = 4.6
    # ell = 0.9
    
    p0 = 1

    pr = p0 * (10**(slr/20)) # pa
    pl = p0 * (10**(sll/20)) # pa
    rr = Nr0 * ((pr/p0)**lam)
    rl = Nr0 * ((pl/p0)**lam)
    # norm_term = rr**(ell) + rl**(ell)
    norm_term = 1

    return rr/norm_term, rl/norm_term


# Parameters
ABL = 20
theta = 40
ild_range = np.arange(0.1, 100, 0.1)

# To store results
mur_values = []
mul_values = []
skellam_rts = []
ddm_rts = []

# Calculate mur, mul and mean RTs for the ild range
for ild in ild_range:
    mur, mul = calc_mu_r_l(ABL, ild)
    mur_values.append(mur)
    mul_values.append(mul)
    
    skellam_rt = skellam_mean_rt(theta, mur, mul)
    skellam_rts.append(skellam_rt)
    
    ddm_rt = ddm_mean_rt(theta, mur, mul)
    ddm_rts.append(ddm_rt)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(ild_range, skellam_rts, label='Skellam Mean RT')
plt.plot(ild_range, ddm_rts, label='DDM Mean RT', ls='--', lw=2)
plt.xlabel('ILD (dB)')
plt.ylabel('Mean RT (s)')
plt.title(f'Mean RT vs. ILD for Skellam and DDM Models\nNr0 = {Nr0 :.2f}, lam = {lam :.2f}, theta={theta}')
plt.xlim(0,17)
plt.legend()
plt.show()


