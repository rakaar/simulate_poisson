"""
Parameters for Poisson (correlated) vs DDM (uncorrelated) analysis
"""

N_sim = 100  # Number of trials for jump distribution analysis
N_sim_rtd = int(2*50e3)  # Number of trials for RTD (reaction time distribution)

# instead of N, c is hardcoded. 
# N_right_and_left = 50
c = 0.01
corr_factor = 5
# 
N_right_and_left = round(((corr_factor - 1)/c) + 1)
N_right = N_right_and_left  
N_left = N_right_and_left   
if N_right_and_left < 1:
    raise ValueError("N_right_and_left must be greater than 1")
# tweak
theta = 2
# theta = 10
theta_scaled = theta * corr_factor

# get from model fits
lam = 1.3
l = 0.9
Nr0 = 13.3 
# Nr0 = 100

# correlation and base firing rate
# c = (corr_factor - 1) / (N_right_and_left - 1)
# if c < 0 or c >1:
#     raise ValueError("Correlation must be between 0 and 1")
r0 = Nr0/N_right_and_left
r0_scaled = r0 * corr_factor

abl = 20
ild = 0
r_db = (2*abl + ild)/2
l_db = (2*abl - ild)/2
pr = (10 ** (r_db/20))
pl = (10 ** (l_db/20))

den = (pr ** (lam * l) ) + ( pl ** (lam * l) )
rr = (pr ** lam) / den
rl = (pl ** lam) / den

# Scaled firing rates (for Poisson simulation)
r_right_scaled = r0_scaled * rr
r_left_scaled = r0_scaled * rl


# Unscaled firing rates (for DDM)
r_right = r0 * rr
r_left = r0 * rl


T = 20  # Max duration of a single trial (seconds)

# DDM parameters
N_neurons = N_right
mu = N_neurons * (r_right - r_left)
# corr_factor_ddm = 1 + ((N_neurons - 1) * c)
corr_factor_ddm = 1
sigma_sq = N_neurons * (r_right + r_left) * corr_factor_ddm
sigma = sigma_sq**0.5
theta_ddm = theta
