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
ild = 4
dt = 1e-6  # Time step for continuous DDM simulation
dB = 1e-3

rate_scale_factor = 1
theta_increment = 0

# %%
N_sim = int(50e3)


theta = 2 + theta_increment
# NC_pairs = [(int(5/1e-3), 1e-3), (int(5/2e-3), 2e-3), (int(5/1e-2), 1e-2), (int(5/0.1), 0.1)]
# NC_pairs = [(1000, 1e-3), (int(2/1e-2), 1e-2), (int(3/2e-3), 2e-3), (int(5/0.1), 0.1)]
NC_pairs = [(100, 1e-5), (10, 1e-4)]
poisson_sim_data_NC_wise = {}
for nc_idx, nc_pair in enumerate(NC_pairs):
    N, rho = nc_pair
    poisson_sim_data_NC_wise[nc_idx] = {}
    
    Nr0 = Nr0_base  * rate_scale_factor
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

    poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(delayed(run_poisson_trial)(N, rho, r_right, r_left, theta) for _ in tqdm(range(N_sim)))
    poisson_sim_data_NC_wise[nc_pair] = poisson_data

# %%
bins = np.arange(0,2,0.01)
plt.figure(figsize=(10, 6))
for pair in NC_pairs:
    rt_choice_data = poisson_sim_data_NC_wise[pair]
    rts_per_NC = np.array([d[0] for d in rt_choice_data])
    rt_mean = np.mean(rts_per_NC)
    plt.hist(rts_per_NC, bins=bins, label=f'N={pair[0]}, rho={pair[1]} (mean={rt_mean:.3f})', density=True, histtype='step')
    plt.axvline(rt_mean, color='black', linestyle='--', alpha=0.5)
plt.legend()
plt.xlabel('RT')
plt.ylabel('density')
plt.title(f'N x rho = varies')
plt.xlim(0,1)

# %%
from scipy import optimize

# is it in h0 ?
def find_h0(r_right, r_left, N, rho):
    def f(t):
        lambda_p = r_right
        lambda_n = r_left
        
        term1 = (((1 + rho * (np.exp(t) - 1)) ** N) - 1) * lambda_p
        term2 = (((1 + rho * (np.exp(-t) - 1)) ** N) - 1) * lambda_n
        
        return term1 + term2
    # Use a bracket method to find the root
    # Start with a reasonable negative range
    a = -10.0  # Lower bound
    b = -0.001  # Upper bound (slightly negative)
    
    # Check if the function changes sign in this interval
    if f(a) * f(b) > 0:
        print("Function might not have a root in the specified interval.")
        # Try to find a better interval by sampling
        t_values = np.linspace(-15, -0.0001, 100)
        f_values = [f(t) for t in t_values]
        
        for i in range(len(t_values)-1):
            if f_values[i] * f_values[i+1] <= 0:
                a = t_values[i]
                b = t_values[i+1]
                break
    
    try:
        # Use scipy's root finding method
        h0 = optimize.brentq(f, a, b)
        # print(f"Found root h0 = {h0}")
        return h0
    except ValueError as e:
        print(f"Error finding root: {e}")
        # Try another approach using a general-purpose method
        try:
            result = optimize.root_scalar(f, bracket=[a, b], method='brentq')
            h0 = result.root
            print(f"Found root h0 = {h0} using alternative method")
            return h0
        except Exception as e:
            print(f"Failed to find root: {e}")
            return None

NC_pairs = [(int(2/1e-3), 1e-3), (2*750, 1/750), (int(2/2e-3), 2e-3), (int(2/1e-2), 1e-2), (250*2, 1/250), (int(1*2/0.1), 0.1)]
h0_dict_wise = {}
for nc_idx, nc_pair in enumerate(NC_pairs):
    N, rho = nc_pair
    poisson_sim_data_NC_wise[nc_idx] = {}
    
    Nr0 = Nr0_base  * rate_scale_factor
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
    # print(f'approx sol will be {(r_left/r_right)}')

    h0_sol = find_h0(r_right, r_left, N, rho)
    print(f'N={N}, rho={rho}, h0={h0_sol}')
    h0_dict_wise[nc_pair] = h0_sol
    
# %%
