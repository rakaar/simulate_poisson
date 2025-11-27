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

abl = 40
ild = 4
dt = 1e-6  # Time step for continuous DDM simulation
dB = 1e-3


# %%
N_sim = int(10e3)


theta = 2
N_rho_ratex_bound_plus_quads = [
    (100, 1e-5, 1, 0),
    (10, 1e-4, 1, 0)   
]
poisson_sim_data_NC_wise = {}
timing_results = []
for nc_idx, ncxb in enumerate(N_rho_ratex_bound_plus_quads):
    N, rho, rate_scaling_factor, bound_increment = ncxb
    poisson_sim_data_NC_wise[nc_idx] = {}
    
    new_theta = theta + bound_increment
    Nr0 = Nr0_base  * rate_scaling_factor
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

    start_time = time.time()
    poisson_data = Parallel(n_jobs=multiprocessing.cpu_count()-2)(delayed(run_poisson_trial)(N, rho, r_right, r_left, new_theta) for _ in tqdm(range(N_sim)))
    elapsed_time = time.time() - start_time
    timing_results.append((ncxb, elapsed_time))
    poisson_sim_data_NC_wise[ncxb] = poisson_data

# Save timing results to txt file
with open('NC_diff_timing.txt', 'w') as f:
    f.write(f'Timing results for N_sim={N_sim}\n')
    f.write('=' * 50 + '\n')
    for ncxb, elapsed in timing_results:
        f.write(f'N={ncxb[0]}, rho={ncxb[1]}, rate_scale={ncxb[2]}, bound_inc={ncxb[3]}: {elapsed:.2f}s\n')

# %%
# ddm data
N = 100
Nr0 = Nr0_base
r0 = Nr0/N
r_db = (2*abl + ild)/2
l_db = (2*abl - ild)/2
pr = (10 ** (r_db/20))
pl = (10 ** (l_db/20))
N_sim_rtd = int(50e3)
T = 10
dt = 1e-4
dB = 1e-2
n_steps = int(T/dt)
den = (pr ** (lam * l)) + (pl ** (lam * l))
rr = (pr ** lam) / den
rl = (pl ** lam) / den
r_right_ddm = r0 * rr
r_left_ddm = r0 * rl

mu = N * (r_right_ddm - r_left_ddm)
sigma_sq = N * (r_right_ddm + r_left_ddm)
sigma = sigma_sq**0.5
theta_ddm = 2
def simulate_single_ddm_trial(trial_idx, mu, sigma, theta_ddm, dt, dB, n_steps):
    """Simulate a single DDM trial."""
    DV = 0.0
    
    for step in range(n_steps):
        # Generate single evidence step
        evidence_step = mu*dt + (sigma)*np.random.normal(0, dB)
        DV += evidence_step
        
        # Check for boundary crossing
        if DV >= theta_ddm:
            return (step * dt, 1)
        elif DV <= -theta_ddm:
            return (step * dt, -1)
    
    # No decision made within time limit
    return (np.nan, 0)

ddm_results = Parallel(n_jobs=-1)(
    delayed(simulate_single_ddm_trial)(i, mu, sigma, theta_ddm, dt, dB, n_steps) 
    for i in tqdm(range(N_sim_rtd), desc='Simulating DDM')
)
ddm_data = np.array(ddm_results)
ddm_rts = ddm_data[:, 0]
# %%
bins = np.arange(0,2,0.01)
plt.figure(figsize=(10, 6))
for ncxb in N_rho_ratex_bound_plus_quads:
    rt_choice_data = poisson_sim_data_NC_wise[ncxb]
    rts_per_NC = np.array([d[0] for d in rt_choice_data])
    rt_mean = np.mean(rts_per_NC)
    plt.hist(rts_per_NC, bins=bins, label=f'N={ncxb[0]}, rho={ncxb[1]} (mean={rt_mean:.3f})', density=True, histtype='step')
    plt.axvline(rt_mean, color='black', linestyle='--', alpha=0.5)
plt.hist(ddm_rts, bins=bins, label='DDM', color='k', linewidth=2, histtype='step', density=True)
plt.axvline(np.mean(ddm_rts), color='g', linewidth=2, alpha=0.3, label='ddm mean')
plt.legend()
plt.xlabel('RT')
plt.ylabel('density')
plt.title(f'N x rho = varies,abl={abl},ild={ild}')
plt.xlim(0,1)
plt.savefig('NC_diff_distributions.png', dpi=150, bbox_inches='tight')
