# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import multiprocessing
from mgf_helper_utils import ddm_fc_dt
from scipy import stats
import time
from joblib import Parallel,delayed

# %%
N = 100
# ddm rates parameters
lam = 1.3
l = 0.9
Nr0 = 13.3 * 1
abl = 20
ild = 1
dt = 1e-4  # Time step for continuous DDM simulation
dB = 1e-2
# Calculate rates
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

def ddm_simulator(N,rr,rl,theta):
    mu = N *(rr - rl)
    sigma = np.sqrt(N*(rr+rl))

    dv = 0
    t = 0
    while abs(dv) < theta:
        dv += mu*dt + sigma*np.random.normal(0,dB)
        t += dt

        if dv >= theta:
            return t,1
        elif dv <= -theta:
            return t,-1
    
# %%
N_sim = int(50e3)
theta_range = [2,4,6, 7]
ddm_theta_data = {}
for theta in theta_range:
    print(f'theta = {theta} data gen')
    ddm_data = Parallel(n_jobs=multiprocessing.cpu_count() - 2)\
        (delayed(ddm_simulator)(N,r_right,r_left,theta) for _ in range(N_sim))
    ddm_theta_data[theta] = np.array(ddm_data)
# %%
accuracy_theta_wise = {}
mean_rt_theta_wise = {}

theory_accuracy_theta_wise = {}
theory_mean_rt_theta_wise = {}
# ddm_fc_dt(lam, l, Nr0, N, abl, ild, theta, dt):
for theta in theta_range:
    data_per_theta = ddm_theta_data[theta]
    
    accuracy = np.mean(data_per_theta[:,1] == 1)
    accuracy_theta_wise[theta] = accuracy
    
    mean_rt = np.mean(data_per_theta[:,0])
    mean_rt_theta_wise[theta] = mean_rt

    theory_acc, theory_rt  = ddm_fc_dt(lam, l, Nr0, N, abl, ild, theta, dt)
    theory_accuracy_theta_wise[theta] = theory_acc
    theory_mean_rt_theta_wise[theta] = theory_rt
    # mu = N*(r_right - r_left)
    # sigma_sq = N*(r_right + r_left)
    # theory_mean_rt_theta_wise[theta] = (theta/mu) * (np.tanh(mu*theta/sigma_sq))
    

# %%
plt.scatter(theta_range,accuracy_theta_wise.values(), label='data', color='r')
plt.plot(theta_range,theory_accuracy_theta_wise.values(), label='theory')
plt.xlabel('Theta')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Theta')
plt.legend()
plt.show()

# %%
plt.scatter(theta_range,mean_rt_theta_wise.values(), label='data', color='r')
plt.plot(theta_range,theory_mean_rt_theta_wise.values(), label='theory')
plt.xlabel('Theta')
plt.ylabel('Mean RT')
plt.title('Mean RT vs Theta')
plt.legend()
plt.show()

# %%
