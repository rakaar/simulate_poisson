# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import multiprocessing
import time
from scipy.stats import entropy
from scipy.special import rel_entr
from joblib import Parallel, delayed



# %%
# poisson spikes times
r1 = 10
r2 = 8

T = 100

# poisson spike time till T with rate r1, and r2
n1 = np.random.poisson(r1 * T)
n2 = np.random.poisson(r2 * T)

spk_t1 = np.random.uniform(0, 1, size=int(n1)) * T
spk_t2 = np.random.uniform(0, 1, size=int(n2)) * T


# %%
t1_diff = np.diff(np.sort(spk_t1))
t2_diff = np.diff(np.sort(spk_t2))

# %%
plt.hist(t1_diff, bins=50)
plt.show()

plt.hist(t2_diff, bins=50)
plt.show()
# %%

# calculate variance of spk_
