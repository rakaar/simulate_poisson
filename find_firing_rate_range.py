# %%
import numpy as np

lam = 2.13
l = 0.9
r0 = 7.7


def rate_calc(abl, ild):
    r_db = (2*abl + ild)/2
    l_db = (2*abl - ild)/2
    
    p0 = 20e-6
    pr = p0 * (10 ** (r_db/20))
    pl = p0 * (10 ** (l_db/20))
    
    den = (pr ** (lam * l) ) + ( pl ** (lam * l) )
    rr = (pr ** lam) / den
    rl = (pl ** lam) / den
    
    r0 = 7.7
    n_rr = r0 * rr
    n_rl = r0 * rl
    return n_rr, n_rl

print(f'ABL = 20, ILD = 1')
print(rate_calc(20, 1))
print(f'ABL = 60, ILD = 16')
print(rate_calc(60, 16))
