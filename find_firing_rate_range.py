# %%
import numpy as np

lam = 1.3
l = 0.9
Nr0 = 1/(75 * 1e-3)

print(f'Nr0 = {Nr0 : .2f}')

def rate_calc(abl, ild):
    r_db = (2*abl + ild)/2
    l_db = (2*abl - ild)/2
    
    p0 = 20e-6
    pr = p0 * (10 ** (r_db/20))
    pl = p0 * (10 ** (l_db/20))
    
    den = (pr ** (lam * l) ) + ( pl ** (lam * l) )

    rr = (pr ** lam) / den
    rl = (pl ** lam) / den
    
    print(f'rr = {rr : .2f}')
    print(f'rl = {rl : .2f}')
    
    n_rr = Nr0 * rr
    n_rl = Nr0 * rl
    return n_rr, n_rl

print(f'ABL = 20, ILD = 1')
print(rate_calc(20, 1))
print(f'ABL = 60, ILD = 16')
print(rate_calc(60, 16))
