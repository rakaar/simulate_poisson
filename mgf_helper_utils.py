# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
# %%

def ddm_rates(lam, l, Nr0, N, abl, ild):
    r0 = Nr0 / N
    r_db = (2*abl + ild)/2
    l_db = (2*abl - ild)/2

    pr = (10 ** (r_db/20))
    pl = (10 ** (l_db/20))

    den = (pr ** (lam * l) ) + ( pl ** (lam * l) )
    rr = (pr ** lam) / den
    rl = (pl ** lam) / den
    r_right = r0 * rr
    r_left = r0 * rl

    return r_right, r_left



    
def FC(h0, theta):
    return 1 / (1 + np.exp(h0 * theta))

def DT(E_W, h0, theta, dt):
    return  (theta * dt / E_W) * np.tanh(-h0 * theta / 2)

def poisson_fc_dt(N, rho, theta, lam, l, Nr0, abl, ild, dt):
    r_right, r_left = ddm_rates(lam, l, Nr0, N, abl, ild)
    
    E_W = dt * N * (r_right - r_left)
    h0 = find_h0(r_right, r_left, N, rho)

    fc = FC(h0, theta)
    dt = DT(E_W, h0, theta, dt)
    
    return fc, dt


def ddm_fc_dt(lam, l, Nr0, N, abl, ild, theta, dt):
    r_right, r_left = ddm_rates(lam, l, Nr0, N, abl, ild)
    mu = N * (r_right - r_left)
    sigma_sq = N * (r_right + r_left)
    
    E_W = mu * dt
    h0 = -2 * mu / sigma_sq
    
    fc = FC(h0, theta)
    dt = DT(E_W, h0, theta, dt)

    return fc, dt
    
    
    


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

### ADDITIVE ###
# def find_h0(r_right, r_left, N, rho):
#     lambda_p = r_right
#     lambda_n = r_left

#     def f(t):
#         # Eq. 85: additive (SIP) correlations + spike integration
#         term1 = lambda_p * (rho * (np.exp(N*t)   - 1) + (1 - rho) * N * (np.exp(t)   - 1))
#         term2 = lambda_n * (rho * (np.exp(-N*t)  - 1) + (1 - rho) * N * (np.exp(-t)  - 1))
#         return term1 + term2

#     a, b = -10.0, -0.001  # search interval for the negative root
#     # (same bracketing/Brent code as you already have)
#     h0 = optimize.brentq(f, a, b)
#     return h0

    