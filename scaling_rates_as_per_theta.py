# scaling R, L rates seperately as per theta scaling
# theta - > k *theta
# %%
def scale_rates(k, R, L):
    # return scaled R, L rates as per the formula:
    # m = ((k^2 + k)R + (k^2 - k)L)/(2R) = (k/2)[(k + 1) + (k - 1)(L/R)]
    # n = ((k^2 - k)R + (k^2 + k)L)/(2L) = (k/2)[(k + 1) + (k - 1)(R/L)]
    
    m = (k/2) * ((k + 1) + (k - 1) * (L/R))
    n = (k/2) * ((k + 1) + (k - 1) * (R/L))
    
    return m*R, n*L


# %%
# check 
L = 3; R = 5; theta = 29
k = 7

def mu_sigma2(R,L):
    return R - L, R + L

def gamma_fn(mu, sigma2, theta):
    return theta * (mu/sigma2)

def omega_fn(mu, sigma2, theta):
    return sigma2 / theta**2



mu_1, sigma2_1 = mu_sigma2(R,L)
gamma_1 = gamma_fn(mu_1, sigma2_1, theta)
omega_1 = omega_fn(mu_1, sigma2_1, theta)
print(f'G={gamma_1}, O={omega_1}')

R2, L2 = scale_rates(k, R, L)
mu_2, sigma2_2 = mu_sigma2(R2, L2)
gamma_2 = gamma_fn(mu_2, sigma2_2, theta*k)
omega_2 = omega_fn(mu_2, sigma2_2, theta*k)
print(f'G={gamma_2}, O={omega_2}')


# Test multiple combinations of L and R
def test_combinations(theta, k):
    # Generate 10 different combinations of L and R values
    L_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    R_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    print("Testing 10 combinations of L and R values:")
    print("L\tR\tGamma1\tGamma2\tOmega1\tOmega2\tMatch_Gamma\tMatch_Omega")
    
    for i in range(10):
        L = L_values[i]
        R = R_values[i]
        
        # Calculate original values
        mu_1, sigma2_1 = mu_sigma2(R, L)
        gamma_1 = gamma_fn(mu_1, sigma2_1, theta)
        omega_1 = omega_fn(mu_1, sigma2_1, theta)
        
        # Scale rates
        R2, L2 = scale_rates(k, R, L)
        
        # Calculate scaled values
        mu_2, sigma2_2 = mu_sigma2(R2, L2)
        gamma_2 = gamma_fn(mu_2, sigma2_2, theta*k)
        omega_2 = omega_fn(mu_2, sigma2_2, theta*k)
        
        # Check if they match (within a small tolerance)
        match_gamma = abs(gamma_1 - gamma_2) < 1e-10
        match_omega = abs(omega_1 - omega_2) < 1e-10
        
        print(f"{L}\t{R}\t{gamma_1:.6f}\t{gamma_2:.6f}\t{omega_1:.6f}\t{omega_2:.6f}\t{match_gamma}\t\t{match_omega}")

test_combinations(theta, k)


# %%
