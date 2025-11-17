# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# %%
N = 100
theta = 2
rho = 0.01

# ddm rates
lam = 1.3
l = 0.9
Nr0 = 13.3 * 1

r0 = Nr0/N
abl = 20
ild = 1
r_db = (2*abl + ild)/2
l_db = (2*abl - ild)/2
pr = (10 ** (r_db/20))
pl = (10 ** (l_db/20))

den = (pr ** (lam * l) ) + ( pl ** (lam * l) )  
rr = (pr ** lam) / den
rl = (pl ** lam) / den
r_right = r0 * rr
r_left = r0 * rl


# %%
# Define the function from the equation
# (((1 + ρ(e^t - 1))^N - 1)) λ_p + (((1 + ρ(e^(-t) - 1))^N - 1)) λ_n = 0
def f(t):
    lambda_p = r_right
    lambda_n = r_left
    
    term1 = (((1 + rho * (np.exp(t) - 1)) ** N) - 1) * lambda_p
    term2 = (((1 + rho * (np.exp(-t) - 1)) ** N) - 1) * lambda_n
    
    return term1 + term2

# %%
# find h0 such that h0 < 0 and h0 is root of f(t)

# Search for the root in the negative domain
def find_h0():
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

# %%
# Run the root finding function
h0 = find_h0()

# Verify the result by checking if f(h0) is close to zero
if h0 is not None:
    print(f"Verification: f(h0) = {f(h0)}")
    
    # Plot the function near the root for visualization
    t_values = np.linspace(h0*1.5, min(h0*0.5, -0.001), 100)
    f_values = [f(t) for t in t_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, f_values)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.axvline(x=h0, color='g', linestyle='--', label=f'h0 = {h0:.6f}')
    plt.grid(True)
    plt.title('Function f(t) near the root h0')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend()
    plt.savefig('h0_root_plot.png')
    plt.show()

# %%
dt = 1e-4
E_W = dt * N * (r_right - r_left)

# %%
print(f'h0 = {h0}, E_W = {E_W}')

# %%
mu_ddm = N * (r_right - r_left)
sigma_sq_ddm = N * (r_right + r_left)

h0_ddm = - 2 * mu_ddm/sigma_sq_ddm
E_W_ddm = mu_ddm * dt
# %%
# plot FC and DT as function of theta

# Define a range of theta values
theta_range = np.arange(1, 11, 1)  # From 1 to 10 in steps of 1

# Use the existing dt parameter defined earlier
# dt = 1e-4  (already defined above)

# Calculate FC and DT for each theta - Poisson model
fc_values = []
dt_values = []

# Calculate FC and DT for DDM
fc_ddm_values = []
dt_ddm_values = []

for t in theta_range:
    # Calculate FC (Fraction Correct) for Poisson
    fc = 1 / (1 + np.exp(h0 * t))
    fc_values.append(fc)
    
    # Calculate DT (Decision Time) for Poisson
    decision_time = (t * dt / E_W) * np.tanh(-h0 * t / 2)
    dt_values.append(decision_time)
    
    # Calculate FC (Fraction Correct) for DDM
    fc_ddm = 1 / (1 + np.exp(h0_ddm * t))
    fc_ddm_values.append(fc_ddm)
    
    # Calculate DT (Decision Time) for DDM
    dt_ddm_value = (t * dt / E_W_ddm) * np.tanh(-h0_ddm * t / 2)
    dt_ddm_values.append(dt_ddm_value)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot FC as a function of theta
ax1.plot(theta_range, fc_values, 'o-', color='blue', label='Poisson')
ax1.plot(theta_range, fc_ddm_values, 's--', color='darkblue', label='DDM')
ax1.set_xlabel('Theta (θ)')
ax1.set_ylabel('accuracy (FC)')
ax1.set_title('accuracy vs Theta')
ax1.grid(True)
ax1.legend()

# Plot DT as a function of theta
ax2.plot(theta_range, dt_values, 'o-', color='red', label='Poisson')
ax2.plot(theta_range, dt_ddm_values, 's--', color='darkred', label='DDM')
ax2.set_xlabel('Theta (θ)')
ax2.set_ylabel('Mean RT (DT)')
ax2.set_title('mean RT vs Theta')
ax2.grid(True)
ax2.legend()

# Print comparison values
print("\nPoisson Model: h0 = {:.6f}, E_W = {:.6f}".format(h0, E_W))
print("DDM Model: h0_ddm = {:.6f}, E_W_ddm = {:.6f}".format(h0_ddm, E_W_ddm))

plt.tight_layout()
plt.savefig('FC_DT_vs_theta_comparison.png')
plt.show()

# %%
