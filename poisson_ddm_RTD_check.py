# check poisson hit RTD
# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from scipy.stats import ks_2samp


def fpt_density_skellam(t, mu1, mu2, theta):
    """
    Calculates the exact First Passage Time (FPT) density for a Skellam process.
    The process is dx = dN1 - dN2, with absorbing boundaries at +/- theta.

    Parameters
    ----------
    t : array_like
        Time points at which to evaluate the density (must be >= 0).
    mu1 : float
        Rate of the upward Poisson process (N1).
    mu2 : float
        Rate of the downward Poisson process (N2).
    theta : int
        Symmetric absorbing boundary location (positive integer).

    Returns
    -------
    ft : np.ndarray
        PDF values at times `t`.
    """
    t = np.asarray(t, dtype=float)

    if np.any(t < 0):
        raise ValueError("Time t cannot be negative.")
    if not (isinstance(theta, (int, np.integer)) and theta > 0):
        raise ValueError("Boundary theta must be a positive integer.")

    ft = np.zeros_like(t, dtype=float)
    M = 2 * theta - 1  # size of interior state space
    j = np.arange(1, M + 1, dtype=float)  # 1..M

    # sum over odd k = 1,3,5,...,M
    for k in range(1, M + 1, 2):
        # eigenvalue
        lambda_k = -(mu1 + mu2) + 2.0 * np.sqrt(mu1 * mu2) * np.cos(k * np.pi / (2.0 * theta))

        # right/left eigenvectors (unnormalized)
        v_k = (mu1 / mu2) ** (j / 2.0) * np.sin(k * np.pi * j / (2.0 * theta))
        u_k = (mu2 / mu1) ** (j / 2.0) * np.sin(k * np.pi * j / (2.0 * theta))

        # projection coefficient c_k = (u_k' * 1) / (u_k' * v_k)
        numerator = np.sum(u_k)
        denominator = np.sum(u_k * v_k)

        if np.abs(denominator) < 1e-12:
            c_k = 0.0
        else:
            c_k = numerator / denominator

        # start state x=0 corresponds to the theta-th component (1-based) -> index theta-1 (0-based)
        v_k_theta = v_k[theta - 1]

        # amplitude
        A_k = -lambda_k * c_k * v_k_theta

        # mode contribution
        ft += A_k * np.exp(lambda_k * t)

    # clamp tiny negative values due to floating point
    ft[ft < 0] = 0.0
    return ft


def simulate_fpt(mu1, mu2, theta, rng=None):
    """
    Simulates one trial of the Skellam process dX = dN1 - dN2
    with absorbing boundaries at +/- theta. Returns the first passage time.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = 0
    time = 0.0
    total_rate = mu1 + mu2
    prob_up = mu1 / total_rate

    while abs(x) < theta:
        # exponential waiting time for next jump
        dt = rng.exponential(1.0 / total_rate)
        time += dt

        # decide jump direction
        if rng.random() < prob_up:
            x += 1
        else:
            x -= 1

    return time


def simulate_cont_ddm_fpt(mu, sigma_sq, theta, N_sim=50_000, T=2.0, dt=1e-4, rng=None):
    """
    Simulate first-passage times for the continuous DDM:
        dX = mu dt + sqrt(sigma_sq) dW,
    with absorbing boundaries at +/- theta.

    Parameters
    ----------
    mu : float
        Drift term.
    sigma_sq : float
        Diffusion variance parameter (i.e., dX has variance sigma_sq * dt).
    theta : float
        Symmetric absorbing boundary at +/- theta.
    N_sim : int
        Number of simulated trajectories.
    T : float
        Maximum simulation time (censoring time).
    dt : float
        Time step for Euler-Maruyama integration.
    rng : np.random.Generator or None
        Random number generator.

    Returns
    -------
    hit_times : np.ndarray
        Array of first-passage times (<= T). Trajectories which do not hit by T are excluded.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_steps = int(np.ceil(T / dt))
    x = np.zeros(N_sim, dtype=float)
    t_accum = np.zeros(N_sim, dtype=float)
    alive = np.ones(N_sim, dtype=bool)
    hit_times = np.full(N_sim, np.nan, dtype=float)

    sq_dt = np.sqrt(dt)
    sig = np.sqrt(sigma_sq)

    for _ in range(n_steps):
        if not np.any(alive):
            break
        idx = np.where(alive)[0]
        # Euler-Maruyama step
        x_step = mu * dt + sig * rng.normal(0.0, sq_dt, size=idx.size)
        x[idx] += x_step
        t_accum[idx] += dt

        crossed = (x[idx] >= theta) | (x[idx] <= -theta)
        if np.any(crossed):
            crossed_idx = idx[crossed]
            hit_times[crossed_idx] = t_accum[crossed_idx]
            alive[crossed_idx] = False

    # Drop censored (NaN) trials
    return hit_times[~np.isnan(hit_times)]


def fpt_density_continuous_ddm(t, mu, sigma_sq, theta, M=401):
    """
    Numerical FPT density for continuous DDM with absorbing boundaries at +/- theta
    using a finite-difference spectral decomposition of the Fokker-Planck operator.

    Model: dX = mu dt + sqrt(sigma_sq) dW
    PDE:   dp/dt = (sigma_sq/2) d^2p/dx^2 - mu dp/dx, with p(±theta,t)=0, p(x,0)=δ(x)

    We discretize the interior of (-theta, theta) with M points and compute the
    eigen-decomposition of the generator L, then use p(t) = V exp(Λ t) V^{-1} p0 and
    f(t) = -d/dt ∫ p dx = -1^T L p(t) h.

    Parameters
    ----------
    t : array_like
        Times at which to evaluate the density (t >= 0)
    mu : float
        Drift
    sigma_sq : float
        Diffusion variance parameter
    theta : float
        Symmetric boundary location
    M : int
        Number of interior grid points (use odd to place x=0 on a grid node)

    Returns
    -------
    ft : np.ndarray
        Density values at times t
    """
    t = np.asarray(t, dtype=float)
    if np.any(t < 0):
        raise ValueError("Time t cannot be negative.")
    if theta <= 0:
        raise ValueError("Boundary theta must be positive.")

    # Ensure M is odd so that x=0 lies on grid
    if M % 2 == 0:
        M += 1

    # Grid spacing and interior points
    h = (2.0 * theta) / (M + 1)
    # Build generator L (M x M) for interior nodes
    # dp/dt = (sigma_sq/2) * (p_{i+1} - 2 p_i + p_{i-1})/h^2 - mu * (p_{i+1} - p_{i-1})/(2h)
    alpha = (sigma_sq / 2.0) / (h * h)
    beta = mu / (2.0 * h)

    # Tridiagonal coefficients
    main = -2.0 * alpha * np.ones(M)
    upper = (alpha - beta) * np.ones(M - 1)
    lower = (alpha + beta) * np.ones(M - 1)

    # Assemble dense matrix (small M so OK)
    L = np.zeros((M, M), dtype=float)
    np.fill_diagonal(L, main)
    np.fill_diagonal(L[1:], lower)
    np.fill_diagonal(L[:, 1:], upper)

    # Initial condition approximating delta at x=0
    p0 = np.zeros(M, dtype=float)
    center_index = M // 2  # since M is odd
    p0[center_index] = 1.0 / h  # so that sum(h * p0) = 1

    # Eigen-decomposition L = V Λ V^{-1}
    lam, V = np.linalg.eig(L)
    # Solve V a = p0 without explicitly inverting V
    a = np.linalg.solve(V, p0)
    # b^T = 1^T V (row vector)
    b = V.sum(axis=0)

    # Compute f(t) = -h * sum_k (b_k * a_k * lam_k * exp(lam_k * t))
    # Handle complex arithmetic; result should be real
    lam_col = lam[:, None]
    exp_term = np.exp(lam_col * t[None, :])
    coeff = (-h) * (b * a * lam)
    ft = np.real(coeff[:, None] * exp_term).sum(axis=0)

    ft[ft < 0] = 0.0
    return ft


# def main():
# %%
#  1. Define Parameters and Plot the Analytical PDF
def two_theta_sim(theta_true, theta_cont):
    A = 20; I = 4
    # r0 = (1000/0.3)
    # r0 = 13.3
    # rate_lambda = 1.3
    # rR = r0 * (10 ** (rate_lambda * A / 20)) * (10 ** (rate_lambda * I / 40))
    # rL = r0 * (10 ** (rate_lambda * A / 20)) * (10 ** (-rate_lambda * I / 40))
    r0 = 13.3
    lam = 1.3
    l = 0.9
    abl = A; ild = I
    r_db = (2*abl + ild)/2
    l_db = (2*abl - ild)/2
    pr = (10 ** (r_db/20))
    pl = (10 ** (l_db/20))
    den = (pr ** (lam * l) ) + ( pl ** (lam * l) )
    rr = (pr ** lam) / den
    rl = (pl ** lam) / den
    rR = r0 * rr
    rL = r0 * rl

    mu1_true = rR
    mu2_true = rL
    print(f"mu1_true={mu1_true}, mu2_true={mu2_true}")
    # theta_true = 3   # Boundary (integer)

    # Time vector for plotting
    t = np.linspace(0.0, 2.0, 500)

    # Calculate the PDF
    # pdf_values = fpt_density_skellam(t, mu1_true, mu2_true, theta_true)

    #  2. Simulate Data from the Process
    num_trials = 100_000
    rng = np.random.default_rng()
    rt_data = np.empty(num_trials, dtype=float)
    for i in range(num_trials):
        rt_data[i] = simulate_fpt(mu1_true, mu2_true, theta_true, rng=rng)

    #  3. We will plot after continuous simulation to overlay all in one figure

    #  4. Continuous DDM parameters from Poisson params
    mu_cont = mu1_true - mu2_true
    sigma_sq_cont = mu1_true + mu2_true
    # theta_cont = float(theta_true)
    # theta_cont = 3

    #  5. Simulate continuous DDM FPT data
    cont_rt_data = simulate_cont_ddm_fpt(mu_cont, sigma_sq_cont, theta_cont, N_sim=num_trials, T=2.0, rng=rng)
    return rt_data, cont_rt_data

# %%
sk_2, dd_2 = two_theta_sim(2, 2)
sk_3, dd_3 = two_theta_sim(5, 5)

# %%
bins = np.arange(0.0, 2.0 + 0.01, 0.01)  # 0:.01:2

plt.hist(sk_2, bins=bins, color='b', ls='--', label='sk 2', histtype='step', density=True)
plt.hist(sk_3, bins=bins, color='r', ls='--', label='sk 3', histtype='step', density=True)

plt.hist(dd_2, bins=bins, color='b', label='dd 2', histtype='step', density=True)
plt.hist(dd_3, bins=bins, color='r', label='dd 3', histtype='step', density=True)
plt.legend()

# %%
#  6. Single plot: Poisson sim RTD + Skellam theory + Continuous DDM sim RTD
plt.figure(figsize=(7, 4), facecolor="white")
bins = np.arange(0.0, 2.0 + 0.01, 0.01)  # 0:.01:2
# Use the first color from the current cycle for the theory line
cycle_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])
plt.hist(rt_data, bins=bins, density=True, alpha=0.45, label="Poisson sim", histtype="step")
plt.hist(cont_rt_data, bins=bins, density=True, alpha=0.45, label="Continuous DDM sim", histtype="step")
# plt.plot(t, pdf_values, linewidth=1.5, color=cycle_colors[0], label="Poisson theory")
plt.xlabel("Time")
plt.ylabel("Density")
plt.title(f"Poisson DDM: mu1={mu1_true :.2f}, mu2={mu2_true :.2f}, theta sk={theta_true :.2f}, theta dd={theta_cont :.2f}")
plt.legend()
plt.xlim(0,1.5)
plt.tight_layout()
plt.show()


# %%
# calculate KS statistic btn sk_2 and dd_2
ks_result = ks_2samp(sk_2, dd_2)
ks_stat, p_value = ks_result.statistic, ks_result.pvalue
print(f'?? = {ks_stat}')
baseline_payload = {
    "sk_2": sk_2,
    "dd_2": dd_2,
    "ks_statistic": ks_stat,
    "p_value": p_value,
}

output_path = Path("theta_2_no_corr_baseline_data.pkl")
with output_path.open("wb") as f:
    pickle.dump(baseline_payload, f)

print(
    f"KS statistic (sk_2 vs dd_2): {ks_stat:.4f}, p-value: {p_value:.4g}. "
    f"Saved baseline data to {output_path}."
)
# %%
ks2, _ = ks_2samp(sk_2, dd_2)
print(f'KS statistic (sk_2 vs dd_2): {ks2:.4f}')

ks3, _ = ks_2samp(sk_3, dd_3)
print(f'KS statistic (sk_3 vs dd_3): {ks3:.4f}')
# %%


def plot_ecdf(data, label, **kwargs):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, label=label, **kwargs)

# Plot CDF for theta=2
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plot_ecdf(sk_2, 'Skellam θ=2', linestyle='--', color='k', alpha=0.3, lw=2)
plot_ecdf(dd_2, 'DDM θ=2', color='blue')
plt.xlabel('Time')
plt.ylabel('CDF')
plt.title('Empirical CDF: θ=2')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot CDF for theta=3
plt.subplot(1, 2, 2)
plot_ecdf(sk_3, 'Skellam θ=3', linestyle='--', color='k', alpha=0.3, lw=2)
plot_ecdf(dd_3, 'DDM θ=3', color='red')
plt.xlabel('Time')
plt.ylabel('CDF')
plt.title('Empirical CDF: θ=3')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# %%


def compute_ecdf(data):
    """Return sorted x values and corresponding CDF y values"""
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

# Compute ECDFs for theta=2
x_sk_2, y_sk_2 = compute_ecdf(sk_2)
x_dd_2, y_dd_2 = compute_ecdf(dd_2)

# Compute ECDFs for theta=3
x_sk_3, y_sk_3 = compute_ecdf(sk_3)
x_dd_3, y_dd_3 = compute_ecdf(dd_3)

# Plot absolute differences
plt.figure(figsize=(8, 6))

# Plot difference for theta=2
plt.subplot(1, 2, 1)
# Interpolate to same x-grid for comparison
x_grid_2 = np.linspace(0, max(x_sk_2[-1], x_dd_2[-1]), 1000)
y_sk_2_interp = np.interp(x_grid_2, x_sk_2, y_sk_2)
y_dd_2_interp = np.interp(x_grid_2, x_dd_2, y_dd_2)
diff_2 = np.abs(y_sk_2_interp - y_dd_2_interp)

plt.plot(x_grid_2, diff_2, color='blue', linewidth=2)
plt.xlabel('Time')
plt.ylabel('|CDF_Skellam - CDF_DDM|')
plt.title('Absolute CDF Difference: θ=2')
plt.grid(True, alpha=0.3)
max_diff_2 = np.max(diff_2)
plt.axhline(y=max_diff_2, color='red', linestyle='--', alpha=0.5, label=f'Max diff: {max_diff_2:.4f}')
plt.legend()

# Plot difference for theta=3
plt.subplot(1, 2, 2)
x_grid_3 = np.linspace(0, max(x_sk_3[-1], x_dd_3[-1]), 1000)
y_sk_3_interp = np.interp(x_grid_3, x_sk_3, y_sk_3)
y_dd_3_interp = np.interp(x_grid_3, x_dd_3, y_dd_3)
diff_3 = np.abs(y_sk_3_interp - y_dd_3_interp)

plt.plot(x_grid_3, diff_3, color='red', linewidth=2)
plt.xlabel('Time')
plt.ylabel('|CDF_Skellam - CDF_DDM|')
plt.title('Absolute CDF Difference: θ=3')
plt.grid(True, alpha=0.3)
max_diff_3 = np.max(diff_3)
plt.axhline(y=max_diff_3, color='blue', linestyle='--', alpha=0.5, label=f'Max diff: {max_diff_3:.4f}')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Maximum CDF differences:")
print(f"  θ=2: {max_diff_2:.4f}")
print(f"  θ=3: {max_diff_3:.4f}") 