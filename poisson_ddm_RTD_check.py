# check poisson hit RTD
# %%
import numpy as np
import matplotlib.pyplot as plt


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


def simulate_cont_ddm_fpt(mu, sigma_sq, theta, N_sim=50_000, T=2.0, dt=2e-3, rng=None):
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


def main():
    #  1. Define Parameters and Plot the Analytical PDF
    A = 60; I = 4
    r0 = (1000/0.3)
    rate_lambda = 0.1
    rR = r0 * (10 ** (rate_lambda * A / 20)) * (10 ** (rate_lambda * I / 40))
    rL = r0 * (10 ** (rate_lambda * A / 20)) * (10 ** (-rate_lambda * I / 40))
    mu1_true = rR
    mu2_true = rL
    theta_true = 30      # Boundary (integer)

    # Time vector for plotting
    t = np.linspace(0.0, 2.0, 500)

    # Calculate the PDF
    pdf_values = fpt_density_skellam(t, mu1_true, mu2_true, theta_true)

    #  2. Simulate Data from the Process
    num_trials = 50_000
    rng = np.random.default_rng()
    rt_data = np.empty(num_trials, dtype=float)
    for i in range(num_trials):
        rt_data[i] = simulate_fpt(mu1_true, mu2_true, theta_true, rng=rng)

    #  3. Compare: histogram vs true PDF (Poisson/Discrete DDM)
    plt.figure(figsize=(7, 4), facecolor="white")
    bins = np.arange(0.0, 2.0 + 0.01, 0.01)  # 0:.01:2
    plt.hist(rt_data, bins=bins, density=True, alpha=0.5, edgecolor="none", label="Poisson sim")
    # Use the first color from the current cycle for the PDF line
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])
    plt.plot(t, pdf_values, linewidth=1.5, color=cycle_colors[0], label="Poisson theory")
    plt.xlabel("Time")
    plt.ylabel("Density")
    plt.title(f"Poisson DDM: mu1={mu1_true}, mu2={mu2_true}, theta={theta_true}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #  4. Continuous DDM parameters from Poisson params
    mu_cont = mu1_true - mu2_true
    sigma_sq_cont = mu1_true + mu2_true
    theta_cont = float(theta_true)

    #  5. Simulate continuous DDM FPT data
    cont_rt_data = simulate_cont_ddm_fpt(mu_cont, sigma_sq_cont, theta_cont, N_sim=num_trials, T=2.0, dt=2e-3, rng=rng)

    #  6. Theoretical FPT density for continuous DDM (numerical spectral method)
    cont_pdf_values = fpt_density_continuous_ddm(t, mu_cont, sigma_sq_cont, theta_cont, M=401)

    #  7. Plot RTD of continuous DDM: histogram + theoretical PDF
    plt.figure(figsize=(7, 4), facecolor="white")
    plt.hist(cont_rt_data, bins=bins, density=True, alpha=0.5, edgecolor="none", label="Cont. DDM sim")
    plt.plot(t, cont_pdf_values, linewidth=1.5, color=cycle_colors[1] if len(cycle_colors) > 1 else 'C1', label="Cont. DDM theory")
    plt.xlabel("Time")
    plt.ylabel("Density")
    plt.title(f"Continuous DDM: mu={mu_cont}, sigma^2={sigma_sq_cont}, theta={theta_cont}")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
