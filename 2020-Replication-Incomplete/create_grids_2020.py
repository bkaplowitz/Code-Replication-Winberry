import numpy as np
from numba import njit, jit
from scipy.stats import norm

# Idiosyncratic shocks-- quadrature to integrate


# @njit
def create_grids_calc(
    approx_N_cap,
    approx_N_prod,
    prod_min,
    prod_max,
    cap_min,
    cap_max,
    n_shocks,
    rho,
    sigma_prod,
    n_state_fine,
    n_prod_fine,
    n_cap_fine,
    n_prod_quad,
    n_cap_quad,
):
    """
    Approximates value function/conditional expectation for next period capital and productivity shocks given how much to approximate capital and productivity.
    """
    # TODO: can this be innovated to be chebyshev quadrature method to take advantage of chebyshev techniques?
    shocks_grid, shocks_weight = quad_idiosyn_shocks_compute(n_shocks)
    # compute states num again for robustness
    n_states = np.int(approx_N_cap * approx_N_prod)
    # Compute zeros of chebyshev function (chebyshev nodes to mitigate runge's phenomena)
    prod_cheb_zeros, cap_cheb_zeros = compute_state_cheby_zeros(
        approx_N_prod, approx_N_cap
    )

    # compute grid of chebyshev nodes
    m_state_grid, m_state_grid_zeros = cheby_nodes_compute(
        prod_cheb_zeros, cap_cheb_zeros, n_states, prod_min, prod_max, cap_min, cap_max
    )

    # Construct grid of future productivity shocks (used for computing expectations)
    prod_prime_zeros = compute_future_prod_shocks(
        rho,
        m_state_grid,
        n_shocks,
        sigma_prod,
        shocks_grid,
        n_state_fine,
        prod_min,
        prod_max,
    )

    # Grids for approximating histogram alternative (Young, 2010 method)
    # normal grids in this case, that will be exponentially modified since hist doesn't automatically choose grid points optimally.
    m_state_grid_fine, prod_grid_fine, prod_step_size = compute_hist_grid(
        prod_min, prod_max, n_prod_fine, cap_min, cap_max, n_cap_fine
    )
    # scales down to unit interval for polynomial computations
    m_state_grid_zeros_fine = np.hstack(
        scaledown(m_state_grid_fine[0], prod_min, prod_max),
        scaledown(m_state_grid_fine[1], prod_min, prod_max),
    )
    # grid of future productivity shocks
    prod_prime_zeros_fine = compute_future_prod_shocks(
        rho,
        m_state_grid_fine,
        n_shocks,
        sigma_prod,
        shocks_grid,
        n_state_fine,
        prod_min,
        prod_max,
    )
    # TODO: Tauchen transition matrix (not Rouenwhorst?)
    # trick to vectorize computation of each probability matrix.
    prod_trans = tauchen_prod_compute(prod_grid_fine, prod_step_size, rho, sigma_prod)
    # compute nodes and weights for integrating exponential family of distribution
    prod_quad_zeros, weights_prod = np.polynomial.legendre.leggauss(n_prod_quad)
    cap_quad_zeros, weights_cap = np.polynomial.legendre.leggauss(n_cap_quad)
    n_quad = n_prod_quad * n_cap_quad
    # scale up
    prod_quad = scaleup(prod_quad_zeros, prod_min, prod_max)
    cap_quad = scaleup(cap_quad_zeros, cap_min, cap_max)
    temp_quad = np.meshgrid(prod_quad, cap_quad)
    state_quad_grid = np.hstack(
        (temp_quad[0].reshape((n_quad, 1)), temp_quad[1].reshape((n_quad, 1)))
    )
    state_quad_grid_zeros = np.hstack(
        (
            scaledown(state_quad_grid[:, 0], prod_min, prod_max),
            scaledown(state_quad_grid[:, 1], cap_min, cap_max),
        )
    )

    # make an outer/tensor product of all weights
    # rescaling so weights sum to full measure in new scale
    weights_prod = (prod_max - prod_min) / 2 * weights_prod
    weights_cap = (cap_max - cap_min) / 2 * weights_cap
    weights_quad = np.outer(weights_prod, weights_cap).reshape((n_quad, 1))

    # make a grid over future productivity shocks for quadrature which helps with optimal policies
    prod_prime_quad = rho * np.tile(
        state_quad_grid[:, 0].T, (n_shocks, 1)
    ) + sigma_prod * np.tile(shocks_grid, (1, n_shocks))
    prod_prime_quad_zeros = scaledown(prod_prime_quad, prod_min, prod_max)
    # makes tensor product grid analogous to before
    # @njit
    # returns first arguments for main code, second set for create_polynomials.py.
    return [
        shocks_grid,
        shocks_weight,
        m_state_grid,
        state_quad_grid,
        weights_quad,
        prod_prime_quad,
        m_state_grid_zeros,
        state_quad_grid_zeros,
        prod_cheb_zeros,
        cap_cheb_zeros,
        prod_prime_zeros,
        m_state_grid_zeros_fine,
        prod_prime_zeros_fine,
        state_quad_grid_zeros,
        prod_prime_quad_zeros,
    ]


def scaleup(x, minval, maxval):
    """
    Converts [-1,1] to [minval,maxval] range with points always lying inside new range. Returns scaled up version of values.
    """
    return np.min(
        (
            np.max(
                (
                    0.5 * (x + 1) * (maxval - minval) + minval,
                    minval * np.ones(np.len(x)),
                )
            ),
            maxval * np.ones(np.len(x)),
        )
    )


# @njit
def scaledown(x, minval, maxval):
    """
    Converts [minval,maxval] to [-1,1] with points always lying inside [-1,1]. Returns scaled down values.
    """
    return np.min(
        (
            np.max(
                (
                    2 * (x - minval) / (maxval - minval) - 1,
                    -1 * np.ones(np.len(x)),
                )
            ),
            np.ones(np.len(x)),
        )
    )


def tauchen_prod_compute(prod_grid_fine, prod_step_size, rho, sigma_prod):
    prod_right_step_grid = (
        np.tile(prod_grid_fine.T, len(prod_grid_fine))
        + 0.5 * prod_step_size * np.ones(len(prod_grid_fine))
        - rho * np.tile(prod_grid_fine, len(prod_grid_fine))
    )
    prod_left_step_grid = (
        np.tile(prod_grid_fine.T, len(prod_grid_fine))
        - 0.5 * prod_step_size * np.ones(len(prod_grid_fine))
        - rho * np.tile(prod_grid_fine, len(prod_grid_fine))
    )
    return norm.cdf(prod_right_step_grid, 0, sigma_prod) - norm.cdf(
        prod_left_step_grid, 0, sigma_prod
    )


def cheby_nodes_compute(
    prod_cheb_zeros, cap_cheb_zeros, n_states, prod_min, prod_max, cap_min, cap_max
):
    # compute grid of chebyshev nodes
    temp_grid = np.meshgrid(prod_cheb_zeros, cap_cheb_zeros)
    print(temp_grid)
    # reshape grid to get a tensor
    print(temp_grid[0])
    print(temp_grid[0].reshape((n_states, 1)))
    m_state_grid_zeros = np.hstack(
        (temp_grid[0].reshape((n_states, 1)), temp_grid[1].reshape((n_states, 1)))
    )
    m_state_grid = np.hstack(
        scaleup(m_state_grid_zeros[0, :], prod_min, prod_max),
        scaleup(m_state_grid_zeros[1, :], cap_min, cap_max),
    )
    return m_state_grid, m_state_grid_zeros


def compute_state_cheby_zeros(approx_N_prod, approx_N_cap):
    """
    Computes chebyshev zero collocation points for each state variable.
    """
    prod_cheb_zeros = -np.cos(
        ((2 * np.arange(1, approx_N_prod + 1) - 1) * np.pi) / (2 * approx_N_prod)
    )
    cap_cheb_zeros = -np.cos(
        ((2 * np.arange(1, approx_N_cap + 1) - 1) * np.pi) / (2 * approx_N_cap)
    )
    return prod_cheb_zeros, cap_cheb_zeros


def compute_future_prod_shocks(
    rho,
    m_state_grid_fine,
    n_shocks,
    sigma_prod,
    shocks_grid,
    n_state_fine,
    prod_min,
    prod_max,
):
    """
    Computes a grid of next period productivity shocks and their chebyshev zeros for expectation estimation.
    """
    prod_prime_zeros_fine = rho * np.matlib.tile(
        m_state_grid_fine[:, 0].T, (n_shocks, 1)
    ) + sigma_prod * np.tile(shocks_grid, (1, n_state_fine))
    prod_prime_zeros_fine = scaledown(prod_prime_zeros_fine, prod_min, prod_max)
    return prod_prime_zeros_fine


def quad_idiosyn_shocks_compute(n_shocks):
    """ 
    Computes grid to integrate idiosyncratic shocks faced by firms by Gauss-Hermite quadrature.
    """
    shocks_grid, shocks_weight = np.polynomial.hermite.hermgauss(n_shocks)
    # ensures this integrates to 1
    shocks_grid = np.srt(2) * shocks_grid
    shocks_weight = np.pi ** (-0.5) * shocks_weight
    return shocks_grid, shocks_weight


def compute_hist_grid(prod_min, prod_max, n_prod_fine, cap_min, cap_max, n_cap_fine):
    """
    Computes the grid for histograms used in Young, 2010 technique.
    """
    prod_grid_fine, prod_step_size = np.linspace(
        prod_min, prod_max, n_prod_fine, step=True
    )
    cap_grid_fine = np.linspace(cap_min, cap_max, n_cap_fine)
    temp_grid_fine = np.meshgrid(prod_grid_fine, cap_grid_fine)
    m_state_grid_fine = np.hstack(
        temp_grid_fine[0].reshape((n_prod_fine, 1)),
        temp_grid_fine[1].reshape((n_cap_fine, 1)),
    )
    return m_state_grid_fine, prod_grid_fine, prod_step_size

