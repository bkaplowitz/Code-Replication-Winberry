import numpy as np
from numba import njit, jit

#@njit
def create_grids(
    n_assets,
    n_assets_fine,
    n_epsilon,
    assets_min,
    assets_max,
    beta_max,
    beta_min,
    n_states,
    n_beta,
    n_states_fine,
    n_assets_quad,
    epsilon_grid
):
    """
    Creates grids for chebyshev interpolation and estimation of density/integration.
    """
    beta_grid = np.geomspace(beta_max, beta_min, num = n_beta).reshape((n_beta,1),order='F') #creates grid of beta values spaced out logarithmically
    beta_weights = beta_grid/(n_beta) #mass at each point, assumed uniform and fixed.
    asset_cheb_zeros = -np.cos(
        ((2 * np.arange(1, n_assets + 1) - 1) * np.pi) / (2 * n_assets)
    ).reshape((n_assets, 1), order='F')
    assets_grid = scaleup(asset_cheb_zeros, assets_min, assets_max)
    # matrix versions of grid to use later for outerproduct (not necessary and can be ommited)
    beta_mat_grid = np.tile(beta_grid, (n_epsilon,1, n_assets))
    epsilon_mat_grid = np.tile(epsilon_grid, (n_assets,n_beta, 1)).T
    assets_mat_grid = np.tile(assets_grid.T, (n_epsilon,n_beta, 1))
    # grid of shocks tomorrow conditional on all possible state pairs today
    epsilon_grid_prime = np.tile(epsilon_grid, (n_states, 1)).T
    # fine grids for histogram method of young 2010 for initial guess
    # equally spaced now for hist
    assets_grid_fine = np.linspace(assets_min, assets_max, n_assets_fine).reshape((n_assets_fine, 1))
    assets_grid_fine_zeros = scaledown(assets_grid_fine, assets_min, assets_max)
    # matrix versions of grid to use later for outer product
    epsilon_mat_grid_fine = np.tile(epsilon_grid, (n_assets_fine, n_beta, 1)).T
    assets_mat_grid_fine = np.tile(assets_grid_fine.T, (n_epsilon, n_beta, 1))
    beta_mat_grid_fine = np.tile(beta_grid, (n_epsilon, 1, n_assets_fine))
    # grid of shocks tomorrow conditional on all possible state pairs today
    epsilon_grid_prime_fine = np.tile(epsilon_grid, (n_states_fine, 1)).T

    # Quadrature grids to integrate densities of dist

    assets_grid_quad_zeros, quad_weights = np.polynomial.legendre.leggauss(
        n_assets_quad
    )
    assets_grid_quad_zeros = assets_grid_quad_zeros.reshape((n_assets_quad, 1))
    # scale up grid
    # push away from assets_min when scaling up so that the point mass is correctly captured during integration
    assets_grid_quad = scaleup(assets_grid_quad_zeros, assets_min + 1e-1, assets_max)

    # matrix version of grids for use in outer product (not necessary)
    epsilon_grid_mat_quad = np.tile(epsilon_grid, (n_assets_quad,1)).T
    assets_grid_mat_quad = np.tile(assets_grid_quad.T, (n_epsilon, 1))
    assets_grid_fine  = assets_grid_fine.reshape((1,n_assets_fine), order='F')
    return [
        asset_cheb_zeros,
        assets_grid,
        epsilon_mat_grid,
        assets_mat_grid,
        epsilon_grid_prime,
        assets_grid_fine,
        assets_grid_fine_zeros,
        epsilon_mat_grid_fine,
        assets_mat_grid_fine,
        epsilon_grid_prime_fine,
        assets_grid_quad_zeros,
        quad_weights,
        assets_grid_quad,
        epsilon_grid_mat_quad,
        assets_grid_mat_quad,
        beta_mat_grid,
        beta_mat_grid_fine,
        beta_grid,
        beta_weights
    ]

@njit
def scaleup(x, minval: float, maxval: float):
    """
    Converts [-1,1] to [minval,maxval] range with points always lying inside new range. Returns scaled up version of values.
    """
    return np.minimum(
        np.maximum(
            0.5 * (x + 1) * (maxval - minval) + minval,
            minval * np.ones(np.shape(x)),
        ),
        maxval * np.ones(np.shape(x)),
    )

@njit
def scaledown(x, minval, maxval):
    """
    Converts [minval,maxval] to [-1,1] with points always lying inside [-1,1]. Returns scaled down values.
    """
    return np.minimum(
        np.maximum(
            2 * (x - minval) / (maxval - minval) - 1, -1 * np.ones(np.shape(x))
        ),
        np.ones(np.shape(x)),
    )
@njit
def tile_h(x,n_dim):
    return x.repeat(n_dim).reshape((-1,n_dim), order='F')
@njit
def tile_v(x,n_dim):
    return ((x).repeat(n_dim).reshape((-1,n_dim), order='F')).T
