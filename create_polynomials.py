import numpy as np
from numba import jit, njit

#@njit
def compute_poly(
    n_assets: int, assets_grid_zeros, assets_grid_fine_zeros, assets_grid_quad_zeros
):
    """
    Computes the polynomials necessary to approximate conditional expectation by grid of chebyshev points
    """
    # grid for conditional expectation
    assets_poly = compute_chebyshev(n_assets, assets_grid_zeros)
    assets_poly_sq = np.sum(np.power(assets_poly, 2),0).T  # square all terms

    # fine grid for histogram/Young
    assets_poly_fine = compute_chebyshev(n_assets, assets_grid_fine_zeros)

    # quadrature grid for integrating
    assets_poly_quad = compute_chebyshev(n_assets, assets_grid_quad_zeros)

    # grid just at borrowing constraint
    bound = -1.0*np.ones(1)
    assets_poly_BC = compute_chebyshev(n_assets,bound)
    # since chebyshev defined on -1,1, -1 is edge of state space on left.
    return [
        assets_poly,
        assets_poly_sq,
        assets_poly_fine,
        assets_poly_quad,
        assets_poly_BC]


#@njit
def compute_chebyshev(power, grid):
    """
    Computes polynomial terms of chebyshev
    """
    n_grid = np.shape(grid)[0]
    # generating series for cheby of first type
    cheby = np.ones((n_grid, power))
    cheby[:, 1:2] = grid
    for n_power in range(2, power):
        cheby[:, n_power:n_power+1] = (
            2 * np.multiply(grid, cheby[:, (n_power - 1):n_power]) - cheby[:, (n_power - 2):(n_power-1)]
        )
    return cheby
