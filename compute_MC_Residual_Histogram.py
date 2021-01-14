from create_polynomials import compute_chebyshev
import numpy as np
from create_grids import scaledown
import scipy.spatial as spat
import scipy.sparse as sparse
import scipy as sp


def compute_MC_Residual_Histogram(
    capital,
    alpha,
    N,
    delta,
    beta,
    sigma,
    epsilon_grid,
    assets_grid,
    assets_poly,
    mu,
    tau,
    n_epsilon,
    n_assets,
    n_states,
    n_assets_fine,
    n_states_fine,
    assets_poly_sq,
    assets_poly_fine,
    assets_mat_grid_fine,
    epsilon_grid_fine,
    epsilon_prime_grid,
    epsilon_trans_mat,
    tol,
    max_iter,
    damp,
    a_bar,
    assets_min,
    assets_max, assets_grid_fine,
    optional_return=False,
):
    """
    Computes residual of market-clearing condition to compute an initial guess for exponential distribution family.
    """

    # solve for market clearing capital level
    # first compute prices of representative firm in winberry 2018, make an initial guess via capital level.
    r = alpha * (capital) ** (alpha - 1) * N ** (1 - alpha) - delta
    w = (1-alpha)*capital ** (alpha) * N ** (-alpha)

    # estimate policy rules for individuals via polynomial interpolation
    init_opt_asset_grid = np.log(
        beta
        * (1 + r)
        * (w * (mu * (1 - epsilon_grid) + (1 - tau) * epsilon_grid) + r * assets_grid)
        ** (-sigma)
    )
    # Find coefficients for estimate  using guessed policy rule
    # do 1-d interpolation
    coefs_mat = np.zeros((n_epsilon, n_assets))
    for i in np.arange(0, n_epsilon):
        # uses trick where i:i+1 indexing for numpy does not reduce dimmension while [i,:] reduces dimmension by 1.
        coefs = np.sum(
            np.multiply(
                assets_poly.T,
                np.ones((n_assets, 1)) @ init_opt_asset_grid[i : i + 1, :],
            ),
            1,
        )
        coefs_mat[i : i + 1, :] = (coefs / assets_poly_sq)
    # Iterate and update coefficients
    err = 100
    iter = 0
    while (err > tol) & (iter <= max_iter):
        coefs_mat_new = update_coefs_poly(
            coefs_mat,
            assets_poly,
            w,
            mu,
            epsilon_grid,
            assets_grid,
            epsilon_prime_grid,
            epsilon_trans_mat,
            beta,
            tau,
            sigma,
            r,
            a_bar,
            n_epsilon,
            n_assets,
            n_states,
            assets_min,
            assets_max,
            assets_poly_sq,
        )
        err = np.max((np.abs(coefs_mat_new - coefs_mat)))
        iter += 1
        coefs_mat = (1 - damp) * coefs_mat_new + damp * coefs_mat

    # comptue policies over histogram grid

    # compute decision rule along fine grid (constructed below as exp of chebyshev vals on fine grid) comes directly from EE defining cond_expec as beta*(1+r)*E(v(epsilon',a'))
    cond_expec = np.exp(coefs_mat @ assets_poly_fine.T)
    # compute savings policy/a'
    a_prime_opt = (
        w * (mu * (1 - epsilon_grid_fine) + (1 - tau) * epsilon_grid_fine)
        + (1 + r) * assets_mat_grid_fine
        - np.power(cond_expec, (-1 / sigma))
    )
    assets_prime_fine = np.maximum(
        a_prime_opt, a_bar * np.ones((n_epsilon, n_assets_fine))
    )

    # consumption optimal rule for consumption
    con_fine = (
        w * (mu * (1 - epsilon_grid_fine) + (1 - tau) * epsilon_grid_fine)
        + (1 + r) * assets_mat_grid_fine
        - assets_prime_fine
    )
    if np.isnan(np.amax(assets_prime_fine)):
        print('uhoh')
    # compute transition matrix for assets
    # TODO: If not done today, move on to dynare portion temporarily
    # compute weighting matrices
    [index_left, index_right, weight_left, weight_right] = compute_linear_weights(
        assets_grid_fine, (assets_prime_fine).ravel(order='F'), n_epsilon
    )
    # assets tomorrow estimate to right and left
    transition_right = np.zeros([n_states_fine, n_assets_fine])
    transition_left = np.zeros([n_states_fine, n_assets_fine])
    for i in np.arange(0, n_assets_fine):
        # since we unravelled we need to now restore the points, which we do by capturing via transition matrix, assigning weight
        transition_left[index_left == i, i] = weight_left[index_left == i]
        transition_right[index_right == i, i] = weight_right[index_right == i]
    # one step transition matrix (weighting of both left and right for complete estimate) with last term just tiling into n_assets by n_epsilon matrix for later multiplication.
    assets_trans = np.kron(transition_left + transition_right, np.ones([1, n_epsilon]))
    # transition matrix for all idiosyncratic shocks
    epsilon_trans_hist = np.tile(epsilon_trans_mat, (n_assets_fine, n_assets_fine))

    # compute full transition matrix in sparse form
    trans_mat = sparse.bsr_matrix(np.multiply(assets_trans, epsilon_trans_hist))

    # compute invariant hist by large iteration
    err_hist = 100
    iter_hist = 0
    # uniform
    hist = sparse.bsr_matrix(np.ones((n_states_fine, 1)) / n_states_fine)
    while (err_hist > 1e-12) & (iter_hist < 1e4):
        hist_new = (trans_mat.T @ (hist))
        diff_hist = hist_new - hist
        err_hist = np.abs(diff_hist).max()
        iter_hist += 1
        hist = hist_new
    hist_mat = hist.todense().reshape((n_epsilon, n_assets_fine), order='F')

    # return the market clearing residual from guess
    # subtracts total capital in order to account fo residual of guess and find new market clearing capital
    residual = capital - np.sum(np.multiply(assets_grid_fine.ravel(order='F'),  (hist_mat[0:1, :].ravel(order='F') + hist_mat[1:2, :].ravel(order='F'))))
    # return [residual, hist_mat, assets_prime_fine, con_fine]
    return_val = np.array([residual, hist_mat,assets_prime_fine, con_fine])
    return return_val

def compute_linear_weights(grid, vals, n_epsilon):
    # TODO: go back in and revise?
    """
    Used for linear interpolation. Requires unravelled coords. Given points off a grid finds the right and left gridpoint to these points and the distance from each side. Uses the standard KDTree algorithm for efficient implementation.
    """
    # if ndim smaller than 20, or no gpu use kdtree. Else use knn as more parallelizable. Here, since we are in 2 dims, we use kdtree.
    # We want to only fit across the asset space not asset,epsilon space pairs, given epsilon is discrete jump here.
    # If epsilon continuously drawn and grid itself,convert to coordinates below by transposing vals and grid and then run kdtree.
    n_grid = np.size(grid)
    grid_new = grid.reshape((n_grid, 1), order='F')
    vals_new = vals.reshape((n_grid * n_epsilon, 1), order='F')
    grid_lookup = spat.kdtree.KDTree(grid_new)
    dist_nearest_neighbors, index_nearest_neighbors = grid_lookup.query(vals_new, k=1)
    # print(index_nearest_neighbors.shape)
    grid_nearest_neighbors = np.take(grid, index_nearest_neighbors)
    # print('grid nearest neighbor')
    # print(grid_nearest_neighbors.shape)
    # print('index nearest neighbor')
    # print(index_nearest_neighbors.shape)
    # print(index_nearest_neighbors)
    # index below and above created as independent new objects
    index_left = index_nearest_neighbors.copy(order='F')
    index_right = index_nearest_neighbors.copy(order='F')
    # ensures index_left that are always left of vals, so if it is right, subtracts 1
    index_left[grid_nearest_neighbors > vals] = (
        index_left[grid_nearest_neighbors > vals] - 1
    )
    # if too small, smallest can be (index value) is 0
    index_left[index_left <= 0] = 0
        # if too large, largest can be is n_grid-1

    index_left[index_left>= (n_grid -1) ] = n_grid-1
    # finds gridpoints of index_left
    grid_left = np.take(grid, index_left)
    # ensures index_right that are always right of vals, so if it is right, adds 1
    index_right[grid_nearest_neighbors <= vals] = (
        index_right[grid_nearest_neighbors <= vals] + 1
    )
    index_right[index_right >= (n_grid - 1)] = n_grid - 1
    index_right[index_right<= 0] =0 
    # finds gridpoints to right
    grid_right = np.take(grid, index_right)

    # weights for interpolation, just linear weights of each value at gridpoints unless outside grid values in which case it gets 1 on whatever point is closest
    weights_left = (grid_right - vals) / (grid_right - grid_left)
    weights_left[vals <= grid_left] = 1
    weights_left[vals >= grid_right] = 0
    weights_right = (vals - grid_left) / (grid_right - grid_left)
    weights_right[vals <= grid_left] = 0
    weights_right[vals >= grid_right] = 1
    return [index_left, index_right, weights_left, weights_right]

#updates coefficients

def update_coefs_poly(
    coefs,
    assets_poly,
    w,
    mu,
    epsilon_grid,
    assets_grid,
    epsilon_prime_grid,
    epsilon_trans_mat,
    beta,
    tau,
    sigma,
    r,
    a_bar,
    n_epsilon,
    n_assets,
    n_states,
    assets_min,
    assets_max,
    assets_poly_sq,
):
    """
    Updates coefs across iterations to find SS rule. 
    """
    # computes cond_expec, under assumed form of exp(sum coefs_i cheby_poly_i). Expectation transform used to make positive... but may not be minimax optimal anymore?
    cond_expec = np.exp(coefs @ assets_poly.T)

    # compute optimal savings rule
    assets_prime_opt = (
        w * (mu * (1 - epsilon_grid) + (1 - tau) * epsilon_grid)
        + (1 + r) * assets_grid
        - np.power(cond_expec, (-1 / sigma))
    )

    # actual savings (including borrowing constraint)
    assets_prime = np.maximum(assets_prime_opt, a_bar * np.ones((n_epsilon, n_assets)))
    # compute next period grid (endogeneous grid point method)
    # need this duplicated to account for savings decisions as savings decision depends on all states pairs today (epsilon, assets)
    assets_prime_grid = np.tile(assets_prime.reshape((1, n_states), order='F'), (n_epsilon, 1))
    # compute next period poly +grid
    assets_prime_zeros = scaledown(assets_prime, assets_min, assets_max)
    poly_assets_prime = compute_chebyshev(
        n_assets, assets_prime_zeros.reshape((n_states, 1),order='F')
    )

    # compute next period optimal savings policy
    cond_expec_prime = np.exp(coefs @ poly_assets_prime.T)

    # computes optimal savings rule for next period
    assets_prime_prime_opt = (
        w * (mu * (1 - epsilon_prime_grid) + (1 - tau) * epsilon_prime_grid)
        + (1 + r) * assets_prime_grid
        - np.power(cond_expec_prime,(-1 / sigma))
    )

    # computes savings w/ BC
    assets_prime_prime_grid = np.maximum(
        assets_prime_prime_opt, a_bar * np.ones((n_epsilon, n_epsilon * n_assets))
    )

    # update conditional expectation/coefs

    cons_prime = (
        w * (mu * (1 - epsilon_prime_grid) + (1 - tau) * epsilon_prime_grid)
        + (1 + r) * assets_prime_grid
        - assets_prime_prime_grid
    )

    # compute estimated EMUC from cond_expectation estimation as inner prods
    # consum prime conditional on future epsilon prime shocks, want it to be conditional on current epsilon prime shocks (hence premultiplying by epsilon_trans_mat)
    cond_expec_est = np.reshape(
        beta * (1 + r) * epsilon_trans_mat @ np.power(cons_prime, -sigma),
        (n_epsilon, n_epsilon, n_assets), order='F'
    )
    # extract diags, which is entries we want, for new conditional expectation since this says state realization of epsilon today is same as cond expec entry under epsilon today
    cond_expec_mat = np.zeros((n_epsilon,n_assets))
    for i_epsilon in np.arange(0,n_epsilon):
        cond_expec_mat[i_epsilon,:] = cond_expec_est[i_epsilon,i_epsilon,:] 
    #cond_expec_mat = np.einsum("ii...->i...", cond_expec_est)

    # estimate new coefficients
    coefs_new = np.zeros((n_epsilon, n_assets))
    for i in np.arange(0, n_epsilon):
        coefs_new_temp = np.sum(
            np.multiply(
                assets_poly.T,
                (np.ones((n_assets, 1)) @ np.log(cond_expec_mat[i : i + 1, :]))
            ),1
        )
        coefs_new[i, :] = coefs_new_temp / assets_poly_sq

    return coefs_new
