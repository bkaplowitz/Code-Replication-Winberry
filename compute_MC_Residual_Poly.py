from compute_MC_Residual_Histogram import update_coefs_poly
import numpy as np
import scipy as sp

def compute_MC_residual_poly(
    capital,
    moments,
    grid_moments,
    constrained,
    epsilon_grid,
    epsilon_grid_quad,
    epsilon_prime_grid,
    epsilon_trans_mat,
    assets_grid,
    N,
    n_epsilon,
    n_assets,
    n_assets_quad,
    n_states,
    n_measure,
    assets_min,
    assets_max,
    assets_poly,
    assets_poly_BC,
    assets_poly_sq,
    assets_poly_quad,
    assets_grid_quad,
    epsilon_invariant,
    quad_weights,
    a_bar,
    alpha,
    beta,
    delta,
    mu,
    sigma,
    tau,
    damp,
    tol,
    max_iter,
    Fast=False,
):
    """
    Computes the residual for aggregate capital under the parameterized form.
    """
    # compute prices
    r = alpha * (capital ** (alpha - 1)) * (N ** (1 - alpha)) - delta
    w = capital ** alpha * (1 - alpha) * (N ** (-alpha))

    # sets different default options depending on use case
    if Fast == True:
        err_2 = 1e-4
        tol_2 = 200
    elif Fast == False:
        err_2 = 1e-6
        tol_2 = 500

    # approximate cond expectation function

    # initial coefs using rule of thumb savings
    init_opt_asset_grid = np.log(
        beta
        * (1 + r)
        * (w * (mu * (1 - epsilon_grid) + (1 - tau) * epsilon_grid) + r * assets_grid)
        ** (-sigma)
    )
    # fit/find coefs using chebyshev interpolation trick
    # also uses trick from i:i+1 to return 1xn vec
    coefs_mat = np.zeros((n_epsilon, n_assets))
    for i in np.arange(0, n_epsilon):
        coefs = np.sum(
            np.multiply(
                assets_poly.T,
                (np.ones((n_assets, 1)) @ init_opt_asset_grid[i : i + 1, :]),
            ),
            1,
        )
        coefs_mat[i, :] = coefs / assets_poly_sq

    # iterate
    err = 10
    iter = 0
    while (err > tol) & (iter <= max_iter):
        coefs_new = update_coefs_poly(
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
        err = np.max(np.abs(coefs_new - coefs_mat))
        iter += 1
        coefs_mat = (1 - damp) * coefs_new + damp * coefs_mat
    coefs = coefs_mat.copy()

    # compute EMUC over quadrature grid for integration later on
    cond_expec = np.exp(coefs @ assets_poly_quad.T)
    # compute optimal savings
    assets_prime_opt = (
        w * (mu * (1 - epsilon_grid_quad) + (1 - tau) * epsilon_grid_quad)
        + (1 + r) * assets_grid_quad
        - cond_expec ** (-1 / sigma)
    )
    assets_prime_quad = np.maximum(
        assets_prime_opt, a_bar * np.ones((n_epsilon, n_assets_quad))
    )

    # compute policies at borrowing constraint:

    # compute cond expec approx at borrowing constraint
    cond_expec = np.exp(coefs @ assets_poly_BC.T)
    # compute optimal savings rule at borrowing constraint (assets_min) given cond_expec estimate
    assets_prime_opt = (
        w * (mu * (1 - epsilon_grid) + (1 - tau) * epsilon_grid)
        + (1 + r) * assets_min
        - cond_expec ** (-1 / sigma)
    )
    assets_prime_BC = np.maximum(assets_prime_opt, a_bar * np.ones((n_epsilon, 1)))
    # compute stationary distribution from decision rules
    err = 10
    iter = 0
    # defines exponential family function to fit
    # quad_weights true value, and this corresponds to FOC from the full nonlinear equation setting the moments (grid moments) equal to the integral of the
    # first order approach only! But this is okay here as exponential family is globally

    
    params_new = np.zeros((n_epsilon, n_measure + 1))
    params_temp = np.zeros((n_epsilon, n_measure))
    while (err > err_2) & (iter <= tol_2):
        for i_epsilon in np.arange(0, n_epsilon):
            # computes minimizing params
            optim_result = sp.optimize.minimize(
                exp_func_error,
                np.zeros((n_measure, 1)),
                (grid_moments[i_epsilon, :, :].squeeze(), quad_weights, n_measure),
                jac=deriv_exp_func,
            )
            params_temp[i_epsilon, :] = optim_result.x
            # computes value of func at optimal params
            normalization = exp_func_error(
                params_temp[i_epsilon,:], grid_moments[i_epsilon, :, :].squeeze(), quad_weights, n_measure
            ).reshape((1,), order='F')
            # make sure works.
            params_new[i_epsilon, :] = np.hstack((1 / normalization, params_temp[i_epsilon,:]))

        # compute new moments/ moments grid
        moments_new = np.zeros((n_epsilon, n_measure))
        grid_moments_new = np.zeros((n_epsilon, n_assets_quad, n_measure))
        # construct new moments across all (epsilon,epsilon_prime) pairs
        # next period first order moments, uncentered (no current mean subtraction)
        # TODO: If time, change to two vectorized ops, or make Numba compatible.
        for i_epsilon_prime in np.arange(0, n_epsilon):
            moments_new[i_epsilon_prime,0] = 0
            for i_epsilon in np.arange(0, n_epsilon):
                # first term is 1st moments of frac who remain unconstrained times prob become unconstrained next period. Second is moments of frac who become constrained
                moments_new[i_epsilon_prime, 0] = (
                    moments_new[i_epsilon_prime, 0]
                    + (1 - constrained[i_epsilon, 0])
                    * epsilon_invariant[i_epsilon]
                    * epsilon_trans_mat[i_epsilon, i_epsilon_prime]
                    # this entire term is just the expected value of next period assets conditional on i_epsilon realized today
                    * params_new[i_epsilon, 0]
                    * (quad_weights.T
                    @ (
                       np.multiply(assets_prime_quad[i_epsilon, :].T,
                         np.exp(
                            np.matmul(
                                np.squeeze(grid_moments[i_epsilon, :, :]),
                                params_new[i_epsilon, 1 : n_measure + 1].T,
                            ))
                        )
                    ))
                    + constrained[i_epsilon, 0]
                    * epsilon_invariant[i_epsilon]
                    * epsilon_trans_mat[i_epsilon, i_epsilon_prime]
                    * assets_prime_BC[i_epsilon, 0]
                )
            # rescaling by epsilon invariant to rescale to conditional on shock realization moment (which is what we want to use)
            moments_new[i_epsilon_prime, 0] = (
                moments_new[i_epsilon_prime, 0] / epsilon_invariant[i_epsilon_prime]
            )
            # new grids for the moments computed by subtracting moments from assets (as before)
            grid_moments_new[i_epsilon_prime, :, 0] = (
                assets_grid_quad - moments_new[i_epsilon_prime, 0]
            )
            # computed all first order moments... now we redo for higher order moments, centered (current mean subtraction)
            for i_moments in np.arange(1, n_measure):
                moments_new[i_epsilon_prime, i_moments] = 0
                for i_epsilon in np.arange(0, n_epsilon):
                    # plug in and estimate higher order moments using the current estimation of the cdf now one period forward
                    # first term is moments of frac who remain unconstrained. Second is moments of frac who become constrained

                    moments_new[i_epsilon_prime, i_moments] = (
                        moments_new[i_epsilon_prime, i_moments]
                        + (1 - constrained[i_epsilon, 0])
                        * epsilon_invariant[i_epsilon]
                        * epsilon_trans_mat[i_epsilon, i_epsilon_prime]
                        * params_new[i_epsilon, 0]
                        * quad_weights.T
                        @ (
                            np.multiply((
                                (
                                    assets_prime_quad[i_epsilon, :].T
                                    - moments_new[i_epsilon_prime, 0]
                                )
                                ** (i_moments + 1)
                            )
                            , np.exp(
                                np.squeeze(grid_moments[i_epsilon, :, :])
                                @ params_new[i_epsilon, 1 : n_measure + 1].T
                            ))
                        )
                        + constrained[i_epsilon, 0]
                        * epsilon_invariant[i_epsilon]
                        * epsilon_trans_mat[i_epsilon, i_epsilon_prime]
                        * (
                            (
                                assets_prime_BC[i_epsilon, 0]
                                - moments_new[i_epsilon_prime, 0]
                            )
                            ** (i_moments + 1)
                        )
                    )
                # renormalize
                moments_new[i_epsilon_prime, i_moments] = (
                    moments_new[i_epsilon_prime, i_moments]
                    / epsilon_invariant[i_epsilon_prime]
                )
                # compute new grid of moments for updating estimate of density, centered
                grid_moments_new[i_epsilon_prime,:, i_moments] = (
                    assets_grid_quad.T - moments_new[i_epsilon_prime, 0]
                ) ** (i_moments + 1) - moments_new[i_epsilon_prime, i_moments]
        # update mass at borrowing constraint
        constrained_new = np.zeros([n_epsilon, 1])
        for i_epsilon_prime in np.arange(0, n_epsilon):
            for i_epsilon in np.arange(0, n_epsilon):
                constrained_new[i_epsilon_prime, 0] = (
                    constrained_new[i_epsilon, 0]
                    + (1 - constrained[i_epsilon_prime, 0])
                    * epsilon_invariant[i_epsilon]
                    * epsilon_trans_mat[i_epsilon, i_epsilon_prime]
                    * params_new[i_epsilon, 0]
                    * quad_weights.T
                    @ (
                        (assets_prime_quad[i_epsilon, :].T <= (a_bar + 1e-8))
                        * np.exp(
                            np.squeeze(grid_moments[i_epsilon, :, :])
                            @ params_new[i_epsilon, 1 : n_measure + 1].T
                        )
                    )
                    + constrained[i_epsilon, 0]
                    * epsilon_invariant[i_epsilon]
                    * epsilon_trans_mat[i_epsilon, i_epsilon_prime]
                    * (assets_prime_BC[i_epsilon, 0] <= (a_bar + 1e-8))
                )
                constrained_new[i_epsilon, 0] = (
                    constrained_new[i_epsilon, 0] / epsilon_invariant[i_epsilon]
                )
        # Update Iteration
        err = np.max(
            [
                np.max(np.abs(moments_new - moments)),
                np.max(np.abs(constrained_new - constrained)),
            ]
        )
        iter += 1
        moments = moments_new
        grid_moments = grid_moments_new
        constrained = constrained_new
        # compute returns
        # new capital estimate
    capital_new = np.multiply(epsilon_invariant , (1 - constrained)).T @ moments[:, 0:1] + a_bar * (
        np.multiply(epsilon_invariant ,constrained)
    ).T @ np.ones((n_epsilon, 1))
    residual = capital - capital_new
    residual_return = residual.flatten()
    params_opt = params_new
    return_val = np.array([residual_return, params_opt, moments, constrained])
    return return_val

def exp_func_error(params, grid_moments,quad_weights, n_measure): 
    '''
    Gives the first order condition that is sufficient for matching moments.
    '''
    params_vec=np.array(params).reshape((n_measure,1),order='F')
    return_val = quad_weights.T @ np.exp(grid_moments @ params_vec)
    return return_val

def deriv_exp_func(params, grid_moments,quad_weights, n_measure):
    '''
    Gives the first order condition derivative to feed into the optimizer fmin as the jacobian. 
    '''
    params_vec = np.array(params).reshape((n_measure,1), order='F')
    deriv  = np.sum(
            np.multiply(
                np.tile(quad_weights, [1, n_measure]),
                np.multiply(
                    grid_moments, np.tile(np.exp(grid_moments @ params_vec), [1, n_measure])
                ),
            ),
            0,
        )
    jac = deriv.T 
    return jac
