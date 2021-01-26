# main SS file for convenient debugging. Replicates SS portion of ipython notebook
import numpy as np
import scipy.optimize as opt
from icecream import ic
import logging
import time
import matplotlib.pyplot as plt
import create_grids, create_polynomials, compute_MC_Residual_Histogram, compute_MC_Residual_Poly, compute_moments

def main():
    # preferences
    beta = .96 # discount rate
    sigma = 1 # CRRA
    a_bar = 0 # borrowing constraint

    #tech
    alpha = .36 # capital share
    delta = .1  # depreciation

    #idiosyncratic shocks
    epsilon_grid = np.array([0,1]).reshape((1,2)) #idiosyncratic shocks
    N = .93 # Aggregate employment
    u_duration = 1
    frac_u = u_duration/(1+u_duration)
    frac_N = (1-N)/N
    epsilon_transition_mat = np.array([frac_u,1-frac_u,frac_N*(1-frac_u), 1-(frac_N*(1-frac_u))]).reshape((2,2))
    epsilon_invariant = np.array([1-N,N]).reshape((2,1))

    #unemployment benefits
    mu = .15
    tau = mu*frac_N

    #Aggregate Shocks
    rho_TFP = .859
    sigma_TFP = .014

        #Approximation Parameters
    n_epsilon = 2 # number of shocks
    n_assets = 25  # number of gridpoints for use in asset grid/polynomial interpolation
    n_states = n_epsilon*n_assets

    #Bounds on Grid space
    K_rep_SS = ((alpha*(N**(1-alpha)))/((1/beta)-(1-delta)))**(1/(1-alpha)) #from firm capital FOC
    assets_min = a_bar
    assets_max = 3*K_rep_SS #ad-hoc

    #Finger grid for analyzing policy funcs/histograms
    n_assets_fine = 100
    n_states_fine = n_epsilon*n_assets_fine

    #Degree Approx of distribution for integrating over via quadrature rule
    n_measure = 3
    n_assets_quad = 8 # number of interpolant points to use for quadrature
    n_state_quad = n_epsilon*n_assets_quad
    n_measure_coeff = n_epsilon*n_measure

    # Optimization params
    max_iter = 2e3
    tol = 1e-5
    damp = .95
    [   
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
    assets_grid_mat_quad] = create_grids.create_grids(
                                    n_assets,
                                n_assets_fine,
                                n_epsilon,
                                assets_min,
                                assets_max,
                                n_states,
                                n_states_fine,
                                n_assets_quad,
                                epsilon_grid)
    [assets_poly,
        assets_poly_sq,
        assets_poly_fine,
        assets_poly_quad,
        assets_poly_BC]=create_polynomials.compute_poly(n_assets,asset_cheb_zeros,assets_grid_fine_zeros, assets_grid_quad_zeros )


    f= lambda capital: np.take(compute_MC_Residual_Histogram.compute_MC_Residual_Histogram(capital,alpha,
        N,
        delta,
        beta,
        sigma,
        epsilon_mat_grid,
        assets_mat_grid,
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
        epsilon_mat_grid_fine,
        epsilon_grid_prime,
        epsilon_transition_mat,
        tol,
        max_iter,
        damp,
        a_bar,
        assets_min,
        assets_max, assets_grid_fine ),0)
    K_guess,dict_details,ier,msg = opt.fsolve(f, 1.01*K_rep_SS,full_output=True)
    if ier ==1:
        print(msg)
    else: 
        print('Initial guess successfully computed with guess K={:f}'.format(K_guess[0]))
    return_val=compute_MC_Residual_Histogram.compute_MC_Residual_Histogram(K_guess,alpha,
    N,
    delta,
    beta,
    sigma,
    epsilon_mat_grid,
    assets_mat_grid,
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
    assets_grid_fine,
    epsilon_mat_grid_fine,
    epsilon_grid_prime,
    epsilon_transition_mat,
    tol,
    max_iter,
    damp,
    a_bar,
    assets_min,
    assets_max, assets_grid_fine)
    # gets histogram_matrix 
    hist_mat=return_val[1]

    assets_grid_quad = assets_grid_quad.flatten()
    moments_hist, grid_moments, constrained = compute_moments.compute_moments_hist(hist_mat, n_epsilon,
    n_measure,
    n_assets_quad,
    assets_grid_fine,
    assets_grid_quad)

    quad_weights_vec = quad_weights.reshape((n_assets_quad,1))

    f= lambda capital: np.take(compute_MC_Residual_Poly.compute_MC_residual_poly(capital,
        moments_hist,
        grid_moments,
        constrained,
        epsilon_mat_grid,
        epsilon_grid_mat_quad,
        epsilon_grid_prime,
        epsilon_transition_mat,
        assets_mat_grid,
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
        quad_weights_vec,
        a_bar,
        alpha,
        beta,
        delta,
        mu,
        sigma,
        tau,
        damp,
        tol,
        max_iter),0)
    print("Computing Steady State from exponential family:")
    if (np.abs(f(K_guess))>1e-4):
        k_ss, infodict2, ier2, msg2 = opt.fsolve(f,K_guess, full_output=True)
    _,constrained_est, params_est,moments_est, constrained_est,   = compute_MC_Residual_Poly.compute_MC_residual_poly(k_ss, moments_hist,
    grid_moments,
    constrained,
    epsilon_mat_grid,
    epsilon_grid_mat_quad,
    epsilon_grid_prime,
    epsilon_transition_mat,
    assets_mat_grid,
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
    quad_weights_vec,
    a_bar,
    alpha,
    beta,
    delta,
    mu,
    sigma,
    tau,
    damp,
    tol,
    max_iter)


    _,hist_mat, assets_prime, consumption = compute_MC_Residual_Histogram.compute_MC_Residual_Histogram(k_ss,alpha,
    N,
    delta,
    beta,
    sigma,
    epsilon_mat_grid,
    assets_mat_grid,
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
    assets_grid_fine,
    epsilon_mat_grid_fine,
    epsilon_grid_prime,
    epsilon_transition_mat,
    tol,
    max_iter,
    damp,
    a_bar,
    assets_min,
    assets_max, assets_grid_fine)
    # compute density among grid
    density_fine = np.zeros((n_epsilon,n_assets_fine))
    for i_epsilon in np.arange(0,n_epsilon):
        # compute first moment
        grid_moments_fine = np.zeros((n_assets_fine,n_measure))
        grid_moments_fine[:,0] = assets_grid_fine - moments_est[i_epsilon,0]

        # higher order moments (centered)
        for i_moments in np.arange(1, n_measure):
            grid_moments_fine[:,i_moments] = (assets_grid_fine.ravel()  - moments_est[i_epsilon,0])**(i_moments+1) -  moments_est[i_epsilon,i_moments]
            #compute  density away from the borrowing constraint
            density_fine[i_epsilon,:] = params_est[i_epsilon,0]*(np.exp(grid_moments_fine@params_est[i_epsilon:i_epsilon+1,1:n_measure+1].T)).flatten()

        

if __name__ == "__main__":
    main()
