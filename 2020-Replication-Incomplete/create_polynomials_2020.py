import numpy as np
def create_polynomials(n_prod,n_cap,state_grid_zeros, n_states,prod_cheby_zeros, cap_cheby_zeros,prod_prime_grid_zeros, fine_grid_zeros, prod_prime_fine_grid_zeros, quad_grid_zeros, prod_prime_quad_zeros, n_shocks, n_states_fine, n_states_quad ):
    """
    Constructs polynomials for value function and capital accumulation over collocation nodes
    """
    prod_poly = compute_chebyshev(n_prod,state_grid_zeros[:,0])
    capital_poly = compute_chebyshev(n_cap, state_grid_zeros[:,1])
    n_grid = np.size(state_grid_zeros[:,0])
    # create a tensor product of two chebyshev, representing all interactions/coef possible
    state_poly_temp =np.einsum('ij,ik->ijk', prod_poly,capital_poly)
    state_poly = state_poly_temp.reshape((n_grid, n_states))
    
    # compute squared terms for interpolation formulas (first two terms)
    prod_poly_sq = compute_chebyshev(n_prod, prod_cheby_zeros)
    cap_poly_sq = compute_chebyshev(n_cap, cap_cheby_zeros)
    prod_sq_val = np.sum(np.power(prod_poly_sq,2), axis=0) 
    cap_sq_val = np.sum(np.power(cap_poly_sq,2), axis=0)
    state_poly_squared = ((prod_sq_val.T)@cap_sq_val).reshape((n_states,1))

    # compute poly over future productivity shocks (for interpolating expected value/CE )
    prod_prime_poly_temp= compute_chebyshev(n_prod,prod_prime_grid_zeros.reshape((n_states,1)))
    prod_prime_poly=prod_prime_poly_temp.reshape((n_shocks,n_states,n_prod))
    
    # Compute chebyshev polynomials for finer grid using histogram
    prod_fine_poly = compute_chebyshev(n_prod, fine_grid_zeros[:,0]) 
    cap_fine_poly = compute_chebyshev(n_prod, fine_grid_zeros[:,1]) 
    # compute tensor product of two chebyshev 

    state_fine_poly_temp=np.einsum('ij,ik->ijk', prod_fine_poly,cap_fine_poly)
    state_fine_poly = state_fine_poly_temp.reshape((n_grid, n_states))

    # compute fine polynomials over future shocks
    prod_prime_poly_fine_temp= compute_chebyshev(n_prod,prod_prime_fine_grid_zeros.reshape((n_states_fine*n_shocks,1)))
    prod_prime_poly_fine = prod_prime_poly_fine_temp.reshape((n_shocks,n_states_fine,n_prod))

    # compute chebyshev polynomials over quadrature grid (for LoM for distribution)
    prod_quad_poly = compute_chebyshev(n_prod,quad_grid_zeros[:,0])
    cap_quad_poly = compute_chebyshev(n_prod,quad_grid_zeros[:,1])
    # compute tensor of polynomials
    quad_poly_temp = np.einsum('ij,ik->ijk', prod_quad_poly,cap_quad_poly)
    quad_poly = quad_poly_temp.reshape((n_states,1))

    # compute polynomials for quadrature over future shocks
    prod_prime_quad_poly_temp = compute_chebyshev(n_prod,prod_prime_quad_zeros.reshape((n_states_quad*n_shocks,1)))
    prod_prime_quad_poly = prod_prime_quad_poly_temp.reshape(n_shocks, n_states_quad, n_prod)

    # compute derivative of chebyshev polynomials at each of the collocation nodes (for FOC in k)
    prod_poly_deriv= compute_chebyshev(n_prod, state_grid_zeros[:,0])
    #-1 as now of lower order
    cap_poly_deriv = compute_chebyshev(n_cap-1, state_grid_zeros[:,1])

    #create tensor product
    state_poly_deriv_temp = np.einsum('ij,ik->ijk', prod_poly_deriv,cap_poly_deriv)
    # -1 becomes -n_prod dim under cartesian product
    state_poly_deriv = state_poly_deriv_temp.reshape((n_states-n_prod,1)) 

    # compute derivatives of chebyshev polynomial at each of collocation points for hist (for marginal value func)

    prod_fine_deriv = compute_chebyshev(n_prod, fine_grid_zeros[:,0])
    cap_fine_deriv = compute_chebyshev(n_cap-1, fine_grid_zeros[:,1])
    
    #create tensor product
    state_poly_fine_deriv_temp = np.einsum('ij,ik->ijk', prod_fine_deriv,cap_fine_deriv)
    state_poly_fine_deriv = state_poly_fine_deriv_temp.reshape((n_states-n_prod,1))

    # returns all computed grids
    return [state_poly,state_poly_squared,prod_prime_poly, state_fine_poly,prod_prime_poly_fine , quad_poly, prod_prime_quad_poly, state_poly_deriv, state_poly_fine_deriv]
    #TODO: Finish and check code works ok
    #TODO: Then do core.


#@njit
def compute_chebyshev(power,grid):
    '''
    Computes polynomial terms of chebyshev
    '''
    n_grid = np.size(grid)
    #generating series for cheby of first type
    cheby = np.ones((n_grid,power))
    cheby[:,1] = grid
    for n_power in range(2,power):
        cheby[:,n_power] = 2*np.multiply(grid,cheby[:,n_power-1])-cheby[:,n_power-2]
    return cheby


    #create a polynomial of chebyshev

