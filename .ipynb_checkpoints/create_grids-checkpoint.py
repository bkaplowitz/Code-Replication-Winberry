import numpy as np
# Idiosyncratic shocks-- quadrature to integrate

def GH_quad(self, n_shocks):
    """
    docstring
    """
    raise NotImplementedError


# 
def create_grids(self, approx_N_cap, approx_N_prod):
    """
    Approximates value function/conditional expectation for next period capital and productivity shocks given how much to approximate capital and productivity.
    """
    # Compute zeros of chebyshev function
    prod_cheb_zeros = -np.cos(((2*np.arange(1,approx_N_prod)-1)*np.pi)/(2*approx_N_prod))
    K_cheb_zeros = -np.cos(((2*np.arange(1,approx_N_cap)-1)*np.pi)/(2*approx_N_cap))

    #compute grid of two:
    temp_grid = np.meshgrid((prod_cheb_zeros,K_cheb_zeros))
    print(temp_grid)
    # reshape grid to get a tensor 