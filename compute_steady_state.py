from numba.core.errors import NotDefinedError
import numpy as np
from numba import jit, njit
from compute_MC_Residual_Histogram import compute_MC_Residual_Histogram
from create_polynomials import compute_poly
from create_grids import create_grids
def compute_steady_state(self, capital):
    """
    This function is a wrapper to compute the steady state of the equilibrium in the base along the lines of Winberry, 2018.
    """
#TODO
    NotDefinedError=create_grids(s)
    NotDefinedError=compute_poly(s)
    else:
        NotImplementedError
    # Make an initial guess of market clearing capital levels from Young, 2010 method using histograms 
    np.fsolve(lambda a, b, c: compute_MC_Residual_Histogram(),x0,args)





########### Stationary equilibrium ##################################
## Guess a stationary K,L
# Make a good guess using residual 
## Compute r,w according to representative firm
## Solve for conditional expectation function
## Solve for invariant distribution approx
## Update K'
## Iterate until |K'-K|<tol