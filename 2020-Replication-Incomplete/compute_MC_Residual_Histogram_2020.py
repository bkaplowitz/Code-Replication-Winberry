import numpy as np
def compute_MC_Residual_Histogram(capital, m,state_grid, state_poly_grid, epsilon_grid, asset_grid, alpha, N, delta, beta, mu,tau, sigma, n_epsilon, n_assets ):
    """
    Computes residual of market-clearing condition to compute an initial guess for exponential distribution family.
    """

    # solve for market clearing capital level
    #first compute prices of representative firm in winberry 2018.
    r= alpha * (capital **(alpha -1))*N**(1-alpha)-delta
    w = capital**(1-alpha)*N**(-alpha)
    # mu unemployment benefits
    init_val_grid = np.log(beta*(1+r)*w*(mu*(1-epsilon_grid)+(1-tau)*epsilon_grid+r*asset_grid**(-sigma))
    coef =  np.zeros(n_epsilon, n_assets)
    for i in np.range(0,n_epsilon):
        

    #guess prices

    # compute grids for firms (only if doing multi-firm version)
    #labor_demand_grid =(np.exp(m_state_grid[:,0])*(m_state_grid[:,1])**(θ)/wage_guess)**(1/(1-ν))
    #profit_grid = (1-τ)*(np.exp(m_state_grid[:,1])*(m_state_grid[:,1])**(θ))*(labor_demand_grid**(ν)) - wage_guess * labor_demand_grid

    # initalize value function  init_est of value=pi+(1-d)*K
    #init_val_grid = profit_grid + (1-δ)*m_state_grid[:,1]
    #estimated 
    #coeff = np.sum(state_poly)
  