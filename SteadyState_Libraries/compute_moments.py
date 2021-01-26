import numpy as np


def compute_moments_hist(
    histogram_mat,
    n_epsilon,
    n_measure,
    n_assets_quad,
    assets_grid_fine,
    assets_grid_quad
):
    """
    Computes moments from given histogram matrix corresponding to initially estimated probability mass.
    """ 
    moments_hist = np.zeros((n_epsilon, n_measure))
    grid_moments = np.zeros((n_epsilon, n_assets_quad, n_measure))
    sum_hist = np.zeros((n_epsilon,n_measure))
    # TODO: can probably vectorize via fapply and extended moments_hist with zeroth entry being 0, but less readable not sure speedup is worth it
    for i_epsilon in np.arange(0, n_epsilon):
        # computes first moment
        # this step is wrong! 
        sum_hist = np.sum(histogram_mat[i_epsilon,:],1)
        moments_hist[i_epsilon, 0] = np.sum(
           (np.multiply(
                assets_grid_fine,
                histogram_mat[i_epsilon:i_epsilon+1, :])/sum_hist)
            ,1
        )
        grid_moments[i_epsilon, :, 0] = assets_grid_quad - moments_hist[i_epsilon, 0]
        # computes centered higher moments of the form (a'-moment_i)^i-moment_1
        for i_moment in np.arange(1, n_measure):
            moments_hist[i_epsilon, i_moment] = (
                np.sum(
                    np.multiply(
                        np.power(
                            (assets_grid_fine - moments_hist[i_epsilon, 0]), i_moment+1
                        ),
                        histogram_mat[i_epsilon, :]
                    )
                    / (np.sum(histogram_mat[i_epsilon, :],1)),1
                )
                - moments_hist[i_epsilon, i_moment]
            )
            grid_moments[i_epsilon,:,i_moment] = np.power((assets_grid_quad - moments_hist[i_epsilon, 0]),(i_moment+1))- moments_hist[i_epsilon,i_moment]
    constrained = np.array(
        [
            histogram_mat[0, 0] / np.sum(histogram_mat[0, :],1),
            histogram_mat[1, 0] / np.sum(histogram_mat[1, :],1)
        ]
    ).reshape((2, 1), order='F')

    return [moments_hist, grid_moments, constrained]

