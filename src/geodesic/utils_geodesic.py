import numpy as np
import scipy.stats as stat

from pyriemann.utils.tangentspace import exp_map_riemann, log_map_riemann

def sample_opposite_matrices(dim):
    """
    Generates two matrices with opposite eigenvalue distributions.

    This function generates two random orthogonal matrices and uses them to
    create two matrices with eigenvalues that are oppositely distributed.
    The first half of the eigenvalues are sampled from a uniform distribution
    in the range [3, 4) and the second half from a uniform distribution in
    the range [0, 1). The second matrix has the same eigenvalues but in
    reversed order.

    Parameters:
    dim (int): The dimension of the square matrices to be generated.

    Returns:
    tuple: A tuple containing two numpy arrays (M1, M2) which are the generated matrices.
    """

    U1 = stat.ortho_group.rvs(dim=dim)
    lamb1 = stat.uniform.rvs(size=dim // 2, loc=3, scale=1)
    lamb2 = stat.uniform.rvs(size=dim - dim // 2, loc=0, scale=1)
    M1 = U1.T @ np.diag(np.concatenate((lamb1, lamb2))) @ U1

    # U2 = stat.ortho_group.rvs(dim = dim)
    lamb1_ = stat.uniform.rvs(size=dim // 2, loc=3, scale=1)
    lamb2_ = stat.uniform.rvs(size=dim - dim // 2, loc=0, scale=1)
    M2 = U1.T @ np.diag(np.concatenate((lamb2_, lamb1_))) @ U1

    return M1, M2


def sample_geodesic_points(M1, M2, n):
    """
    Samples points along the geodesic between two points on a Riemannian manifold.
    Parameters:
    M1 (array-like): The starting point on the manifold.
    M2 (array-like): The ending point on the manifold.
    n (int): The number of points to sample along the geodesic.
    Returns:
    np.ndarray: An array of sampled points along the geodesic from M1 to M2.
    """

    all_t = np.linspace(0, 1, n)
    v = log_map_riemann(M1, M2, C12=True)
    return np.array([exp_map_riemann(t * v, M1, Cm12=True) for t in all_t])
