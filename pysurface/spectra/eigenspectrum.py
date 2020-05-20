import numpy as np
from scipy import sparse

def spectrum(L, k=6):

    """
    Compute the smallest k eigenvalues and corresponding eigenvectors
    of the graph laplacian.

    Parameters:
    - - - - - 
    L: float, array
        sparse laplacian matrix
    k: int
        number of eigenvectors / values to compute

    Returns:
    - - - -
    E: float, array
        eigenvectors
    Lambda: float, array
        eigenvalues
    """

    [Lambda, E] = sparse.linalg.eigs(L, k=k, which='SM')
    Lambda = np.real(Lambda)
    E = np.real(E)

    # ensure that eigenvalues and vectors are sorted in ascending order
    idx = np.argsort(Lambda)
    Lambda = Lambda[idx]
    E = E[:, idx]

    # scale eigenvectors by inverse sqare root of eigenvales
    E[:,1:] = np.dot(E[:, 1:], np.diag(Lambda[1:]**(-0.5)))

    signf = 1-2*(E[0,:]<0)
    E = E*signf[None, :]

    return [E, Lambda]