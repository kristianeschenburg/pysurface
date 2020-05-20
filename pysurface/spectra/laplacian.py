import numpy as np
from scipy import sparse

def degree(W):
    """
    Compute the weighted degree of each vertex in a surface mesh.

    Parameters:
    - - - - -
    W: float, sparse array
        weighted adjacency matrix of surface mesh

    Returns:
    - - - -
    D: float, array
        degree of each vertex
    Dinv: float, array
        inverse degree of each vertex
    """

    n = W.shape[0]
    D = sparse.spdiags(W.sum(1).squeeze(), diags=0, m=n, n=n)
    Dinv = sparse.spdiags(1/W.sum(1).squeeze(), diags=0, m=n, n=n)

    return [D, Dinv]

def generallaplacian(W, G=None):

    """
    Compute the general laplacian of the surface mesh.

    Parameters:
    - - - - -
    W: float, array
        weight adjacency matrix

    Returns:
    - - - -
    L: float, sparse array
        surface mesh Laplacian matrix
    """

    [D, Dinv] = degree(W)
    n = D.shape[0]

    if not G:
        G = D
        Ginv = Dinv
    
    L = np.dot(Ginv, (D - W))

    return L

def weightlaplacian(L, T):

    """
    Weight laplacian matrix using surface features.

    Parameters:
    - - - - -
    L: float, array
        sparse laplacian matrix
    T: float, array
        scalar surface map (sulcal depth, cortical thickness, myelin density)

    Returns:
    - - - -
    L: float, sparse array
        surface mesh Laplacian matrix, weighted by surface scalar map
    """

    n = L.shape[0]
    a = 1
    G = sparse.spdiags(np.exp(T*a) - np.exp(T*a).min() + 1e-2, diags=0, m=n, n=n)
    L = np.dot(L, G)

    return L
