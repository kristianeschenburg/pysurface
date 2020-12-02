import numpy as np
from scipy import sparse
from .. import graphs


def normal(X, F, A=None):
    """
    Compute normals at each vertex.

    Parameters:
    - - - - -
    X: float, array
    vertex coordinates (m, 3)
    F: int, array
    triangles of mesh
    A: int, sparse array
    adjacency matrix of surface mesh

    Returns:
    - - - -
    N: float, array
    normal vectors at each vertex in mesh
    """

    n = X.shape[0]
    eps = 1e-6

    # generate adjacency matrix
    if not A:
        A = adjacency(F)

    # compute vertex degree
    D = np.asarray(A.sum(1)).squeeze()

    # compute normals of each face
    Nf = ncrossp(X[F[:, 1], :]-X[F[:, 0], :],
                X[F[:, 2], :]-X[F[:, 0], :])

    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    cols = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])

    d0 = np.concatenate([Nf[:, 0], Nf[:, 0], Nf[:, 0]])
    d1 = np.concatenate([Nf[:, 1], Nf[:, 1], Nf[:, 1]])
    d2 = np.concatenate([Nf[:, 2], Nf[:, 2], Nf[:, 2]])

    N = np.zeros((n, 3))
    N[:, 0] = sparse.csr_matrix((d0, (rows, cols)), shape=(n, n)).diagonal()
    N[:, 1] = sparse.csr_matrix((d1, (rows, cols)), shape=(n, n)).diagonal()
    N[:, 2] = sparse.csr_matrix((d2, (rows, cols)), shape=(n, n)).diagonal()
    N = N / D[:, None]

    dnorm = np.sqrt((N**2).sum(1))
    dnorm[dnorm < eps] = 1
    N = N / dnorm[:, None]

    return N

def ncrossp(x, y):
    """
    Compute cross product.

    Parameters:
    - - - - -
    x, y: float, array
    3-dimensional (m, 3)
    """

    assert x.shape[1] == 3
    assert y.shape[1] == 3

    eps = 1e-6

    xnorm = np.sqrt((x**2).sum(1))
    xnorm[xnorm < eps] = 1
    x = x/xnorm[:, None]

    ynorm = np.sqrt((y**2).sum(1))
    ynorm[ynorm < eps] = 1
    y = y/ynorm[:, None]

    z = np.zeros((x.shape))
    z[:, 0] = x[:, 1]*y[:, 2] - x[:, 2]*y[:, 1]
    z[:, 1] = x[:, 2]*y[:, 0] - x[:, 0]*y[:, 2]
    z[:, 2] = x[:, 0]*y[:, 1] - x[:, 1]*y[:, 0]

    znorm = np.sqrt((z**2).sum(1))
    znorm[znorm < eps] = 1
    z = z / znorm[:, None]

    return z

def adjacency(F):

    """
    Compute adjacency matrix of surface mesh

    Parameters:
    - - - - -
    F: int, array
    mesh triangles

    Returns:
    - - - -
    A: int, sparse array
    adjacency matrix of surface mesh
    """

    n = F.max()+1

    rows = np.concatenate([F[:, 0], F[:, 0],
                        F[:, 1], F[:, 1], 
                        F[:, 2], F[:, 2]])

    cols = np.concatenate([F[:, 1], F[:, 2], 
                        F[:, 0], F[:, 2], 
                        F[:, 0], F[:, 1]])

    combos = np.column_stack([rows, cols])

    [_, idx] = np.unique(combos, axis=0, return_index=True)
    A = sparse.csr_matrix((np.ones(len(idx)), (combos[idx, 0], combos[idx, 1])), shape=(n, n))

    return A

def weightedadjacency(X, F, inverse=False):
    """
    Compute weighted adjacency matrix.

    Parameters:
    - - - - - -
    X: float, array
    vertex coordinates
    F: int, array
    mesh triangles

    Returns:
    - - - -
    W: float, sparse array
    weighted adjacency matrix
    """

    n = F.max()+1

    # Compute weights for all links (euclidean distance)
    weights = np.sqrt(np.concatenate([((X[F[:, 0], :]-X[F[:, 1], :])**2).sum(1),
                            ((X[F[:, 0], :]-X[F[:, 2], :])**2).sum(1),
                            ((X[F[:, 1], :]-X[F[:, 0], :])**2).sum(1),
                            ((X[F[:, 1], :]-X[F[:, 2], :])**2).sum(1),
                            ((X[F[:, 2], :]-X[F[:, 0], :])**2).sum(1),
                            ((X[F[:, 2], :]-X[F[:, 1], :])**2).sum(1)]))

    # penalize small distances (avoid division by zero)
    eps = 1e-6

    if inverse:
        weights = 1/weights

    weights = (weights + eps)**(-1)

    # remove duplicated edges
    rows = np.concatenate([F[:, 0], F[:, 0],
                        F[:, 1], F[:, 1],
                        F[:, 2], F[:, 2]])

    cols = np.concatenate([F[:, 1], F[:, 2],
                        F[:, 0], F[:, 2],
                        F[:, 0], F[:, 1]])

    combos = np.column_stack([rows, cols])

    [rc, idx] = np.unique(combos, axis=0, return_index=True)
    weights = weights[idx]

    W = sparse.csr_matrix((weights, (rc[:, 0], rc[:, 1])), shape=(n, n))
    W = (W + W.transpose())/2

    return W

def weightedadjacencynormal(X, F):

    """
    Compute weighted adjacency matrix.

    Parameters:
    - - - - - -
    X: float, array
    vertex coordinates
    F: int, array
    mesh triangles

    Returns:
    - - - -
    W: float, sparse array
    weighted adjacency matrix
    """

    eps = 1e-6
    N = normal(X, F)
    n = X.shape[0]

    # compute weights for all links (euclidean distance)
    wdist = np.sqrt(np.concatenate([((X[F[:, 0], :]-X[F[:, 1], :])**2).sum(1),
                            ((X[F[:, 0], :]-X[F[:, 2], :])**2).sum(1),
                            ((X[F[:, 1], :]-X[F[:, 0], :])**2).sum(1),
                            ((X[F[:, 1], :]-X[F[:, 2], :])**2).sum(1),
                            ((X[F[:, 2], :]-X[F[:, 0], :])**2).sum(1),
                            ((X[F[:, 2], :]-X[F[:, 1], :])**2).sum(1)]))

    # compute weights for all links (euclidean distance)
    wnormal = np.sqrt(np.concatenate([((N[F[:, 0], :]-N[F[:, 1], :])**2).sum(1),
                            ((N[F[:, 0], :]-N[F[:, 2], :])**2).sum(1),
                            ((N[F[:, 1], :]-N[F[:, 0], :])**2).sum(1),
                            ((N[F[:, 1], :]-N[F[:, 2], :])**2).sum(1),
                            ((N[F[:, 2], :]-N[F[:, 0], :])**2).sum(1),
                            ((N[F[:, 2], :]-N[F[:, 1], :])**2).sum(1)]))

    wdist   /= wdist.mean()
    wnormal /= wnormal.mean()
    weights = (wdist + wnormal + eps)**(-1)

    rows = np.concatenate([F[:, 0], F[:, 0], 
                        F[:, 1], F[:, 1], 
                        F[:, 2], F[:, 2]])

    cols = np.concatenate([F[:, 1], F[:, 2], 
                        F[:, 0], F[:, 2], 
                        F[:, 0], F[:, 1]])

    combos = np.column_stack([rows, cols])

    [rc, idx] = np.unique(combos, axis=0, return_index=True)
    weights = weights[idx]

    W = sparse.csr_matrix((weights, (rc[:, 0], rc[:, 1])), shape=(n, n))

    return W
