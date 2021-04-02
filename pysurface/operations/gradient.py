import numpy as np

class Gradient(object):

    """
    Class to compute gradients of a scalar field distributed over a regular mesh.
    """

    def __init__(self, graph):

        """
        Parameters:
        - - - - -
        graph: pysurface Graph object
        field: scalar field
        """

        self.graph = graph

    def fit(self, field, mask=None):

        """
        Compute the gradient vectors, and normalized gradient magnitudes of the scalar field.
        This function iterates over vertices in the mesh, and compute the gradient vector for each.

        Parameters:
        - - - - -
        mask: float, array
            binary array indicating which indices to compute the gradient at
        """

        N = self.graph.n

        if not hasattr(self, 'v_adj'):
            print('Does not yet have vertex adjacency list.')
            self.v_adj = self.graph.vertex_adjacency()
        
        if not hasattr(self, 'v_norm'):
            print('Does not yet have vertex normals')
            self.v_norm = self.graph.normals(kind='vertex')

        # if mask is supplied, make sure the domain of the mask
        # is the same as that of the field
        if np.any(mask):
            assert mask.shape == field.shape

        if np.any(mask):

            bins = np.where(mask)[0]
            v_adj = {k: None for k in bins}

            for key in bins:
                values = self.v_adj[key]
                v_adj[key] = list(set(bins) & set(values))
        else:
            bins = np.arange(self.graph.n)
            v_adj = self.v_adj
        
        v_norm = self.v_norm

        gradients = np.zeros((self.graph.n, 3))

        for i in bins:
            
            # compute difference vector vertex of interest and its neighbors
            pos_diff = self.graph.vertices[i] - self.graph.vertices[v_adj[i]]
            X = np.row_stack([[0,0,0], pos_diff])

            # compute difference in field between vertex of interest and its neighbors
            field_diff = field[i] - field[v_adj[i]]
            Y = np.asarray([0] + list(field_diff))

            # compute gradient at vertex
            try:
                g_vec = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
            except np.linalg.LinAlgError:
                gradients[i] = np.nan
            else:
                
                # compute tangent plane of normal vector
                tangent = null_projector(v_norm[i])
                # project gradient vector onto tangent plane
                g_vec = tangent.dot(g_vec)
                # save projected gradient
                gradients[i] = g_vec
        
        magnitude = np.linalg.norm(gradients, axis=1)

        self.magnitude = magnitude
        self.gradients = gradients


def column_projector(V):

    """
    Compute the orthogonal projection matrix onto the column space of a vertex.

    Parameters:
    - - - - -
    V: float, array
        spatial coordinates of a vertex
    """

    if V.ndim == 1:
        V=V[:,None]
    
    P = V.dot(np.linalg.inv(V.T.dot(V))).dot(V.T)

    return P

def null_projector(V):

    """
    Compute the orthogonal projection matrix onto the null space of a vertex.

    Parameters:
    - - - - -
    V: float, array
        spatial coordinates of a vertex
    """

    P = column_projector(V)
    Q = np.eye(P.shape[0]) - P

    return Q

class Angle(object):

    """
    Class to compute the angles between two sets of gradient vectors.
    """

    def __init__(self):

        pass


    def fit(self, g1, g2):

        """
        
        """

        n1 = np.linalg.norm(g1, axis=1)
        n2 = np.linalg.norm(g2, axis=1)

        g1 = g1 / n1[:,None]
        g2 = g2 / n2[:,None]

        inner = (g1*g2).sum(1)
        radians = np.arccos(inner)
        degrees = np.rad2deg(radians)

        self.degrees = degrees



