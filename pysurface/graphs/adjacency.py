import networkx as nx
import numpy as np


class SurfaceAdjacency(object):
    """
    Class to generate an adjancey list  of a surface mesh representation
    of the brain.

    Initialize SurfaceAdjacency object.

    Parameters:
    - - - - -
    vertices : array
        vertex coordinates
    faces : list
        list of faces in surface
    """

    def __init__(self, faces):

        self.faces = faces

    def generate(self, subset=False, indices=[]):

        """
        Compute adjacency matrix of surface mesh

        Parameters:
        - - - - -
        F: int, array
            mesh triangles

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


        G = nx.from_scipy_sparse_matrix(A)

        if subset:
            nodes = indices
        else:
            nodes = G.nodes

        adj = {node: [] for node in nodes}
        for node in nodes:
            for j in G.neighbors(n=node):
                adj[node].append(j)
            adj[node].sort()
        
        self.adj = adj


    @staticmethod
    def filtration(adj, filter_indices, toArray=False, remap=False):
        """
        Generate a local adjacency list, constrained to a subset of vertices on
        the surface.  For each vertex in 'vertices', retain neighbors
        only if they also exist in 'vertices'.

        Parameters:
        - - - - -
        adj : dictionary
            adjacency list to filter
        fitler_indices : array
            indices to include in sub-graph.  If none, returns original graph.
        to_array : bool
            return adjacency matrix of filter_indices
        remap : bool
            remap indices to 0-len(filter_indices)
        Returns:
        - - - -
        G : array / dictionary
            down-sampled adjacency list / matrix
        """

        accepted = np.zeros((len(adj.keys()),))
        accepted[filter_indices] = True

        filter_indices = np.sort(filter_indices)

        G = {}.fromkeys(filter_indices)

        for v in filter_indices:
            neighbors = adj[v]
            neighbors = [n for n in neighbors if n in filter_indices]
            G[v] = list(set(adj[v]).intersection(set(filter_indices)))

        ind2sort = dict(zip(
            filter_indices,
            np.arange(len(filter_indices))))

        if remap:
            remapped = {
                ind2sort[fi]: [ind2sort[nb] for nb in G[fi]]
                for fi in filter_indices}

            G = remapped

        if toArray:
            G = nx.from_dict_of_lists(G)
            nodes = G.nodes()
            nodes = np.argsort(nodes)
            G = nx.to_numpy_array(G)
            G = G[nodes, :][:, nodes]

        return G