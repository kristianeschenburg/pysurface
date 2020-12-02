import numpy as np
from .adjacency import SurfaceAdjacency

class Graph(object):

    """
    Graph class with functionality to compute basis characteristics about the graph, including
    vertex and triangle normal vectors, and triangle area.
    """

    def __init__(self, vertices, triangles):

        """
        Parameters:
        - - - - -
        vertices: N x 3, float array
            spatial coordinates of vertices
        triangles: T x 3, int array
            vertex IDs of each triangle
        """

        assert triangles.max() <= vertices.shape[0], 'Triangles cannot include non-existent vertices.'

        self.vertices = vertices
        self.triangles = triangles
        self.n = self.vertices.shape[0]


    def vertex_adjacency(self):

        """
        Compute the vertex adjacency list.

        Returns:
        - - - -
        v_adj: dictionary of lists
            each key is a vertex ID and each value is a list of verices
        """

        S = SurfaceAdjacency(faces=self.triangles)
        S.generate()

        v_adj = S.adj

        return v_adj
    
    def triangle_membership(self):
    
        """
        For each vertex, compute which triangles it participates in.

        Returns:
        - - - -
        t_adj: dictionary of lists
            each key is a vertex ID and each value is a list of triangle indices
        """
        
        t_adj = {k: [] for k in np.arange(self.triangles.max()+1)}
        for j, t in enumerate(self.triangles):
            for i in t:
                t_adj[i].append(j)
        
        return t_adj

    def normals(self, kind='vertex'):

        """
        
        """

        if kind == 'vertex':

            N = self._vertex_normals()

        elif kind == 'triangle':

            N = self._triangle_normals()
        
        return N
    
    def _triangle_normals(self):
    
        """
        Compute the normal vertex for each triangle.

        Returns:
        - - - -
        normals: T x 3, float array
            each row is the normal vector for a single triangle
        
        """
    
        normals = np.zeros((self.triangles.shape[0], 3))
        components = self.vertices[self.triangles]
        
        for i, c in enumerate(components):
            d0 = c[0] - c[1]
            d1 = c[0] - c[2]
            
            n = np.cross(d0, d1)
            normals[i] = n/np.linalg.norm(n)
        
        return normals

    def _vertex_normals(self):

        """
        Compute the normal vector for each vertex.  Each normal
        is the averaged of each triangle to which a vertex belongs
        to -- the contribution of each triangle normal is weighted 
        by the area of each triangle.

        Returns:
        - - - -
        normals: N x 3, float array
            each row is the normal vector for a single vertex
        """

        t_norms = self.normals(kind='triangle')
        t_areas = self.areas()
        t_adj = self.triangle_membership()
        
        v_norms = np.zeros((32492, 3))

        for i in np.arange(self.vertices.shape[0]):
            
            v_norm = (t_areas[t_adj[i]]/t_areas[t_adj[i]].sum())[:, None]*t_norms[t_adj[i]]
            v_norm = v_norm.mean(0)
            v_norm = v_norm / np.linalg.norm(v_norm)
            v_norms[i] = v_norm
            
        return v_norms

    def areas(self):
    
        """
        Compute the surface area of each triangle.

        Returns:
        - - - -
        areas: T x 1, float array
            each row is the area of a triangle
        """
        
        components = self.vertices[self.triangles]
        
        areas = np.zeros((self.triangles.shape[0],))
        
        for i in np.arange(areas.shape[0]):
            areas[i] = self._area(components[i])
        
        return areas

    def _area(self, vertices):
    
        """
        Compute the area of a single triangle.

        Parameters:
        - - - - -
        vertices: 3 x 3, float array
            each row are the spatial coordinate of a single vertex

        Returns:
        - - - -
        area: float
            area of the triangle
        """
        
        PQ = vertices[0] - vertices[1]
        PR = vertices[0] - vertices[2]
        
        return 0.5*np.linalg.norm(np.cross(PQ, PR))