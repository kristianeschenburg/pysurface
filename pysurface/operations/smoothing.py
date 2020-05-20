import numpy as np
from nibabel import gifti


class LaplacianSmoothing(object):
    """
    Perform Laplacian smoothing on a surface mesh.

    Parameters:
    ------------

    vertices: array
        3d coordinates of surface mesh

    faces: array
        list of faces of surface mesh

    adj_list dictionary
        adjacency list of surface mesh

    itererations: int
        number of iterations of Laplacian Smoothing
    """

    def __init__(self, vertices, faces, itererations, adj_list):

        # Initialize vertex, face, and epsilon fields
        self.vertices = vertices
        self.faces = faces
        self.adj_list = adj_list
        self.iter = itererations

    # PRIVATE
    # Compute a single iteration of Laplacian smoothing
    def fit(self, adj_list):

        verts, adj_list = self.vertices, self.adj_list
        smoothed = np.zeros(np.shape(verts))

        for k in range(0, np.shape(verts)[0]):

            coords = list(adj_list[k])

            smoothed[k, :] = verts[coords, :].mean(0)

        self.vertices = smoothed

    # Perform .__smooth for iter rounds of smoothing
    def _laplacian(self):

        iter = self.iter

        for k in range(0, iter):

            self.fit()

    # Write new surface to file specified by outName
    def write_surface(self, vertices, outName):

        self.surface.darrays[0].data = vertices.astype(np.float32)
        gifti.giftiio.write(self.surface, outName)
