import numpy as np
from nibabel import gifti
from numpy.random import uniform


class SampleMesh(object):
    """
    Sample points uniformly from a triangular 2-simplices.

    Parameters:
    - - - - -
    vertices: array
        3d coordinates of vertices in mesh

    face: array
        list of faces in mesh

    n_samples: int
        number of coordinates to sample from each face

    """

    def __init__(self, vertices, faces, n_samples):

        """
        Initializes SampleMesh class

        surface = Gifti surface file
        nS = number of samples to generate from each face on surface
        """

        # Initialize vertex, face, and epsilon fields
        self.vertices = vertices
        self.faces = faces
        self.n_samples = n_samples

    def face_areas(self):

        """
        Calculates the area of each face
        """

        verts, faces = self.vertices, self.faces

        e1 = verts[faces[:, 0], :] - verts[faces[:, 1], :]
        e2 = verts[faces[:, 0], :] - verts[faces[:, 2], :]

        temp = np.cross(e1, e2, axis=1)

        areas = (0.5)*np.linalg.norm(temp, axis=1)

        self.areas = areas

    def sample_faces(self):

        verts, faces, nS = self.vertices, self.faces, self.n_samples

        # 3D array of size (ns) x (3) x (num faces)
        samples = np.zeros((nS, 3, np.shape(faces)[0]))

        for k in range(0, np.shape(faces)[0]):

            v1 = verts[faces[k, 0], :]
            v2 = verts[faces[k, 1], :]
            v3 = verts[faces[k, 2], :]

            samples[:, :, k] = self.simplex_sample(v1, v2, v3, nS)

        return samples

    def simplex_sample(self, v1, v2, v3, n_samples):

        r = np.random.uniform(0, 1, n_samples*2)
        r = (np.tile(r, [3, 1])).T

        r1 = r[:n_samples, :]
        r2 = r[n_samples:, :]

        sample_face = (1-np.sqrt(r1))*v1 + np.sqrt(r1)*(1-r2)*v2 + np.sqrt(r1)*r2*v3

        return sample_face
