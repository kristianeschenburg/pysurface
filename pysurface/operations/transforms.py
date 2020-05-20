import numpy as np

def cart2sphere(x, y, z):

    """
    Convert 3d cartesian coordinates to spherical coordinates.

    Parameters:
    - - - - -
    x: int
        x coordinate
    y: int
        y coordinate
    z: int
        z coordinate

    Returns:
    - - - -
    theta: float
        azimuthal angle
    phi: float
        polar angle
    """

    vector = np.asarray([x, y, z])
    x, y, z = vector / np.sqrt((vector*vector).sum(1))[:, None]

    theta = np.arccos(z)
    phi = np.arcsin(y / np.sin(theta))

    return theta, phi
