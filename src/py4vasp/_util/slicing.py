# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

INDICES = {"a": 0, "b": 1, "c": 2}


def plane(cell, cut):
    """Takes a 2d slice of a 3d cell and projects it onto 2d coordinates.

    For simplicity in the documentation, we will assume that the cut is in the plane of
    the a and b lattice vector. We use Greek letter α and β to refer to the vectors in
    the 2d coordinate system. This routine computes the transformation Q such that

    .. math::

        \alpha &= Q a \\
        \beta  &= Q b

    Here, the matrix Q has a 2 × 3 shape so that the vectors α and β have only 2
    dimensions. Furthermore, we want that the matrix Q fulfills certain properties,
    namely that the lengths of the vectors and there angle is not changed. If the
    vectors lie within e.g. the x-y plane, α and β should be the same as a and b just
    with the zero for the z component removed.

    Parameters
    ----------
    cell : np.ndarray
        A 3 × 3 array defining the three lattice vectors of the unit cell.
    cut : str
        Either "a", "b", or "c" to define which lattice vector is removed to get the slice.

    Returns
    -------
    np.ndarray
        A 2 × 2 array defining the two lattice vectors spanning the plane.
    """
    vectors = np.delete(cell, INDICES[cut], axis=0)
    axis = np.cross(*vectors).astype(np.float_)
    axis /= np.linalg.norm(axis)
    closest_cartesian_axis = np.argmax(np.abs(axis))
    cartesian_axis = np.zeros(3)
    cartesian_axis[closest_cartesian_axis] = np.sign(axis[closest_cartesian_axis])
    rotation_matrix = _calculate_rotation_matrix((axis, cartesian_axis))
    new_vectors = vectors @ rotation_matrix.T
    return np.delete(new_vectors, closest_cartesian_axis, axis=1)


def _calculate_rotation_matrix(vectors):
    cos_angle = np.dot(*vectors)
    v = np.cross(*vectors)
    if np.linalg.norm(v) < 1e-10:
        return np.eye(3)
    V = np.cross(np.eye(3), v)
    return np.eye(3) + V + V @ V / (1 + cos_angle)
