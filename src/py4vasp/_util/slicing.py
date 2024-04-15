# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception

INDICES = {"a": 0, "b": 1, "c": 2}
AXIS = ("x", "y", "z")


def plane(cell, cut, normal="auto"):
    """Takes a 2d slice of a 3d cell and projects it onto 2d coordinates.

    For simplicity in the documentation, we will assume that the cut is in the plane of
    the a and b lattice vector. We use Greek letter α and β to refer to the vectors in
    the 2d coordinate system. This routine computes the transformation Q such that

    .. math::

        \alpha &= Q a \\
        \beta  &= Q b

    Here, the matrix Q has a 2 × 3 shape so that the vectors α and β have only 2
    dimensions. Furthermore, we want that the matrix Q fulfills certain properties,
    namely that the lengths of the vectors and their angle is not changed. If the
    vectors lie within e.g. the x-y plane, α and β should be the same as a and b just
    with the zero for the z component removed. However, to avoid irregular behavior
    where small changes to the unit cell cause large changes to the selected vectors
    this mode only works if the normal vector is close (~15°) to a Cartesian axis.
    Otherwise the user needs to provide the axis with the *normal* argument.
    Alternatively, one can align the first remaining lattice vector with the x axis by
    passing None for *normal*.

    Parameters
    ----------
    cell : np.ndarray
        A 3 × 3 array defining the three lattice vectors of the unit cell.
    cut : str
        Either "a", "b", or "c" to define which lattice vector is removed to get the slice.
    normal : str
        Set the Cartesian direction "x", "y", or "z" parallel to which the normal of
        the plane is rotated. Alteratively, set it to "auto" to rotate to the closest
        Cartesian axis. If you set it to None, the normal will not be considered and
        the first remaining lattice vector will be aligned with the x axis instead.

    Returns
    -------
    np.ndarray
        A 2 × 2 array defining the two lattice vectors spanning the plane.
    """
    _raise_error_if_cut_unknown(cut)
    vectors = np.delete(cell, INDICES[cut], axis=0)
    if normal is not None:
        return _rotate_normal_to_cartesian_axis(vectors, normal)
    else:
        return _rotate_first_vector_to_x_axis(vectors)


def _rotate_first_vector_to_x_axis(vectors):
    u, v = np.linalg.norm(vectors, axis=1)
    x = np.dot(*vectors) / (u * v)
    return np.array([[u, 0], [v * x, v * np.sqrt(1 - x**2)]])


def _rotate_normal_to_cartesian_axis(vectors, normal):
    old_normal = _get_old_normal(vectors)
    index_axis = _get_index_axis(old_normal, normal)
    new_normal = _get_new_normal_from_cartesian_axis(old_normal, index_axis)
    rotation_matrix = _get_rotation_matrix((old_normal, new_normal))
    new_vectors = vectors @ rotation_matrix.T
    return np.delete(new_vectors, index_axis, axis=1)


def _get_old_normal(vectors):
    old_normal = np.cross(*vectors).astype(np.float_)
    return old_normal / np.linalg.norm(old_normal)


def _get_index_axis(old_normal, normal):
    if normal in AXIS:
        return AXIS.index(normal)
    elif normal == "auto":
        index = np.argmax(np.abs(old_normal))
        _raise_error_if_direction_is_not_obvious(np.abs(old_normal[index]))
        return index
    else:
        _raise_unknown_normal_error(normal)

def _get_new_normal_from_cartesian_axis(old_normal, index):
    new_normal = np.zeros(3)
    # do not use sign function because it is 0 if old_normal[index] == 0
    new_normal[index] = 1 if old_normal[index] >= 0 else -1
    return new_normal


def _get_rotation_matrix(vectors):
    cos_angle = np.dot(*vectors)
    v = np.cross(*vectors)
    if np.linalg.norm(v) < 1e-10:
        return np.eye(3)
    V = np.cross(np.eye(3), v)
    return np.eye(3) + V + V @ V / (1 + cos_angle)


def _raise_error_if_cut_unknown(cut):
    if cut in INDICES:
        return
    message = """\
The selected choice {cut} is invalid. Please select one of the three lattice vectors
with passing *cut* ("a", "b", or "c").""".format(
        cut=cut
    )
    raise exception.IncorrectUsage(message)


def _raise_error_if_direction_is_not_obvious(largest_element):
    if largest_element > 0.95:
        return
    message = """\
You did not specify the Cartesian direction to which the normal of the cut plane will
be rotated to. py4vasp tries to determine the axis automatically but in this case no
axis is close to the normal of the plane. Please pass the additional argument *normal*
("x", "y", or "z") to specify to which axis py4vasp should use as normal vector for the
plane."""
    raise exception.IncorrectUsage(message)

def _raise_unknown_normal_error(normal):
    message = """\
The selected normal {normal} is invalid. Please select one of the Cartesian axis ("x",
"y", or "z") or "auto" to specify to which axis the normal is rotated. "auto" will use
a closeby Cartesian axis if possible.""".format(normal=normal)
    raise exception.IncorrectUsage(message)
