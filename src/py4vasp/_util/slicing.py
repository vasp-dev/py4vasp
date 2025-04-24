# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses

import numpy as np

from py4vasp import exception

INDICES = {"a": 0, "b": 1, "c": 2}
AXIS = ("x", "y", "z")
PLANE = """\
You need to specify a plane defined by two of the lattice vectors by selecting
a *cut* along the third one. You must only select a single cut and the value
should correspond to the fractional length along this third lattice vector.
py4vasp will then create a plane from the other two lattice vectors and
generate a contour plot within this plane.

Usually, the first remaining lattice vector is aligned with the x-axis and the
second one such that the angle between the vectors is preserved. You can
overwrite this choice by defining a normal direction. Then py4vasp will rotate
the normal vector of the plane to align with the specified direction. This is
particularly useful if the lattice vector you cut is aligned with a Cartesian
direction.
"""
PARAMETERS = """\
a, b, c : float
    You must select exactly one of these to specify which of the three lattice
    vectors you want to remove to form a plane. The assigned value represents
    the fractional length along this lattice vector, so `a = 0.3` will remove
    the first lattice vector and then take the grid points at 30% of the length
    of the first vector in the b-c plane. The fractional height uses periodic
    boundary conditions.

normal : str or None
    If not set, py4vasp will align the first remaining lattice vector with the
    x-axis and the second one such that the angle between the lattice vectors
    is preserved. You can set it to "x", "y", or "z"; then py4vasp will rotate
    the plane in such a way that the normal direction aligns with the specified
    Cartesian axis. This may look better if the normal direction is close to a
    Cartesian axis. You may also set it to "auto" so that py4vasp chooses a
    close Cartesian axis if it can find any.
"""


@dataclasses.dataclass
class Plane:
    """Defines a plane in 2d produced from cutting the 3d unit cell by removing one
    lattice vector."""

    vectors: np.array
    "2d vectors spanning the plane."
    cell: np.array = None
    "Lattice vectors of the unit cell."
    cut: str = None
    "Lattice vector cut to get the plane, if not set, no labels will be added"


def get_cut(a, b, c):
    raise_error_cut_selection_incorrect(a, b, c)
    if a is not None:
        return "a", a
    if b is not None:
        return "b", b
    return "c", c


def raise_error_cut_selection_incorrect(*selections):
    # only a single element may be selected
    selected_elements = sum(selection is not None for selection in selections)
    if selected_elements == 0:
        raise exception.IncorrectUsage(
            "You have not selected a lattice vector along which the slice should be "
            "constructed. Please set exactly one of the keyword arguments (a, b, c) "
            "to a real number that specifies at which fraction of the lattice vector "
            "the plane is."
        )
    if selected_elements > 1:
        raise exception.IncorrectUsage(
            "You have selected more than a single element. Please use only one of "
            "(a, b, c) and not multiple choices."
        )


def grid_scalar(data, plane, fraction):
    """Takes a 2d slice of a 3d grid data.

    Often, we want to generate a 2d slice of a 3d data on a grid. One example would be
    to visualize a slice through a plane as a contour plot. This routine facilitates
    this task implementing taking the cut of the 3d data.

    Parameters
    ----------
    data : np.ndarray
        3d data on a grid from which a 2d plane is extracted.
    plane : Plane
        Defines the 2d plane to which the data is reduced.
    fraction : float
        Determines which plane of the grid data is used. Periodic boundaries are assumed.

    Returns
    -------
    np.ndarray
        A 2d array where the dimension selected by cut has been removed by selecting
        the plane according to the specificied fraction.
    """
    _raise_error_if_cut_unknown(plane.cut)
    index = INDICES[plane.cut]
    length = data.shape[index]
    slice_ = [slice(None), slice(None), slice(None)]
    slice_[index] = np.round(length * fraction).astype(np.int_) % length
    return data[tuple(slice_)]


def grid_vector(data, plane, fraction):
    """Takes a 2d slice of grid data where every datapoint is a 3d vector.

    Often, we want to generate a 2d slice of data on a grid. One example would be to
    visualize a slice through a plane as a quiver plot. This routine facilitates this
    task implementing taking the cut of the 3d data.

    Parameters
    ----------
    data : np.ndarray
        Data on a grid where every point is a vector. The dimensions should be
        (3d vector, grid_a, grid_b, grid_c) and will be reduced to a 2d vector on a
        2d grid.
    plane : Plane
        Defines the 2d plane to which the data is reduced.
    fraction : float
        Determines which plane of the grid data is used. Periodic boundaries are assumed.

    Returns
    -------
    np.ndarray
        A 3d array where the dimension selected by cut has been removed by selecting
        the plane according to the specificied fraction. The vector is projected onto
        the plane for visualization.
    """
    _raise_error_if_cut_unknown(plane.cut)
    index = INDICES[plane.cut]
    length = data.shape[index + 1]  # add 1 to account for the vector dimension
    slice_ = [slice(None), slice(None), slice(None), slice(None)]
    slice_[index + 1] = np.round(length * fraction).astype(np.int_) % length
    return _project_vectors_to_plane(plane, data[tuple(slice_)])


def _project_vectors_to_plane(plane, data):
    # We want to want to project the vector r onto the plane spanned by the vectors
    # u and v. Let the result be s = a u + b v. We can obtain the projected vector by
    # minimizing the length |r - s| which leads to two conditions
    # (r - a u - b v).u = 0
    # (r - a u - b v).v = 0
    # solving for a and b yields
    # a = (v^2 u.r - u.v v.r) / N
    # b = (u^2 v.r - u.v u.r) / N
    # N = u^2 v^2 - (u.v)^2
    u2 = np.dot(plane.vectors[0], plane.vectors[0])
    v2 = np.dot(plane.vectors[1], plane.vectors[1])
    uv = np.dot(plane.vectors[0], plane.vectors[1])
    vectors = np.delete(plane.cell, INDICES[plane.cut], axis=0)
    ur = np.tensordot(vectors[0], data, axes=(0, 0))
    vr = np.tensordot(vectors[1], data, axes=(0, 0))
    N = u2 * v2 - uv**2
    a = (v2 * ur - uv * vr) / N
    b = (u2 * vr - uv * ur) / N
    au = np.multiply.outer(plane.vectors[0], a)
    bv = np.multiply.outer(plane.vectors[1], b)
    return au + bv


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
    Plane
        A 2d representation of the plane with some information to transform data to it.
    """
    _raise_error_if_cut_unknown(cut)
    vectors = np.delete(cell, INDICES[cut], axis=0)
    if normal is not None:
        return Plane(_rotate_normal_to_cartesian_axis(vectors, normal), cell, cut)
    else:
        return Plane(_rotate_first_vector_to_x_axis(vectors), cell, cut)


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
    old_normal = np.cross(*vectors).astype(np.float64)
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
with passing *cut* ("a", "b", or "c")."""
    raise exception.IncorrectUsage(message.format(cut=cut))


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
a closeby Cartesian axis if possible."""
    raise exception.IncorrectUsage(message.format(normal=normal))
