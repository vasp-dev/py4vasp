# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np

from py4vasp._calculation.structure import Structure
from py4vasp._third_party.graph import Contour, Graph
from py4vasp._third_party.view import GridQuantity, View
from py4vasp._util import documentation, index, slicing
from py4vasp.exception import IncorrectUsage

_DATA_DICT_PARAMETER = """\
data_dict : dict[str, Any]
    A dictionary containing labels as keys and the data as values (if required, already 
    transposed and reshaped) for all selections, in the form of 
    `{make_label(selection): get_data(selector, selection) for selection in selections}`.
"""
_SLICE_ARGS_PARAMETER = """\
slice_args : SliceArguments
    A wrapper around a, b, c, normal, and supercell that also allows to compute the
    plane at the specified cut.
"""


@dataclass
class SliceArguments:
    """Helper dataclass for storing a, b, c (the specifications for slicing along which vector),
    normal (how the plane should be aligned), and supercell (expanding the lattice).

    Define like so:
    >>> slice_args = SliceArguments(a=1.3)

    Note that exactly one of a, b, c must be given and not None, otherwise,
    py4vasp will raise an IncorrectUsage exception.
    """

    a: Optional[float] = None
    """You must select exactly one of a,b,c to specify which of the three lattice
    vectors you want to remove to form a plane. The assigned value represents
    the fractional length along this lattice vector, so `a = 0.3` will remove
    the first lattice vector and then take the grid points at 30% of the length
    of the first vector in the b-c plane. The fractional height uses periodic
    boundary conditions."""
    b: Optional[float] = None
    """You must select exactly one of a,b,c to specify which of the three lattice
    vectors you want to remove to form a plane. The assigned value represents
    the fractional length along this lattice vector, so `b = 0.3` will remove
    the second lattice vector and then take the grid points at 30% of the length
    of the second vector in the a-c plane. The fractional height uses periodic
    boundary conditions."""
    c: Optional[float] = None
    """You must select exactly one of a,b,c to specify which of the three lattice
    vectors you want to remove to form a plane. The assigned value represents
    the fractional length along this lattice vector, so `c = 0.3` will remove
    the third lattice vector and then take the grid points at 30% of the length
    of the third vector in the a-b plane. The fractional height uses periodic
    boundary conditions."""
    normal: Optional[str] = None
    """If not set or None, py4vasp will align the first remaining lattice vector with the
    x-axis and the second one such that the angle between the lattice vectors
    is preserved. You can set it to "x", "y", or "z"; then py4vasp will rotate
    the plane in such a way that the normal direction aligns with the specified
    Cartesian axis. This may look better if the normal direction is close to a
    Cartesian axis. You may also set it to "auto" so that py4vasp chooses a
    close Cartesian axis if it can find any."""
    supercell: Optional[Union[int, tuple]] = None
    """Stored here to replicate the lattice periodically a given number of times
    when passed to Visualizer. If you provide two different numbers, the resulting 
    cell will be the two remaining lattice vectors multiplied by the specific number."""

    def __post_init__(self):
        """__post_init__ is a special @dataclass function, called automatically
        after __init__. Here, we check if a, b, c are valid."""
        # Count how many of a, b, c are not None
        provided = [x is not None for x in (self.a, self.b, self.c)].count(True)
        if provided != 1:
            raise IncorrectUsage(
                "Exactly one of a, b, or c must be provided (not None)."
            )

    def slice_plane(self, lattice_vectors) -> tuple[slicing.Plane, float]:
        """Cuts along a fraction of the selected vector (`a`, `b`, or `c`)
        of the `lattice_vectors` argument, then defines the plane at that
        fraction spanned by the remaining two lattice vectors and rotates
        that plane according to `normal`.

        Parameters
        ----------
        lattice_vectors : np.ndarray
            A 3 × 3 array defining the three lattice vectors of the unit cell.

        Returns
        -------
        tuple
            slicing.Plane
                A 2d representation of the plane with some information to transform data to it.
            float
                The fraction at which the plane is defined, in the direction of `a`, `b`, or `c`.
        """
        cut, fraction = slicing.get_cut(self.a, self.b, self.c)
        plane = slicing.plane(lattice_vectors, cut, self.normal)
        return plane, fraction


class Visualizer:
    def __init__(self, structure: Structure):
        self._structure = structure

    @documentation.format(data_dict=_DATA_DICT_PARAMETER)
    def to_view(
        self, data_dict: dict[str, Any], supercell: Union[int, np.ndarray, None] = 1
    ) -> View:
        """Constructs a `View` object and manages setting of `grid_scalars`
        with data and labels.

        Parameters
        ----------
        {data_dict}

        supercell : Union[int, np.ndarray, None]
            Replicate the plot periodically a given number of times. If you
            provide two different numbers, the resulting cell will be the two remaining
            lattice vectors multiplied by the specific number.

        Returns
        -------
        View
            A visualization of the quantity within the crystal structure.
        """
        viewer = self._structure.plot(supercell)
        viewer.grid_scalars = [
            GridQuantity(
                data,
                label=label if label else "",
            )
            for label, data in data_dict.items()
        ]
        return viewer

    @documentation.format(
        data_dict=_DATA_DICT_PARAMETER,
        slice_args=_SLICE_ARGS_PARAMETER,
    )
    def to_contour(
        self,
        data_dict: dict[str, Any],
        slice_args: SliceArguments,
        isolevels: bool = True,
    ) -> Graph:
        """Constructs a `Graph` object and manages slicing and constructing
        the contours.

        Parameters
        ----------
        {data_dict}

        {slice_args}

        isolevels : bool
            Defines whether isolevels should be added or a heatmap is used.

        Returns
        -------
        Graph
            A contour plot in the plane spanned by the 2 remaining lattice vectors.
        """
        plane, fraction = slice_args.slice_plane(self._structure.lattice_vectors())
        contours = [
            self._make_contour(
                data, label, plane, fraction, slice_args.supercell, isolevels
            )
            for label, data in data_dict.items()
        ]
        return Graph(contours)

    def _make_contour(self, data, label, plane, fraction, supercell, isolevels):
        """Helper function for creating and modifying a `Contour` object.

        Parameters
        ----------
        data : Any
            The unsliced data to be represented.

        label : str
            The label for the data.

        plane : slicing.Plane
            The plane along which to slice the data.

        fraction : float
            The fraction at which the slice is taken along the chosen lattice vector.

        supercell : Union[int, np.ndarray, None]
            The specification of the supercell for duplicating the lattice.

        isolevels : bool
            Defines whether isolevels should be added or a heatmap is used.

        Returns
        -------
        Contour
            Represents data on a 2d slice through the unit cell.
        """
        contour = Contour(
            slicing.grid_scalar(data, plane, fraction),
            plane,
            label=label if label else "",
            isolevels=isolevels,
        )
        if supercell is not None:
            contour.supercell = np.ones(2, dtype=np.int_) * supercell
        return contour

    @documentation.format(
        data_dict=_DATA_DICT_PARAMETER,
        slice_args=_SLICE_ARGS_PARAMETER,
    )
    def to_quiver(
        self,
        data_dict: dict[str, Any],
        slice_args: SliceArguments,
    ) -> Graph:
        """Constructs a `Graph` object and manages slicing and constructing
        the quivers.

        Parameters
        ----------
        {data_dict}

        {slice_args}

        Returns
        -------
        Graph
            A quiver plot in the plane spanned by the 2 remaining lattice vectors.
        """
        plane, fraction = slice_args.slice_plane(self._structure.lattice_vectors())
        quivers = [
            self._make_quiver(data, label, plane, fraction, slice_args.supercell)
            for label, data in data_dict.items()
        ]
        return Graph(quivers)

    def _make_quiver(self, sel, label, plane, fraction, supercell):
        """Helper function for creating and modifying a `Contour` object for quivers.

        Parameters
        ----------
        sel : Any
            The unsliced data to be represented.

        label : str
            The label for the data.

        plane : slicing.Plane
            The plane along which to slice the data.

        fraction : float
            The fraction at which the slice is taken along the chosen lattice vector.

        supercell : Union[int, np.ndarray, None]
            The specification of the supercell for duplicating the lattice.

        Returns
        -------
        Contour
            Represents data on a 2d slice through the unit cell.
        """
        if sel.ndim == 3 or len(sel) == 1:
            data = slicing.grid_scalar(sel, plane, fraction)
            data = np.array((np.zeros_like(data), data))
        else:
            data = slicing.grid_vector(sel, plane, fraction)
        contour = Contour(
            data,
            plane,
            label=label if label else "",
            isolevels=True,
        )
        if supercell is not None:
            contour.supercell = np.ones(2, dtype=np.int_) * supercell
        return contour
