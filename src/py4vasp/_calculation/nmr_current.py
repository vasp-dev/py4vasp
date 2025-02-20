# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

from py4vasp import exception
from py4vasp._calculation import base, structure
from py4vasp._third_party import graph
from py4vasp._util import documentation, slicing


class NmrCurrent(base.Refinery, structure.Mixin):
    """The NMR (Nuclear Magnetic Resonance) current refers to the electrical response
    induced in a detection coil by the precessing magnetic moments of nuclear spins in a
    sample. When the nuclei are exposed to a strong external magnetic field and excited
    by a radiofrequency pulse, they generate an oscillating magnetization that induces
    a weak but detectable voltage in the coil. This signal, known as the free induction
    decay (FID), contains information about the chemical environment, molecular
    structure, and dynamics of the sample.
    """

    @base.data_access
    def to_dict(self):
        """Read the NMR current and structural information into a Python dictionary.

        Returns
        -------
        dict
            Contains the NMR current data for all magnetic fields selected in the INCAR
            file as well as structural data.
        """
        return {"structure": self._structure.read(), **self._read_nmr_currents()}

    def _read_nmr_currents(self):
        return dict(self._read_nmr_current(key) for key in self._raw_data)

    def _read_nmr_current(self, key):
        return f"nmr_current_B{key}", self._raw_data[key].nmr_current[:].T

    @base.data_access
    @documentation.format(plane=slicing.PLANE, parameters=slicing.PARAMETERS)
    def to_quiver(
        self, selection=None, *, a=None, b=None, c=None, normal=None, supercell=None
    ):
        """Generate a quiver plot of NMR current.

        {plane}

        For a collinear calculation, the magnetization density will be aligned with the
        y axis of the plane. For noncollinear calculations, the magnetization density
        is projected into the plane.

        Parameters
        ----------
        selection : Direction of the magnetic field for which the NMR current is read.
            If only a single direction is computed, it will default to that direction,
            otherwise to the "z" axis.

        {parameters}

        supercell : int or np.ndarray
            Replicate the contour plot periodically a given number of times. If you
            provide two different numbers, the resulting cell will be the two remaining
            lattice vectors multiplied by the specific number.

        Returns
        -------
        graph
            A quiver plot in the plane spanned by the 2 remaining lattice vectors.

        Examples
        --------

        Cut a plane at the origin of the third lattice vector.

        >>> calc.nmr_current.to_quiver(c=0)

        Replicate a plane in the middle of the second lattice vector 2 times in each
        direction.

        >>> calc.nmr_current.to_quiver(b=0.5, supercell=2)

        Take a slice along the first lattice vector and rotate it such that the normal
        of the plane aligns with the x axis.

        >>> calc.nmr_current.to_quiver(a=0.3, normal="x")
        """
        cut, fraction = self._get_cut(a, b, c)
        plane = slicing.plane(self._structure.lattice_vectors(), cut, normal)
        selection = selection or self._raw_data.valid_indices[-1]
        label, data = self._read_nmr_current(selection)
        sliced_data = slicing.grid_vector(np.moveaxis(data, -1, 0), plane, fraction)
        quiver_plot = graph.Contour(0.003 * sliced_data, plane, label)
        if supercell is not None:
            quiver_plot.supercell = np.ones(2, dtype=np.int_) * supercell
        return graph.Graph([quiver_plot])

    def _get_cut(self, a, b, c):
        _raise_error_cut_selection_incorrect(a, b, c)
        if a is not None:
            return "a", a
        if b is not None:
            return "b", b
        return "c", c


def _raise_error_cut_selection_incorrect(*selections):
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
