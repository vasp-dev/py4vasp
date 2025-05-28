from typing import Callable, Mapping

import numpy as np

from py4vasp._calculation.structure import Structure
from py4vasp._third_party.graph import Contour, Graph
from py4vasp._third_party.view import GridQuantity, View
from py4vasp._util import index, slicing


class Visualizer:
    def __init__(self, structure: Structure, mapping: Mapping, label_function: Callable=None):
        self._mapping = mapping
        self._structure = structure
        if label_function is not None:
            self._label = label_function
        else:
            self._label = getattr(mapping, "label", (lambda _: ""))

    def to_view(self, selections, supercell=1) -> View:
        viewer = self._structure.plot(supercell)
        viewer.grid_scalars = [
            GridQuantity(
                (self._mapping[selection].T)[np.newaxis],
                label=self._label(selection)
            )
            for selection in selections
        ]
        return viewer

    def to_contour(
        self, selections, a=None, b=None, c=None, normal=None, supercell=None
    ) -> Graph:
        cut, fraction = slicing.get_cut(a, b, c)
        plane = slicing.plane(self._structure.lattice_vectors(), cut, normal)

        def _make_contour(selection):
            contour = Contour(
                slicing.grid_scalar(self._mapping[selection].T, plane, fraction),
                plane,
                label=self._label(selection),
                isolevels=True,
            )
            if supercell is not None:
                contour.supercell = np.ones(2, dtype=np.int_) * supercell
            return contour

        contours = [_make_contour(selection) for selection in selections]
        return Graph(contours)

    def to_quiver(
        self, selections_or_data, *, a=None, b=None, c=None, normal=None, supercell=None, 
    ) -> Graph:
        cut, fraction = slicing.get_cut(a, b, c)
        plane = slicing.plane(self._structure.lattice_vectors(), cut, normal)

        def _make_contour(selection):
            if (isinstance(selection, tuple)):
                sel = self._mapping[selection].T
            else:
                sel = selection
            if sel.ndim == 3 or len(sel) == 1:
                data = slicing.grid_scalar(sel, plane, fraction)
                data = np.array((np.zeros_like(data), data))
            else:
                data = slicing.grid_vector(sel, plane, fraction)
            contour = Contour(
                data,
                plane,
                label=self._label(selection),
                isolevels=True,
            )
            if supercell is not None:
                contour.supercell = np.ones(2, dtype=np.int_) * supercell
            return contour

        if (isinstance(selections_or_data, list)):
            contours = [_make_contour(selection) for selection in selections_or_data]
        else:
            contours = [_make_contour(selections_or_data)]
        return Graph(contours)