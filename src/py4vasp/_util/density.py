from typing import Callable

import numpy as np

from py4vasp._calculation.structure import Structure
from py4vasp._third_party.graph import Contour, Graph
from py4vasp._third_party.view import GridQuantity, View
from py4vasp._util import index, slicing


class Visualizer:
    def __init__(self, structure: Structure, selector: index.Selector):
        self._selector = selector
        self._structure = structure

    def to_view(self, selections, supercell=1) -> View:
        viewer = self._structure.plot(supercell)
        viewer.grid_scalars = [
            GridQuantity(
                (self._selector[selection].T)[np.newaxis],
                label=self._selector.label(selection),
            )
            for selection in selections
        ]
        return viewer

    def to_contour(
        self, selections, a=None, b=None, c=None, normal=None, supercell=None
    ) -> Graph:
        return self._make_contour_graph(
            selections, slicing.grid_scalar, a, b, c, supercell, normal
        )

    def to_quiver(
        self, selections, a=None, b=None, c=None, supercell=None, normal=None
    ) -> Graph:
        return self._make_contour_graph(
            selections, slicing.grid_vector, a, b, c, supercell, normal
        )

    def _make_contour_graph(
        self,
        selections,
        data_slicer: Callable,
        a=None,
        b=None,
        c=None,
        supercell=None,
        normal=None,
    ) -> Graph:
        cut, fraction = slicing.get_cut(a, b, c)
        plane = slicing.plane(self._structure.lattice_vectors(), cut, normal)

        def _make_contour(selection):
            contour = Contour(
                data_slicer(self._selector[selection].T, plane, fraction),
                plane,
                label=self._selector.label(selection) or "",
                isolevels=True,
            )
            if supercell is not None:
                contour.supercell = np.ones(2, dtype=np.int_) * supercell
            return contour

        contours = [_make_contour(selection) for selection in selections]
        return Graph(contours)
