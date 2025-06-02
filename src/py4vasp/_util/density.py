from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import numpy as np

from py4vasp._calculation.structure import Structure
from py4vasp._third_party.graph import Contour, Graph
from py4vasp._third_party.view import GridQuantity, View
from py4vasp._util import index, slicing
from py4vasp.exception import IncorrectUsage


@dataclass
class SliceArguments:
    a: Optional[float]=None
    b: Optional[float]=None
    c: Optional[float]=None
    normal: Optional[float]=None
    supercell: Optional[float]=None

    def __post_init__(self):
        # Count how many of a, b, c are not None
        provided = [x is not None for x in (self.a, self.b, self.c)].count(True)
        if provided != 1:
            raise IncorrectUsage("Exactly one of a, b, or c must be provided (not None).")

    def slice_plane(self, lattice_vectors):
        cut, fraction = slicing.get_cut(self.a, self.b, self.c)
        plane = slicing.plane(lattice_vectors, cut, self.normal)
        return plane, fraction



class Visualizer:
    def __init__(self, structure: Structure, label_function: Callable = None):
        self._structure = structure
        self._label = label_function

    def to_view(self, mapping, selections, supercell=1) -> View:
        if self._label is None:
            self._label = getattr(mapping, "label", (lambda _: ""))
        viewer = self._structure.plot(supercell)
        viewer.grid_scalars = [
            GridQuantity(
                (mapping[selection].T)[np.newaxis], label=self._label(selection)
            )
            for selection in selections
        ]
        return viewer

    def to_contour_from_mapping(
        self,
        mapping,
        selections,
        slice_args: SliceArguments,
        isolevels: bool=True,
    ) -> Graph:
        if self._label is None:
            self._label = getattr(mapping, "label", (lambda _: ""))
        plane, fraction = slice_args.slice_plane(self._structure.lattice_vectors())
        contours = [
            self._make_contour(
                mapping[selection].T, selection, plane, fraction, slice_args.supercell, isolevels
            )
            for selection in selections
        ]
        return Graph(contours)

    def _make_contour(self, data, selection, plane, fraction, supercell, isolevels):
        contour = Contour(
            slicing.grid_scalar(data, plane, fraction),
            plane,
            label=self._label(selection),
            isolevels=isolevels,
        )
        if supercell is not None:
            contour.supercell = np.ones(2, dtype=np.int_) * supercell
        return contour

    def to_contour_from_data(
        self,
        data,
        slice_args: SliceArguments,
        isolevels: bool=True,
    ) -> Graph:
        if self._label is None:
            self._label = lambda _: ""
        plane, fraction = slice_args.slice_plane(self._structure.lattice_vectors())
        contours = [
            self._make_contour(data, None, plane, fraction, slice_args.supercell, isolevels)
        ]
        return Graph(contours)

    def to_quiver_from_mapping(
        self,
        mapping,
        selections,
        slice_args: SliceArguments,
    ) -> Graph:
        if self._label is None:
            self._label = getattr(mapping, "label", (lambda _: ""))
        plane, fraction = slice_args.slice_plane(self._structure.lattice_vectors())
        contours = [
            self._make_quiver(
                mapping[selection].T, selection, plane, fraction, slice_args.supercell
            )
            for selection in selections
        ]
        return Graph(contours)

    def _make_quiver(self, sel, selection, plane, fraction, supercell):
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

    def to_quiver_from_data(
        self,
        data,
        slice_args: SliceArguments,
    ) -> Graph:
        if self._label is None:
            self._label = lambda _: ""
        plane, fraction = slice_args.slice_plane(self._structure.lattice_vectors())
        contours = [self._make_quiver(data, None, plane, fraction, slice_args.supercell)]
        return Graph(contours)
