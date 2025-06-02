from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np

from py4vasp._calculation.structure import Structure
from py4vasp._third_party.graph import Contour, Graph
from py4vasp._third_party.view import GridQuantity, View
from py4vasp._util import index, slicing
from py4vasp.exception import IncorrectUsage


@dataclass
class SliceArguments:
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    normal: Optional[float] = None
    supercell: Optional[float] = None

    def __post_init__(self):
        # Count how many of a, b, c are not None
        provided = [x is not None for x in (self.a, self.b, self.c)].count(True)
        if provided != 1:
            raise IncorrectUsage(
                "Exactly one of a, b, or c must be provided (not None)."
            )

    def slice_plane(self, lattice_vectors):
        cut, fraction = slicing.get_cut(self.a, self.b, self.c)
        plane = slicing.plane(lattice_vectors, cut, self.normal)
        return plane, fraction


class Visualizer:
    def __init__(self, structure: Structure):
        self._structure = structure

    def to_view(self, dataDict: dict[str, Any], supercell=1) -> View:
        viewer = self._structure.plot(supercell)
        viewer.grid_scalars = [
            GridQuantity(
                data,
                label=label if label else "",
            )
            for label, data in dataDict.items()
        ]
        return viewer

    def to_contour(
        self,
        dataDict: dict[str, Any],
        slice_args: SliceArguments,
        isolevels: bool = True,
    ) -> Graph:
        plane, fraction = slice_args.slice_plane(self._structure.lattice_vectors())
        contours = [
            self._make_contour(
                data, label, plane, fraction, slice_args.supercell, isolevels
            )
            for label, data in dataDict.items()
        ]
        return Graph(contours)

    def _make_contour(self, data, label, plane, fraction, supercell, isolevels):
        contour = Contour(
            slicing.grid_scalar(data, plane, fraction),
            plane,
            label=label if label else "",
            isolevels=isolevels,
        )
        if supercell is not None:
            contour.supercell = np.ones(2, dtype=np.int_) * supercell
        return contour

    def to_quiver(
        self,
        dataDict,
        slice_args: SliceArguments,
    ) -> Graph:
        plane, fraction = slice_args.slice_plane(self._structure.lattice_vectors())
        contours = [
            self._make_quiver(data, label, plane, fraction, slice_args.supercell)
            for label, data in dataDict.items()
        ]
        return Graph(contours)

    def _make_quiver(self, sel, label, plane, fraction, supercell):
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
