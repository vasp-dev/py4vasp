from typing import NamedTuple
import nglview
import numpy as np


class _Arrow3d(NamedTuple):
    tail: np.ndarray
    tip: np.ndarray
    color: np.ndarray
    radius: float = 0.2


_x_axis = _Arrow3d(tail=np.zeros(3), tip=np.array((3, 0, 0)), color=[1, 0, 0])
_y_axis = _Arrow3d(tail=np.zeros(3), tip=np.array((0, 3, 0)), color=[0, 1, 0])
_z_axis = _Arrow3d(tail=np.zeros(3), tip=np.array((0, 0, 3)), color=[0, 0, 1])


class Viewer3d:
    """Collection of data and elements to be displayed in a structure viewer"""

    def __init__(self, structure, supercell=None):
        self._structure = structure
        self._axes = None
        self._arrows = []
        self.supercell = supercell
        self._ngl = self.show()

    def _ipython_display_(self):
        self._ngl._ipython_display_()

    def show_cell(self):
        self._ngl.add_unitcell()

    def hide_cell(self):
        self._ngl.remove_unitcell()

    def show_axes(self):
        if self._axes is not None:
            return
        self._axes = (
            self._make_arrow(_x_axis),
            self._make_arrow(_y_axis),
            self._make_arrow(_z_axis),
        )

    def hide_axes(self):
        if self._axes is None:
            return
        for axis in self._axes:
            self._ngl.remove_component(axis)
        self._axes = None

    def show_arrows_at_atoms(self, arrows, color=[0.1, 0.1, 0.8]):
        structure = self._structure.to_pymatgen()
        for tail, arrow in zip(structure.cart_coords, arrows):
            tip = tail + arrow
            arrow = _Arrow3d(tail, tip, color)
            self._arrows.append(self._make_arrow(arrow))

    def hide_arrows_at_atoms(self):
        for arrow in self._arrows:
            self._ngl.remove_component(arrow)
        self._arrows = []

    def _make_arrow(self, arrow):
        return self._ngl.shape.add_arrow(*arrow)

    def show(self):
        structure = self._structure.to_pymatgen()
        if self.supercell is not None:
            structure.make_supercell(self.supercell)
        view = nglview.show_pymatgen(structure)
        return view
