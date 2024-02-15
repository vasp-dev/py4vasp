# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools
import os
import tempfile
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np
import numpy.typing as npt

from py4vasp._util import import_

ase = import_.optional("ase")
ase_cube = import_.optional("ase.io.cube")
nglview = import_.optional("nglview")

CUBE_FILENAME = "quantity.cube"


class _Arrow3d(NamedTuple):
    tail: np.ndarray
    tip: np.ndarray

    def to_serializable(self):
        return list(self.tail), list(self.tip)


@dataclass
class Isosurface:
    isolevel: float
    "The isosurface moves through points where the interpolated data has this value."
    color: str
    "Color with which the isosurface should be drawn"
    opacity: float
    "Amount of light blocked by the isosurface."


@dataclass
class GridQuantity:
    quantity: npt.ArrayLike
    """The quantity which is to be plotted as an isosurface"""
    label: str
    """Name of the quantity"""
    isosurfaces: Sequence[Isosurface]


@dataclass
class IonArrow:
    quantity: npt.ArrayLike
    """Vector quantity to be used to draw arrows at the ion positions"""
    label: str
    """Name of the quantity"""


@dataclass
class View:
    elements: npt.ArrayLike
    """Elements for all structures in the trajectory"""
    lattice_vectors: npt.ArrayLike
    """Lattice vectors for all structures in the trajectory"""
    positions: npt.ArrayLike
    """Ion positions for all structures in the trajectory"""
    grid_scalars: Sequence[GridQuantity] = None
    """This sequence stores quantities that are generated on a grid."""
    ion_arrows: Sequence[IonArrow] = None
    """This sequence stores arrows at the atom-centers."""
    supercell: npt.ArrayLike = (1, 1, 1)
    "Defines how many multiple of the cell are drawn along each of the coordinate axis."

    def _ipython_display_(self):
        widget = self.to_ngl()
        widget._ipython_display_()

    def _create_atoms(self, idx_traj):
        symbols = "".join(self.elements[idx_traj])
        atoms = ase.Atoms(symbols)
        atoms.cell = self.lattice_vectors[idx_traj]
        atoms.set_scaled_positions(self.positions[idx_traj])
        atoms.set_pbc(True)
        return atoms

    def to_ngl(self):
        trajectory = []
        for idx_traj in range(len(self.lattice_vectors)):
            atoms = self._create_atoms(idx_traj)
            trajectory.append(atoms)
        ngl_trajectory = nglview.ASETrajectory(trajectory)
        widget = nglview.NGLWidget(ngl_trajectory)
        return widget

    def show_isosurface(self):
        widget = self.to_ngl()
        iter_traj = list(range(len(self.lattice_vectors)))
        for grid_scalar, idx_traj in itertools.product(self.grid_scalars, iter_traj):
            atoms = self._create_atoms(idx_traj)
            data = grid_scalar.quantity[idx_traj]
            with tempfile.TemporaryDirectory() as tmp:
                filename = os.path.join(tmp, CUBE_FILENAME)
                ase_cube.write_cube(open(filename, "w"), atoms=atoms, data=data)
                widget.add_component(filename)
        return widget

    def show_arrows_at_atoms(self):
        widget = self.to_ngl()
        iter_traj = list(range(len(self.lattice_vectors)))
        for arrow, idx_traj in itertools.product(self.ion_arrows, iter_traj):
            tail = self.positions[idx_traj]
            tip = arrow.quantity + tail
            arrow_3d = _Arrow3d(tail, tip)
            widget.shape.add_arrow(*(arrow_3d.to_serializable()))
        return widget
