# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools
import os
import tempfile
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np
import numpy.typing as npt

from py4vasp import exception
from py4vasp._util import convert, import_

ase = import_.optional("ase")
ase_cube = import_.optional("ase.io.cube")
nglview = import_.optional("nglview")

CUBE_FILENAME = "quantity.cube"


class _Arrow3d(NamedTuple):
    tail: npt.ArrayLike
    """Tail, which is usually the atom centers"""
    tip: npt.ArrayLike
    """Tip, which is usually the atom centers + arrows"""
    color: str = "#2FB5AB"
    """Color of each arrow"""
    radius: float = 0.2

    def to_serializable(self):
        return (
            list(self.tail),
            list(self.tip),
            list(convert.to_rgb(self.color)),
            self.radius,
        )


def _rotate(arrow, transformation):
    return _Arrow3d(
        transformation @ arrow.tail,
        transformation @ arrow.tip,
        arrow.color,
        arrow.radius,
    )


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
    isosurfaces: Sequence[Isosurface] = None


@dataclass
class IonArrow:
    quantity: npt.ArrayLike
    """Vector quantity to be used to draw arrows at the ion positions"""
    label: str
    """Name of the quantity"""
    color: str
    "Color with which the arrows should be drawn"
    radius: float
    "Radius of the arrows"


_x_axis = _Arrow3d(tail=np.zeros(3), tip=np.array((3, 0, 0)), color="#000000")
_y_axis = _Arrow3d(tail=np.zeros(3), tip=np.array((0, 3, 0)), color="#000000")
_z_axis = _Arrow3d(tail=np.zeros(3), tip=np.array((0, 0, 3)), color="#000000")


def _recenter(arrow, origin=None):
    if origin is not None:
        return _Arrow3d(
            arrow.tail + origin,
            arrow.tip + origin,
            arrow.color,
            arrow.radius,
        )
    else:
        return arrow


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
    show_cell: bool = True
    """Defines if a cell is shown in ngl."""
    show_axes: bool = False
    """Defines if the axes is shown in the viewer"""
    show_axes_at: Sequence[float] = None
    """Defines where the axis is shown, defaults to the origin"""

    def _ipython_display_(self):
        widget = self.to_ngl()
        widget._ipython_display_()

    def _create_atoms(self, idx_traj):
        symbols = "".join(self.elements[idx_traj])
        atoms = ase.Atoms(symbols)
        atoms.cell = self.lattice_vectors[idx_traj]
        atoms.set_scaled_positions(self.positions[idx_traj])
        atoms.set_pbc(True)
        atoms = atoms.repeat(self.supercell)
        return atoms

    def _warning_message(self, attribute):
        for _attribute in attribute:
            if len(self.positions) > 1 and len(_attribute.quantity) > 1:
                raise exception.NotImplemented(
                    """\
Currently isosurfaces and ion arrows are implemented only for cases where there is only
one frame in the trajectory. Make sure that either only one frame for the positions
attribute is supplied with its corresponding grid scalar or ion arrow component."""
                )

    def _iterate_trajectory_frames(self):
        if self.grid_scalars:
            self._warning_message(self.grid_scalars)
        if self.ion_arrows:
            self._warning_message(self.ion_arrows)
        return range(len(self.positions))

    def to_ngl(self):
        """Create a widget with NGL

        This method creates the widget required to view a structure, isosurfaces and
        arrows at atom centers. The attributes of View are used as a starting point to
        determine which methods are called (either isosurface, arrows, etc).
        """
        trajectory = [self._create_atoms(i) for i in self._iterate_trajectory_frames()]
        ngl_trajectory = nglview.ASETrajectory(trajectory)
        widget = nglview.NGLWidget(ngl_trajectory)
        if self.grid_scalars:
            self._show_isosurface(widget, trajectory)
        if self.ion_arrows:
            self._show_arrows_at_atoms(widget, trajectory)
        if self.show_cell:
            self._show_cell(widget)
        if self.show_axes:
            self._show_axes(widget, trajectory)
        return widget

    def _show_cell(self, widget):
        widget.add_unitcell()

    def _show_axes(self, widget, trajectory):
        _, transformation = trajectory[0].cell.standard_form()
        x_axis = _rotate(_recenter(_x_axis, self.show_axes_at), transformation)
        y_axis = _rotate(_recenter(_y_axis, self.show_axes_at), transformation)
        z_axis = _rotate(_recenter(_z_axis, self.show_axes_at), transformation)
        widget.shape.add_arrow(*(x_axis.to_serializable()))
        widget.shape.add_arrow(*(y_axis.to_serializable()))
        widget.shape.add_arrow(*(z_axis.to_serializable()))

    def _skip_if_not_first_trajectory(self, idx_traj):
        """Boolean to describe if a trajectory is to be skipped.

        Skip anything but the first trajectory for both arrows and densities because the
        we currently only show this quantity for the first frame of the trajectory based
        on the nglviewer.
        """
        if idx_traj != 0:
            return True
        else:
            return False

    def _set_atoms_in_standard_form(self, atoms):
        cell, _ = atoms.cell.standard_form()
        atoms.set_cell(cell)

    def _repeat_isosurface(self, quantity):
        quantity_repeated = np.tile(quantity, self.supercell)
        return quantity_repeated

    def _show_isosurface(self, widget, trajectory):
        iter_traj = self._iterate_trajectory_frames()
        for grid_scalar, idx_traj in itertools.product(self.grid_scalars, iter_traj):
            atoms = trajectory[idx_traj]
            self._set_atoms_in_standard_form(atoms)
            if self._skip_if_not_first_trajectory(idx_traj):
                continue
            quantity = self._repeat_isosurface(grid_scalar.quantity[idx_traj])
            with tempfile.TemporaryDirectory() as tmp:
                filename = os.path.join(tmp, CUBE_FILENAME)
                ase_cube.write_cube(open(filename, "w"), atoms=atoms, data=quantity)
                component = widget.add_component(filename)
            if grid_scalar.isosurfaces:
                for isosurface in grid_scalar.isosurfaces:
                    isosurface_options = {
                        "isolevel": isosurface.isolevel,
                        "color": isosurface.color,
                        "opacity": isosurface.opacity,
                    }
                    component.add_surface(**isosurface_options)

    def _show_arrows_at_atoms(self, widget, trajectory):
        iter_traj = self._iterate_trajectory_frames()
        for _arrows, idx_traj in itertools.product(self.ion_arrows, iter_traj):
            _, transformation = trajectory[idx_traj].cell.standard_form()
            if self._skip_if_not_first_trajectory(idx_traj):
                continue
            arrows = _arrows.quantity[idx_traj]
            positions = trajectory[idx_traj].get_positions()
            for arrow, tail in zip(itertools.cycle(arrows), positions):
                tip = arrow + tail
                arrow_3d = _rotate(
                    _Arrow3d(tail, tip, color=_arrows.color, radius=_arrows.radius),
                    transformation,
                )
                widget.shape.add_arrow(*(arrow_3d.to_serializable()))
