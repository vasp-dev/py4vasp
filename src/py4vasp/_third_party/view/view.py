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
vaspview = import_.optional("vasp_viewer")

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
    shift: npt.ArrayLike = None
    """Defines the shift of the origin"""
    camera: str = "orthographic"
    """Defines the camera view type (orthographic or perspective)"""

    def __post_init__(self):
        self._verify()

    def _ipython_display_(self, mode="auto"):
        if mode == "auto":
            if import_.is_imported(vaspview):
                mode = "vasp_viewer"
            elif import_.is_imported(nglview):
                mode = "ngl"
            else:
                raise exception.IncorrectUsage(
                    "No supported viewer found. Please install either 'vasp_viewer' or 'nglview' to visualize the structure."
                )

        if mode == "ngl":
            widget = self.to_ngl()
            widget._ipython_display_()
        elif mode == "vasp_viewer":
            widget = self.to_vasp_viewer()
        else:
            raise exception.IncorrectUsage(
                f"Mode '{mode}' is not supported. Choose either 'auto', 'ngl' or 'vasp_viewer'."
            )

    def to_ngl(self):
        """Create a widget with NGL

        This method creates the widget required to view a structure, isosurfaces and
        arrows at atom centers. The attributes of View are used as a starting point to
        determine which methods are called (either isosurface, arrows, etc).
        """
        self._verify()
        trajectory = [self._create_atoms(i) for i in self._iterate_trajectory_frames()]
        ngl_trajectory = nglview.ASETrajectory(trajectory)
        widget = nglview.NGLWidget(ngl_trajectory)
        widget.camera = self.camera
        if self.grid_scalars:
            self._show_isosurface(widget, trajectory)
        if self.ion_arrows:
            self._show_arrows_at_atoms(widget, trajectory)
        if self.show_cell:
            self._show_cell(widget)
        if self.show_axes:
            self._show_axes(widget, trajectory)
        return widget

    def to_vasp_viewer(self):
        """Create a widget with VASP Viewer

        This method creates the widget required to view a structure, isosurfaces and
        arrows at atom centers. The attributes of View are added to a dictionary with which
        to call initialize a VASP Viewer widget."""
        self._verify()
        structure: dict = {
            "positions": self._convert_to_list(self.positions),
            "types": (
                self._convert_to_list(self.elements[0])
            ),  # TODO enable trajectory for elements
            "lattice": (
                self._convert_to_list(self.lattice_vectors[0])
            ),  # TODO enable trajectory for lattice_vectors
        }

        # === Atoms options ===
        # if self.selective_dynamics:
        #    structure["selective_dynamics"] = (
        #        self.selective_dynamics
        #    )  # TODO query selective dynamics and check type

        # === Vector Group options ===
        if self.ion_arrows is not None:
            # TODO handle list of vector groups instead of magnetization only
            # TODO adjust UI to support this
            structure["ion_arrows"] = [
                {
                    "label": arrow.label,
                    "quantity": self._convert_to_list(arrow.quantity),
                    "base_color": arrow.color,  # TODO allow a default fallback color for each Vector Group for colorMode = "off"
                    "base_radius": arrow.radius,
                }
                for arrow in self.ion_arrows
            ]
        if self.grid_scalars is not None:
            # TODO merge isosurface branch
            # TODO handle list of grid scalars instead of single grid scalar only
            # TODO adjust UI to support this
            structure["grid_scalars"] = [
                {
                    "label": grid_quantity.label,
                    "data": grid_quantity.quantity,  # TODO check type
                    "isosurfaces": [  # TODO hook this list to isosurface settings
                        {
                            "isolevel": isosurface.isolevel,
                            "color": isosurface.color,  # TODO interpret this as base color of isosurface
                            "opacity": isosurface.opacity,  # TODO tie this to opacity on isosurface
                        }
                        for isosurface in grid_quantity.isosurfaces
                    ],
                }
                for grid_quantity in self.grid_scalars
            ]

        # === Lattice options ===
        if self.shift is not None:
            structure["constant_shift"] = self._convert_to_list(self.shift)
        # if self.scaling_factor:
        #    structure["scaling_factor"] = (
        #        self.scaling_factor
        #    )  # TODO query scaling factor
        # if self.lattice_vectors_primitive:
        #    structure["lattice_primitive"] = (
        #        self.lattice_vectors_primitive
        #    )  # TODO query primitive lattice
        if self.supercell is not None:
            structure["supercell"] = self._convert_to_list(self.supercell)

        # === Visualization options ===
        if self.camera is not None:
            structure["camera_mode"] = self.camera
        if self.show_cell is not None:
            structure["show_lattice"] = self.show_cell
        if self.show_axes is not None:
            structure["show_xyz"] = False
            structure["show_abc"] = self.show_axes
            structure["show_xyz_aside"] = self.show_axes
            structure["show_abc_aside"] = self.show_axes
        if self.show_axes_at is not None:
            structure["axes_scene_shift"] = self._convert_to_list(self.show_axes_at)

        # === Meta options ===
        # if self.title:
        #    structure["descriptor"] = self.title  # TODO query structure name

        return vaspview.Widget(structure)

    def _verify(self):
        self._raise_error_if_present_on_multiple_steps(self.grid_scalars)
        self._raise_error_if_present_on_multiple_steps(self.ion_arrows)
        self._raise_error_if_number_steps_inconsistent()
        self._raise_error_if_any_shape_is_incorrect()

    def _raise_error_if_present_on_multiple_steps(self, attributes):
        if not attributes:
            return
        for attribute in attributes:
            if len(attribute.quantity) > 1:
                raise exception.NotImplemented(
                    """\
Currently isosurfaces and ion arrows are implemented only for cases where there is only
one frame in the trajectory. Make sure that either only one frame for the positions
attribute is supplied with its corresponding grid scalar or ion arrow component."""
                )

    def _raise_error_if_number_steps_inconsistent(self):
        if len(self.elements) == len(self.lattice_vectors) == len(self.positions):
            return
        raise exception.IncorrectUsage(
            "The shape of the arrays is inconsistent. Each of 'elements' (length = "
            f"{len(self.elements)}), 'lattice_vectors' (length = "
            f"{len(self.lattice_vectors)}), and 'positions' (length = "
            f"{len(self.positions)}) should have a leading dimension of the number of"
            "steps."
        )

    def _raise_error_if_any_shape_is_incorrect(self):
        number_elements = len(self.elements[0])
        _, number_positions, vector_size = np.shape(self.positions)
        if number_elements != number_positions:
            raise exception.IncorrectUsage(
                f"Number of elements ({number_elements}) inconsistent with number of positions ({number_positions})."
            )
        if vector_size != 3:
            raise exception.IncorrectUsage(
                f"Positions must have three components and not {vector_size}."
            )
        cell_shape = np.shape(self.lattice_vectors)[1:]
        if any(length != 3 for length in cell_shape):
            raise exception.IncorrectUsage(
                f"Lattice vectors must be a 3x3 unit cell but have the shape {cell_shape}."
            )

    def _convert_to_list(self, attribute):
        if isinstance(attribute, list):
            return attribute
        elif isinstance(attribute, tuple):
            return list(attribute)
        elif isinstance(attribute, np.ndarray):
            return attribute.tolist()
        else:
            raise exception.NotImplemented(
                f"Safe conversion of type {type(attribute)} to list is not implemented."
            )

    def _create_atoms(self, step):
        symbols = "".join(self.elements[step])
        atoms = ase.Atoms(symbols, cell=self.lattice_vectors[step], pbc=True)
        shift = np.zeros(3) if self.shift is None else self.shift
        atoms.set_scaled_positions(np.add(self.positions[step], shift))
        atoms.wrap()
        atoms = atoms.repeat(self.supercell)
        return atoms

    def _iterate_trajectory_frames(self):
        return range(len(self.positions))

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

    def _set_atoms_in_standard_form(self, atoms):
        cell, _ = atoms.cell.standard_form()
        atoms.set_cell(cell)

    def _repeat_isosurface(self, quantity):
        quantity_repeated = np.tile(quantity, self.supercell)
        return quantity_repeated

    def _show_isosurface(self, widget, trajectory):
        step = 0
        for grid_scalar in self.grid_scalars:
            if not grid_scalar.isosurfaces:
                continue
            quantity = grid_scalar.quantity[step]
            quantity = self._shift_quantity(quantity)
            quantity = self._repeat_isosurface(quantity)
            atoms = trajectory[step]
            self._set_atoms_in_standard_form(atoms)
            with tempfile.TemporaryDirectory() as tmp:
                filename = os.path.join(tmp, CUBE_FILENAME)
                ase_cube.write_cube(open(filename, "w"), atoms=atoms, data=quantity)
                component = widget.add_component(filename)
            for isosurface in grid_scalar.isosurfaces:
                isosurface_options = {
                    "isolevel": isosurface.isolevel,
                    "color": isosurface.color,
                    "opacity": isosurface.opacity,
                }
                component.add_surface(**isosurface_options)

    def _shift_quantity(self, quantity):
        if self.shift is None:
            return quantity
        new_grid_center = np.multiply(quantity.shape, self.shift)
        shift_indices = np.round(new_grid_center).astype(np.int32)
        return np.roll(quantity, shift_indices, axis=(0, 1, 2))

    def _show_arrows_at_atoms(self, widget, trajectory):
        step = 0
        for _arrows in self.ion_arrows:
            _, transformation = trajectory[step].cell.standard_form()
            arrows = _arrows.quantity[step]
            positions = trajectory[step].get_positions()
            for arrow, tail in zip(itertools.cycle(arrows), positions):
                tip = arrow + tail
                arrow_3d = _rotate(
                    _Arrow3d(tail, tip, color=_arrows.color, radius=_arrows.radius),
                    transformation,
                )
                widget.shape.add_arrow(*(arrow_3d.to_serializable()))
