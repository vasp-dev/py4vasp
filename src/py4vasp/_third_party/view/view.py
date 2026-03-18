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
    """Dataclass to store the settings for an isosurface to be plotted for a grid quantity.

    This class does not hold the data for the isosurface itself, only the settings for how
    an isosurface should be plotted as a :class:`~py4vasp.view.GridQuantity`.

    Examples
    --------
    Each :class:`~py4vasp.view.GridQuantity` object needs matching
    :class:`~py4vasp.view.Isosurface` objects to specify how the isosurfaces for this
    quantity should be plotted.
    For example, if you want to plot an isosurface for a charge density with an isolevel
    of 0.1, red color and 50% opacity, you can create an :class:`~py4vasp.view.Isosurface`
    object like this:
    
    >>> from py4vasp.view import Isosurface
    >>> Isosurface(isolevel=0.1, color='red', opacity=0.5)
    Isosurface(isolevel=0.1, color='red', opacity=0.5)
    """

    isolevel: float
    """The isosurface moves through points where the interpolated data has this value. For example, if the quantity is a charge density, an isolevel of 0.1 means that the isosurface will be drawn through points where the charge density has a value of 0.1."""
    color: str
    """Color with which the isosurface should be drawn. For example, '#2FB5AB' or 'red'."""
    opacity: float
    """Amount of light blocked by the isosurface. Must be a value between 0 and 1, where 0 means fully transparent and 1 means fully opaque."""


@dataclass
class GridQuantity:
    """Dataclass to store a quantity defined on a grid and the settings for the isosurfaces to be plotted for this quantity.

    Examples
    --------
    If you want to plot an isosurface for a charge density with an isolevel of 0.1, red
    color and 50% opacity, you can create a :class:`~py4vasp.view.GridQuantity` object
    with a single step like this:
    
    >>> from py4vasp.view import GridQuantity, Isosurface
    >>> quantity = [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0.1, 0], [0, 0.15, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]
    >>> isosurface = Isosurface(isolevel=0.1, color='red', opacity=0.5)
    >>> GridQuantity(quantity=quantity, label='Charge Density', isosurfaces=[isosurface])
    GridQuantity(quantity=[[[[...]]]], label=..., isosurfaces=[Isosurface(...)])
    """

    quantity: npt.ArrayLike
    """The quantity which is to be plotted as an isosurface. Expected shape is (number of steps, grid size x, grid size y, grid size z)."""
    label: str
    """Name of the quantity"""
    isosurfaces: Sequence[Isosurface] = None
    """Sequence of isosurfaces to be plotted for this quantity."""


@dataclass
class IonArrow:
    """Dataclass to store a vectorial quantity defined at the ion positions and the settings for the arrows to be plotted for this quantity.

    Positions for these arrows will be inferred in the context of the structure / ion
    positions, as provided, e.g., in the :class:`~py4vasp.view.View` class.

    Examples
    --------
    If you want to plot arrows for a spin quantity with red color and a radius of 0.2,
    you can create an :class:`~py4vasp.view.IonArrow` object with a single step like this:
    
    >>> from py4vasp.view import IonArrow
    >>> quantity = [[[[1, 0, 0], [-1, 0, 0]], [[0.5, 0.5, 0], [-0.5, -0.5, 0]]]]
    >>> IonArrow(quantity=quantity, label='Spin', color='red', radius=0.2)
    IonArrow(quantity=[[[[...]]]], label=..., color=..., radius=...)
    """

    quantity: npt.ArrayLike
    """Vector quantity to be used to draw arrows at the ion positions.
    The shape is expected to be (number of steps, number of ions, 3) and the vectors are expected to be in Cartesian coordinates."""
    label: str
    """Name of the quantity"""
    color: str
    """Color with which the arrows should be drawn. For example, '#2FB5AB' or 'red'."""
    radius: float
    """Radius of the arrows"""


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
    """Class to store all information required for 3D visualization of a structure, isosurfaces and vectorial quantities (e.g., spins, magnetization etc.).

    The View class acts as a unified data container for 3D structural visualization,
    supporting crystal structures with optional supercell expansion, scalar field
    isosurfaces, and vector quantities drawn as arrows at ion positions. It supports
    multi-step ionic trajectories and two interactive viewer backends.

    Key Features
    ------------
    - Visualization of crystal structures with optional supercell expansion
    - Isosurface rendering for grid-based scalar quantities, e.g., charge density 
      (see :class:`~py4vasp.view.GridQuantity`)
    - Arrow visualization of vector quantities at ion positions, e.g., spins or forces
      (see :class:`~py4vasp.view.IonArrow`)
    - Support for multi-step ionic trajectories
    - Compatible with NGL and VASP Viewer backends
    - Configurable cell display, coordinate axes, camera mode, and unit cell origin shift

    Notes
    -----
    - The methods :meth:`to_ngl` and :meth:`to_vasp_viewer` convert the stored information
      to an interactive widget; which method is available depends on the installed packages.
    - Typically, :class:`~py4vasp.view.View` objects are obtained by calling the `plot`
      method on a calculation quantity, which internally calls `to_view` and automatically
      selects the viewer.
    - In NGL mode, isosurfaces and ion arrows are currently supported only for single-frame
      trajectories.
    - The created `View` object will automatically display as a widget in supported
      environments, e.g., in Jupyter notebooks.

    See Also
    --------
    :class:`~py4vasp.view.IonArrow` : Vectorial quantity at ion positions, visualized as arrows.
    :class:`~py4vasp.view.GridQuantity` : Scalar quantity on a grid, visualized as isosurfaces.
    :class:`~py4vasp.view.Isosurface` : Settings for a single isosurface within a :class:`~py4vasp.view.GridQuantity`.

    Examples
    --------
    Generally speaking, you might obtain :class:`~py4vasp.view.View` objects by calling
    the corresponding `plot` methods on the different quantities:

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)
    >>> calculation.structure.plot()
    View(...)

    >>> calculation.nics.plot()
    View(...)

    But you can also create :class:`~py4vasp.view.View` objects directly. For example:

    >>> from py4vasp.view import View

    The created `View` object will automatically display as a widget in supported
    environments, e.g., in Jupyter notebooks.

    >>> elements = [["Si", "Si"], ["Na", "Cl"]]
    >>> lattice_vectors = [[[3, 0, 0], [0, 3, 0], [0, 0, 3]], [[4, 0, 0], [0, 4, 0], [0, 0, 4]]]
    >>> positions = [[[0, 0, 0], [0.25, 0.25, 0.25]], [[0, 0, 0], [0.5, 0.5, 0.5]]]
    >>> view = View(elements=elements, lattice_vectors=lattice_vectors, positions=positions)
    >>> view
    View(elements=[[...]], lattice_vectors=[[[...]]], positions=[[[...]]], ...)

    It is possible to add quantities or change settings after creating the
    :class:`~py4vasp.view.View` object:

    >>> view.supercell = (2, 2, 2)
    >>> view
    View(elements=[[...]], lattice_vectors=[[[...]]], positions=[[[...]]], ... supercell=(2, 2, 2), ...)

    You may wish to add arrows at the atom centers, for example to visualize spins or
    forces. You can do this by creating an :class:`~py4vasp.view.IonArrow` object and
    adding it to the `ion_arrows` attribute of the :class:`~py4vasp.view.View` object:

    >>> from py4vasp.view import IonArrow
    >>> quantity = [[[[1, 0, 0], [-1, 0, 0]], [[0.5, 0.5, 0], [-0.5, -0.5, 0]]]]
    >>> ion_arrow = IonArrow(quantity=quantity, label='Spin', color='red', radius=0.2)
    >>> view.ion_arrows = [ion_arrow]
    >>> view
    View(elements=[[...]], lattice_vectors=[[[...]]], positions=[[[...]]], ... ion_arrows=[IonArrow(...)], ...)

    Similarly, you can add isosurfaces for quantities defined on a grid by creating
    :class:`~py4vasp.view.GridQuantity` objects and adding them to the `grid_scalars`
    attribute of the :class:`~py4vasp.view.View` object:

    >>> from py4vasp.view import GridQuantity, Isosurface
    >>> quantity = [[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0.1, 0], [0, 0.15, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]
    >>> isosurface = Isosurface(isolevel=0.1, color='red', opacity=0.5)
    >>> grid_quantity = GridQuantity(quantity=quantity, label='Charge Density', isosurfaces=[isosurface])
    >>> view.grid_scalars = [grid_quantity]
    >>> view
    View(elements=[[...]], lattice_vectors=[[[...]]], positions=[[[...]]], ... grid_scalars=[GridQuantity(...)], ...)
    """

    elements: npt.ArrayLike
    """Elements for all structures in the trajectory. Expected shape is (number of steps, number of ions)."""
    lattice_vectors: npt.ArrayLike
    """Lattice vectors for all structures in the trajectory. Expected shape is (number of steps, 3, 3)."""
    positions: npt.ArrayLike
    """Ion positions for all structures in the trajectory. Expected shape is (number of steps, number of ions, 3)."""
    grid_scalars: Sequence[GridQuantity] = None
    """This sequence stores quantities that are generated on a grid. Expected shape is (number of quantities,)."""
    ion_arrows: Sequence[IonArrow] = None
    """This sequence stores arrows at the atom-centers. Expected shape is (number of quantities,)."""
    supercell: npt.ArrayLike = (1, 1, 1)
    """Defines how many multiples of the cell are drawn along each coordinate axes, in integer values. Valid shapes are (1,), or (3,), and valid dtype=int."""
    show_cell: bool = True
    """Defines if a cell is shown in ngl."""
    show_axes: bool = False
    """Defines if the axes is shown in the viewer."""
    show_axes_at: Sequence[float] = None
    """Defines where the axis is shown, defaults to the origin. Given in Direct coordinates. Expected shape is (3,)."""
    shift: npt.ArrayLike = None
    """Defines the shift of the origin, defaults to no shift. Given in Direct coordinates. Expected shape is (3,). This is useful if you want to change the origin of the unit cell, e.g., to the center of the cell."""
    camera: str = "orthographic"
    """Defines the camera view type (orthographic or perspective)."""
    atom_radius: float = None
    """Defines the radius of the atoms (only available for VASP Viewer). Must be a positive number."""
    structure_title: str = None
    """Title of the structure to be shown (only available for VASP Viewer)."""

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
        self._verify("ngl")
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
            "atoms_trajectory": self._convert_to_list(self.positions),
            "atoms_types": self._convert_to_list(self.elements),
            "lattice_vectors": self._convert_to_list(self.lattice_vectors),
        }

        # === Atoms options ===
        if self.atom_radius is not None:
            structure["selections_atom_radius"] = self.atom_radius

        # === Vector Group options ===
        if self.ion_arrows is not None:
            structure["ion_arrow_groups"] = [
                {
                    "label": arrow.label,
                    "quantity": self._convert_to_list(arrow.quantity),
                    "base_color": arrow.color,
                    "base_radius": arrow.radius,
                }
                for arrow in self.ion_arrows
            ]
        if self.grid_scalars is not None:
            # TODO merge isosurface branch
            # TODO handle list of grid scalars instead of single grid scalar only
            # TODO adjust UI to support this
            structure["grid_scalar_groups"] = [
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
            structure["selections_constant_shift"] = self._convert_to_list(self.shift)
        if self.supercell is not None:
            structure["selections_supercell"] = self._convert_to_list(self.supercell)

        # === Visualization options ===
        if self.camera is not None:
            structure["selections_camera_mode"] = self.camera
        if self.show_cell is not None:
            structure["selections_show_lattice"] = self.show_cell
        if self.show_axes is not None:
            structure["selections_show_xyz"] = False
            structure["selections_show_abc"] = self.show_axes
            structure["selections_show_xyz_aside"] = self.show_axes
            structure["selections_show_abc_aside"] = self.show_axes
        if self.show_axes_at is not None:
            structure["selections_axes_abc_shift"] = self._convert_to_list(
                self.show_axes_at
            )
            structure["selections_axes_xyz_shift"] = self._convert_to_list(
                self.show_axes_at
            )

        # === Meta options ===
        if self.structure_title:
            structure["selections_descriptor"] = self.structure_title

        return vaspview.Widget(structure)

    def _verify(self, mode=None):
        self._raise_error_if_present_on_multiple_steps(self.grid_scalars, mode)
        self._raise_error_if_present_on_multiple_steps(self.ion_arrows, mode)
        self._raise_error_if_number_steps_inconsistent()
        self._raise_error_if_any_shape_is_incorrect()

    def _raise_error_if_present_on_multiple_steps(self, attributes, mode=None):
        if not attributes:
            return
        for attribute in attributes:
            try:
                if len(attribute.quantity) > 1:
                    if mode == "ngl":
                        raise exception.NotImplemented("""\
    Currently isosurfaces and ion arrows are implemented only for cases where there is only
    one frame in the trajectory. Make sure that either only one frame for the positions
    attribute is supplied with its corresponding grid scalar or ion arrow component.""")
            except AttributeError:
                pass

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
            if len(attribute) == 0 or not isinstance(attribute[0], np.ndarray):
                return attribute
            else:
                return [a.tolist() for a in attribute]
        if isinstance(attribute, tuple):
            if len(attribute) == 0 or not isinstance(attribute[0], np.ndarray):
                return list(attribute)
            else:
                return [a.tolist() for a in attribute]
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
