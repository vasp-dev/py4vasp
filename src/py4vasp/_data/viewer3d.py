# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import collections
import os
import tempfile
from typing import NamedTuple

import mrcfile
import nglview
import numpy as np

from py4vasp import exception


class _Arrow3d(NamedTuple):
    tail: np.ndarray
    tip: np.ndarray
    color: np.ndarray
    radius: float = 0.2

    def to_serializable(self):
        return list(self.tail), list(self.tip), list(self.color), float(self.radius)


_x_axis = _Arrow3d(tail=np.zeros(3), tip=np.array((3, 0, 0)), color=[1, 0, 0])
_y_axis = _Arrow3d(tail=np.zeros(3), tip=np.array((0, 3, 0)), color=[0, 1, 0])
_z_axis = _Arrow3d(tail=np.zeros(3), tip=np.array((0, 0, 3)), color=[0, 0, 1])


class Viewer3d:
    """Collection of data and elements to be displayed in a structure viewer.

    Parameters
    ----------
    viewer : nglview.NGLWidget
        The raw viewer used to display the structure. Currently we are only
        supporting the nglview package.
    """

    _positions = None
    _lengths = None
    _angles = None
    _number_cells = 1
    _axes = None

    def __init__(self, viewer):
        self._ngl = viewer
        self._arrows = []

    @classmethod
    def from_structure(cls, structure, supercell=None):
        """Generate a new Viewer3d from a structure.

        Parameters
        ----------
        structure : data.Structure
            Defines the structure of the Vasp calculation.
        supercell : int or np.ndarray
            If present the cell is extended by the specified factor along each axis.
        """
        ase = structure.to_ase(supercell)
        # ngl works with the standard form, so we need to store the positions in the same format
        standard_cell, _ = ase.cell.standard_form()
        ase.set_cell(standard_cell, scale_atoms=True)
        res = cls(nglview.show_ase(ase))
        res._lengths = tuple(ase.cell.lengths())
        res._angles = tuple(ase.cell.angles())
        res._positions = ase.positions
        res._number_cells = res._calculate_number_cells(supercell)
        return res

    def _calculate_number_cells(self, supercell):
        if isinstance(supercell, collections.abc.Iterable):
            return np.prod(supercell)
        if isinstance(supercell, int):
            return supercell**3
        if isinstance(supercell, float) and supercell.is_integer():
            return supercell**3
        return 1

    @classmethod
    def from_trajectory(cls, trajectory):
        """Generate a new Viewer3d from a trajectory.

        Parameters
        ----------
        trajectory : data.Structure
            Defines the trajectory of the Vasp MD run.
        supercell : int or np.ndarray
            If present the cell is extended by the specified factor along each axis.
        """
        ngl_trajectory = nglview.MDTrajTrajectory(trajectory.to_mdtraj())
        return cls(nglview.NGLWidget(ngl_trajectory))

    def _ipython_display_(self):
        self._ngl._ipython_display_()

    def show_cell(self):
        """Show the unit cell of the crystal."""
        self._ngl.add_unitcell()

    def hide_cell(self):
        """Hide the unit cell of the crystal."""
        self._ngl.remove_unitcell()

    def show_axes(self):
        """Show the cartesian axis in the corner of the figure."""
        if self._axes is not None:
            return
        self._axes = (
            self._make_arrow(_x_axis),
            self._make_arrow(_y_axis),
            self._make_arrow(_z_axis),
        )

    def hide_axes(self):
        """Hide the cartesian axis."""
        if self._axes is None:
            return
        for axis in self._axes:
            self._ngl.remove_component(axis)
        self._axes = None

    def show_arrows_at_atoms(self, arrows, color=[0.1, 0.1, 0.8]):
        """Add arrows at all the atoms.

        Parameters
        ----------
        arrows : np.ndarray
            An array containing the direction of an arrow for every atom in the
            unit cell. This arrow will be drawn in the figure.
        color : np.ndarray
            rgb values of the arrow, defaulting to blue.

        Notes
        -----
        If you are working on a supercell, the code will automatically extend the
        size of the array to show arrows in the supercell, too.
        """
        if self._positions is None:
            raise exception.RefinementError("Positions of atoms are not known.")
        arrows = np.repeat(arrows, self._number_cells, axis=0)
        for tail, arrow in zip(self._positions, arrows):
            tip = tail + arrow
            arrow = _Arrow3d(tail, tip, color)
            self._arrows.append(self._make_arrow(arrow))

    def hide_arrows_at_atoms(self):
        """Remove all arrows from the atoms.

        Notes
        -----
        If two different kind of atoms have been added to the system, there is
        currently no option to distinguish between them."""
        for arrow in self._arrows:
            self._ngl.remove_component(arrow)
        self._arrows = []

    def _make_arrow(self, arrow):
        return self._ngl.shape.add_arrow(*(arrow.to_serializable()))

    def show_isosurface(self, volume_data, **kwargs):
        """Add an isosurface to the structure.

        Parameters
        ----------
        volume_data : np.ndarray
            The raw data represented on a 3d grid. Make sure the grid aligns with
            the FFT grid used in the Vasp calculation.
        kwargs
            Additional parameters passed on to the visualizer. Most relevant is
            the isolevel changing the position at which the isosurface is drawn.
        """
        with tempfile.TemporaryDirectory() as tmp:
            filename = os.path.join(tmp, "data.mrc")
            self._make_mrc_file(filename, volume_data)
            self._make_isosurface(filename, **kwargs)

    def _make_mrc_file(self, filename, volume_data):
        with mrcfile.new(filename, overwrite=True) as data_file:
            data_file.set_data(volume_data.astype(np.float32))
            data_file.header.cella = self._lengths
            data_file.header.cellb = self._angles

    def _make_isosurface(self, filename, **kwargs):
        if len(kwargs) == 0:
            self._ngl.add_component(filename)
        else:
            component = self._ngl.add_component(filename, defaultRepresentation=False)
            component.add_surface(**kwargs)
