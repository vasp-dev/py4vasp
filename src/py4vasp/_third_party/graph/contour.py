# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

import dataclasses
import itertools

import numpy as np

from py4vasp import _config
from py4vasp._third_party.graph import trace
from py4vasp._util import import_

ff = import_.optional("plotly.figure_factory")
go = import_.optional("plotly.graph_objects")
interpolate = import_.optional("scipy.interpolate")


@dataclasses.dataclass
class Contour(trace.Trace):
    """Represents data on a 2d slice through the unit cell.

    This class creates a visualization of the data within the unit cell based on its
    configuration. Currently it supports the creation of heatmaps and quiver plots.
    For heatmaps each data point corresponds to one point on the grid. For quiver plots
    each data point should be a 2d vector within the plane.
    """

    _interpolation_factor = 2
    """If the lattice does not align with the cartesian axes, the data is interpolated
    to by approximately this factor along each line."""
    _shift_label_pixels = 10
    "Shift the labels by this many pixels to avoid overlap."

    data: np.array
    """2d or 3d grid data in the plane spanned by the lattice vectors. If the data is
    the dimensions should be the ones of the grid, if the data is 3d the first dimension
    should be a 2 for a vector in the plane of the grid and the other two dimensions
    should be the grid."""
    lattice: Plane
    """Lattice plane in which the data is represented spanned by 2 vectors.
    Each vector should have two components, so remove any element normal to
    the plane. Can be generated with the 'plane' function in py4vasp._util.slicing."""
    label: str
    "Assign a label to the visualization that may be used to identify one among multiple plots."
    isolevels: bool = False
    "Defines whether isolevels should be added or a heatmap is used."
    supercell: np.array = (1, 1)
    "Multiple of each lattice vector to be drawn."
    show_cell: bool = True
    "Show the unit cell in the resulting visualization."

    def to_plotly(self):
        lattice_supercell = np.diag(self.supercell) @ self.lattice.vectors
        # swap a and b axes because that is the way plotly expects the data
        data = np.tile(self.data, self.supercell).T
        if self._is_contour():
            yield self._make_contour(lattice_supercell, data), self._options()
        elif self._is_heatmap():
            yield self._make_heatmap(lattice_supercell, data), self._options()
        else:
            yield self._make_quiver(lattice_supercell, data), self._options()

    def _is_contour(self):
        return self.data.ndim == 2 and self.isolevels

    def _is_heatmap(self):
        return self.data.ndim == 2 and not self.isolevels

    def _make_contour(self, lattice, data):
        x, y, z = self._interpolate_data_if_necessary(lattice, data)
        return go.Contour(x=x, y=y, z=z, name=self.label, autocontour=True)

    def _make_heatmap(self, lattice, data):
        x, y, z = self._interpolate_data_if_necessary(lattice, data)
        return go.Heatmap(x=x, y=y, z=z, name=self.label, colorscale="turbid_r")

    def _make_quiver(self, lattice, data):
        u = data[:, :, 0].flatten()
        v = data[:, :, 1].flatten()
        meshes = [
            np.linspace(np.zeros(2), vector, num_points, endpoint=False)
            for vector, num_points in zip(reversed(lattice), data.shape)
            # remember that b and a axis are swapped
        ]
        x, y = np.array([sum(points) for points in itertools.product(*meshes)]).T
        fig = ff.create_quiver(x, y, u, v, scale=1)
        return fig.data[0]

    def _interpolate_data_if_necessary(self, lattice, data):
        if self._interpolation_required():
            x, y, z = self._interpolate_data(lattice, data)
        else:
            x, y, z = self._use_data_without_interpolation(lattice, data)
        return x, y, z

    def _interpolation_required(self):
        y_position_first_vector = self.lattice.vectors[0, 1]
        x_position_second_vector = self.lattice.vectors[1, 0]
        return not np.allclose((y_position_first_vector, x_position_second_vector), 0)

    def _interpolate_data(self, lattice, data):
        area_cell = abs(np.cross(lattice[0], lattice[1]))
        points_per_area = data.size / area_cell
        points_per_line = np.sqrt(points_per_area) * self._interpolation_factor
        lengths = np.sum(np.abs(lattice), axis=0)
        shape = np.ceil(points_per_line * lengths).astype(int)
        line_mesh_a = self._make_mesh(lattice, data.shape[1], 0)
        line_mesh_b = self._make_mesh(lattice, data.shape[0], 1)
        x_in, y_in = (line_mesh_a[:, np.newaxis] + line_mesh_b[np.newaxis, :]).T
        x_in = x_in.flatten()
        y_in = y_in.flatten()
        z_in = data.flatten()
        x_out, y_out = np.meshgrid(
            np.linspace(x_in.min(), x_in.max(), shape[0]),
            np.linspace(y_in.min(), y_in.max(), shape[1]),
        )
        z_out = interpolate.griddata((x_in, y_in), z_in, (x_out, y_out), method="cubic")
        return x_out[0], y_out[:, 0], z_out

    def _use_data_without_interpolation(self, lattice, data):
        x = self._make_mesh(lattice, data.shape[1], 0)
        y = self._make_mesh(lattice, data.shape[0], 1)
        return x, y, data

    def _make_mesh(self, lattice, num_point, index):
        vector = index if self._interpolation_required() else (index, index)
        return (
            np.linspace(0, lattice[vector], num_point, endpoint=False)
            + 0.5 * lattice[vector] / num_point
        )

    def _options(self):
        return {
            "shapes": self._create_unit_cell(),
            "annotations": self._label_unit_cell_vectors(),
        }

    def _create_unit_cell(self):
        if not self.show_cell:
            return ()
        pos_to_str = lambda pos: f"{pos[0]} {pos[1]}"
        vectors = self.lattice.vectors
        corners = (vectors[0], vectors[0] + vectors[1], vectors[1])
        to_corners = (f"L {pos_to_str(corner)}" for corner in corners)
        path = f"M 0 0 {' '.join(to_corners)} Z"
        color = _config.VASP_COLORS["dark"]
        unit_cell = {"type": "path", "line": {"color": color}, "path": path}
        return (unit_cell,)

    def _label_unit_cell_vectors(self):
        if self.lattice.cut is None:
            return []
        vectors = self.lattice.vectors
        labels = tuple("abc".replace(self.lattice.cut, ""))
        return [
            {
                "text": label,
                "showarrow": False,
                "x": 0.5 * vectors[i, 0],
                "y": 0.5 * vectors[i, 1],
                **self._shift_label(vectors[i], vectors[1 - i]),
            }
            for i, label in enumerate(labels)
        ]

    def _shift_label(self, current_vector, other_vector):
        invert = np.cross(current_vector, other_vector) < 0
        norm = np.linalg.norm(current_vector)
        shifts = self._shift_label_pixels * current_vector[::-1] / norm
        if invert:
            return {
                "xshift": -shifts[0],
                "yshift": shifts[1],
            }
        else:
            return {
                "xshift": shifts[0],
                "yshift": -shifts[1],
            }
