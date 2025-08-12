# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

import dataclasses
import itertools

import numpy as np

from py4vasp import _config
from py4vasp._third_party.graph import trace
from py4vasp._util import import_
from py4vasp._util.slicing import Plane

ff = import_.optional("plotly.figure_factory")
go = import_.optional("plotly.graph_objects")
px = import_.optional("plotly.express")
interpolate = import_.optional("scipy.interpolate")

_COLORMAP_LIST = px.colors.named_colorscales()
_COLORMAP_LIST_R = [c + "_r" for c in _COLORMAP_LIST]


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
    colorbar_label: str = None
    """Label to show at the colorbar."""
    isolevels: bool = False
    "Defines whether isolevels should be added or a heatmap is used."
    show_contour_values: bool = None
    "Defines whether contour values should be shown along contour plot lines."
    color_scheme: str = "stm"
    """The color_scheme argument informs the chosen color map and parameters for the contours plot.
    It should be chosen according to the nature of the data to be plotted, as one of the following:
    - "auto": py4vasp will try to infer the color scheme on its own.
    - "monochrome" OR "stm": (Default) Standard colorscheme for stm.
    - "positive": Values are only positive.
    - "signed": Values are mixed - positive and negative.
    - "negative": Values are only negative.
    
    Additionally, any of the color maps listed in _COLORMAP_LIST (and their names appended with 
    "_r") are also valid inputs, but the list itself might be subject to change on future releases.
    """
    color_limits: tuple = None
    """Is a tuple that sets the minimum and maximum of the color scale. Can be:
    - None | (None, None): No limits are imposed.
    - (float, None): Sets the minimum of the color scale.
    - (None, float): Sets the maximum of the color scale.
    - (float, float): Sets minimum and maximum of the color scale."""
    traces_as_periodic: bool = False
    """If True, traces (contour and quiver) are shifted so that quiver and heatmap 'cell' 
    centers align with the positions they were computed at. Periodic images will be drawn
    so that the supercell still appears completely covered on all sides.

    If False, traces (contour and quiver) are shifted so that the heatmap cells visually
    align with the supercell instead. No periodic images are required, but the visual
    presentation might be misleading."""
    num_periodic_add: int = 0
    """The number of periodic rows and columns (>= 0) of heatmap/contour cells added to the plot 
    if and only if interpolation is required and traces_as_periodic is True. 
    by default, traces_as_periodic will cause the first row and first column to be repeated.
    num_periodic_add can be used to repeat additional rows and columns.
    Periodicity will be enforced in the direction of lattice vectors first, then alternate.

    Example:
    num_periodic_add = 2
    
    ```
    o4|o1o2o3o4|o1o2
       --------
    m4|m1m2m3m4|m1m2
    n4|n1n2n3n4|n1n2
    o4|o1o2o3o4|o1o2
       --------
    m4|m1m2m3m4|m1m2
    n4|n1n2n3n4|n1n2
    ```
    """
    supercell: np.array = (1, 1)
    "Multiple of each lattice vector to be drawn."
    show_cell: bool = True
    "Show the unit cell in the resulting visualization."
    max_number_arrows: int = None
    "Subsample the data until the number of arrows falls below this limit."
    scale_arrows: float = None
    """Scale arrows by this factor when converting their length to Å. None means
    autoscale them so that the arrows do not overlap."""

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
        zmin, zmax = self._get_color_range(z)
        return go.Contour(
            x=x,
            y=y,
            z=z,
            name=self.label,
            autocontour=True,
            colorscale=self._get_color_scale(z),
            colorbar=self._get_color_bar(),
            zmin=zmin,
            zmax=zmax,
            contours={"showlabels": self.show_contour_values},
        )

    def _make_heatmap(self, lattice, data):
        x, y, z = self._interpolate_data_if_necessary(lattice, data)
        zmin, zmax = self._get_color_range(z)
        return go.Heatmap(
            x=x,
            y=y,
            z=z,
            name=self.label,
            colorscale=self._get_color_scale(z),
            colorbar=self._get_color_bar(),
            zmin=zmin,
            zmax=zmax,
        )

    def _interpolate_data_if_necessary(self, lattice, data):
        if self._interpolation_required():
            x, y, z = self._interpolate_data(lattice, data)
        else:
            x, y, z = self._use_data_without_interpolation(lattice, data)
        return x, y, z

    def _make_quiver(self, lattice, data):
        subsamples = self._limit_number_of_arrows(data.size)
        # remember that b and a axis are swapped
        vectors = reversed(lattice)
        meshes_raw = [
            np.linspace(
                np.zeros(2),
                vector,
                num_points + (1 if self.traces_as_periodic else 0),
                endpoint=self.traces_as_periodic,
            )
            for vector, num_points in zip(vectors, data.shape)
        ]
        dx = np.linalg.norm(meshes_raw[0][1])
        dy = np.linalg.norm(meshes_raw[1][1])
        meshes = [v[::subsample] for v, subsample in zip(meshes_raw, subsamples)]
        subsampled_data = (
            self._subsample_data_quiver(data, subsamples)
            if self.traces_as_periodic
            else data[:: subsamples[0], :: subsamples[1]]
        )
        if self.scale_arrows is None:
            # arrows may be at most as long as the shorter lattice vector
            max_length = min(np.linalg.norm(meshes[0][1]), np.linalg.norm(meshes[1][1]))
            current_max_length = np.max(np.linalg.norm(subsampled_data, axis=-1))
            scale = max_length / current_max_length
        else:
            scale = self.scale_arrows
        x, y = np.array([sum(points) for points in itertools.product(*meshes)]).T
        u = scale * subsampled_data[:, :, 0].flatten()
        v = scale * subsampled_data[:, :, 1].flatten()
        fig = ff.create_quiver(
            x - 0.5 * u + ((0.5 * dy) if not (self.traces_as_periodic) else 0.0),
            y - 0.5 * v + ((0.5 * dx) if not (self.traces_as_periodic) else 0.0),
            u,
            v,
            scale=1,
        )
        fig.data[0].line.color = _config.VASP_COLORS["dark"]
        return fig.data[0]

    def _subsample_data_quiver(self, data, subsamples):
        xdim, ydim, _ = data.shape

        # Create index arrays with "wrapped" boundaries
        x_indices = (
            np.arange(0, xdim + (1 if self.traces_as_periodic else 0), subsamples[0])
            % xdim
        )
        y_indices = (
            np.arange(0, ydim + (1 if self.traces_as_periodic else 0), subsamples[1])
            % ydim
        )

        # Access data using advanced indexing
        subsampled_data = data[np.ix_(x_indices, y_indices)]
        return subsampled_data

    def _extend_data_contour(self, data, periodic_expand=1):
        xdim, ydim = data.shape

        periodic_left = 0
        if (self.traces_as_periodic) and (periodic_expand > 1):
            periodic_left = int(np.floor((periodic_expand - 1) / 2))
        # Create index arrays with "wrapped" boundaries
        x_indices = (np.arange(0, xdim + periodic_expand) % xdim) - periodic_left
        y_indices = (np.arange(0, ydim + periodic_expand) % ydim) - periodic_left

        # Access data using advanced indexing
        subsampled_data = data[np.ix_(x_indices, y_indices)]
        return subsampled_data

    def _limit_number_of_arrows(self, data_size):
        subsamples = [1, 1]
        if self.max_number_arrows is None:
            return subsamples
        data_size /= 2  # ignore dimension of arrow
        while data_size / np.prod(subsamples) > self.max_number_arrows:
            if subsamples[0] <= subsamples[1]:
                subsamples[0] += 1
            else:
                subsamples[1] += 1
        return subsamples

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
        # obtain min and max for final grid
        line_mesh_a = self._make_mesh(lattice, data.shape[1], 0, periodic_expand=1)
        line_mesh_b = self._make_mesh(lattice, data.shape[0], 1, periodic_expand=1)
        x_in, y_in = (line_mesh_a[:, np.newaxis] + line_mesh_b[np.newaxis, :]).T
        x_in = x_in.flatten()
        y_in = y_in.flatten()
        xmin, xmax = x_in.min(), x_in.max()
        ymin, ymax = y_in.min(), y_in.max()

        periodic_expand = 1 + self.num_periodic_add
        line_mesh_a = self._make_mesh(
            lattice, data.shape[1], 0, periodic_expand=periodic_expand
        )
        line_mesh_b = self._make_mesh(
            lattice, data.shape[0], 1, periodic_expand=periodic_expand
        )
        x_in, y_in = (line_mesh_a[:, np.newaxis] + line_mesh_b[np.newaxis, :]).T
        x_in = x_in.flatten()
        y_in = y_in.flatten()
        z_in = (
            self._extend_data_contour(data, periodic_expand=periodic_expand).flatten()
            if (self.traces_as_periodic)
            else data.flatten()
        )

        # make sure the actual grid aligns with shifts
        x_line_mesh = np.linspace(
            xmin,
            xmax,
            shape[0] + (1 if self.traces_as_periodic else 0),
            endpoint=self.traces_as_periodic,
        )
        y_line_mesh = np.linspace(
            ymin,
            ymax,
            shape[1] + (1 if self.traces_as_periodic else 0),
            endpoint=self.traces_as_periodic,
        )
        x_out, y_out = np.meshgrid(
            x_line_mesh,
            y_line_mesh,
        )

        z_out = interpolate.griddata((x_in, y_in), z_in, (x_out, y_out), method="cubic")
        return x_out[0], y_out[:, 0], z_out

    def _use_data_without_interpolation(self, lattice, data):
        x = self._make_mesh(lattice, data.shape[1], 0)
        y = self._make_mesh(lattice, data.shape[0], 1)
        return (
            x,
            y,
            self._extend_data_contour(data) if (self.traces_as_periodic) else data,
        )

    def _make_mesh(self, lattice, num_point, index, periodic_expand: int = 1):
        vector = index if self._interpolation_required() else (index, index)

        endpoint = lattice[vector]
        if (self.traces_as_periodic) and (periodic_expand > 0):
            endpoint = endpoint + float(periodic_expand - 1) * (
                lattice[vector] / float(num_point)
            )
        mesh = np.linspace(
            0,
            endpoint,
            num_point + (periodic_expand if (self.traces_as_periodic) else 0),
            endpoint=self.traces_as_periodic,
        )

        periodic_left = 0
        if (self.traces_as_periodic) and (periodic_expand > 1):
            periodic_left = np.floor((periodic_expand - 1) / 2)

        if not (self.traces_as_periodic):
            mesh = mesh + (0.5 * lattice[vector] / num_point)
        else:
            mesh = mesh - periodic_left * (lattice[vector] / float(num_point))
        return mesh

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

    def _get_color_scale(self, z: np.ndarray):
        selected_color_scheme = None
        color_lower = _config.VASP_COLORS["blue"]
        color_center = "white"
        color_upper = _config.VASP_COLORS["red"]
        zmin, zmax = self._get_color_range(z)
        if self.color_scheme in ["monochrome", "stm"]:
            selected_color_scheme = "turbid"
        elif (self.color_scheme == "signed") or (
            self.color_scheme == "auto" and (zmin < 0 and zmax > 0)
        ):
            selected_color_scheme = [
                [0, color_lower],
                [0.5, color_center],
                [1, color_upper],
            ]
        elif (self.color_scheme == "positive") or (
            self.color_scheme == "auto" and (zmin >= 0)
        ):
            selected_color_scheme = [[0, color_center], [1, color_upper]]
        elif (self.color_scheme == "negative") or (
            self.color_scheme == "auto" and (zmax <= 0)
        ):
            selected_color_scheme = [[0, color_lower], [1, color_center]]
        # Defaulting to color map if not yet set
        if selected_color_scheme is None:
            if self.color_scheme in (_COLORMAP_LIST + _COLORMAP_LIST_R):
                selected_color_scheme = self.color_scheme
            else:
                selected_color_scheme = [
                    [0, color_lower],
                    [0.5, color_center],
                    [1, color_upper],
                ]

        return selected_color_scheme

    def _get_color_range(self, z: np.ndarray) -> tuple:
        if self.color_limits is None:
            return (np.min(z), np.max(z))
        else:
            assert len(self.color_limits) == 2
            zmin, zmax = self.color_limits
            if zmin is None and zmax is not None:
                return (np.min(z), zmax)
            elif zmin is not None and zmax is None:
                return (zmin, np.max(z))
            elif zmin is None and zmax is None:
                return (np.min(z), np.max(z))
            else:
                return (zmin, zmax)

    def _get_color_bar(self):
        if (self.colorbar_label is not None) and (self.colorbar_label):
            return {"title": {"text": f"{self.colorbar_label}", "side": "right"}}
        else:
            return None

    def _label_unit_cell_vectors(self):
        if self.lattice.cut is None:
            return []
        vectors = self.lattice.vectors
        labels = self.lattice.map_raw_labels(tuple("abc".replace(self.lattice.cut, "")))
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
