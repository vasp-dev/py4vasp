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
px = import_.optional("plotly.express")
interpolate = import_.optional("scipy.interpolate")


@dataclasses.dataclass
class Contour(trace.Trace):
    """Represents data on a 2d slice through the unit cell.

    This class creates a visualization of the data within the unit cell based on its
    configuration. It supports the creation of heatmaps, contour plots, and quiver plots
    for 2D data representations.

    For scalar data (heatmaps/contours), each data point corresponds to one point on the grid.
    For vector data (quiver plots), each data point should be a 2D vector within the plane.

    Examples
    --------
    Create a simple heatmap:

    >>> from py4vasp.graph import Lattice, Graph
    >>> lattice = Lattice(vectors=np.array([[3.0, 0.0], [0.0, 3.0]]))
    >>> data = np.random.rand(50, 50)
    >>> contour = Contour(
    ...     data=data,
    ...     lattice=lattice,
    ...     label="Charge Density",
    ...     colorbar_label="e/Å³",
    ... )
    >>> Graph(contour).show()

    Create a contour plot with isolevels:

    >>> contour = Contour(
    ...     data=data,
    ...     lattice=lattice,
    ...     label="Potential",
    ...     isolevels=True,
    ...     show_contour_values=True,
    ...     color_scheme="diverging",
    ... )
    >>> Graph(contour).show()

    Create a quiver plot for vector data:

    >>> vector_data = np.random.rand(2, 20, 20)
    >>> quiver = Contour(
    ...     data=vector_data,
    ...     lattice=lattice,
    ...     label="Current Density",
    ...     max_number_arrows=100,
    ... )
    >>> Graph(quiver).show()

    Use a custom color scheme with limits:

    >>> contour = Contour(
    ...     data=data,
    ...     lattice=lattice,
    ...     label="Energy",
    ...     color_scheme="positive",
    ...     color_limits=(0, 1.0),
    ... )
    >>> Graph(contour).show()
    """

    _interpolation_factor = 2
    """If the lattice does not align with the cartesian axes, the data is interpolated
    to by approximately this factor along each line."""
    _shift_label_pixels = 10
    "Shift the labels by this many pixels to avoid overlap."
    _interpolation_method = "linear"
    """Can be linear or cubic to determine interpolation behavior."""

    data: np.array
    """Grid data representing values in the plane spanned by the lattice vectors.

    - For scalar data (2D array): Shape should match the grid dimensions (ny, nx).
      Used for heatmaps and contour plots.
    - For vector data (3D array): Shape should be (2, ny, nx) where the first dimension
      contains the x and y components of vectors in the plane. Used for quiver plots.
    """

    lattice: "Lattice"
    """Lattice plane defining the visualization coordinate system.

    Should contain exactly 2 lattice vectors, each with 2 components (x, y).
    Any components normal to the visualization plane should be removed beforehand.
    The data grid points are distributed along these lattice vectors.
    """

    label: str
    """Descriptive label for this visualization.

    Used to identify this plot among multiple visualizations and appears in legends.
    Example: "Charge Density", "Potential", "Current Density"
    """

    colorbar_label: str = None
    """Label displayed on the colorbar axis.

    Typically includes the physical quantity and units.
    Example: "e/Å³", "eV", "electrons"
    If None, no colorbar label is shown.
    """

    isolevels: bool = False
    """Display mode for scalar data visualization.

    - True: Show contour lines with isolevels (constant value curves)
    - False: Show heatmap with continuous color gradients (default)
    Only applies to 2D scalar data, ignored for vector data.
    """

    show_contour_values: bool = None
    """Whether to display numerical values along contour lines.

    Only relevant when isolevels=True. If None, uses plotly's default behavior.
    Set to True to show the isovalues, False to hide them.
    """

    color_scheme: str = "auto"
    """Color mapping strategy for the visualization.

    Available options:
    - "auto" (default): Automatically select based on data range:
        * "diverging" if data spans negative and positive values
        * "positive" if all values are non-negative
        * "negative" if all values are non-positive
        * "default" otherwise
    - "monochrome" or "stm": Single-color gradient (suitable for STM images)
    - "sequential": Perceptually uniform progression (Viridis)
    - "positive": Red gradient for non-negative data
    - "negative": Reverse blue gradient for non-positive data
    - "diverging": Red-white-blue for data crossing zero (RdBu_r)

    Choose based on your data's physical meaning and range.
    """

    color_limits: tuple = None
    """Explicit bounds for the color scale mapping.

    Controls which data values map to the minimum and maximum colors:
    - None or (None, None): Use data's actual min/max (default)
    - (vmin, None): Set minimum, auto-detect maximum
    - (None, vmax): Auto-detect minimum, set maximum
    - (vmin, vmax): Set both bounds explicitly

    Useful for:
    - Comparing multiple plots on the same scale
    - Emphasizing specific data ranges
    - Clipping outliers
    """

    traces_as_periodic: bool = False
    """Alignment mode for visualization elements relative to the computational grid.

    - True: Align visualization elements (contours, arrows) with the actual grid points
      where data was computed. Periodic images are drawn to fully cover the supercell,
      providing physically accurate representation but potentially less aesthetic appearance.

    - False (default): Align heatmap cells with supercell boundaries for clean visual
      appearance. Grid points appear at cell corners rather than centers. No periodic
      images needed, but may be slightly misleading about where data was computed.

    Recommended: True for quantitative analysis, False for presentation graphics.
    """

    supercell: np.array = (1, 1)
    """Number of unit cell repetitions along each lattice vector.

    Array of 2 integers (na, nb) specifying how many times to tile the unit cell
    along the first and second lattice vectors. Useful for visualizing periodic
    patterns or showing context around a single cell.
    Default (1, 1) shows one unit cell.
    """

    show_cell: bool = True
    """Whether to draw the unit cell boundaries.

    - True (default): Draw outline of the unit cell as a box/parallelogram
    - False: Hide unit cell boundaries

    Helpful for understanding the periodicity and lattice geometry.
    """

    max_number_arrows: int = None
    """Maximum arrow count for quiver plots (vector data only).

    If the vector field grid has more points than this limit, data is automatically
    subsampled to reduce visual clutter. Subsampling is done uniformly in both
    directions, attempting to keep arrows evenly distributed.

    None (default): Show all arrows without subsampling.
    Recommended: ~100-500 for readable visualizations.
    """

    scale_arrows: float = None
    """Arrow length scaling factor for quiver plots (vector data only).

    Multiplier applied to vector magnitudes when converting to visual arrow lengths:
    - None (default): Automatically scale so the longest arrow equals the smaller
      of the two grid spacings (prevents overlap)
    - float > 0: Manual scaling factor. Larger values = longer arrows.
      Value of 1.0 means arrow length in Ångströms equals vector magnitude.

    Use manual scaling when comparing multiple plots or adjusting readability.
    """

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

        periodic_left = self._get_periodic_left(periodic_expand)
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
        corners = np.array([[0, 0], lattice[0], lattice[1], lattice[0] + lattice[1]])
        xmin, xmax = (min(corners[:, 0]), max(corners[:, 0]))
        ymin, ymax = (min(corners[:, 1]), max(corners[:, 1]))

        _num_periodic_add = min(data.shape) - 1
        periodic_expand = 1 + _num_periodic_add
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

        z_out = interpolate.griddata(
            (x_in, y_in), z_in, (x_out, y_out), method=self._interpolation_method
        )
        if self.traces_as_periodic:
            z_out = self._mask_outside_supercell(x_out, y_out, z_out, lattice)
        return x_out[0], y_out[:, 0], z_out

    def _use_data_without_interpolation(self, lattice, data):
        x = self._make_mesh(lattice, data.shape[1], 0)
        y = self._make_mesh(lattice, data.shape[0], 1)
        return (
            x,
            y,
            self._extend_data_contour(data) if (self.traces_as_periodic) else data,
        )

    def _get_periodic_left(self, periodic_expand: int) -> int:
        # When we periodically expand the data, we may wish to do so symmetrically.
        # This function returns the integer number of points to prepend to the line mesh,
        # data row or column. Generally, line meshes will need to be shifted by this number.
        periodic_left = 0
        if (self.traces_as_periodic) and (periodic_expand > 1):
            periodic_left = int(np.floor((periodic_expand - 1) / 2))
        return periodic_left

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

        periodic_left = self._get_periodic_left(periodic_expand)

        if not (self.traces_as_periodic):
            # shift the mesh by 0.5*cell_length so that heatmap cells are bottom-left anchored
            # rather than centered on the computed point, which makes it so rectangular boxes
            # are filled exactly and in a visually appealing way
            mesh = mesh + (0.5 * lattice[vector] / num_point)
        else:
            # shift the mesh by -periodic_left*cell_length to accommodate repeats in other direction
            # this is necessary to ensure that the heatmap cells are center-anchored and aligned correctly
            # especially in the presence of repeated rows/columns
            mesh = mesh - periodic_left * (lattice[vector] / float(num_point))
        return mesh

    def _mask_outside_supercell(self, x_out, y_out, z_out, lattice_supercell):
        # Mask points that are outside the supercell area.
        # Convert Cartesian coordinates to lattice coordinates
        # lattice_supercell has vectors as rows, so we need its inverse
        lattice_inv = np.linalg.inv(lattice_supercell)

        # For each point, get its position in lattice coordinates
        points_cart = np.column_stack([x_out.flatten(), y_out.flatten()])
        points_lattice = points_cart @ lattice_inv

        # Calculate adaptive tolerance based on grid resolution
        # Get the spacing between grid points in lattice coordinates
        if x_out.shape[1] > 1 and x_out.shape[0] > 1:
            # Calculate grid spacing in Cartesian coordinates
            dx_cart = abs(x_out[0, 1] - x_out[0, 0])
            dy_cart = abs(y_out[1, 0] - y_out[0, 0])

            # Convert grid spacing to lattice coordinates
            # A small displacement in Cartesian becomes this in lattice coordinates
            dx_lattice = abs(np.array([dx_cart, 0]) @ lattice_inv).max()
            dy_lattice = abs(np.array([0, dy_cart]) @ lattice_inv).max()

            # Use half the largest grid spacing as tolerance
            tolerance = 0.5 * max(dx_lattice, dy_lattice)
        else:
            # Fallback for edge cases
            tolerance = 0.025

        # Check if points are inside the unit cell with adaptive tolerance
        inside_mask = (
            (points_lattice[:, 0] >= -tolerance)
            & (points_lattice[:, 0] <= 1 + tolerance)
            & (points_lattice[:, 1] >= -tolerance)
            & (points_lattice[:, 1] <= 1 + tolerance)
        )

        # Create masked output
        z_out_masked = z_out.copy()
        z_out_masked.flat[~inside_mask] = np.nan

        return z_out_masked

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
        zmin, zmax = self._get_color_range(z)
        target_scheme = self.color_scheme

        tolerance = 1e-12 * (zmax - zmin)

        if self.color_scheme == "auto":
            if zmin == zmax:
                target_scheme = "default"
            elif zmin < -tolerance and zmax > tolerance:
                target_scheme = "diverging"
            elif zmin >= -tolerance:
                target_scheme = "positive"
            elif zmax <= tolerance:
                target_scheme = "negative"
            else:
                target_scheme = "default"

        colormaps_dict = Contour._get_colormap_themes()
        selected_color_map = colormaps_dict.get(
            target_scheme,
            colormaps_dict.get("default", Contour._get_fallback_colormap()),
        )

        return selected_color_map

    @staticmethod
    def _get_fallback_colormap() -> list[tuple[float, str]]:
        return [
            (0, _config.VASP_COLORS["blue"]),
            (0.5, "white"),
            (1, _config.VASP_COLORS["red"]),
        ]

    @staticmethod
    def _get_colormap_themes() -> dict[str, list[tuple[float, str]] | str]:
        return {
            "default": Contour._get_fallback_colormap(),
            "monochrome": "turbid_r",
            "positive": "Reds",
            "negative": "Blues_r",
            "sequential": "Viridis",
            "diverging": "RdBu_r",
        }

    def _get_color_range(self, z: np.ndarray) -> tuple:
        z_finite = z[np.isfinite(z)]
        if self.color_limits is None:
            return (np.min(z_finite), np.max(z_finite))
        else:
            assert len(self.color_limits) == 2
            zmin, zmax = self.color_limits
            if zmin is None and zmax is not None:
                return (np.min(z_finite), zmax)
            elif zmin is not None and zmax is None:
                return (zmin, np.max(z_finite))
            elif zmin is None and zmax is None:
                return (np.min(z_finite), np.max(z_finite))
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
