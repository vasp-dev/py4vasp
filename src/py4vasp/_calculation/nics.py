# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
from typing import Optional, Union

import numpy as np

from py4vasp import _config, exception
from py4vasp._calculation import _stoichiometry
from py4vasp._calculation.dispatch import DataSource, merge_default, merge_strings, quantity
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw import data as raw
from py4vasp._raw.data_db import Nics_DB
from py4vasp._third_party import graph, view
from py4vasp._util import check, documentation, import_, index, select, slicing

pretty = import_.optional("IPython.lib.pretty")

_DEFAULT_SELECTION: str = "isotropic"


class NicsHandler:
    """Handler for NICS data — performs all data access and transformation."""

    def __init__(self, raw_nics: raw.Nics):
        self._raw_nics = raw_nics

    @classmethod
    def from_data(cls, raw_nics: raw.Nics) -> "NicsHandler":
        return cls(raw_nics)

    def __str__(self) -> str:
        raw_stoichiometry = self._raw_nics.structure.stoichiometry
        stoichiometry = _stoichiometry.Stoichiometry.from_data(raw_stoichiometry)
        if self._data_is_on_grid:
            data_string = self._grid_to_string()
        else:
            data_string = self._points_to_string()
        return f"""\
nucleus-independent chemical shift:
    structure: {pretty.pretty(stoichiometry)}
{data_string}"""

    def read(self) -> dict:
        return self.to_dict()

    def to_dict(self) -> dict:
        result = {
            "structure": self._structure().read(),
            "nics": self.to_numpy(),
            **self._get_method_and_positions(),
        }
        return result

    def to_database(self) -> dict:
        method = "grid" if self._data_is_on_grid else "positions"
        return {"nics": Nics_DB(method=method)}

    def to_numpy(self, selection: Optional[str] = None):
        selected_data = self._read_selected_data(selection)
        return np.squeeze(list(selected_data.values()))

    def to_view(
        self,
        selection: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
        **user_options,
    ):
        self._raise_error_if_used_in_points_mode()
        selection = selection or _DEFAULT_SELECTION
        viewer = self._structure().to_view(supercell)
        viewer.grid_scalars = [
            self._make_grid_quantity(*item, user_options)
            for item in self._read_selected_data(selection).items()
        ]
        return viewer

    def to_contour(
        self,
        selection: Optional[str] = None,
        *,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        normal: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
    ):
        self._raise_error_if_used_in_points_mode()
        selection = selection or _DEFAULT_SELECTION
        cut, fraction = slicing.get_cut(a, b, c)
        plane = slicing.plane(self._structure().lattice_vectors(), cut, normal)
        contour_plots = [
            self._make_contour(*item, plane, fraction, supercell)
            for item in self._read_selected_data(selection).items()
        ]
        return graph.Graph(contour_plots)

    def _structure(self):
        return StructureHandler.from_data(self._raw_nics.structure)

    @property
    def _data_is_on_grid(self):
        return check.is_none(self._raw_nics.positions)

    def _read_selected_data(self, selection: Optional[str]):
        if self._data_is_on_grid:
            nics_data = np.array(self._raw_nics.nics_grid).T
        else:
            nics_data = np.array(self._raw_nics.nics_points)
            nics_data = nics_data.reshape((len(nics_data), 9))
        if selection is None:
            new_shape = (*nics_data.shape[:-1], 3, 3)
            return {None: nics_data.reshape(new_shape)}
        tree = select.Tree.from_selection(selection)
        maps = {nics_data.ndim - 1: self._init_directions_dict()}
        selector = index.Selector(maps, nics_data, reduction=_TensorReduction)
        return {
            selector.label(selection): selector[selection]
            for selection in tree.selections()
        }

    @staticmethod
    def _init_directions_dict():
        return {
            "isotropic": [0, 4, 8],
            "xx": 0,
            "xy": 1,
            "xz": 2,
            "yx": 3,
            "yy": 4,
            "yz": 5,
            "zx": 6,
            "zy": 7,
            "zz": 8,
            "11": slice(None),
            "22": slice(None),
            "33": slice(None),
            "span": slice(None),
            "skew": slice(None),
            "anisotropy": slice(None),
            "asymmetry": slice(None),
        }

    def _get_method_and_positions(self):
        if self._data_is_on_grid:
            return {"method": "grid"}
        else:
            return {"method": "positions", "positions": self._raw_nics.positions[:].T}

    def _grid_to_string(self):
        grid = self._raw_nics.nics_grid.shape[1:]
        return f"""    grid: {grid[2]}, {grid[1]}, {grid[0]}
    tensor shape: 3x3"""

    def _points_to_string(self):
        positions = self._raw_nics.positions[:].T
        tensors = self.to_numpy()
        return "\n\n".join(self._format_nics(*item) for item in zip(positions, tensors))

    def _format_nics(self, position, tensor):
        position_string = " ".join(f"{x:10.6f}" for x in position)
        newline_with_indent = "\n        "
        tensor = np.round(tensor, 14)
        tensor_string = newline_with_indent.join(
            "   ".join(f"{x:+.6e}" for x in column) for column in tensor
        )
        return f"""\
    NICS at {position_string}: |
        {tensor_string}"""

    def _make_grid_quantity(self, key, quantity, user_options):
        return view.GridQuantity(
            quantity=quantity[np.newaxis],
            label=f"{key} NICS",
            isosurfaces=self._isosurfaces(**user_options),
        )

    def _isosurfaces(self, isolevel=1.0, opacity=0.6):
        return [
            view.Isosurface(isolevel, _config.VASP_COLORS["blue"], opacity),
            view.Isosurface(-isolevel, _config.VASP_COLORS["red"], opacity),
        ]

    def _make_contour(self, key, data, plane, fraction, supercell):
        grid_scalar = slicing.grid_scalar(data, plane, fraction)
        label = f"{key} NICS contour ({plane.cut})"
        contour_plot = graph.Contour(grid_scalar, plane, label, isolevels=True)
        if supercell is not None:
            contour_plot.supercell = np.ones(2, dtype=np.int_) * supercell
        return contour_plot

    def _raise_error_if_used_in_points_mode(self):
        if self._data_is_on_grid:
            return
        raise exception.IncorrectUsage(
            "You set LNICSALL = .FALSE. in the INCAR file. This mode is incompatible with the plotting routines."
        )


@quantity("nics")
class Nics(view.Mixin):
    """This class accesses information on the nucleus-independent chemical shift (NICS)."""

    def __init__(self, source, quantity_name="nics"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_nics):
        return cls(source=DataSource(raw_nics))

    def _handler_factory(self, raw):
        return NicsHandler.from_data(raw)

    def __str__(self):
        return merge_strings(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            NicsHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection=None) -> dict:
        """Read NICS into a dictionary."""
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            NicsHandler.read,
        )

    def to_dict(self, selection=None) -> dict:
        """Alias for read()."""
        return self.read(selection=selection)

    def to_numpy(self, selection: Optional[str] = None):
        """Convert NICS to a numpy array."""
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            NicsHandler.to_numpy,
            selection,
        )

    def to_view(
        self,
        selection: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
        **user_options,
    ):
        """Plot the selected chemical shift as a 3d isosurface within the structure."""
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            NicsHandler.to_view,
            selection,
            supercell=supercell,
            **user_options,
        )

    def to_contour(
        self,
        selection: Optional[str] = None,
        *,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        normal: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
    ):
        """Generate a contour plot of chemical shift."""
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            NicsHandler.to_contour,
            selection,
            a=a,
            b=b,
            c=c,
            normal=normal,
            supercell=supercell,
        )


class _TensorReduction(index.Reduction):
    def __init__(self, keys):
        keys_using_average = "isotropic xx xy xz yx yy yz zx zy zz"
        self._use_average = keys[-1] in keys_using_average
        self._selection = keys[-1]

    def __call__(self, array, axis):
        if self._use_average:
            return np.average(array, axis=axis)
        else:
            return self._reduce(array, axis)

    def _reduce(self, array, axis):
        array = array.reshape((*array.shape[:-1], 3, 3))
        symmetric_array = 0.5 * (array + np.moveaxis(array, -2, -1))
        eigenvalues = np.linalg.eigvalsh(array)
        if self._selection == "11":
            return eigenvalues[..., 2]
        if self._selection == "22":
            return eigenvalues[..., 1]
        if self._selection == "33":
            return eigenvalues[..., 0]
        if self._selection == "span":
            return eigenvalues[..., 2] - eigenvalues[..., 0]
        if self._selection == "skew":
            span = eigenvalues[..., 2] - eigenvalues[..., 0]
            return (3 * eigenvalues[..., 1] - np.sum(eigenvalues, axis=-1)) / span
        if self._selection in ("anisotropy", "asymmetry"):
            return self._haeberlen_mehring(eigenvalues)[self._selection]
        message = f"The reduction for selection '{self._selection}' is not implemented."
        raise exception.NotImplemented(message)

    def _haeberlen_mehring(self, eigenvalues):
        delta_iso = np.average(eigenvalues, axis=-1)
        mask = delta_iso < eigenvalues[..., 1]
        delta_xx = np.where(mask, eigenvalues[..., 2], eigenvalues[..., 0])
        delta_zz = np.where(mask, eigenvalues[..., 0], eigenvalues[..., 2])
        anisotropy = delta_zz - delta_iso
        asymmetry = (eigenvalues[..., 1] - delta_xx) / anisotropy
        return {"anisotropy": anisotropy, "asymmetry": asymmetry}
