# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

import pathlib
from typing import Any, Iterable, List, Optional

import numpy as np
from numpy.typing import ArrayLike

from py4vasp import exception
from py4vasp._calculation import projector
from py4vasp._calculation._dispersion import DispersionHandler
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._calculation.kpoint import KpointHandler
from py4vasp._calculation.projector import ProjectorHandler
from py4vasp._raw import data as raw
from py4vasp._raw.data_db import Band_DB
from py4vasp._third_party import graph
from py4vasp._util import check, database, import_, index, select, slicing

pd = import_.optional("pandas")
pretty = import_.optional("IPython.lib.pretty")

_OCCUPATION_CUTOFF = 1e-2


class BandHandler:
    """Handler for electronic band structure data."""

    def __init__(self, raw_band: raw.Band):
        self._raw_band = raw_band

    @classmethod
    def from_data(cls, raw_band: raw.Band) -> "BandHandler":
        return cls(raw_band)

    def __str__(self) -> str:
        return f"""
{"spin polarized" if self._is_collinear() else ""} band data:
    {self._raw_band.dispersion.eigenvalues.shape[1]} k-points
    {self._raw_band.dispersion.eigenvalues.shape[2]} bands
{str(self._projector())}
    """.strip()

    def read(self, selection=None, fermi_energy=None) -> dict:
        return self.to_dict(selection, fermi_energy)

    def to_dict(self, selection=None, fermi_energy=None) -> dict[str, Any]:
        dispersion = self._dispersion().to_dict()
        eigenvalues = dispersion.pop("eigenvalues")
        return {
            **dispersion,
            "fermi_energy": self._raw_band.fermi_energy,
            **self._shift_dispersion_by_fermi_energy(eigenvalues, fermi_energy),
            **self._read_occupations(),
            **self._read_projections(selection),
        }

    def to_database(self, selection=None, fermi_energy=None) -> dict:
        dispersion = self._dispersion().to_database()

        occupations = self._read_occupations()
        num_total_occupied = occupations.get("occupations", None)
        num_checked_bands = None
        if num_total_occupied is not None:
            num_checked_bands = np.shape(num_total_occupied)[-1]
            num_total_occupied = int(
                np.max(np.sum(num_total_occupied > _OCCUPATION_CUTOFF, axis=-1))
            )
        num_occupied_up = occupations.get("occupations_up", None)
        num_occupied_down = occupations.get("occupations_down", None)
        if num_occupied_up is not None:
            num_checked_bands = np.shape(num_occupied_up)[-1]
            num_occupied_up = int(
                np.max(np.sum(num_occupied_up > _OCCUPATION_CUTOFF, axis=-1))
            )
        if num_occupied_down is not None:
            num_occupied_down = int(
                np.max(np.sum(num_occupied_down > _OCCUPATION_CUTOFF, axis=-1))
            )

        raw_fermi_energy = (
            self._raw_band.fermi_energy
            if not check.is_none(self._raw_band.fermi_energy)
            else None
        )

        return database.combine_db_dicts(
            {
                "band": Band_DB(
                    num_considered_bands=num_checked_bands,
                    num_occupied_bands=num_total_occupied,
                    num_occupied_bands_up=num_occupied_up,
                    num_occupied_bands_down=num_occupied_down,
                    fermi_energy_raw=raw_fermi_energy,
                    fermi_energy=fermi_energy or raw_fermi_energy,
                ),
            },
            dispersion,
        )

    def to_graph(self, selection=None, fermi_energy=None, width=0.5) -> graph.Graph:
        projections = self._projections(selection, width)
        result = self._dispersion().plot(projections)
        result = self._shift_series_by_fermi_energy(result, fermi_energy)
        result.ylabel = "Energy (eV)"
        return result

    def to_frame(self, selection=None, fermi_energy=None):
        return pd.DataFrame(self._extract_relevant_data(selection, fermi_energy))

    def to_quiver(self, selection, normal=None, supercell=None) -> graph.Graph:
        reciprocal_lattice_vectors = self._kpoint()._reciprocal_lattice_vectors()
        nkp1, nkp2, cut = self._kmesh()
        options = {
            "lattice": slicing.plane(
                reciprocal_lattice_vectors, cut, normal, axis_labels=("b1", "b2", "b3")
            )
        }
        if supercell is not None:
            options["supercell"] = np.ones(2, dtype=np.int_) * supercell
        selector = self._make_selector(self._raw_band.projections)
        tree = select.Tree.from_selection(selection)
        quiver_plots = [
            graph.Contour(**self._quiver_plot(selector, sel, nkp1, nkp2), **options)
            for sel in tree.selections()
        ]
        return graph.Graph(quiver_plots, title="Spin Texture")

    def selections(self) -> dict:
        return self._projector().selections()

    def _is_collinear(self):
        return len(self._raw_band.dispersion.eigenvalues) == 2

    def _is_noncollinear(self):
        assert not check.is_none(self._raw_band.projections)
        return len(self._raw_band.projections) == 4

    def _dispersion(self):
        return DispersionHandler.from_data(self._raw_band.dispersion)

    def _projector(self):
        return ProjectorHandler.from_data(self._raw_band.projectors)

    def _kpoint(self):
        return KpointHandler.from_data(self._raw_band.dispersion.kpoints)

    def _projections(self, selection, width):
        if selection is None:
            return None
        check.raise_error_if_not_number(
            width, "Width of fat band structure must be a number."
        )
        projections = self._read_projections(selection)
        spin_projections = projections.get(projector.SPIN_PROJECTION, [])
        for label, weight in projections.items():
            if label == projector.SPIN_PROJECTION or label in spin_projections:
                continue
            weight *= width
        return projections

    def _read_projections(self, selection):
        return self._projector().project(selection, self._raw_band.projections)

    def _read_occupations(self):
        if self._is_collinear():
            return {
                "occupations_up": self._raw_band.occupations[0],
                "occupations_down": self._raw_band.occupations[1],
            }
        else:
            return {"occupations": self._raw_band.occupations[0]}

    def _shift_dispersion_by_fermi_energy(self, eigenvalues, fermi_energy):
        shifted = self._shift_array_by_fermi_energy(eigenvalues, fermi_energy)
        if len(shifted) == 2:
            return {"bands_up": shifted[0], "bands_down": shifted[1]}
        else:
            return {"bands": shifted[0]}

    def _shift_series_by_fermi_energy(self, g, fermi_energy):
        for series in g.series:
            series.y = self._shift_array_by_fermi_energy(series.y, fermi_energy)
        return g

    def _shift_array_by_fermi_energy(self, array, fermi_energy):
        if fermi_energy is None:
            fermi_energy = self._raw_band.fermi_energy
        return array - fermi_energy

    def _extract_relevant_data(self, selection, fermi_energy):
        need_to_be_repeated = ("kpoint_distances", "kpoint_labels")
        relevant_keys = (
            "bands",
            "bands_up",
            "bands_down",
            "occupations",
            "occupations_up",
            "occupations_down",
        )
        data = {}
        for key, value in self.to_dict(selection, fermi_energy).items():
            if key in need_to_be_repeated:
                data[key] = np.repeat(value, self._raw_band.occupations[0].shape[-1])
            if key in relevant_keys:
                data[key] = _to_series(value)
        for key, value in self._read_projections(selection).items():
            if key == projector.SPIN_PROJECTION:
                continue
            data[key] = _to_series(value)
        return data

    def _kmesh(self):
        try:
            nkpx = self._raw_band.dispersion.kpoints.number_x
            nkpy = self._raw_band.dispersion.kpoints.number_y
            nkpz = self._raw_band.dispersion.kpoints.number_z
            if nkpx == 1:
                return (nkpy, nkpz, "a")
            elif nkpy == 1:
                return (nkpx, nkpz, "b")
            elif nkpz == 1:
                return (nkpx, nkpy, "c")
            else:
                raise exception.DataMismatch(
                    f"For spin texture visualisation, the plane normal (a,b,c) to the desired cutting plane must have exactly 1 k-point, but the k-point mesh is {nkpx},{nkpy},{nkpz}. Please adjust the KPOINTS file and re-run VASP."
                )
        except exception.NoData:
            raise exception.DataMismatch(
                "For spin texture visualisation, a k-point grid is assumed, but could not be found for this VASP run."
            )

    def _quiver_plot(self, selector, selection, nkp1, nkp2):
        data = selector[selection]
        data = data.reshape(2, nkp1, nkp2)
        return {"data": data, "label": selector.label(selection)}

    def _make_selector(self, projections):
        maps = self._projector().to_dict()
        maps = {
            1: maps["atom"],
            2: maps["orbital"],
            0: self._spin_map(maps["spin"]),
            4: self._band_map(projections.shape[-1]),
        }
        return index.Selector(
            maps, projections, reduction=_ToQuiverReduction, use_number_labels=True
        )

    def _spin_map(self, spin_map):
        if "sigma_x" not in spin_map:
            raise exception.DataMismatch(
                "System is not noncollinear which is required to visualize spin texture."
            )
        return {
            "sigma_x~sigma_y": slice(1, 3),
            "sigma_x~sigma_z": slice(1, 4, 2),
            "sigma_y~sigma_z": slice(2, 4),
            "x~y": slice(1, 3),
            "x~z": slice(1, 4, 2),
            "y~z": slice(2, 4),
            "sigma_1~sigma_2": slice(1, 3),
            "sigma_1~sigma_3": slice(1, 4, 2),
            "sigma_2~sigma_3": slice(2, 4),
        }

    def _band_map(self, num_bands):
        return {"band": {i + 1: i for i in range(num_bands)}}


@quantity("band")
class Band(graph.Mixin):
    """The band structure contains the k point resolved eigenvalues."""

    def __init__(self, source, quantity_name="band"):
        self._source = source
        self._quantity_name = quantity_name
        self._path = pathlib.Path.cwd()

    @classmethod
    def from_data(cls, raw_band):
        return cls(source=DataSource(raw_band))

    def _handler_factory(self, raw):
        return BandHandler.from_data(raw)

    def __str__(self):
        return merge_strings(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            BandHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection=None, fermi_energy=None) -> dict:
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            BandHandler.read,
            selection,
            fermi_energy=fermi_energy,
        )

    def to_dict(self, selection=None, fermi_energy=None) -> dict:
        return self.read(selection=selection, fermi_energy=fermi_energy)

    def to_graph(self, selection=None, fermi_energy=None, width=0.5) -> graph.Graph:
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            BandHandler.to_graph,
            selection,
            fermi_energy=fermi_energy,
            width=width,
        )

    def to_frame(self, selection=None, fermi_energy=None):
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            BandHandler.to_frame,
            selection,
            fermi_energy=fermi_energy,
        )

    def to_quiver(self, selection, normal=None, supercell=None) -> graph.Graph:
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            BandHandler.to_quiver,
            selection,
            normal=normal,
            supercell=supercell,
        )

    def selections(self) -> dict:
        from py4vasp._raw import definition as raw_module

        handler_selections = merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            BandHandler.selections,
        )
        sources = list(raw_module.selections(self._quantity_name))
        return {self._quantity_name: sources, **handler_selections}


def _to_series(array):
    return array.T.flatten()


class _ToQuiverReduction(index.Reduction):
    def __init__(self, keys: List):
        if not (keys[0]):
            raise exception.IncorrectUsage(
                "Spin Elements must be chosen, but none are given. Please adapt your `selection` argument to include, e.g., `x~y`. You can combine arguments by `arg1(arg2(arg3(...)))`."
            )
        if not (keys[4]):
            raise exception.IncorrectUsage(
                "A band must be chosen, but none are given. Please adapt your `selection` argument to include, e.g., `band[1]`. You can combine arguments by `arg1(arg2(arg3(...)))`."
            )
        pass

    def __call__(self, array: ArrayLike, axis: Iterable):
        axis = tuple(filter(None, axis))
        return np.sum(array, axis=axis)
