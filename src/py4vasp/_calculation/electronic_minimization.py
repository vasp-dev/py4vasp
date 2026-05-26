# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import pathlib
from contextlib import suppress

import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation import slice_
from py4vasp._calculation.dispatch import (
    DataSource,
    FileSource,
    merge_default,
    merge_graphs,
    merge_strings,
    quantity,
)
from py4vasp._raw.data_db import ElectronicMinimization_DB
from py4vasp._third_party import graph
from py4vasp._util import check

_TO_DATABASE_SUPPRESSED_EXCEPTIONS = (
    exception.Py4VaspError,
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
)


class ElectronicMinimizationHandler:
    """Handler for electronic minimization data — all data access and transformation."""

    def __init__(self, raw_elmin: raw.ElectronicMinimization, steps=None):
        self._raw_data = raw_elmin
        self._steps = steps

    @classmethod
    def from_data(
        cls, raw_elmin: raw.ElectronicMinimization, steps=None
    ) -> "ElectronicMinimizationHandler":
        return cls(raw_elmin, steps)

    def __str__(self) -> str:
        format_rep = "{0:g}\t{1:0.12E}\t{2:0.6E}\t{3:0.6E}\t{4:g}\t{5:0.3E}\t{6:0.3E}\n"
        label_rep = "{}\t\t{}\t\t{}\t\t{}\t\t{}\t{}\t\t{}\n"
        string = ""
        labels = [label.decode("utf-8") for label in getattr(self._raw_data, "label")]
        data = self.to_dict()
        electronic_iterations = data["N"]
        if not self._more_than_one_ionic_step(electronic_iterations):
            electronic_iterations = [electronic_iterations]
        ionic_steps = len(electronic_iterations)
        for ionic_step in range(ionic_steps):
            string += label_rep.format(*labels)
            electronic_steps = len(electronic_iterations[ionic_step])
            for electronic_step in range(electronic_steps):
                _data = []
                for label in self._raw_data.label:
                    _values_electronic = data[label.decode("utf-8")]
                    if not self._more_than_one_ionic_step(_values_electronic):
                        _values_electronic = [_values_electronic]
                    _value = _values_electronic[ionic_step][electronic_step]
                    _data.append(_value)
                _data = [float(_value) for _value in _data]
                string += format_rep.format(*_data)
        return string

    def read(self, selection=None) -> dict:
        return self.to_dict(selection)

    def to_dict(self, selection=None) -> dict:
        """Extract convergence data and return as a dict."""
        return_data = {}
        if selection is None:
            keys_to_include = self._from_bytes_to_utf(self._raw_data.label)
        else:
            labels_as_str = self._from_bytes_to_utf(self._raw_data.label)
            if selection not in labels_as_str:
                message = """\
Please choose a selection including at least one of the following keywords:
N, E, dE, deps, ncg, rms, rms(c)"""
                raise exception.RefinementError(message)
            keys_to_include = [selection]
        for key in keys_to_include:
            return_data[key] = self._read(key)
        return return_data

    def to_graph(self, selection="E") -> graph.Graph:
        """Graph the change in parameter with iteration number."""
        data = self.to_dict()
        series = graph.Series(data["N"], data[selection], selection)
        from py4vasp._util import select as sel_util

        ylabel = " ".join(s.capitalize() for s in selection.split("_"))
        return graph.Graph(
            series=[series],
            xlabel="Iteration number",
            ylabel=ylabel,
        )

    def is_converged(self) -> np.ndarray:
        is_elmin_converged = self._raw_data.is_elmin_converged[self._steps_or_last]
        converged = is_elmin_converged == 0
        if isinstance(converged, bool):
            converged = np.array([converged])
        return converged.flatten()

    def to_database(self) -> dict:
        """Serialize electronic minimization data for database storage."""
        num_max_electronic_steps_per_ionic = None
        num_min_electronic_steps_per_ionic = None
        num_electronic_steps = None
        elmin_is_converged_all = None
        elmin_is_converged_final = None

        with suppress(*_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
            if not check.is_none(self._raw_data.is_elmin_converged):
                elmin_is_converged_all = bool(
                    np.all(np.array(self._raw_data.is_elmin_converged[:]) == 0.0)
                )
                elmin_is_converged_final = bool(
                    self._raw_data.is_elmin_converged[-1] == 0.0
                )

        with suppress(exception.NoData, *_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
            (
                num_max_electronic_steps_per_ionic,
                num_min_electronic_steps_per_ionic,
                num_electronic_steps,
            ) = self._get_electronic_steps_info()

        return {
            "electronic_minimization": ElectronicMinimization_DB(
                num_electronic_steps=num_electronic_steps,
                elmin_is_converged_all=elmin_is_converged_all,
                elmin_is_converged_final=elmin_is_converged_final,
                num_max_electronic_steps_per_ionic_step=num_max_electronic_steps_per_ionic,
                num_min_electronic_steps_per_ionic_step=num_min_electronic_steps_per_ionic,
            )
        }

    @property
    def _steps_or_last(self):
        return -1 if self._steps is None else self._steps

    def _more_than_one_ionic_step(self, data):
        return any(isinstance(_data, list) for _data in data)

    def _read(self, key):
        data = getattr(self._raw_data, "convergence_data")
        iteration_number = data[:, 0]
        split_index = np.where(iteration_number == 1)[0]
        data = np.vsplit(data, split_index)[1:][self._steps_or_last]
        if isinstance(self._steps, slice):
            data = [raw.VaspData(_data) for _data in data]
        else:
            data = [raw.VaspData(data)]
        labels = [label.decode("utf-8") for label in self._raw_data.label]
        data_index = labels.index(key)
        return_data = [list(_data[:, data_index]) for _data in data]
        is_none = [_data.is_none() for _data in data]
        if len(return_data) == 1:
            return_data = return_data[0]
        return return_data if not np.all(is_none) else {}

    def _get_electronic_steps_info(self) -> tuple:
        if check.is_none(self._raw_data.convergence_data):
            return None, None, None

        data = getattr(self._raw_data, "convergence_data")
        iteration_number = data[:, 0]
        split_index = np.where(iteration_number == 1)[0]
        data = [raw.VaspData(_data) for _data in np.vsplit(data, split_index)[1:][:]]

        labels = [label.decode("utf-8") for label in self._raw_data.label]
        data_index = labels.index("N")
        N_data = [list(_data[:, data_index]) for _data in data]
        num_electronic_steps_per_ionic = [len(_data) for _data in N_data]
        is_none = [_data.is_none() for _data in data]
        if np.all(is_none):
            return None, None, None

        if len(num_electronic_steps_per_ionic) == 0:
            return None, None, None

        num_max_electronic_steps_per_ionic = max(num_electronic_steps_per_ionic)
        num_min_electronic_steps_per_ionic = min(num_electronic_steps_per_ionic)
        num_electronic_steps = sum(num_electronic_steps_per_ionic)

        return (
            num_max_electronic_steps_per_ionic,
            num_min_electronic_steps_per_ionic,
            num_electronic_steps,
        )

    def _from_bytes_to_utf(self, quantity):
        return [_quantity.decode("utf-8") for _quantity in quantity]


@quantity("electronic_minimization")
class ElectronicMinimization(graph.Mixin):
    """Access the convergence data for each electronic step.

    The OSZICAR file written out by VASP stores information related to convergence.
    Please check the `vasp-wiki <https://www.vasp.at/wiki/index.php/OSZICAR>`__ for more
    details about the exact outputs generated for each combination of INCAR tags."""

    def __init__(
        self, source, quantity_name: str = "electronic_minimization", steps=None
    ):
        self._source = source
        self._quantity_name = quantity_name
        self._steps = steps

    @classmethod
    def from_data(cls, raw_elmin: raw.ElectronicMinimization):
        """Create an ElectronicMinimization dispatcher from raw data."""
        return cls(source=DataSource(raw_elmin))

    @classmethod
    def from_path(cls, path="."):
        """Create an ElectronicMinimization dispatcher from HDF5 files at *path*."""
        return cls(source=FileSource(path))

    @classmethod
    def from_file(cls, file_name):
        """Create an ElectronicMinimization dispatcher from a specific HDF5 file."""
        resolved = pathlib.Path(file_name).expanduser().resolve()
        return cls(source=FileSource(resolved.parent, file=file_name))

    @property
    def _path(self):
        return self._source.path or pathlib.Path.cwd()

    def __getitem__(self, steps) -> "ElectronicMinimization":
        new = copy.copy(self)
        new._steps = steps
        return new

    def _handler_factory(self, raw_data):
        return ElectronicMinimizationHandler.from_data(raw_data, steps=self._steps)

    def read(self, selection=None) -> dict:
        """Extract convergence data from the HDF5 file and make it available in a dict."""
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ElectronicMinimizationHandler.read,
            selection,
        )

    def to_dict(self, selection=None) -> dict:
        """Convenient alias for :py:meth:`read`."""
        return self.read(selection=selection)

    def to_graph(self, selection="E") -> graph.Graph:
        """Graph the change in parameter with iteration number."""
        return merge_graphs(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ElectronicMinimizationHandler.to_graph,
            selection,
        )

    def is_converged(self) -> np.ndarray:
        """Return whether the electronic minimization converged."""
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            ElectronicMinimizationHandler.is_converged,
        )

    def __str__(self, selection: str | None = None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ElectronicMinimizationHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

