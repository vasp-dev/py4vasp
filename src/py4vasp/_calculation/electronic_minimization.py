# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import copy
from contextlib import suppress

import numpy as np

from py4vasp import exception, raw
from py4vasp._calculation import slice_
from py4vasp._calculation.dispatch import (
    DataSource,
    _dispatch,
    merge_default,
    merge_graphs,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._raw.data_db import ElectronicMinimization_DB
from py4vasp._third_party import graph
from py4vasp._util import check, index, select

# VASP labels that clash with the selection grammar (parentheses mean nesting) are
# exposed under a grammar-safe alias. The raw label is kept for display / printing.
_LABEL_TOKEN_OVERRIDES = {"rms(c)": "rms_c"}

_SELECTION_ERROR_MESSAGE = """\
Please choose a selection including at least one of the following keywords:
N, E, dE, deps, ncg, rms, rms_c"""

# Energy-change series shown on the left axis of the convergence overview, beside the
# "E" distance to the converged energy. Given as (column token, display label) pairs.
_ENERGY_CHANGE_TOKENS = [("dE", "dE"), ("deps", "d eps")]
# Columns not plotted on the secondary residual axis: the iteration index, the energies,
# and the number of Hamiltonian evaluations. Every remaining column (rms and the density
# / orthonormalization residual, which VASP labels rms(c) or ort) is a residual.
_NON_RESIDUAL_TOKENS = {"N", "E", "dE", "deps", "ncg"}

# Convergence entries whose magnitude is below this threshold are treated as
# not-yet-computed and reported as NaN. This most notably affects rms(c), which VASP
# reports as zero until density updates begin after the NELMDL delay.
_SANITY_THRESHOLD = 1e-16

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
        # print the raw OSZICAR data verbatim, without the read() sanity filter
        data = self._extract(selection=None, sanitize=False)
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
                    token = _LABEL_TOKEN_OVERRIDES.get(
                        label.decode("utf-8"), label.decode("utf-8")
                    )
                    _values_electronic = data[token]
                    if not self._more_than_one_ionic_step(_values_electronic):
                        _values_electronic = [_values_electronic]
                    _value = _values_electronic[ionic_step][electronic_step]
                    _data.append(_value)
                _data = [float(_value) for _value in _data]
                string += format_rep.format(*_data)
        return string

    def to_dict(self, selection=None) -> dict:
        """Extract convergence data and return as a dict.

        The selection is parsed with the standard :mod:`py4vasp._util.select` grammar and
        evaluated with :class:`py4vasp._util.index.Selector`, so multiple columns
        (``"E, dE"``) and compositions (``"E + dE"``) are supported. Entries whose
        magnitude is below a small threshold are treated as not-yet-computed and reported
        as NaN (see :data:`_SANITY_THRESHOLD`).
        """
        return self._extract(selection, sanitize=True)

    def _extract(self, selection, sanitize) -> dict:
        tokens = self._tokens()
        step_arrays = self._step_arrays()
        column_map = {1: {token: column for column, token in enumerate(tokens)}}
        selectors = [index.Selector(column_map, array) for array in step_arrays]
        is_none = np.all([array.is_none() for array in step_arrays])
        if selection is None:
            selections = [(token,) for token in tokens]
        else:
            selections = list(select.Tree.from_selection(selection).selections())
        return_data = {}
        for sel in selections:
            try:
                key = selectors[0].label(sel)
                columns = [selector[sel] for selector in selectors]
            except exception.IncorrectUsage as error:
                raise exception.RefinementError(_SELECTION_ERROR_MESSAGE) from error
            if sanitize:
                columns = [self._sanitize(column) for column in columns]
            values = [list(column) for column in columns]
            if len(values) == 1:
                values = values[0]
            return_data[key] = values if not is_none else {}
        return return_data

    def _sanitize(self, column):
        """Report not-yet-computed (near-zero) entries as NaN."""
        column = np.array(column, dtype=float)
        column[np.abs(column) < _SANITY_THRESHOLD] = np.nan
        return column

    def to_graph(self, selection=None) -> graph.Graph:
        """Graph the convergence data against the iteration number.

        Without a selection this produces a convergence overview: the energy changes
        on a logarithmic left axis and the residuals on a logarithmic secondary axis.
        With a selection, the chosen columns are plotted as-is on a logarithmic axis.
        """
        if selection is None:
            return self._overview_graph()
        return self._selection_graph(selection)

    def _overview_graph(self) -> graph.Graph:
        data = self.to_dict()
        iterations = np.array(data["N"], dtype=float)
        # distance to the converged energy; positive while E is above its final value
        series = [self._make_series(iterations, self._energy_distance(data), "E - E_final")]
        series += [
            self._make_series(iterations, np.array(data[token], dtype=float), label)
            for token, label in _ENERGY_CHANGE_TOKENS
        ]
        series += [
            self._make_series(iterations, np.array(data[token], dtype=float), token, y2=True)
            for token in self._tokens()
            if token not in _NON_RESIDUAL_TOKENS
        ]
        return graph.Graph(
            series=series,
            xlabel="Iteration number",
            ylabel="Energy change (eV)",
            y2label="Residual",
            yscale="log",
            y2scale="log",
        )

    def _energy_distance(self, data):
        # E - E_final is positive while the energy is above its converged value; the
        # final point is zero and therefore dropped so it does not vanish on the axis
        values = np.array(data["E"], dtype=float)
        values = values - values[-1]
        values[-1] = np.nan
        return values

    def _selection_graph(self, selection) -> graph.Graph:
        iterations = np.array(self.to_dict("N")["N"], dtype=float)
        series = [
            self._make_series(iterations, np.array(values, dtype=float), label)
            for label, values in self.to_dict(selection).items()
        ]
        return graph.Graph(
            series=series,
            xlabel="Iteration number",
            ylabel="Convergence data",
            yscale="log",
        )

    def _make_series(self, x, values, label, y2=False) -> graph.Series:
        """Plot |values| labelled "|label|" if any value is negative (a log axis cannot
        show negative numbers); otherwise plot the values as they are."""
        if np.any(values < 0):
            values = np.abs(values)
            label = f"|{label}|"
        return graph.Series(x, values, label, y2=y2)

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

        return ElectronicMinimization_DB(
            num_electronic_steps=num_electronic_steps,
            elmin_is_converged_all=elmin_is_converged_all,
            elmin_is_converged_final=elmin_is_converged_final,
            num_max_electronic_steps_per_ionic_step=num_max_electronic_steps_per_ionic,
            num_min_electronic_steps_per_ionic_step=num_min_electronic_steps_per_ionic,
        )

    @property
    def _steps_or_last(self):
        return -1 if self._steps is None else self._steps

    def _more_than_one_ionic_step(self, data):
        return any(isinstance(_data, list) for _data in data)

    def _tokens(self):
        """Selectable tokens: the raw labels with grammar-clashing ones aliased."""
        labels = self._from_bytes_to_utf(self._raw_data.label)
        return [_LABEL_TOKEN_OVERRIDES.get(label, label) for label in labels]

    def _step_arrays(self):
        """Split the convergence data into per-ionic-step arrays for the chosen steps."""
        data = getattr(self._raw_data, "convergence_data")
        iteration_number = data[:, 0]
        split_index = np.where(iteration_number == 1)[0]
        data = np.vsplit(data, split_index)[1:][self._steps_or_last]
        if isinstance(self._steps, slice):
            return [raw.VaspData(_data) for _data in data]
        return [raw.VaspData(data)]

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

    def __getitem__(self, steps) -> "ElectronicMinimization":
        new = copy.copy(self)
        new._steps = steps
        return new

    def _handler_factory(self, raw_data):
        return ElectronicMinimizationHandler.from_data(raw_data, steps=self._steps)

    def read(self, selection=None) -> dict:
        """Extract convergence data from the HDF5 file and make it available in a dict

        Parameters
        ----------
        selection: str
            Choose from either iteration_number, free_energy, free_energy_change,
            bandstructure_energy_change, number_hamiltonian_evaluations, norm_residual,
            difference_charge_density to get specific columns of the OSZICAR file. In
            case no selection is provided, supply all columns.

        Returns
        -------
        dict
            Contains a dict from the HDF5 related to OSZICAR convergence data
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ElectronicMinimizationHandler.to_dict,
        )

    def to_dict(self, selection=None) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read(selection=selection)

    def to_graph(self, selection=None) -> graph.Graph:
        """Graph the convergence data against the iteration number.

        Parameters
        ----------
        selection: str
            Choose strings consistent with the OSZICAR format (N, E, dE, deps, ncg, rms,
            rms_c), optionally composed with the standard selection grammar (e.g.
            "E + dE"). Without a selection a convergence overview is produced with the
            energy changes on a logarithmic left axis and the residuals on a logarithmic
            secondary axis.

        Returns
        -------
        Graph
            The Graph with the quantity plotted on y-axis and the iteration number of
            the x-axis.
        """
        return merge_graphs(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ElectronicMinimizationHandler.to_graph,
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

    def __str__(self, selection=None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ElectronicMinimizationHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            ElectronicMinimizationHandler.from_data,
            ElectronicMinimizationHandler.to_database,
        )
