# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
import pathlib

import numpy as np

from py4vasp import raw
from py4vasp._calculation import slice_
from py4vasp._calculation.dispatch import (
    DataSource,
    FileSource,
    merge_default,
    merge_graphs,
    merge_strings,
    quantity,
)
from py4vasp._raw.data_db import Energy_DB
from py4vasp._third_party import graph
from py4vasp._util import convert, documentation, index, select


def _selection_string(default):
    return f"""\
selection : str or None
    String specifying the labels of the energy to be read. If no energy is selected
    this will default to selecting {default}. Separate distinct labels by commas or
    whitespace. You can add or subtract different contributions e.g. `TOTEN + EKIN`.
    For a complete list of all possible selections, please use

    >>> calculation.energy.selections()
"""


_SELECTIONS = {
    "ion-electron   TOTEN": ["ion_electron", "TOTEN"],
    "kinetic energy EKIN": ["kinetic_energy", "EKIN"],
    "kin. lattice   EKIN_LAT": ["kinetic_lattice", "EKIN_LAT"],
    "temperature    TEIN": ["temperature", "TEIN"],
    "nose potential ES": ["nose_potential", "ES"],
    "nose kinetic   EPS": ["nose_kinetic", "EPS"],
    "total energy   ETOTAL": ["total_energy", "ETOTAL"],
    "free energy    TOTEN": ["free_energy", "TOTEN"],
    "energy without entropy": ["without_entropy", "ENOENT"],
    "energy(sigma->0)": ["sigma_0", "ESIG0"],
    "step            STEP": ["step", "STEP"],
    "One el. energy  E1": ["one_electron", "E1"],
    "Hartree energy  -DENC": ["Hartree", "hartree", "DENC"],
    "exchange        EXHF": ["exchange", "EXHF"],
    "free energy     TOTEN": ["free_energy", "TOTEN"],
    "free energy cap TOTENCAP": ["cap", "TOTENCAP"],
    "weight          WEIGHT": ["weight", "WEIGHT"],
}

_DB_KEYS = {
    "ion-electron   TOTEN": "ion_electron",
    "kinetic energy EKIN": "kinetic_energy",
    "kin. lattice   EKIN_LAT": "kinetic_energy_lattice",
    "temperature    TEIN": "temperature",
    "nose potential ES": "nose_potential",
    "nose kinetic   EPS": "nose_kinetic",
    "total energy   ETOTAL": "total_energy",
    "free energy    TOTEN": "free_energy",
    "energy without entropy": "energy_without_entropy",
    "energy(sigma->0)": "energy_sigma_0",
    "step            STEP": "step",
    "One el. energy  E1": "one_electron_energy",
    "Hartree energy  -DENC": "hartree_energy",
    "exchange        EXHF": "exchange_energy",
    "free energy     TOTEN": "free_energy",
    "free energy cap TOTENCAP": "free_energy_cap",
    "weight          WEIGHT": "weight",
}


@documentation.format(examples=slice_.examples("energy"))
class EnergyHandler:
    """Handler for energy data — performs all data access and transformation logic."""

    def __init__(self, raw_energy: raw.Energy, steps=None):
        self._raw_energy = raw_energy
        self._steps = steps

    @classmethod
    def from_data(cls, raw_energy: raw.Energy, steps=None) -> "EnergyHandler":
        return cls(raw_energy, steps)

    def __str__(self) -> str:
        text = f"Energies at {self._step_string()}:"
        values = self._raw_energy.values[self._last_step_in_slice]
        for label, value in zip(self._raw_energy.labels, values):
            label_str = f"{convert.text_to_string(label):23.23}"
            text += f"\n   {label_str}={value:17.6f}"
        return text

    def to_dict(self, selection=None) -> dict:
        if selection is None:
            return self._default_dict()
        tree = select.Tree.from_selection(selection)
        return dict(self._read_data(tree, self._steps_or_last))

    def to_graph(self, selection="TOTEN") -> graph.Graph:
        tree = select.Tree.from_selection(selection)
        yaxes = _YAxes(tree)
        return graph.Graph(
            series=self._make_series(yaxes, tree),
            xlabel="Step",
            ylabel=yaxes.ylabel,
            y2label=yaxes.y2label,
        )

    def to_numpy(self, selection="TOTEN") -> np.ndarray:
        tree = select.Tree.from_selection(selection)
        return np.squeeze(
            [values for _, values in self._read_data(tree, self._steps_or_last)]
        )

    def selections(self) -> dict:
        components = list(self._init_selection_dict().keys())
        return {"energy": [], "component": components}

    def to_database(self) -> dict:
        default_dict = self._default_dict_all()
        energy_dict = {}
        for original_label, db_key in _DB_KEYS.items():
            v = default_dict.get(original_label, None)
            if v is not None:
                v = np.array(v)
            energy_dict[f"{db_key}_initial"] = None if v is None else float(v[0])
            if (db_key != "step") and (v is not None):
                energy_dict[f"{db_key}_min"] = float(np.min(v))
                energy_dict[f"{db_key}_step_min"] = int(np.argmin(v))
            energy_dict[f"{db_key}_final"] = None if v is None else float(v[-1])
        extra_dict = {}
        for k, v in default_dict.items():
            if k not in _DB_KEYS:
                vs = np.array(v) if v is not None else None
                key = convert.text_to_string(k).strip().lower().replace(" ", "_")
                extra_dict[f"{key}_initial"] = None if vs is None else float(vs[0])
                extra_dict[f"{key}_min"] = None if vs is None else float(np.min(vs))
                extra_dict[f"{key}_step_min"] = (
                    None if vs is None else int(np.argmin(vs))
                )
                extra_dict[f"{key}_final"] = None if vs is None else float(vs[-1])
        return {"energy": Energy_DB(**energy_dict, other_energy_data=extra_dict)}

    @property
    def _steps_or_last(self):
        if self._steps is None:
            return -1
        return self._steps

    @property
    def _last_step_in_slice(self):
        if self._steps is None or self._steps == -1:
            return -1
        if isinstance(self._steps, slice):
            return (self._steps.stop or 0) - 1
        return self._steps

    @property
    def _to_slice(self):
        if self._steps is None or self._steps == -1:
            return slice(-1, None)
        if isinstance(self._steps, slice):
            return self._steps
        return slice(self._steps, self._steps + 1)

    def _step_string(self):
        if isinstance(self._steps, slice):
            n = len(self._raw_energy.values)
            range_ = range(n)[self._steps]
            start = range_.start + 1
            stop = range_.stop
            return f"step {stop} of range {start}:{stop}"
        elif self._steps == -1 or self._steps is None:
            return "final step"
        else:
            return f"step {self._steps + 1}"

    def _default_dict(self):
        raw_values = np.array(self._raw_energy.values).T
        return {
            convert.text_to_string(label).strip(): value[self._steps_or_last]
            for label, value in zip(self._raw_energy.labels, raw_values)
        }

    def _default_dict_all(self):
        raw_values = np.array(self._raw_energy.values).T
        return {
            convert.text_to_string(label).strip(): value[:]
            for label, value in zip(self._raw_energy.labels, raw_values)
        }

    def _read_data(self, tree, steps):
        maps = {1: self._init_selection_dict()}
        selector = index.Selector(maps, self._raw_energy.values)
        for selection in tree.selections():
            yield selector.label(selection), selector[selection][steps]

    def _init_selection_dict(self):
        return {
            selection: idx
            for idx, label in enumerate(self._raw_energy.labels)
            for selection in _SELECTIONS.get(convert.text_to_string(label).strip(), ())
        }

    def _make_series(self, yaxes, tree):
        n = len(self._raw_energy.values)
        slice_ = self._to_slice
        steps = np.arange(n)[slice_] + 1
        return [
            graph.Series(x=steps, y=values, label=label, y2=yaxes.use_y2(label))
            for label, values in self._read_data(tree, slice_)
        ]


@quantity("energy")
@documentation.format(examples=slice_.examples("energy"))
class Energy(graph.Mixin):
    """The energy data for one or several steps of a relaxation or MD simulation.

    You can use this class to inspect how the ionic relaxation converges or
    during an MD simulation whether the total energy is conserved. The total
    energy of the system is one of the most important results to analyze materials.
    Total energy differences of different atom arrangements reveal which structure
    is more stable. Even when the number of atoms are different between two
    systems, you may be able to compare the total energies by adding a corresponding
    amount of single atom energies. In this case, you need to double check the
    convergence because some error cancellation does not happen if the number of
    atoms is changes. Finally, monitoring the total energy can reveal insights
    about the stability of the thermostat.

    {examples}
    """

    def __init__(self, source, quantity_name: str = "energy", steps=None):
        self._source = source
        self._quantity_name = quantity_name
        self._steps = steps

    @classmethod
    def from_data(cls, raw_energy: raw.Energy):
        """Create an Energy dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_energy))

    @classmethod
    def from_path(cls, path="."):
        """Create an Energy dispatcher that reads from HDF5 files at *path*."""
        return cls(source=FileSource(path))

    @classmethod
    def from_file(cls, file_name):
        """Create an Energy dispatcher that reads from a specific HDF5 file."""
        resolved = pathlib.Path(file_name).expanduser().resolve()
        return cls(source=FileSource(resolved.parent, file=file_name))

    @property
    def _path(self):
        """Path used for file-export methods. Falls back to cwd."""
        return self._source.path or pathlib.Path.cwd()

    def __getitem__(self, steps) -> "Energy":
        new = copy.copy(self)
        new._steps = steps
        return new

    def _handler_factory(self, raw_data):
        return EnergyHandler.from_data(raw_data, steps=self._steps)

    @documentation.format(
        selection=_selection_string("all energies"),
        examples=slice_.examples("energy", "to_dict"),
    )
    def read(self, selection=None) -> dict:
        """Read the energy data and store it in a dictionary.

        Parameters
        ----------
        {selection}

        Returns
        -------
        dict
            Contains the exact labels corresponding to the selection and the
            associated energies for every selected ionic step.

        {examples}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            EnergyHandler.to_dict,
        )

    def to_dict(self, selection=None) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read(selection=selection)

    @documentation.format(
        selection=_selection_string("the total energy"),
        examples=slice_.examples("energy", "to_graph"),
    )
    def to_graph(self, selection="TOTEN") -> graph.Graph:
        """Read the energy data and generate a figure of the selected components.

        Parameters
        ----------
        {selection}

        Returns
        -------
        Graph
            figure containing the selected energies for every selected ionic step.

        {examples}
        """
        return merge_graphs(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            EnergyHandler.to_graph,
        )

    @documentation.format(
        selection=_selection_string("the total energy"),
        examples=slice_.examples("energy", "to_numpy"),
    )
    def to_numpy(self, selection="TOTEN") -> np.ndarray:
        """Read the energy of the selected steps.

        Parameters
        ----------
        {selection}

        Returns
        -------
        float or np.ndarray or tuple
            Contains energies associated with the selection for the selected ionic step(s).
            When only a single step is inquired, result is a float otherwise an array.
            If you select multiple quantities a tuple of them is returned.

        {examples}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            EnergyHandler.to_numpy,
        )

    def selections(self, selection: str | None = None) -> dict:
        """Return a dictionary describing what kind of energies are available.

        Returns
        -------
        -
            Dictionary containing available selection options with their possible values.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            EnergyHandler.selections,
        )

    def __str__(self, selection: str | None = None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            EnergyHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")


class _YAxes:
    def __init__(self, tree):
        uses = set(self._is_temperature(selection) for selection in tree.selections())
        use_energy = False in uses
        self.use_both = len(uses) == 2
        self.ylabel = "Energy (eV)" if use_energy else "Temperature (K)"
        self.y2label = "Temperature (K)" if self.use_both else None

    def _is_temperature(self, selection):
        choices = _SELECTIONS["temperature    TEIN"]
        return any(select.contains(selection, choice) for choice in choices)

    def use_y2(self, label):
        choices = _SELECTIONS["temperature    TEIN"]
        return self.use_both and label in choices
