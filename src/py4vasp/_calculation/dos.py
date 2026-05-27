# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib
from contextlib import suppress

import numpy as np

from py4vasp import exception
from py4vasp._calculation import projector
from py4vasp._calculation.dispatch import DataSource, merge_default, merge_strings, quantity
from py4vasp._calculation.projector import ProjectorHandler
from py4vasp._raw import data as raw
from py4vasp._raw.data_db import Dos_DB
from py4vasp._third_party import graph
from py4vasp._util import check, import_

pd = import_.optional("pandas")
pretty = import_.optional("IPython.lib.pretty")

_TO_DATABASE_SUPPRESSED_EXCEPTIONS = (
    exception.Py4VaspError,
    AttributeError,
    TypeError,
    ValueError,
    IndexError,
    ZeroDivisionError,
)


class DosHandler:
    """Handler for density of states data."""

    def __init__(self, raw_dos: raw.Dos):
        self._raw_dos = raw_dos

    @classmethod
    def from_data(cls, raw_dos: raw.Dos) -> "DosHandler":
        return cls(raw_dos)

    def __str__(self):
        energies = self._raw_dos.energies
        if self._is_collinear():
            label = "collinear Dos"
        elif self._is_noncollinear():
            label = "noncollinear Dos"
        else:
            label = "Dos"
        return f"""\
{label}:
    energies: [{energies[0]:0.2f}, {energies[-1]:0.2f}] {len(energies)} points
{str(self._projector())}"""

    def read(self, selection=None) -> dict:
        return self.to_dict(selection)

    def to_dict(self, selection=None) -> dict:
        data = self._read_data(selection)
        data.pop(projector.SPIN_PROJECTION, None)
        return {**data, "fermi_energy": self._raw_dos.fermi_energy}

    def to_database(self, fermi_energy=None) -> dict:
        raw_fermi_energy = (
            self._raw_dos.fermi_energy
            if not check.is_none(self._raw_dos.fermi_energy)
            else None
        )
        dos_at_fermi_dict = self._dos_at_energy(fermi_energy or raw_fermi_energy)
        dos_at_raw_fermi_dict = self._dos_at_energy(raw_fermi_energy)

        dos_at_fermi_total = dos_at_fermi_dict.get("total", None)
        dos_at_raw_fermi_total = dos_at_raw_fermi_dict.get("total", None)
        dos_at_fermi_up = dos_at_fermi_dict.get("up", None)
        dos_at_fermi_down = dos_at_fermi_dict.get("down", None)
        dos_at_raw_fermi_up = dos_at_raw_fermi_dict.get("up", None)
        dos_at_raw_fermi_down = dos_at_raw_fermi_dict.get("down", None)

        return {
            "dos": Dos_DB(
                dos_at_fermi_total=dos_at_fermi_total,
                dos_at_fermi_up=dos_at_fermi_up,
                dos_at_fermi_down=dos_at_fermi_down,
                dos_at_raw_fermi_total=dos_at_raw_fermi_total,
                dos_at_raw_fermi_up=dos_at_raw_fermi_up,
                dos_at_raw_fermi_down=dos_at_raw_fermi_down,
                energy_min=(
                    float(np.min(self._raw_dos.energies[:]))
                    if not check.is_none(self._raw_dos.energies)
                    else None
                ),
                energy_max=(
                    float(np.max(self._raw_dos.energies[:]))
                    if not check.is_none(self._raw_dos.energies)
                    else None
                ),
            )
        }

    def to_graph(self, selection=None) -> graph.Graph:
        data = self._read_data(selection)
        energies = data.pop("energies")
        data.pop(projector.SPIN_PROJECTION, None)
        return graph.Graph(
            series=list(_series(energies, data)),
            xlabel="Energy (eV)",
            ylabel="DOS (1/eV)",
        )

    def to_frame(self, selection=None):
        data = self._read_data(selection)
        data.pop(projector.SPIN_PROJECTION, None)
        df = pd.DataFrame(data)
        df.fermi_energy = self._raw_dos.fermi_energy
        return df

    def selections(self) -> dict:
        return self._projector().selections()

    def _is_collinear(self):
        return len(self._raw_dos.dos) == 2

    def _is_noncollinear(self):
        return len(self._raw_dos.dos) == 4

    def _projector(self):
        return ProjectorHandler.from_data(self._raw_dos.projectors)

    def _read_data(self, selection):
        return {
            **self._read_energies(),
            **self._read_total_dos(),
            **self._projector().project(selection, self._raw_dos.projections),
        }

    def _read_energies(self):
        return {"energies": self._raw_dos.energies[:] - self._raw_dos.fermi_energy}

    def _read_total_dos(self):
        if self._is_collinear():
            return {"up": self._raw_dos.dos[0, :], "down": self._raw_dos.dos[1, :]}
        else:
            return {"total": self._raw_dos.dos[0, :]}

    def _dos_at_energy(self, energy):
        with suppress(*_TO_DATABASE_SUPPRESSED_EXCEPTIONS):
            energies = self._raw_dos.energies[:]
            dos_dict = self._read_total_dos()
            dos_at_energy = {}
            for key, dos in dos_dict.items():
                idx = (np.abs(energies - energy)).argmin()
                if energies[idx] == energy:
                    dos_at_energy[key] = float(dos[idx])
                else:
                    if energies[idx] < energy:
                        idx_low = idx
                        idx_high = idx + 1
                    else:
                        idx_low = idx - 1
                        idx_high = idx
                    if (idx_low < 0) or (idx_high >= len(energies)):
                        dos_at_energy[key] = None
                        continue
                    dos_low = dos[idx_low]
                    dos_high = dos[idx_high]
                    energy_low = energies[idx_low]
                    energy_high = energies[idx_high]
                    dos_at_energy[key] = float(
                        dos_low
                        + (dos_high - dos_low)
                        * (energy - energy_low)
                        / (energy_high - energy_low)
                    )
            return dos_at_energy
        return {}


@quantity("dos")
class Dos(graph.Mixin):
    """The density of states (DOS) describes the number of states per energy."""

    def __init__(self, source, quantity_name="dos"):
        self._source = source
        self._quantity_name = quantity_name
        self._path = pathlib.Path.cwd()

    @classmethod
    def from_data(cls, raw_dos):
        return cls(source=DataSource(raw_dos))

    def _handler_factory(self, raw):
        return DosHandler.from_data(raw)

    def __str__(self):
        return merge_strings(
            self._source, self._quantity_name, None,
            self._handler_factory, DosHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection=None) -> dict:
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, DosHandler.read,
            selection,
        )

    def to_dict(self, selection=None) -> dict:
        return self.read(selection=selection)

    def to_graph(self, selection=None) -> graph.Graph:
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, DosHandler.to_graph,
            selection,
        )

    def to_frame(self, selection=None):
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, DosHandler.to_frame,
            selection,
        )

    def selections(self) -> dict:
        from py4vasp._raw import definition as raw_module
        handler_selections = merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, DosHandler.selections,
        )
        sources = list(raw_module.selections(self._quantity_name))
        return {self._quantity_name: sources, **handler_selections}


def _series(energies, data):
    for name, dos in data.items():
        spin_factor = -1 if _flip_down_component(name) else 1
        yield graph.Series(energies, spin_factor * dos, name)


def _flip_down_component(name):
    return "down" in name and "up" not in name and "total" not in name
