import functools
import itertools
import numpy as np
import pandas as pd
from .projectors import _projectors_or_dummy
from py4vasp.data import _util


class Dos:
    def __init__(self, raw_dos):
        self._raw = raw_dos
        self._fermi_energy = raw_dos.fermi_energy
        self._energies = raw_dos.energies
        self._dos = raw_dos.dos
        self._spin_polarized = self._dos.shape[0] == 2
        self._has_partial_dos = raw_dos.projectors is not None
        self._projectors = _projectors_or_dummy(raw_dos.projectors)
        self._projections = raw_dos.projections

    @classmethod
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "dos")

    def plot(self, selection=None):
        df = self.to_frame(selection)
        if self._spin_polarized:
            for col in filter(lambda col: "down" in col, df):
                df[col] = -df[col]
        default = {
            "x": "energies",
            "xTitle": "Energy (eV)",
            "yTitle": "DOS (1/eV)",
            "asFigure": True,
        }
        return df.iplot(**default)

    def read(self, selection=None):
        return self.to_dict(selection)

    def to_dict(self, selection=None):
        return {**self._read_data(selection), "fermi_energy": self._fermi_energy}

    def to_frame(self, selection=None):
        df = pd.DataFrame(self._read_data(selection))
        df.fermi_energy = self._fermi_energy
        return df

    def _read_data(self, selection):
        return {
            **self._read_energies(),
            **self._read_total_dos(),
            **self._projectors.read(selection, self._raw.projections),
        }

    def _read_energies(self):
        return {"energies": self._energies[:] - self._fermi_energy}

    def _read_total_dos(self):
        if self._spin_polarized:
            return {"up": self._dos[0, :], "down": self._dos[1, :]}
        else:
            return {"total": self._dos[0, :]}
