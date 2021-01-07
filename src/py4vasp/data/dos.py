import functools
import itertools
import numpy as np
import pandas as pd
from IPython.lib.pretty import pretty
from .projectors import _projectors_or_dummy, _selection_doc
from py4vasp.data import _util

_to_dict_doc = (
    """ Read the data into a dictionary.

Parameters
----------
{}

Returns
-------
dict
    Contains the energies at which the DOS was evaluated aligned to the
    Fermi energy and the total DOS or the spin-resolved DOS for
    spin-polarized calculations. If available and a selection is passed,
    the orbital resolved DOS for the selected orbitals is included.
"""
).format(_selection_doc)

_to_plotly_doc = (
    """ Read the data and generate a plotly figure.

Parameters
----------
{}

Returns
-------
plotly.graph_objects.Figure
    plotly figure containing the total DOS. If the calculation was spin
    polarized, the resulting DOS is spin resolved and the spin-down DOS
    is plotted towards negative values. If a selection the orbital
    resolved DOS is given for the specified projectors.
"""
).format(_selection_doc)

_to_frame_doc = (
    """ Read the data into a pandas DataFrame.

Parameters
----------
{}

Returns
-------
pd.DataFrame
    Contains the energies at which the DOS was evaluated aligned to the
    Fermi energy and the total DOS or the spin-resolved DOS for
    spin-polarized calculations. If available and a selection is passed,
    the orbital resolved DOS for the selected orbitals is included.
"""
).format(_selection_doc)


@_util.add_wrappers
class Dos(_util.Data):
    """The electronic density of states (DOS).

    You can use this class to extract the DOS data of a Vasp calculation.
    Typically you want to run a non self consistent calculation with a
    denser mesh for a smoother DOS, but the class will work independent
    of it. If you generated orbital decomposed DOS, you can use this
    class to select which subset of these orbitals to read or plot.

    Parameters
    ----------
    raw_dos : raw.Dos
        Dataclass containing the raw data necessary to produce a DOS.
    """

    def __init__(self, raw_dos):
        error_message = "No DOS data found, please verify that LORBIT flag is set."
        _util.raise_error_if_data_is_none(raw_dos, error_message)
        super().__init__(raw_dos)
        self._fermi_energy = raw_dos.fermi_energy
        self._energies = raw_dos.energies
        self._dos = raw_dos.dos
        self._spin_polarized = self._dos.shape[0] == 2
        self._has_partial_dos = raw_dos.projectors is not None
        self._projectors = _projectors_or_dummy(raw_dos.projectors)
        self._projections = raw_dos.projections

    def _repr_pretty_(self, p, cycle):
        text = f"""
{"spin polarized" if self._spin_polarized else ""} Dos:
   energies: [{self._energies[0]:0.2f}, {self._energies[-1]:0.2f}] {len(self._energies)} points
{pretty(self._projectors)}
        """.strip()
        p.text(text)

    @classmethod
    @_util.add_doc(_util.from_file_doc("electronic DOS"))
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "dos")

    @_util.add_doc(_to_plotly_doc)
    def to_plotly(self, selection=None):
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

    @_util.add_doc(_to_dict_doc)
    def to_dict(self, selection=None):
        return {**self._read_data(selection), "fermi_energy": self._fermi_energy}

    @_util.add_doc(_to_frame_doc)
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
