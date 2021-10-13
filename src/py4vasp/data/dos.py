import functools
import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.lib.pretty import pretty
from .projector import _projectors_or_dummy, _selection_doc
from py4vasp.data._base import DataBase, RefinementDescriptor
import py4vasp.data._export as _export
import py4vasp._util.documentation as _documentation


class Dos(DataBase, _export.Image):
    """The electronic density of states (DOS).

    You can use this class to extract the DOS data of a Vasp calculation.
    Typically you want to run a non self consistent calculation with a
    denser mesh for a smoother DOS, but the class will work independent
    of it. If you generated orbital decomposed DOS, you can use this
    class to select which subset of these orbitals to read or plot.

    Parameters
    ----------
    raw_dos : RawDos
        Dataclass containing the raw data necessary to produce a DOS.
    """

    _missing_data_message = "No DOS data found, please verify that LORBIT flag is set."

    read = RefinementDescriptor("_to_dict")
    to_dict = RefinementDescriptor("_to_dict")
    plot = RefinementDescriptor("_to_plotly")
    to_plotly = RefinementDescriptor("_to_plotly")
    to_frame = RefinementDescriptor("_to_frame")
    __str__ = RefinementDescriptor("_to_string")

    def _to_string(self):
        energies = self._raw_data.energies
        return f"""
{"spin polarized" if self._spin_polarized() else ""} Dos:
    energies: [{energies[0]:0.2f}, {energies[-1]:0.2f}] {len(energies)} points
{pretty(self._projectors())}
    """.strip()

    @_documentation.add(
        f"""Read the data into a dictionary.

Parameters
----------
{_selection_doc}

Returns
-------
dict
    Contains the energies at which the DOS was evaluated aligned to the
    Fermi energy and the total DOS or the spin-resolved DOS for
    spin-polarized calculations. If available and a selection is passed,
    the orbital resolved DOS for the selected orbitals is included."""
    )
    def _to_dict(self, selection=None):
        return {
            **self._read_data(selection),
            "fermi_energy": self._raw_data.fermi_energy,
        }

    @_documentation.add(
        f"""Read the data and generate a plotly figure.

Parameters
----------
{_selection_doc}

Returns
-------
plotly.graph_objects.Figure
    plotly figure containing the total DOS. If the calculation was spin
    polarized, the resulting DOS is spin resolved and the spin-down DOS
    is plotted towards negative values. If a selection the orbital
    resolved DOS is given for the specified projectors."""
    )
    def _to_plotly(self, selection=None):
        df = self._to_frame(selection)
        data = [_scatter_plot(df, col) for col in df if col != "energies"]
        default = {
            "xaxis": {"title": {"text": "Energy (eV)"}},
            "yaxis": {"title": {"text": "DOS (1/eV)"}},
        }
        return go.Figure(data=data, layout=default)

    @_documentation.add(
        f"""Read the data into a pandas DataFrame.

Parameters
----------
{_selection_doc}

Returns
-------
pd.DataFrame
    Contains the energies at which the DOS was evaluated aligned to the
    Fermi energy and the total DOS or the spin-resolved DOS for
    spin-polarized calculations. If available and a selection is passed,
    the orbital resolved DOS for the selected orbitals is included."""
    )
    def _to_frame(self, selection=None):
        df = pd.DataFrame(self._read_data(selection))
        df.fermi_energy = self._raw_data.fermi_energy
        return df

    def _spin_polarized(self):
        return self._raw_data.dos.shape[0] == 2

    def _projectors(self):
        return _projectors_or_dummy(self._raw_data.projectors)

    def _read_data(self, selection):
        return {
            **self._read_energies(),
            **self._read_total_dos(),
            **self._projectors().read(selection, self._raw_data.projections),
        }

    def _read_energies(self):
        return {"energies": self._raw_data.energies[:] - self._raw_data.fermi_energy}

    def _read_total_dos(self):
        if self._spin_polarized():
            return {"up": self._raw_data.dos[0, :], "down": self._raw_data.dos[1, :]}
        else:
            return {"total": self._raw_data.dos[0, :]}


def _scatter_plot(df, column):
    spin_factor = 1 if "down" not in column else -1
    return go.Scatter(x=df["energies"], y=spin_factor * df[column], name=column)
