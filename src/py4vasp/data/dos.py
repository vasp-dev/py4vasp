import functools
import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.lib.pretty import pretty
from .projectors import _projectors_or_dummy, _selection_doc
from py4vasp.data._base import DataBase, RefinementDescriptor
from py4vasp.data import _util


class Dos(DataBase):
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


def _to_string(raw_dos):
    return f"""
{"spin polarized" if _spin_polarized(raw_dos) else ""} Dos:
    energies: [{raw_dos.energies[0]:0.2f}, {raw_dos.energies[-1]:0.2f}] {len(raw_dos.energies)} points
{pretty(_projectors(raw_dos))}
    """.strip()


@_util.add_doc(
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
def _to_dict(raw_dos, selection=None):
    return {**_read_data(raw_dos, selection), "fermi_energy": raw_dos.fermi_energy}


@_util.add_doc(
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
def _to_plotly(raw_dos, selection=None):
    df = _to_frame(raw_dos, selection)
    data = [_scatter_plot(df, col) for col in df if col != "energies"]
    default = {
        "xaxis": {"title": {"text": "Energy (eV)"}},
        "yaxis": {"title": {"text": "DOS (1/eV)"}},
    }
    return go.Figure(data=data, layout=default)


@_util.add_doc(
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
def _to_frame(raw_dos, selection=None):
    df = pd.DataFrame(_read_data(raw_dos, selection))
    df.fermi_energy = raw_dos.fermi_energy
    return df


def _spin_polarized(raw_dos):
    return raw_dos.dos.shape[0] == 2


def _projectors(raw_dos):
    return _projectors_or_dummy(raw_dos.projectors)


def _read_data(raw_dos, selection):
    return {
        **_read_energies(raw_dos),
        **_read_total_dos(raw_dos),
        **_projectors(raw_dos).read(selection, raw_dos.projections),
    }


def _read_energies(raw_dos):
    return {"energies": raw_dos.energies[:] - raw_dos.fermi_energy}


def _read_total_dos(raw_dos):
    if _spin_polarized(raw_dos):
        return {"up": raw_dos.dos[0, :], "down": raw_dos.dos[1, :]}
    else:
        return {"total": raw_dos.dos[0, :]}


def _scatter_plot(df, column):
    spin_factor = 1 if "down" not in column else -1
    return go.Scatter(x=df["energies"], y=spin_factor * df[column], name=column)
