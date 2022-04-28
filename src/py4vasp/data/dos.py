# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import functools
import itertools
import numpy as np
import pandas as pd
from IPython.lib.pretty import pretty
from .projector import _projectors_or_dummy, _selection_doc, _selection_examples
from py4vasp.data._base import DataBase, RefinementDescriptor
import py4vasp.data._export as _export
from py4vasp.data.kpoint import _kpoints_opt_source
import py4vasp._third_party.graph as _graph
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
    plot = RefinementDescriptor("_plot")
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
{_kpoints_opt_source}

Returns
-------
dict
    Contains the energies at which the DOS was evaluated aligned to the
    Fermi energy and the total DOS or the spin-resolved DOS for
    spin-polarized calculations. If available and a selection is passed,
    the orbital resolved DOS for the selected orbitals is included.

{_selection_examples("dos", "read")}"""
    )
    def _to_dict(self, selection=None):
        return {
            **self._read_data(selection),
            "fermi_energy": self._raw_data.fermi_energy,
        }

    @_documentation.add(
        f"""Generate a graph of the selected data reading it from the VASP output.

Parameters
----------
{_selection_doc}
{_kpoints_opt_source}

Returns
-------
Graph
    Graph containing the total DOS. If the calculation was spin polarized,
    the resulting DOS is spin resolved and the spin-down DOS is plotted
    towards negative values. If a selection is given the orbital-resolved
    DOS is given for the specified projectors.

{_selection_examples("dos", "to_plotly")}"""
    )
    def _plot(self, selection=None):
        data = self._read_data(selection)
        return _graph.Graph(
            series=list(_series(data)),
            xlabel="Energy (eV)",
            ylabel="DOS (1/eV)",
        )

    @_documentation.add(
        f"""Read the data and generate a plotly figure.

Parameters
----------
{_selection_doc}
{_kpoints_opt_source}

Returns
-------
plotly.graph_objects.Figure
    plotly figure containing the total DOS. If the calculation was spin
    polarized, the resulting DOS is spin resolved and the spin-down DOS
    is plotted towards negative values. If a selection is passed the orbital
    resolved DOS is given for the specified projectors.

{_selection_examples("dos", "to_plotly")}"""
    )
    def _to_plotly(self, selection=None):
        return self._plot(selection).to_plotly()

    @_documentation.add(
        f"""Read the data into a pandas DataFrame.

Parameters
----------
{_selection_doc}
{_kpoints_opt_source}

Returns
-------
pd.DataFrame
    Contains the energies at which the DOS was evaluated aligned to the
    Fermi energy and the total DOS or the spin-resolved DOS for
    spin-polarized calculations. If available and a selection is passed,
    the orbital resolved DOS for the selected orbitals is included.

{_selection_examples("dos", "to_frame")}"""
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


def _series(data):
    energies = data["energies"]
    for name, dos in data.items():
        if name == "energies":
            continue
        spin_factor = 1 if "down" not in name else -1
        yield _graph.Series(energies, spin_factor * dos, name)
