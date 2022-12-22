# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pandas as pd
from IPython.lib.pretty import pretty

from py4vasp import data
from py4vasp._data import base, projector
from py4vasp._third_party import graph
from py4vasp._util import documentation


class Dos(base.Refinery, graph.Mixin):
    """The electronic density of states (DOS).

    You can use this class to extract the DOS data of a VASP calculation.
    Typically you want to run a non self consistent calculation with a
    denser mesh for a smoother DOS, but the class will work independent
    of it. If you generated orbital decomposed DOS, you can use this
    class to select which subset of these orbitals to read or plot.
    """

    _missing_data_message = "No DOS data found, please verify that LORBIT flag is set."

    @base.data_access
    def __str__(self):
        energies = self._raw_data.energies
        return f"""
{"spin polarized" if self._spin_polarized() else ""} Dos:
    energies: [{energies[0]:0.2f}, {energies[-1]:0.2f}] {len(energies)} points
{pretty(self._projectors())}
    """.strip()

    @base.data_access
    @documentation.format(
        selection_doc=projector.selection_doc,
        examples=projector.selection_examples("dos", "to_dict"),
    )
    def to_dict(self, selection=None):
        """Read the data into a dictionary.

        Parameters
        ----------
        {selection_doc}

        Returns
        -------
        dict
            Contains the energies at which the DOS was evaluated aligned to the
            Fermi energy and the total DOS or the spin-resolved DOS for
            spin-polarized calculations. If available and a selection is passed,
            the orbital resolved DOS for the selected orbitals is included.

        {examples}
        """
        return {
            **self._read_data(selection),
            "fermi_energy": self._raw_data.fermi_energy,
        }

    @base.data_access
    @documentation.format(
        selection_doc=projector.selection_doc,
        examples=projector.selection_examples("dos", "to_graph"),
    )
    def to_graph(self, selection=None):
        """Generate a graph of the selected data reading it from the VASP output.

        Parameters
        ----------
        {selection_doc}

        Returns
        -------
        Graph
            Graph containing the total DOS. If the calculation was spin polarized,
            the resulting DOS is spin resolved and the spin-down DOS is plotted
            towards negative values. If a selection is given the orbital-resolved
            DOS is given for the specified projectors.

        {examples}
        """
        data = self._read_data(selection)
        return graph.Graph(
            series=list(_series(data)),
            xlabel="Energy (eV)",
            ylabel="DOS (1/eV)",
        )

    @base.data_access
    @documentation.format(
        selection_doc=projector.selection_doc,
        examples=projector.selection_examples("dos", "to_frame"),
    )
    def to_frame(self, selection=None):
        """Read the data into a pandas DataFrame.

        Parameters
        ----------
        {selection_doc}

        Returns
        -------
        pd.DataFrame
            Contains the energies at which the DOS was evaluated aligned to the
            Fermi energy and the total DOS or the spin-resolved DOS for
            spin-polarized calculations. If available and a selection is passed,
            the orbital resolved DOS for the selected orbitals is included.

        {examples}
        """
        df = pd.DataFrame(self._read_data(selection))
        df.fermi_energy = self._raw_data.fermi_energy
        return df

    def _spin_polarized(self):
        return self._raw_data.dos.shape[0] == 2

    def _projectors(self):
        return data.Projector.from_data(self._raw_data.projectors)

    def _read_data(self, selection):
        proj = self._projectors()
        args = (selection, self._raw_data.projections)
        proj.read(*args)
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
        yield graph.Series(energies, spin_factor * dos, name)
