# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from contextlib import suppress

from py4vasp import exception, raw
from py4vasp._calculation import bandgap as bandgap_module
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_graphs,
    merge_strings,
    quantity,
)
from py4vasp._raw.data_db import Workfunction_DB
from py4vasp._third_party import graph


class WorkfunctionHandler:
    """Handler for the workfunction quantity. Works with exactly one raw.Workfunction object."""

    def __init__(self, raw_workfunction: raw.Workfunction):
        self._raw_workfunction = raw_workfunction

    @classmethod
    def from_data(cls, raw_workfunction: raw.Workfunction) -> "WorkfunctionHandler":
        return cls(raw_workfunction)

    def read(self) -> dict:
        """Reports useful information about the workfunction as a dictionary.

        In addition to the vacuum potential, the dictionary contains typical reference
        energies such as the valence band maximum, the conduction band minimum, and the
        Fermi energy. Furthermore you obtain the average potential, so you can use a
        different algorithm to determine the vacuum potential if desired.

        Returns
        -------
        dict
            Contains vacuum potential, average potential and relevant reference energies
            within the surface.
        """
        band_extrema = {}
        with suppress(exception.NoData):
            gap = bandgap_module.BandgapHandler.from_data(
                self._raw_workfunction.reference_potential
            )
            band_extrema = {
                "valence_band_maximum": gap.valence_band_maximum(),
                "conduction_band_minimum": gap.conduction_band_minimum(),
            }
        return {
            "direction": f"lattice vector {self._raw_workfunction.idipol}",
            "distance": self._raw_workfunction.distance[:],
            "average_potential": self._raw_workfunction.average_potential[:],
            "vacuum_potential": self._raw_workfunction.vacuum_potential[:],
            "fermi_energy": self._raw_workfunction.fermi_energy,
            **band_extrema,
        }

    def to_dict(self) -> dict:
        """Public alias for read()."""
        return self.read()

    def to_database(self) -> dict:
        """Serialize workfunction data for database storage."""
        return {
            "workfunction": Workfunction_DB(
                direction=self._raw_workfunction.idipol,
                workfunction_value=None,
            )
        }

    def to_graph(self) -> graph.Graph:
        """Plot the average potential along the lattice vector selected by IDIPOL.

        Returns
        -------
        Graph
            A plot where the distance in the unit cell along the selected lattice vector
            is on the x axis and the averaged potential across the plane of the other
            two lattice vectors is on the y axis.
        """
        data = self.read()
        series = graph.Series(data["distance"], data["average_potential"], "potential")
        return graph.Graph(
            series=series,
            xlabel=f"distance along {data['direction']} (Å)",
            ylabel="average potential (eV)",
        )

    def __str__(self) -> str:
        data = self.read()
        return f"""workfunction along {data["direction"]}:
    vacuum potential: {data["vacuum_potential"][0]:.3f} {data["vacuum_potential"][1]:.3f}
    Fermi energy: {data["fermi_energy"]:.3f}
    valence band maximum: {data["valence_band_maximum"]:.3f}
    conduction band minimum: {data["conduction_band_minimum"]:.3f}"""


@quantity("workfunction")
class Workfunction(graph.Mixin):
    """The workfunction describes the energy required to remove an electron to the vacuum.

    The workfunction of a material is the minimum energy required to remove an
    electron from its most loosely bound state and move it to an energy level just
    outside the material's surface. In other words, it represents the energy barrier
    that electrons must overcome to escape the material. The workfunction helps
    understanding electronic emission phenomena in surface science and materials
    engineering. In VASP, you can compute the workfunction by setting the :tag:`IDIPOL`
    flag in the INCAR file. This class provides then the functionality to analyze the
    resulting potential.
    """

    def __init__(self, source, quantity_name: str = "workfunction"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_workfunction: raw.Workfunction) -> "Workfunction":
        """Create a Workfunction dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_workfunction))

    @property
    def path(self):
        """Returns the path from which the output is obtained."""
        return self._path

    def _handler_factory(self, raw_data):
        return WorkfunctionHandler.from_data(raw_data)

    def read(self) -> dict:
        """Reports useful information about the workfunction as a dictionary.

        In addition to the vacuum potential, the dictionary contains typical reference
        energies such as the valence band maximum, the conduction band minimum, and the
        Fermi energy. Furthermore you obtain the average potential, so you can use a
        different algorithm to determine the vacuum potential if desired.

        Returns
        -------
        dict
            Contains vacuum potential, average potential and relevant reference energies
            within the surface.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            WorkfunctionHandler.read,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Public alias for read(). Check that method for examples and optional arguments."""
        return self.read()

    def to_graph(self) -> graph.Graph:
        """Plot the average potential along the lattice vector selected by IDIPOL.

        Returns
        -------
        Graph
            A plot where the distance in the unit cell along the selected lattice vector
            is on the x axis and the averaged potential across the plane of the other
            two lattice vectors is on the y axis.
        """
        return merge_graphs(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            WorkfunctionHandler.to_graph,
        )

    def __str__(self, selection=None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            WorkfunctionHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
