# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import bandgap, base
from py4vasp._third_party import graph


class Workfunction(base.Refinery, graph.Mixin):
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

    @base.data_access
    def __str__(self):
        data = self.to_dict()
        return f"""workfunction along {data["direction"]}:
    vacuum potential: {data["vacuum_potential"][0]:.3f} {data["vacuum_potential"][1]:.3f}
    Fermi energy: {data["fermi_energy"]:.3f}
    valence band maximum: {data["valence_band_maximum"]:.3f}
    conduction band minimum: {data["conduction_band_minimum"]:.3f}"""

    @base.data_access
    def to_dict(self):
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
        gap = bandgap.Bandgap.from_data(self._raw_data.reference_potential)
        return {
            "direction": f"lattice vector {self._raw_data.idipol}",
            "distance": self._raw_data.distance[:],
            "average_potential": self._raw_data.average_potential[:],
            "vacuum_potential": self._raw_data.vacuum_potential[:],
            "valence_band_maximum": gap.valence_band_maximum(),
            "conduction_band_minimum": gap.conduction_band_minimum(),
            "fermi_energy": self._raw_data.fermi_energy,
        }

    @base.data_access
    def to_graph(self):
        """Plot the average potential along the lattice vector selected by IDIPOL.

        Returns
        -------
        Graph
            A plot where the distance in the unit cell along the selected lattice vector
            is on the x axis and the averaged potential across the plane of the other
            two lattice vectors is on the y axis.
        """
        data = self.to_dict()
        series = graph.Series(data["distance"], data["average_potential"], "potential")
        return graph.Graph(
            series=series,
            xlabel=f"distance along {data['direction']} (Å)",
            ylabel="average potential (eV)",
        )
