from dataclasses import dataclass

import numpy as np

from py4vasp._raw.data import CONTCAR, Cell, Structure, Topology
from py4vasp._raw.data_wrapper import VaspData


@dataclass
class ParsePoscar:
    poscar: str
    species_name: str | None = None

    def __post_init__(self):
        self.split_poscar = self.poscar.split("\n")

    @property
    def comment_line(self):
        return self.split_poscar[0]

    @property
    def cell(self):
        scaling_factor = self.split_poscar[1]
        scaling_factor = np.array(scaling_factor.split(), dtype=float)
        if len(scaling_factor) == 1:
            scaling_factor = scaling_factor[0]
        lattice_vectors = np.array(
            [x.split() for x in self.split_poscar[2:5]], dtype=float
        )
        lattice_vectors = VaspData(lattice_vectors)
        cell = Cell(lattice_vectors=lattice_vectors, scale=scaling_factor)
        return cell

    @property
    def topology(self):
        if self.species_name is None:
            species_name = self.split_poscar[5].split()
            assert all(
                s.isalpha() for s in species_name
            ), "Either supply species as an argument or in the POSCAR file."
            number_of_species = self.split_poscar[6].split()
        else:
            species_name = np.array(self.species_name.split())
            number_of_species = self.split_poscar[5].split()
        number_of_species = VaspData(np.array(number_of_species, dtype=int))
        species_name = VaspData(np.array(species_name))
        topology = Topology(number_ion_types=number_of_species, ion_types=species_name)
        return topology

    @property
    def ion_positions(self):
        number_of_species = self.topology.number_ion_types.data.sum()
        if self.species_name is None:
            type_positions = self.split_poscar[7]
            positions = self.split_poscar[8 : 8 + number_of_species]
        else:
            type_positions = self.split_poscar[6]
            positions = self.split_poscar[7 : 7 + number_of_species]
        if type_positions == "Direct":
            positions = np.array([x.split() for x in positions], dtype=float)
            positions = VaspData(positions)
        elif type_positions == "Coordinates":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown type of positions: {type_positions}")
        return positions

    def to_contcar(self):
        structure = Structure(
            topology=self.topology,
            cell=self.cell,
            positions=self.ion_positions,
        )
        contcar = CONTCAR(structure)
        return contcar
