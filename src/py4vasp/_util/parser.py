import re
from dataclasses import dataclass

import numpy as np

from py4vasp._raw.data import CONTCAR, Cell, Structure, Topology
from py4vasp._raw.data_wrapper import VaspData
from py4vasp.exception import ParserError


@dataclass
class ParsePoscar:
    poscar: str
    species_name: str or None = None

    def __post_init__(self):
        self.split_poscar = self.poscar.split("\n")

    @property
    def comment_line(self):
        return self.split_poscar[0]

    @classmethod
    def _get_volume(cls, lattice_vectors):
        return np.dot(
            lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2])
        )

    @property
    def cell(self):
        scaling_factor = self.split_poscar[1]
        if len(scaling_factor.split()) not in [1, 3]:
            raise ParserError(
                "The scaling factor is not specified in the right format."
            )
        scaling_factor = np.array(scaling_factor.split(), dtype=float)
        if scaling_factor.ndim == 0 or (
            scaling_factor.ndim == 1 and len(scaling_factor) == 1
        ):
            scaling_factor = scaling_factor[0]
        if scaling_factor.ndim == 1:
            if np.any(scaling_factor <= 0):
                raise ParserError(
                    "The scaling factor for the cell is either negative or zero."
                )
        lattice_vectors = np.array(
            [x.split() for x in self.split_poscar[2:5]], dtype=float
        )
        if scaling_factor.ndim == 1:
            scaled_lattice_vectors = lattice_vectors * scaling_factor
            cell = Cell(lattice_vectors=VaspData(scaled_lattice_vectors), scale=1.0)
        else:
            if scaling_factor > 0:
                cell = Cell(lattice_vectors=lattice_vectors, scale=scaling_factor)
            else:
                volume = self._get_volume(lattice_vectors)
                cell = Cell(
                    lattice_vectors=lattice_vectors,
                    scale=(abs(scaling_factor) / volume) ** (1 / 3),
                )
        return cell

    @property
    def has_selective_dynamics(self):
        if self.species_name is None:
            possible_selective_dynamics = self.split_poscar[7]
        else:
            possible_selective_dynamics = self.split_poscar[6]
        if possible_selective_dynamics[0] in ["S", "s"]:
            return True
        else:
            return False

    @property
    def topology(self):
        if self.species_name is None:
            species_name = self.split_poscar[5].split()
            if not all(s.isalpha() for s in species_name):
                raise ParserError(
                    "Either supply species as an argument or in the POSCAR file."
                )
            number_of_species = self.split_poscar[6].split()
        else:
            species_name = np.array(self.species_name.split())
            number_of_species = self.split_poscar[5].split()
        number_of_species = VaspData(np.array(number_of_species, dtype=int))
        species_name = VaspData(np.array(species_name))
        topology = Topology(number_ion_types=number_of_species, ion_types=species_name)
        return topology

    @property
    def ion_positions_and_selective_dynamics(self):
        number_of_species = self.topology.number_ion_types.data.sum()
        idx_start = 6
        if self.has_selective_dynamics:
            idx_start += 1
        if self.species_name is None:
            idx_start += 1
        type_positions = self.split_poscar[idx_start]
        positions_and_selective_dyn = self.split_poscar[
            idx_start + 1 : idx_start + 1 + number_of_species
        ]
        if type_positions == "Direct":
            positions = np.array(
                [x.split()[0:3] for x in positions_and_selective_dyn], dtype=float
            )
            if self.has_selective_dynamics:
                selective_dynamics = [
                    x.split()[3:6] for x in positions_and_selective_dyn
                ]
                selective_dynamics = [
                    [True if x == "T" else False for x in y] for y in selective_dynamics
                ]
            else:
                selective_dynamics = False
            positions = VaspData(positions)
            selective_dynamics = VaspData(selective_dynamics)
        elif type_positions == "Coordinates":
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown type of positions: {type_positions}")
        return positions, selective_dynamics

    def to_contcar(self):
        ion_positions, selective_dynamics = self.ion_positions_and_selective_dynamics
        structure = Structure(
            topology=self.topology,
            cell=self.cell,
            positions=ion_positions,
        )
        contcar = CONTCAR(
            structure=structure,
            system=self.comment_line,
            selective_dynamics=selective_dynamics,
        )
        return contcar
