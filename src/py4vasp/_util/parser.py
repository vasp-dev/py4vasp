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
        self.split_poscar = self.poscar.splitlines()

    @property
    def comment_line(self):
        return self.split_poscar[0]

    @classmethod
    def _get_volume(cls, lattice_vectors):
        return np.dot(
            lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2])
        )

    @classmethod
    def get_reciprocal_lattice_vectors(cls, cell):
        """Get the reciprocal lattice vectors from the cell.

        Computes the reciprocal lattice vectors from the cell. The cell must
        be a Cell object. The reciprocal lattice vectors are computed without
        the (2pi) factor.
        """
        lattice_vectors = cell.lattice_vectors.data * cell.scale
        volume = cls._get_volume(lattice_vectors)
        b1 = np.cross(lattice_vectors[1], lattice_vectors[2]) / volume
        b2 = np.cross(lattice_vectors[2], lattice_vectors[0]) / volume
        b3 = np.cross(lattice_vectors[0], lattice_vectors[1]) / volume
        return np.array([b1, b2, b3])

    @property
    def scaling_factor(self):
        """The scaling factor from the POSCAR file.

        Parses the scaling factor, deciding it the scaling factor is a single
        number (all dimensions) or a vector (each dimension). If the scaling
        if a negative number, then it is interpreted as a volume and the
        a single scaling is computed to make sure that the volume of the final
        cell is the same.
        """
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
        return scaling_factor

    @property
    def cell(self):
        """The cell from the POSCAR file.

        Parses the cell from the POSCAR file. The cell is parsed as is and
        the scaling factor is reported in the Cell object. In case volume scaling
        is used, the scaling factor is computed to make sure that the volume of
        the final cell is the same.
        """
        scaling_factor = self.scaling_factor
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
        """Checks if the POSCAR file has selective dynamics.

        Checks if the POSCAR file has selective dynamics. The check is done
        by looking at the 7th line of the POSCAR file. If the first letter
        is 'S' or 's', then it is assumed that the POSCAR file has selective
        dynamics.
        """
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
        """The topology from the POSCAR file.

        Parses the topology from the POSCAR file. The topology is parsed as is
        and the species names are reported in the Topology object. If the species
        names are not specified in the POSCAR file, then the species names must
        be supplied as an argument.
        """
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
        """The ion positions and selective dynamics from the POSCAR file.

        Parses the ion positions and selective dynamics from the POSCAR file.
        The ion positions and selective dynamics are parsed as is and the
        positions are reported in the Structure object. If the positions are
        specified in Cartesian coordinates, then the positions are converted
        to direct coordinates.
        """
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
            positions = VaspData(positions)
        elif type_positions == "Cartesian":
            cartesian_positions = np.array(
                [x.split()[0:3] for x in positions_and_selective_dyn], dtype=float
            )
            scaling_factor = self.scaling_factor
            cartesian_positions = cartesian_positions * scaling_factor
            reciprocal_lattice_vectors = self.get_reciprocal_lattice_vectors(self.cell)
            direct_positions = cartesian_positions @ reciprocal_lattice_vectors.T
            positions = np.remainder(direct_positions, 1)
        else:
            raise ParserError(
                "The type of positions is not specified in the right format. Choose\
                either 'Direct' or 'Cartesian'."
            )
        if self.has_selective_dynamics:
            selective_dynamics = [x.split()[3:6] for x in positions_and_selective_dyn]
            selective_dynamics = [
                [True if x == "T" else False for x in y] for y in selective_dynamics
            ]
        else:
            selective_dynamics = False
        selective_dynamics = VaspData(selective_dynamics)

        return positions, selective_dynamics

    @property
    def has_lattice_velocities(self):
        """Checks if the POSCAR file has lattice velocities.

        Checks if the POSCAR file has lattice velocities. The check is done
        by looking at the 7th line of the POSCAR file. If that line
        is 'Lattice velocities and vectors', then it is assumed that the POSCAR
        file has lattice velocities.
        """
        num_species = self.topology.number_ion_types.data.sum()
        idx_start = 7 + num_species
        if self.has_selective_dynamics:
            idx_start += 1
        if self.species_name is None:
            idx_start += 1
        if len(self.split_poscar) <= idx_start:
            return False
        lattice_velocities_header = self.split_poscar[idx_start]
        if lattice_velocities_header == "Lattice velocities and vectors":
            return True
        else:
            return False

    @property
    def lattice_velocities(self):
        """The lattice velocities from the POSCAR file.

        Parses the lattice velocities from the POSCAR file. The lattice velocities
        are parsed as is and the velocities are reported in the Structure object.
        If the velocities are specified in Direct coordinates, then the velocities
        are converted to Cartesian coordinates.
        """
        num_species = self.topology.number_ion_types.data.sum()
        idx_start = 7 + num_species
        if not self.has_lattice_velocities:
            raise ParserError("No lattice velocities found in POSCAR.")
        if self.has_selective_dynamics:
            idx_start += 1
        if self.species_name is None:
            idx_start += 1
        lattice_velocities = self.split_poscar[idx_start + 2 : idx_start + 2 + 3]
        lattice_velocities = [x.split() for x in lattice_velocities]
        lattice_velocities = VaspData(np.array(lattice_velocities, dtype=float))
        return lattice_velocities

    @classmethod
    def _convert_direct_to_cartesian(cls, cell, x, scale=True):
        if scale:
            lattice_vectors = cell.lattice_vectors.data * cell.scale
        else:
            lattice_vectors = np.array(cell.lattice_vectors.data)

        cartesian_positions = x @ lattice_vectors.T
        return cartesian_positions

    @property
    def has_ion_velocities(self):
        """Checks if the POSCAR file has ion velocities.

        Checks if the POSCAR file has ion velocities. The header for the ion
        velocities can be either 'Cartesian' or 'Direct' or an empty line
        (assumed to be Cartesian). If the header is not one of these, then
        it is assumed that the POSCAR file does not have ion velocities.
        """
        num_species = self.topology.number_ion_types.data.sum()
        idx_start = 7 + num_species
        if self.has_selective_dynamics:
            idx_start += 1
        if self.species_name is None:
            idx_start += 1
        if self.has_lattice_velocities:
            idx_start += 8
        if len(self.split_poscar) <= idx_start:
            return False
        ion_velocities_header = self.split_poscar[idx_start]
        if ion_velocities_header in ["", "Cartesian", "Direct"]:
            return True
        else:
            return False

    @property
    def ion_velocities(self):
        """The ion velocities from the POSCAR file.

        Parses the ion velocities from the POSCAR file. The ion velocities
        are parsed as is and the velocities are reported in the Structure object.
        If the velocities are specified in Direct coordinates, then the velocities
        are converted to Cartesian coordinates.
        """
        num_species = self.topology.number_ion_types.data.sum()
        if not self.has_ion_velocities:
            raise ParserError("No ion velocities found in POSCAR.")
        idx_start = 7 + num_species
        if self.has_selective_dynamics:
            idx_start += 1
        if self.species_name is None:
            idx_start += 1
        if self.has_lattice_velocities:
            idx_start += 8
        coordinate_system = self.split_poscar[idx_start]
        ion_velocities = self.split_poscar[idx_start + 1 : idx_start + 1 + num_species]
        ion_velocities = [x.split() for x in ion_velocities]
        if coordinate_system == "Direct":
            ion_velocities = self._convert_direct_to_cartesian(
                self.cell, np.array(ion_velocities, dtype=float), scale=False
            )
            ion_velocities = ion_velocities.tolist()
        ion_velocities = VaspData(np.array(ion_velocities, dtype=float))
        return ion_velocities

    def to_contcar(self):
        """Converts a string POSCAR file to a CONTCAR object.

        The CONTCAR object contains the Structure object and the system
        name (comment line), and the optional arguments selective_dynamics,
        lattice_velocities, and ion_velocities. The optional arguments are
        only included if they are present in the POSCAR file.
        """
        ion_positions, selective_dynamics = self.ion_positions_and_selective_dynamics
        structure = Structure(
            topology=self.topology,
            cell=self.cell,
            positions=ion_positions,
        )
        optional = {}
        if self.has_selective_dynamics:
            optional["selective_dynamics"] = selective_dynamics
        if self.has_lattice_velocities:
            optional["lattice_velocities"] = self.lattice_velocities
        if self.has_ion_velocities:
            optional["ion_velocities"] = self.ion_velocities

        contcar = CONTCAR(structure=structure, system=self.comment_line, **optional)
        return contcar
