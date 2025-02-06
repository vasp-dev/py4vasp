import itertools
from dataclasses import dataclass
from typing import List

import numpy as np

from py4vasp import exception, raw
from py4vasp._raw.data_wrapper import VaspData


def POSCAR(poscar_string, ion_types=None):
    """Converts a string POSCAR file to a CONTCAR object.

    The CONTCAR object contains the Structure object and the system
    name (comment line), and the optional arguments selective_dynamics,
    lattice_velocities, and ion_velocities. The optional arguments are
    only included if they are present in the POSCAR file.

    Parameters
    ----------
    poscar_string : str
        A string in POSCAR format.
    ion_types : Sequence or None
        If the POSCAR file does not set the ion types you need to provide the ion
        types as an arguments

    Returns
    -------
    CONTCAR
        A CONTCAR object with the data in the string.
    """
    parser = PoscarParser(poscar_string.splitlines(), ion_types)
    contcar_content = parser.parse_lines()
    structure_keys = ["stoichiometry", "cell", "positions"]
    structure_content = {key: contcar_content.pop(key) for key in structure_keys}
    contcar_content["structure"] = raw.Structure(**structure_content)
    return raw.CONTCAR(**contcar_content)


def _put_back(iterator, item):
    return itertools.chain([item], iterator)


@dataclass
class PoscarParser:
    poscar_lines: List[str]
    species_name: str or None = None

    def parse_lines(self):
        remaining_lines = iter(self.poscar_lines)
        result = {"system": next(remaining_lines)}
        scaling_factor, remaining_lines = self._parse_scaling_factor(remaining_lines)
        cell, remaining_lines = self._parse_cell(scaling_factor, remaining_lines)
        result["cell"] = cell
        self.stoichiometry, remaining_lines = self._stoichiometry(remaining_lines)
        number_ions = np.sum(self.stoichiometry.number_ion_types)
        result["stoichiometry"] = self.stoichiometry
        ion_positions, selective_dynamics, remaining_lines = (
            self._ion_positions_and_selective_dynamics(
                number_ions, scaling_factor, cell, remaining_lines
            )
        )
        result["positions"] = ion_positions
        if self.has_selective_dynamics:
            result["selective_dynamics"] = selective_dynamics
        if self.has_lattice_velocities:
            result["lattice_velocities"] = self.lattice_velocities
        if self.has_ion_velocities:
            result["ion_velocities"] = self.ion_velocities(cell)
        return result

    def _parse_scaling_factor(self, remaining_lines):
        """The scaling factor from the POSCAR file.

        Parses the scaling factor, deciding it the scaling factor is a single
        number (all dimensions) or a vector (each dimension). If the scaling
        if a negative number, then it is interpreted as a volume and the
        a single scaling is computed to make sure that the volume of the final
        cell is the same.
        """
        scaling_factor = next(remaining_lines)
        if len(scaling_factor.split()) not in [1, 3]:
            raise exception.ParserError(
                "The scaling factor is not specified in the right format."
            )
        scaling_factor = np.array(scaling_factor.split(), dtype=float)
        if scaling_factor.ndim == 0 or (
            scaling_factor.ndim == 1 and len(scaling_factor) == 1
        ):
            scaling_factor = scaling_factor[0]
        if scaling_factor.ndim == 1:
            if np.any(scaling_factor <= 0):
                raise exception.ParserError(
                    "The scaling factor for the cell is either negative or zero."
                )
        return scaling_factor, remaining_lines

    def _parse_cell(self, scaling_factor, remaining_lines):
        """The cell from the POSCAR file.

        Parses the cell from the POSCAR file. The cell is parsed as is and
        the scaling factor is reported in the Cell object. In case volume scaling
        is used, the scaling factor is computed to make sure that the volume of
        the final cell is the same.
        """
        lattice_vectors = np.array(
            [next(remaining_lines).split() for _ in range(3)], dtype=float
        )
        if scaling_factor.ndim == 1:
            scaled_lattice_vectors = lattice_vectors * scaling_factor
            cell = raw.Cell(lattice_vectors=VaspData(scaled_lattice_vectors), scale=1.0)
        else:
            if scaling_factor > 0:
                cell = raw.Cell(lattice_vectors=lattice_vectors, scale=scaling_factor)
            else:
                volume = self._get_volume(lattice_vectors)
                cell = raw.Cell(
                    lattice_vectors=lattice_vectors,
                    scale=(abs(scaling_factor) / volume) ** (1 / 3),
                )
        return cell, remaining_lines

    def _stoichiometry(self, remaining_lines):
        """The stoichiometry from the POSCAR file.

        Parses the stoichiometry from the POSCAR file. The stoichiometry is parsed as is
        and the species names are reported in the Topology object. If the species
        names are not specified in the POSCAR file, then the species names must
        be supplied as an argument.
        """
        if self.species_name is None:
            species_name = next(remaining_lines).split()
            if not all(s.isalpha() for s in species_name):
                raise exception.ParserError(
                    "Either supply species as an argument or in the POSCAR file."
                )
            number_of_species = next(remaining_lines).split()
        else:
            species_name = np.array(self.species_name)
            number_of_species = next(remaining_lines).split()
        number_of_species = VaspData(np.array(number_of_species, dtype=int))
        species_name = VaspData(np.array(species_name))
        return (
            raw.Stoichiometry(
                number_ion_types=number_of_species, ion_types=species_name
            ),
            remaining_lines,
        )

    def _ion_positions_and_selective_dynamics(
        self, number_ions, scaling_factor, cell, remaining_lines
    ):
        """The ion positions and selective dynamics from the POSCAR file.

        Checks if the POSCAR file has selective dynamics. The check is done
        by looking at the 7th line of the POSCAR file. If the first letter
        is 'S' or 's', then it is assumed that the POSCAR file has selective
        dynamics.

        Parses the ion positions and selective dynamics from the POSCAR file.
        The ion positions and selective dynamics are parsed as is and the
        positions are reported in the Structure object. If the positions are
        specified in Cartesian coordinates, then the positions are converted
        to direct coordinates.
        """
        possible_selective_dynamics = next(remaining_lines)
        has_selective_dynamics = possible_selective_dynamics[0] in "sS"
        if not has_selective_dynamics:
            remaining_lines = _put_back(remaining_lines, possible_selective_dynamics)
        coordinate_system = next(remaining_lines)

        def parse_line(line, positions, selective_dynamics):
            parts = line.split()
            positions.append(np.array(parts[:3], dtype=np.float64))
            selective_dynamics.append(np.array([x == "T" for x in parts[3:]]))

        positions = []
        selective_dynamics = []
        for _ in range(number_ions):
            parse_line(next(remaining_lines), positions, selective_dynamics)
        if coordinate_system[0] in "cCkK":
            if np.all(scaling_factor < 0):
                scaling_factor = cell.scale
            cartesian_positions = np.array(positions) * scaling_factor
            reciprocal_lattice_vectors = self.get_reciprocal_lattice_vectors(cell)
            direct_positions = cartesian_positions @ reciprocal_lattice_vectors.T
            positions = np.remainder(direct_positions, 1)
        else:
            positions = np.array(positions)
        if self.has_selective_dynamics:
            selective_dynamics = np.array(selective_dynamics)
        else:
            selective_dynamics = None
        return VaspData(positions), VaspData(selective_dynamics), remaining_lines

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
    def has_selective_dynamics(self):
        """Checks if the POSCAR file has selective dynamics.

        Checks if the POSCAR file has selective dynamics. The check is done
        by looking at the 7th line of the POSCAR file. If the first letter
        is 'S' or 's', then it is assumed that the POSCAR file has selective
        dynamics.
        """
        if self.species_name is None:
            possible_selective_dynamics = self.poscar_lines[7]
        else:
            possible_selective_dynamics = self.poscar_lines[6]
        if possible_selective_dynamics[0] in ["S", "s"]:
            return True
        else:
            return False

    @property
    def has_lattice_velocities(self):
        """Checks if the POSCAR file has lattice velocities.

        Checks if the POSCAR file has lattice velocities. The check is done
        by looking at the 7th line of the POSCAR file. If that line
        is 'Lattice velocities and vectors', then it is assumed that the POSCAR
        file has lattice velocities.
        """
        num_species = self.stoichiometry.number_ion_types.data.sum()
        idx_start = 7 + num_species
        if self.has_selective_dynamics:
            idx_start += 1
        if self.species_name is None:
            idx_start += 1
        if len(self.poscar_lines) <= idx_start:
            return False
        lattice_velocities_header = self.poscar_lines[idx_start]
        return lattice_velocities_header[0] in "lL"

    @property
    def lattice_velocities(self):
        """The lattice velocities from the POSCAR file.

        Parses the lattice velocities from the POSCAR file. The lattice velocities
        are parsed as is and the velocities are reported in the Structure object.
        If the velocities are specified in Direct coordinates, then the velocities
        are converted to Cartesian coordinates.
        """
        num_species = self.stoichiometry.number_ion_types.data.sum()
        idx_start = 7 + num_species
        if not self.has_lattice_velocities:
            raise exception.ParserError("No lattice velocities found in POSCAR.")
        if self.has_selective_dynamics:
            idx_start += 1
        if self.species_name is None:
            idx_start += 1
        lattice_velocities = self.poscar_lines[idx_start + 2 : idx_start + 2 + 3]
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
        num_species = self.stoichiometry.number_ion_types.data.sum()
        idx_start = 7 + num_species
        if self.has_selective_dynamics:
            idx_start += 1
        if self.species_name is None:
            idx_start += 1
        if self.has_lattice_velocities:
            idx_start += 8
        return len(self.poscar_lines) > idx_start
        # if len(self.poscar_lines) <= idx_start:
        #     return False
        # ion_velocities_header = self.poscar_lines[idx_start]
        # if ion_velocities_header in ["", "Cartesian", "Direct"]:
        #     return True
        # else:
        #     return False

    def ion_velocities(self, cell):
        """The ion velocities from the POSCAR file.

        Parses the ion velocities from the POSCAR file. The ion velocities
        are parsed as is and the velocities are reported in the Structure object.
        If the velocities are specified in Direct coordinates, then the velocities
        are converted to Cartesian coordinates.
        """
        num_species = self.stoichiometry.number_ion_types.data.sum()
        if not self.has_ion_velocities:
            raise exception.ParserError("No ion velocities found in POSCAR.")
        idx_start = 7 + num_species
        if self.has_selective_dynamics:
            idx_start += 1
        if self.species_name is None:
            idx_start += 1
        if self.has_lattice_velocities:
            idx_start += 8
        coordinate_system = self.poscar_lines[idx_start]
        ion_velocities = self.poscar_lines[idx_start + 1 : idx_start + 1 + num_species]
        ion_velocities = [x.split() for x in ion_velocities]
        if not coordinate_system[0] in "cCkK ":
            # I'm not sure this implementation is correct, in VASP there is a factor of
            # POTIM to convert between Cartesian to fractional coordinates. Since this
            # case is not common, let's raise an error instead
            ion_velocities = self._convert_direct_to_cartesian(
                cell, np.array(ion_velocities, dtype=float), scale=False
            )
            ion_velocities = ion_velocities.tolist()
            message = "Velocities can only be parsed in Cartesian coordinates."
            raise exception.ParserError(message)
        ion_velocities = VaspData(np.array(ion_velocities, dtype=float))
        return ion_velocities
