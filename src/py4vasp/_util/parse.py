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


@dataclass
class PoscarParser:
    poscar_lines: List[str]
    species_name: str or None = None

    def parse_lines(self):
        remaining_lines = iter(self.poscar_lines)
        result = {"system": next(remaining_lines)}
        result, remaining_lines = self._parse_scaling_factor(result, remaining_lines)
        result, remaining_lines = self._parse_cell(result, remaining_lines)
        result, remaining_lines = self._parse_stoichiometry(result, remaining_lines)
        result, remaining_lines = self._parse_ion_lines(result, remaining_lines)
        result, remaining_lines = self._parse_lattice_velocity(result, remaining_lines)
        result, remaining_lines = self._parse_ion_velocities(result, remaining_lines)
        del result["scaling_factor"]  # remove unnessary element
        return result

    def _parse_scaling_factor(self, result, remaining_lines):
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
        result["scaling_factor"] = scaling_factor
        return result, remaining_lines

    def _parse_cell(self, result, remaining_lines):
        """The cell from the POSCAR file.

        Parses the cell from the POSCAR file. The cell is parsed as is and
        the scaling factor is reported in the Cell object. In case volume scaling
        is used, the scaling factor is computed to make sure that the volume of
        the final cell is the same.
        """
        scaling_factor = result["scaling_factor"]
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
        result["cell"] = cell
        return result, remaining_lines

    def _parse_stoichiometry(self, result, remaining_lines):
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
        result["stoichiometry"] = raw.Stoichiometry(
            number_ion_types=number_of_species, ion_types=species_name
        )
        return result, remaining_lines

    def _parse_ion_lines(self, result, remaining_lines):
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
        number_ions = np.sum(result["stoichiometry"].number_ion_types)
        possible_selective_dynamics = next(remaining_lines)
        has_selective_dynamics = possible_selective_dynamics[0] in "sS"
        if not has_selective_dynamics:
            remaining_lines = _put_back(remaining_lines, possible_selective_dynamics)
        coordinate_system = next(remaining_lines)
        positions = []
        selective_dynamics = []
        for _ in range(number_ions):
            self._parse_ion_line(next(remaining_lines), positions, selective_dynamics)
        if coordinate_system[0] in "cCkK":
            cell = result["cell"]
            scaling_factor = result["scaling_factor"]
            if np.all(scaling_factor < 0):
                scaling_factor = cell.scale
            cartesian_positions = np.array(positions) * scaling_factor
            reciprocal_lattice_vectors = self._get_reciprocal_lattice_vectors(cell)
            direct_positions = cartesian_positions @ reciprocal_lattice_vectors.T
            positions = np.remainder(direct_positions, 1)
        else:
            positions = np.array(positions)
        result["positions"] = VaspData(positions)
        if has_selective_dynamics:
            result["selective_dynamics"] = VaspData(np.array(selective_dynamics))
        return result, remaining_lines

    @staticmethod
    def _parse_ion_line(line, positions, selective_dynamics):
        parts = line.split()
        positions.append(np.array(parts[:3], dtype=np.float64))
        selective_dynamics.append(np.array([x == "T" for x in parts[3:]]))

    def _parse_lattice_velocity(self, result, remaining_lines):
        """The lattice velocities from the POSCAR file.

        Checks if the POSCAR file has lattice velocities. The check is done
        by looking at the line after the positions. If that line
        is 'Lattice velocities and vectors', then it is assumed that the POSCAR
        file has lattice velocities.

        Parses the lattice velocities from the POSCAR file. The lattice velocities
        are parsed as is and the velocities are reported in the Structure object.
        If the velocities are specified in Direct coordinates, then the velocities
        are converted to Cartesian coordinates.
        """
        try:
            possible_lattice_velocities = next(remaining_lines)
        except StopIteration:  # lattice velocity is optional
            return result, remaining_lines
        if not possible_lattice_velocities[0] in "lL":
            remaining_lines = _put_back(remaining_lines, possible_lattice_velocities)
            return result, remaining_lines
        init_lattice_velocities = next(remaining_lines)
        if init_lattice_velocities.strip() != "1":
            message = "Only init lattice velocities = 1 is implemented!"
            raise exception.ParserError(message)
        lattice_velocities = [next(remaining_lines) for _ in range(6)]
        lattice_velocities = [x.split() for x in lattice_velocities[:3]]
        result["lattice_velocities"] = VaspData(
            np.array(lattice_velocities, dtype=float)
        )
        return result, remaining_lines

    def _parse_ion_velocities(self, result, remaining_lines):
        """The ion velocities from the POSCAR file.

        Checks if the POSCAR file has ion velocities. The header for the ion
        velocities can be either 'Cartesian' or 'Direct' or an empty line
        (assumed to be Cartesian). If the header is not one of these, then
        it is assumed that the POSCAR file does not have ion velocities.

        Parses the ion velocities from the POSCAR file. The ion velocities
        are parsed as is and the velocities are reported in the Structure object.
        If the velocities are specified in Direct coordinates, then the velocities
        are converted to Cartesian coordinates.
        """
        try:
            possible_coordinate_system = next(remaining_lines)
        except StopIteration:  # velocities are optional
            return result, remaining_lines
        coordinate_system = possible_coordinate_system
        number_ions = np.sum(result["stoichiometry"].number_ion_types)
        ion_velocities = [next(remaining_lines) for _ in range(number_ions)]
        ion_velocities = [x.split() for x in ion_velocities]
        if not coordinate_system[0] in "cCkK ":
            # I'm not sure this implementation is correct, in VASP there is a factor of
            # POTIM to convert between Cartesian to fractional coordinates. Since this
            # case is not common, let's raise an error instead
            ion_velocities = self._convert_direct_to_cartesian(
                result["cell"], np.array(ion_velocities, dtype=float), scale=False
            )
            ion_velocities = ion_velocities.tolist()
            message = "Velocities can only be parsed in Cartesian coordinates."
            raise exception.ParserError(message)
        result["ion_velocities"] = VaspData(np.array(ion_velocities, dtype=float))
        return result, remaining_lines

    @staticmethod
    def _get_reciprocal_lattice_vectors(cell):
        """Get the reciprocal lattice vectors from the cell.

        Computes the reciprocal lattice vectors from the cell. The cell must
        be a Cell object. The reciprocal lattice vectors are computed without
        the (2pi) factor.
        """
        lattice_vectors = cell.lattice_vectors.data * cell.scale
        volume = PoscarParser._get_volume(lattice_vectors)
        b1 = np.cross(lattice_vectors[1], lattice_vectors[2]) / volume
        b2 = np.cross(lattice_vectors[2], lattice_vectors[0]) / volume
        b3 = np.cross(lattice_vectors[0], lattice_vectors[1]) / volume
        return np.array([b1, b2, b3])

    @staticmethod
    def _get_volume(lattice_vectors):
        return lattice_vectors[0] @ np.cross(lattice_vectors[1], lattice_vectors[2])

    @staticmethod
    def _convert_direct_to_cartesian(cell, x, scale=True):
        if scale:
            lattice_vectors = cell.lattice_vectors.data * cell.scale
        else:
            lattice_vectors = np.array(cell.lattice_vectors.data)
        cartesian_positions = x @ lattice_vectors.T
        return cartesian_positions


def _put_back(iterator, item):
    return itertools.chain([item], iterator)
