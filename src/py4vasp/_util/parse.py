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
    ion_types: List[str] or None = None

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
        scaling_factor = np.squeeze(scaling_factor.split()).astype(np.float64)
        if scaling_factor.size not in [1, 3]:
            message = "The scaling factor should be one or three numbers"
            raise exception.ParserError(message)
        if scaling_factor.ndim == 1 and np.any(scaling_factor <= 0):
            message = "A negative scaling factor is only allowed for a single value."
            raise exception.ParserError(message)
        result["scaling_factor"] = scaling_factor
        return result, remaining_lines

    def _parse_cell(self, result, remaining_lines):
        """The cell from the POSCAR file.

        Parses the cell from the POSCAR file. The cell is parsed as is and
        the scaling factor is reported in the Cell object. In case volume scaling
        is used, the scaling factor is computed to make sure that the volume of
        the final cell is the same.
        """
        lattice_vectors = [next(remaining_lines).split() for _ in range(3)]
        lattice_vectors = np.array(lattice_vectors, dtype=np.float64)
        scaling_factor = result["scaling_factor"]
        if scaling_factor.ndim == 1:
            lattice_vectors *= scaling_factor
            scaling_factor = 1
        elif scaling_factor < 0:
            volume_ratio = abs(scaling_factor) / self._get_volume(lattice_vectors)
            result["scaling_factor"] = scaling_factor = volume_ratio ** (1 / 3)
        result["cell"] = raw.Cell(VaspData(lattice_vectors), VaspData(scaling_factor))
        return result, remaining_lines

    @staticmethod
    def _get_volume(lattice_vectors):
        return np.abs(np.linalg.det(lattice_vectors))

    def _parse_stoichiometry(self, result, remaining_lines):
        """The stoichiometry from the POSCAR file.

        Parses the stoichiometry from the POSCAR file. The stoichiometry is parsed as is
        and the species names are reported in the Topology object. If the species
        names are not specified in the POSCAR file, then the species names must
        be supplied as an argument.
        """
        possible_ion_types = next(remaining_lines)
        has_ion_types = all(type.isalpha() for type in possible_ion_types.split())
        if not has_ion_types and self.ion_types is None:
            message = "If ion types are not in the POSCAR file, you must provide it as an argument for the parser."
            raise exception.ParserError(message)
        if has_ion_types:
            ion_types = possible_ion_types.split()
        else:
            remaining_lines = _put_back(remaining_lines, possible_ion_types)
            ion_types = self.ion_types
        number_ion_types = np.array(next(remaining_lines).split(), dtype=np.int32)
        result["stoichiometry"] = raw.Stoichiometry(
            number_ion_types=VaspData(number_ion_types),
            ion_types=VaspData(np.array(ion_types)),
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
        possible_selective_dynamics = next(remaining_lines)
        has_selective_dynamics = possible_selective_dynamics[0] in "sS"
        if not has_selective_dynamics:
            remaining_lines = _put_back(remaining_lines, possible_selective_dynamics)

        coordinate_system = next(remaining_lines)
        number_ions = np.sum(result["stoichiometry"].number_ion_types)
        positions = []
        selective_dynamics = []
        for _ in range(number_ions):
            self._parse_ion_line(next(remaining_lines), positions, selective_dynamics)

        positions = self._to_fractional(result, positions, coordinate_system)
        result["positions"] = VaspData(positions)
        if has_selective_dynamics:
            result["selective_dynamics"] = VaspData(np.array(selective_dynamics))
        return result, remaining_lines

    @staticmethod
    def _parse_ion_line(line, positions, selective_dynamics):
        parts = line.split()
        positions.append(np.array(parts[:3], dtype=np.float64))
        selective_dynamics.append(np.array([x == "T" for x in parts[3:]]))

    def _to_fractional(self, result, positions, coordinate_system):
        if not coordinate_system[0] in "cCkK":
            return np.array(positions)
        cartesian_positions = np.array(positions) * result["scaling_factor"]
        inverse_lattice_vectors = self._get_inverse_lattice_vectors(result["cell"])
        direct_positions = cartesian_positions @ inverse_lattice_vectors
        return np.remainder(direct_positions, 1)

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
        lattice_velocities = [next(remaining_lines).split() for _ in range(6)]
        lattice_velocities = np.array(lattice_velocities[:3], np.float64)
        result["lattice_velocities"] = VaspData(lattice_velocities)
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
        ion_velocities = [next(remaining_lines).split() for _ in range(number_ions)]
        ion_velocities = np.array(ion_velocities, dtype=np.float64)
        if not coordinate_system[0] in "cCkK ":
            # I'm not sure this implementation is correct, in VASP there is a factor of
            # POTIM to convert between Cartesian to fractional coordinates. Since this
            # case is not common, let's raise an error instead
            ion_velocities = self._convert_direct_to_cartesian(
                result["cell"], ion_velocities, scale=False
            )
            message = "Velocities can only be parsed in Cartesian coordinates."
            raise exception.ParserError(message)
        result["ion_velocities"] = VaspData(ion_velocities)
        return result, remaining_lines

    @staticmethod
    def _get_inverse_lattice_vectors(cell):
        """Get the inverse of lattice vectors from the cell.

        The inverse of the lattice vectors are related to the reciprocal lattice vectors
        up to a factor of 2 * pi.
        """
        lattice_vectors = cell.lattice_vectors.data * cell.scale
        return np.linalg.inv(lattice_vectors)

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
