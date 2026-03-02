# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from typing import Optional, Union

import numpy as np

from py4vasp import exception
from py4vasp._calculation import base, slice_
from py4vasp._raw import data as raw_data
from py4vasp._util import check, reader

_VACUUM_RATIO = 2.5


class Cell(slice_.Mixin, base.Refinery):
    """Cell parameters of the simulation cell."""

    _raw_data: raw_data.Cell

    def lattice_vectors(self):
        """Lattice vectors of the simulation cell for all selected steps."""
        lattice_vectors = _LatticeVectors(self._raw_data.lattice_vectors)
        return self.scale() * lattice_vectors[self._get_steps()]

    def scale(self):
        """Scale factor of the simulation cell."""
        if isinstance(self._raw_data.scale, np.float64):
            return self._raw_data.scale
        if not check.is_none(self._raw_data.scale):
            return self._raw_data.scale[()]
        else:
            return 1.0

    def lengths(self):
        """Lengths of the simulation cell for all selected steps."""
        lattices = self.lattice_vectors()
        if lattices.ndim == 3:
            lengths_a = np.linalg.norm(lattices[:, 0, :], axis=-1)
            lengths_b = np.linalg.norm(lattices[:, 1, :], axis=-1)
            lengths_c = np.linalg.norm(lattices[:, 2, :], axis=-1)
            lengths = np.array([lengths_a, lengths_b, lengths_c]).T
            return lengths
        else:
            lengths_a = np.linalg.norm(lattices[0, :])
            lengths_b = np.linalg.norm(lattices[1, :])
            lengths_c = np.linalg.norm(lattices[2, :])
            lengths = np.array([lengths_a, lengths_b, lengths_c])
        return lengths

    def angles(self):
        """Angles of the simulation cell for all selected steps."""
        lattices = self.lattice_vectors()
        if lattices.ndim == 3:
            a = lattices[:, 0]
            b = lattices[:, 1]
            c = lattices[:, 2]
            lengths = self.lengths()
            la = lengths[:, 0]
            lb = lengths[:, 1]
            lc = lengths[:, 2]

            alpha = [
                np.degrees(np.arccos(np.dot(bi, ci) / (lb[i] * lc[i])))
                for i, (bi, ci) in enumerate(zip(b, c))
            ]
            beta = [
                np.degrees(np.arccos(np.dot(ai, ci) / (la[i] * lc[i])))
                for i, (ai, ci) in enumerate(zip(a, c))
            ]
            gamma = [
                np.degrees(np.arccos(np.dot(ai, bi) / (la[i] * lb[i])))
                for i, (ai, bi) in enumerate(zip(a, b))
            ]
            angles = np.array([alpha, beta, gamma]).T
        else:
            a = lattices[0]
            b = lattices[1]
            c = lattices[2]
            la = np.linalg.norm(a)
            lb = np.linalg.norm(b)
            lc = np.linalg.norm(c)

            alpha = np.degrees(np.arccos(np.dot(b, c) / (lb * lc)))
            beta = np.degrees(np.arccos(np.dot(a, c) / (la * lc)))
            gamma = np.degrees(np.arccos(np.dot(a, b) / (la * lb)))
            angles = np.array([alpha, beta, gamma])
        return angles

    @property
    def is_suspected_2d_system(self) -> Union[bool, np.ndarray]:
        """Determine if the system is 2D based on the lattice vectors."""
        lengths = self.lengths()
        dipole_direction = _idipol_to_direction(
            self._raw_data.idipol, self._raw_data.ldipol
        )
        if lengths.ndim == 2:
            return np.array([_is_suspected_2d(l, dipole_direction) for l in lengths])
        else:
            lengths = self.lengths()
            return _is_suspected_2d(lengths, dipole_direction)

    @base.data_access
    def _area_2d(self) -> tuple[Union[float, np.ndarray], Union[str, list[str]]]:
        """Area of the 2D cell if the system is 2D."""
        lattices = self.lattice_vectors()
        lengths = self.lengths()
        idipol_direction = _idipol_to_direction(
            self._raw_data.idipol, self._raw_data.ldipol
        )
        if lattices.ndim == 3:
            area_list = [
                _get_area_2d(lattice, length, idipol_direction)
                for lattice, length in zip(lattices, lengths)
            ]
            return np.array([area[0] for area in area_list]), [
                area[1] for area in area_list
            ]
        else:
            return _get_area_2d(lattices, lengths, idipol_direction)

    def _get_steps(self):
        return self._steps if self._is_trajectory else ()

    @property
    def _is_trajectory(self):
        return self._raw_data.lattice_vectors.ndim == 3

    def _find_likely_vacuum_direction(self):
        """Identify likeliest vacuum direction as the lattice vector with the largest length, or from IDIPOL flag."""
        try:
            lattice_vectors = self.lattice_vectors()
            dipole_direction = _idipol_to_direction(
                self._raw_data.idipol, self._raw_data.ldipol
            )
            if lattice_vectors.ndim == 3:
                if dipole_direction is not None:
                    return (
                        np.zeros(lattice_vectors.shape[0], dtype=int) + dipole_direction
                    )
                return np.argmax(np.linalg.norm(lattice_vectors, axis=-1), axis=-1)
            else:
                return dipole_direction or int(
                    np.argmax(np.linalg.norm(lattice_vectors, axis=-1))
                )
        except Exception:
            return None


def _is_suspected_2d(
    lengths: np.ndarray, dipole_direction: Optional[int] = None
) -> bool:
    if (dipole_direction is not None) and (
        dipole_direction >= 0 and dipole_direction <= 2
    ):
        return True
    if lengths.shape != (3,):
        raise exception._Py4VaspInternalError(
            "Lengths must be a 1D array of length 3 or dipole must be specified."
        )
    max_length_idx = (
        dipole_direction if dipole_direction is not None else np.argmax(lengths)
    )
    other_lengths = [l for i, l in enumerate(lengths) if i != max_length_idx]
    return bool(
        np.all([(lengths[max_length_idx] / l) >= _VACUUM_RATIO for l in other_lengths])
    )


def _get_area_2d(
    lattice: np.ndarray, lengths: np.ndarray, idipol_direction: Optional[int] = None
) -> tuple[float, str]:
    if len(lattice.shape) != 2 or lattice.shape != (3, 3):
        raise ValueError("Lattice must be a 3x3 array.")
    if len(lengths.shape) != 1 or lengths.shape != (3,):
        raise ValueError("Lengths must be a 1D array of length 3.")
    max_length_idx = (
        idipol_direction if idipol_direction is not None else np.argmax(lengths)
    )
    idx1 = (max_length_idx + 1) % 3
    idx2 = (max_length_idx + 2) % 3
    area = np.linalg.norm(np.cross(lattice[idx1], lattice[idx2]))
    return area, f"{min(idx1, idx2)+1}{max(idx1, idx2)+1}"


def _idipol_to_direction(
    idipol: Optional[int] = None, ldipol: Optional[int] = None
) -> int:
    dipole_direction = None
    if not check.is_none(idipol):
        dipole_direction = int(idipol) - 1
        if dipole_direction < 0 or dipole_direction > 2:
            dipole_direction = None
    if not check.is_none(ldipol):
        if not bool(ldipol):
            dipole_direction = None
    return dipole_direction


class _LatticeVectors(reader.Reader):
    def error_message(self, key, err):
        key = np.array(key)
        steps = key if key.ndim == 0 else key[0]
        return (
            f"Error reading the lattice vectors. Please check if the steps "
            f"`{steps}` are properly formatted and within the boundaries. "
            "Additionally, you may consider the original error message:\n" + err.args[0]
        )
