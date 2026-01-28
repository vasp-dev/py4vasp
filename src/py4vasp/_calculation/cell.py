# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp._calculation import base, slice_
from py4vasp._util import check, reader


class Cell(slice_.Mixin, base.Refinery):
    """Cell parameters of the simulation cell."""

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
                for i, bi, ci in enumerate(zip(b, c))
            ]
            beta = [
                np.degrees(np.arccos(np.dot(ai, ci) / (la[i] * lc[i])))
                for i, ai, ci in enumerate(zip(a, c))
            ]
            gamma = [
                np.degrees(np.arccos(np.dot(ai, bi) / (la[i] * lb[i])))
                for i, ai, bi in enumerate(zip(a, b))
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

    def _get_steps(self):
        return self._steps if self._is_trajectory else ()

    @property
    def _is_trajectory(self):
        return self._raw_data.lattice_vectors.ndim == 3


class _LatticeVectors(reader.Reader):
    def error_message(self, key, err):
        key = np.array(key)
        steps = key if key.ndim == 0 else key[0]
        return (
            f"Error reading the lattice vectors. Please check if the steps "
            f"`{steps}` are properly formatted and within the boundaries. "
            "Additionally, you may consider the original error message:\n" + err.args[0]
        )
