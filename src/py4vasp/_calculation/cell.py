# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp._calculation import base, slice_
from py4vasp._util import reader


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
        if not self._raw_data.scale.is_none():
            return self._raw_data.scale[()]
        else:
            return 1.0

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
