# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import data
from py4vasp._data import base, slice_
from py4vasp._util import documentation, reader


@documentation.format(examples=slice_.examples("stress"))
class Stress(slice_.Mixin, base.Refinery):
    """The stress acting on the unit cell for selected steps of the simulation.

    You can use this class to analyze the stress on the shape of the cell. In
    particular, you can check whether the stress is small at the end of the
    calculation.

    {examples}
    """

    @base.data_access
    def __str__(self):
        "Convert the stress to a format similar to the OUTCAR file."
        step = self._last_step_in_slice
        eV_to_kB = 1.602176634e3 / self._structure[step].volume()
        stress = _symmetry_reduce(self._stress[step])
        stress_to_string = lambda stress: " ".join(f"{x:11.5f}" for x in stress)
        return f"""
FORCE on cell =-STRESS in cart. coord.  units (eV):
Direction    XX          YY          ZZ          XY          YZ          ZX
-------------------------------------------------------------------------------------
Total   {stress_to_string(stress / eV_to_kB)}
in kB   {stress_to_string(stress)}
""".strip()

    @base.data_access
    @documentation.format(examples=slice_.examples("stress", "to_dict"))
    def to_dict(self):
        """Read the stress and associated structural information for one or more
        selected steps of the trajectory.

        Returns
        -------
        dict
            Contains the stress for all selected steps and the structural information
            to know on which cell the stress acts.

        {examples}
        """
        return {
            "structure": self._structure[self._steps].read(),
            "stress": self._stress[self._steps],
        }

    @property
    def _structure(self):
        return data.Structure.from_data(self._raw_data.structure)

    @property
    def _stress(self):
        return _StressReader(self._raw_data.stress)


class _StressReader(reader.Reader):
    def error_message(self, key, err):
        key = np.array(key)
        steps = key if key.ndim == 0 else key[0]
        return (
            f"Error reading the stress. Please check if the steps "
            f"`{steps}` are properly formatted and within the boundaries. "
            "Additionally, you may consider the original error message:\n" + err.args[0]
        )


def _symmetry_reduce(stress_tensor):
    symmetry_reduced_tensor = [
        stress_tensor[0, 0],
        stress_tensor[1, 1],
        stress_tensor[2, 2],
        0.5 * (stress_tensor[0, 1] + stress_tensor[1, 0]),
        0.5 * (stress_tensor[1, 2] + stress_tensor[2, 1]),
        0.5 * (stress_tensor[0, 2] + stress_tensor[2, 0]),
    ]
    return np.array(symmetry_reduced_tensor)
