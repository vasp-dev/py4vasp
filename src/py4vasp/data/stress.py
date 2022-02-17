# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

import py4vasp.data._base as _base
import py4vasp.data._trajectory as _trajectory
from py4vasp.data import Structure
import py4vasp.exceptions as exception
import py4vasp._util.documentation as _documentation
import py4vasp._util.reader as _reader

_stress_docstring = f"""
The stress acting on the unit cell for selected steps of the simulation.

You can use this class to analyze the stress on the shape of the cell. In
particular, you can check whether the stress is small at the end of the
calculation.

Parameters
----------
raw_stress : RawStress
    Dataclass containing the raw stress tensor and associated structure data.

{_trajectory.trajectory_examples("stress")}
""".strip()


@_documentation.add(_stress_docstring)
class Stress(_trajectory.DataTrajectory):
    read = _base.RefinementDescriptor("_to_dict")
    to_dict = _base.RefinementDescriptor("_to_dict")
    __str__ = _base.RefinementDescriptor("_to_string")

    def _to_string(self):
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

    @_documentation.add(
        f"""Read the stress and associated structural information for one or more
selected steps of the trajectory.

Returns
-------
dict
    Contains the stress for all selected steps and the structural information
    to know on which cell the stress acts.

{_trajectory.trajectory_examples("stress", "read")}"""
    )
    def _to_dict(self):
        return {
            "structure": self._structure[self._steps].read(),
            "stress": self._stress[self._steps],
        }

    @property
    def _structure(self):
        return Structure(self._raw_data.structure)

    @property
    def _stress(self):
        return _StressReader(self._raw_data.stress)


class _StressReader(_reader.Reader):
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
