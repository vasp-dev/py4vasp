# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp.calculation import _base


class PiezoelectricTensor(_base.Refinery):
    """The piezoelectric tensor is the derivative of the energy with respect to strain and field.

    The piezoelectric tensor represents the coupling between mechanical stress and
    electrical polarization in a material. VASP computes the piezoelectric tensor with
    a linear response calculation. The piezoelectric tensor is a 3x3 matrix that relates
    the three components of stress to the three components of polarization.
    Specifically, it describes how the application of mechanical stress induces an
    electric polarization and, conversely, how an applied electric field results in
    a deformation.

    The piezoelectric tensor helps to characterize the efficiency and anisotropy of the
    piezoelectric response. A large piezoelectric tensor is useful e.g. for sensors
    and actuators. Moreover, the tensor's symmetry properties are coupled to the crystal
    structure and symmetry. Therefore a mismatch of the symmetry properties between
    calculations and experiment can reveal underlying flaws in the characterization of
    the crystal structure.
    """

    @_base.data_access
    def __str__(self):
        data = self.to_dict()
        return f"""Piezoelectric tensor (C/m²)
         XX          YY          ZZ          XY          YZ          ZX
---------------------------------------------------------------------------
{_tensor_to_string(data["clamped_ion"], "clamped-ion")}
{_tensor_to_string(data["relaxed_ion"], "relaxed-ion")}"""

    @_base.data_access
    def to_dict(self):
        """Read the ionic and electronic contribution to the piezoelectric tensor
        into a dictionary.

        It will combine both terms as the total piezoelectric tensor (relaxed_ion)
        but also give the pure electronic contribution, so that you can separate the
        parts.

        Returns
        -------
        dict
            The clamped ion and relaxed ion data for the piezoelectric tensor.
        """
        electron_data = self._raw_data.electron[:]
        return {
            "clamped_ion": electron_data,
            "relaxed_ion": electron_data + self._raw_data.ion[:],
        }


def _tensor_to_string(tensor, label):
    compact_tensor = _compact(tensor.T).T
    line = lambda dir_, vec: dir_ + " " + " ".join(f"{x:11.5f}" for x in vec)
    directions = (" x", " y", " z")
    lines = (line(dir_, vec) for dir_, vec in zip(directions, compact_tensor))
    return f"{label:^75}".rstrip() + "\n" + "\n".join(lines)


def _compact(tensor):
    x, y, z = range(3)
    symmetrized = (
        tensor[x, x],
        tensor[y, y],
        tensor[z, z],
        0.5 * (tensor[x, y] + tensor[y, x]),
        0.5 * (tensor[y, z] + tensor[z, y]),
        0.5 * (tensor[z, x] + tensor[x, z]),
    )
    return np.array(symmetrized)
