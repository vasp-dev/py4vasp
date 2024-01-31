# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp.calculation import _base


class ElasticModulus(_base.Refinery):
    """The elastic modulus is the second derivative of the energy with respect to strain.

    The elastic modulus, also known as the modulus of elasticity, is a measure of a
    material's stiffness and its ability to deform elastically in response to an
    applied force. It quantifies the ratio of stress (force per unit area) to strain
    (deformation) in a material within its elastic limit. You can use this class to
    extract the elastic modulus of a linear response calculation. There are two
    variants of the elastic modulus: (i) in the clamped-ion one, the cell is deformed
    but the ions are kept in their positions; (ii) in the relaxed-ion one the
    atoms are allowed to relax when the cell is deformed.
    """

    @_base.data_access
    def to_dict(self):
        """Read the clamped-ion and relaxed-ion elastic modulus into a dictionary.

        Returns
        -------
        dict
            Contains the level of approximation and its associated elastic modulus.
        """
        return {
            "clamped_ion": self._raw_data.clamped_ion[:],
            "relaxed_ion": self._raw_data.relaxed_ion[:],
        }

    @_base.data_access
    def __str__(self):
        return f"""Elastic modulus (kBar)
Direction    XX          YY          ZZ          XY          YZ          ZX
--------------------------------------------------------------------------------
{_elastic_modulus_string(self._raw_data.clamped_ion[:], "clamped-ion")}
{_elastic_modulus_string(self._raw_data.relaxed_ion[:], "relaxed-ion")}"""


def _elastic_modulus_string(tensor, label):
    compact_tensor = _compact(_compact(tensor).T).T
    line = lambda dir_, vec: dir_ + 6 * " " + " ".join(f"{x:11.4f}" for x in vec)
    directions = ("XX", "YY", "ZZ", "XY", "YZ", "ZX")
    lines = (line(dir_, vec) for dir_, vec in zip(directions, compact_tensor))
    return f"{label:^79}".rstrip() + "\n" + "\n".join(lines)


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
