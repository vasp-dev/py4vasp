# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

from py4vasp._data import base, structure


class ForceConstant(base.Refinery, structure.Mixin):
    """The force constants (second derivatives of atomic displacement).

    You can use this class to extract the force constants of a linear
    response calculation.
    """

    @base.data_access
    def __str__(self):
        result = """
Force constants (eV/Å²):
atom(i)  atom(j)   xi,xj     xi,yj     xi,zj     yi,xj     yi,yj     yi,zj     zi,xj     zi,yj     zi,zj
----------------------------------------------------------------------------------------------------------
""".strip()
        number_atoms = self._structure.number_atoms()
        force_constants = self._raw_data.force_constants[:]
        force_constants = 0.5 * (force_constants + force_constants.T)
        slice_ = lambda x: slice(3 * x, 3 * (x + 1))
        for i, j in itertools.combinations_with_replacement(range(number_atoms), 2):
            subsection = force_constants[slice_(i), slice_(j)]
            string_representation = " ".join(f"{x:9.4f}" for x in subsection.flatten())
            result += f"\n{i + 1:6d}   {j + 1:6d}  {string_representation}"
        return result

    @base.data_access
    def to_dict(self):
        """Read structure information and force constants into a dictionary.

        The structural information is added to inform about which atoms are included
        in the array. The force constants array contains the second derivatives with
        respect to atomic displacement for all atoms and directions.

        Returns
        -------
        dict
            Contains structural information as well as the raw force constant data.
        """
        return {
            "structure": self._structure.read(),
            "force_constants": self._raw_data.force_constants[:],
        }
