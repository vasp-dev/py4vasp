import itertools
import py4vasp.data._base as _base
from py4vasp.data import Structure


class ForceConstants(_base.DataBase):
    """The force constants (second derivatives of atomic displacement).

    You can use this class to extract the force constants of a linear
    response calculation.

    Parameters
    ----------
    raw_force_constants : RawForceConstants
        Dataclass containing the raw force constants data.
    """

    read = _base.RefinementDescriptor("_to_dict")
    to_dict = _base.RefinementDescriptor("_to_dict")
    __str__ = _base.RefinementDescriptor("_to_string")

    def _to_string(self):
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

    def _to_dict(self):
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

    @property
    def _structure(self):
        return Structure(self._raw_data.structure)
