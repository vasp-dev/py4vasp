# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.data import _base, Structure
from py4vasp._util import documentation as _documentation

_read_doc = """Read structure information and Born effective charges into a dictionary.

The structural information is added to inform about which atoms are included
in the array. The Born effective charges array contains the mixed second
derivative with respect to an electric field and an atomic displacement for
all atoms and possible directions.

Returns
-------
dict
    Contains structural information as well as the Born effective charges.
"""


class BornEffectiveCharge(_base.Refinery):
    """The Born effective charge tensors coupling electric field and atomic displacement.

    You can use this class to extract the Born effective charges of a linear
    response calculation.
    """

    @_base.data_access
    def __str__(self):
        data = self.to_dict()
        result = """
BORN EFFECTIVE CHARGES (including local field effects) (in |e|, cumulative output)
---------------------------------------------------------------------------------
        """.strip()
        generator = zip(data["structure"]["elements"], data["charge_tensors"])
        vec_to_string = lambda vec: " ".join(f"{x:11.5f}" for x in vec)
        for ion, (element, charge_tensor) in enumerate(generator):
            result += f"""
ion {ion + 1:4d}   {element}
    1 {vec_to_string(charge_tensor[0])}
    2 {vec_to_string(charge_tensor[1])}
    3 {vec_to_string(charge_tensor[2])}"""
        return result

    @_base.data_access
    @_documentation.add(_read_doc)
    def read(self):
        return self.to_dict()

    @_base.data_access
    @_documentation.add(_read_doc)
    def to_dict(self):
        return {
            "structure": self._structure.read(),
            "charge_tensors": self._raw_data.charge_tensors[:],
        }

    @property
    def _structure(self):
        return Structure(self._raw_data.structure)
