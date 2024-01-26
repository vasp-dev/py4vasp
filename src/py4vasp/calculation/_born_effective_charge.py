# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.calculation import _base, _structure


class BornEffectiveCharge(_base.Refinery, _structure.Mixin):
    """The Born effective charge tensors couple electric field and atomic displacement.

    You can use this class to extract the Born effective charges of a linear
    response calculation. The Born effective charges describes the effective charge of
    an ion in a crystal lattice when subjected to an external electric field.
    These charges account for the displacement of the ion positions in response to the
    field, reflecting the distortion of the crystal structure. Born effective charges
    help understanding the material's response to external stimuli, such as
    piezoelectric and ferroelectric behavior.
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
    def to_dict(self):
        """Read structure information and Born effective charges into a dictionary.

        The structural information is added to inform about which atoms are included
        in the array. The Born effective charges array contains the mixed second
        derivative with respect to an electric field and an atomic displacement for
        all atoms and possible directions.

        Returns
        -------
        dict
            Contains structural information as well as the Born effective charges.
        """
        return {
            "structure": self._structure.read(),
            "charge_tensors": self._raw_data.charge_tensors[:],
        }
