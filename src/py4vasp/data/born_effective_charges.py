import py4vasp.data._base as _base
from py4vasp.data import Structure


class BornEffectiveCharges(_base.DataBase):
    read = _base.RefinementDescriptor("_to_dict")
    to_dict = _base.RefinementDescriptor("_to_dict")
    __str__ = _base.RefinementDescriptor("_to_string")

    def _to_string(self):
        data = self._to_dict()
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

    def _to_dict(self):
        return {
            "structure": self._structure.read(),
            "charge_tensors": self._raw_data.charge_tensors[:],
        }

    @property
    def _structure(self):
        return Structure(self._raw_data.structure)
