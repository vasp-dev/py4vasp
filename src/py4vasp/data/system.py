import py4vasp.data._base as _base
import py4vasp._util.convert as _convert


class System(_base.DataBase):
    "Extract the system tag from the INCAR file."
    __str__ = _base.RefinementDescriptor("_to_string")

    def _to_string(self):
        return _convert.text_to_string(self._raw_data.system)
