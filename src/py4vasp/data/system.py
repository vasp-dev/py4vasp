# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import py4vasp.data._base as _base
import py4vasp._util.convert as _convert


class System(_base.Refinery):
    "Extract the system tag from the INCAR file."

    @_base.data_access
    def __str__(self):
        return _convert.text_to_string(self._raw_data.system)

    @_base.data_access
    def to_dict(self):
        "Returns a dictionary containing the system tag."
        return {"system": str(self)}
