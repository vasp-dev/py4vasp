# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._util import convert
from py4vasp.calculation import _base


class System(_base.Refinery):
    "The :tag:`SYSTEM` tag in the INCAR file is a title you choose for a VASP calculation."

    @_base.data_access
    def __str__(self):
        return convert.text_to_string(self._raw_data.system)

    @_base.data_access
    def to_dict(self):
        "Returns a dictionary containing the system tag."
        return {"system": str(self)}
