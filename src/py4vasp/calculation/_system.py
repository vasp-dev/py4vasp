# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._data import base
from py4vasp._util import convert


class System(base.Refinery):
    "Extract the system tag from the INCAR file."

    @base.data_access
    def __str__(self):
        return convert.text_to_string(self._raw_data.system)

    @base.data_access
    def to_dict(self):
        "Returns a dictionary containing the system tag."
        return {"system": str(self)}
