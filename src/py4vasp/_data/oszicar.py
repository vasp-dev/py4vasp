# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np

from py4vasp import data, raw
from py4vasp._data import base
from py4vasp._util import convert


class OSZICAR(base.Refinery):
    """Access the convergence data for each electronic step."""

    @base.data_access
    def to_dict(self):
        return {
            "convergence_data": self._read("convergence_data"),
        }

    def _read(self, key):
        data = getattr(self._raw_data, key)
        return {key: data[:]} if not data.is_none() else {}
