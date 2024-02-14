# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class View:
    elements: array
    lattice_vectors: array
    positions: array
    supercell: array = (1, 1, 1)
    "Defines how many multiple of the cell are drawn along each of the coordinate axis."

    def to_ngl(self):
        pass
