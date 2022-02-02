# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from typing import NamedTuple, Iterable


class Selection(NamedTuple):
    "Helper class specifying which indices to extract their label."
    indices: Iterable[int]
    "Indices from which the specified quantity is read."
    label: str = ""
    "Label identifying the quantity."
