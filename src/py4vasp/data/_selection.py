from typing import NamedTuple, Iterable


class Selection(NamedTuple):
    "Helper class specifying which indices to extract their label."
    indices: Iterable[int]
    "Indices from which the specified quantity is read."
    label: str = ""
    "Label identifying the quantity."
