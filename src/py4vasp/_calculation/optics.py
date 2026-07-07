# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Optical properties (transmission, absorption, reflectivity, color) derived from
the dielectric function that VASP computes."""

from py4vasp import raw
from py4vasp._calculation.dispatch import DataSource, quantity
from py4vasp._third_party import graph

# The optics quantity owns no raw data of its own; it derives every quantity from the
# dielectric function. Dispatch therefore accesses the "dielectric_function" schema
# entry, so a selection picks both the dielectric function source and the direction.
_DATA_QUANTITY = "dielectric_function"


class OpticsHandler:
    """Transforms a single raw.DielectricFunction into optical properties."""

    def __init__(self, raw_dielectric_function: raw.DielectricFunction):
        self._raw_dielectric_function = raw_dielectric_function

    @classmethod
    def from_data(cls, raw_dielectric_function: raw.DielectricFunction) -> "OpticsHandler":
        return cls(raw_dielectric_function)


@quantity("optics")
class Optics(graph.Mixin):
    """Optical properties of a material derived from its dielectric function.

    From the complex dielectric function VASP computes, this quantity derives the
    transmission, absorption, and reflectivity spectra as well as the perceived RGB
    color of the material. Pass a ``selection`` to any method to choose which
    dielectric function (e.g. ``bse``) and which direction (e.g. ``xx``, ``isotropic``)
    you are interested in.
    """

    def __init__(self, source, quantity_name: str = "optics"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_dielectric_function: raw.DielectricFunction) -> "Optics":
        """Create an Optics dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_dielectric_function))

    @property
    def path(self):
        """Returns the path from which the output is obtained."""
        return self._path

    def _handler_factory(self, raw_data):
        return OpticsHandler.from_data(raw_data)

    def to_graph(self, selection: str | None = None) -> graph.Graph:
        raise NotImplementedError

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
