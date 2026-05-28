# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from typing import Optional, Union

import numpy as np

from py4vasp import _config, exception, raw
from py4vasp._calculation.dispatch import (
    DataSource,
    merge_default,
    merge_strings,
    quantity,
)
from py4vasp._calculation._stoichiometry import StoichiometryHandler
from py4vasp._calculation.structure import StructureHandler
from py4vasp._third_party import view
from py4vasp._util import index, select

_DEFAULT_SELECTION = "1"


class ExcitonDensityHandler:
    """Handler for exciton charge density data."""

    def __init__(self, raw_exciton_density: raw.ExcitonDensity):
        self._raw_exciton_density = raw_exciton_density

    @classmethod
    def from_data(cls, raw_exciton_density: raw.ExcitonDensity) -> "ExcitonDensityHandler":
        return cls(raw_exciton_density)

    def __str__(self) -> str:
        _raise_error_if_no_data(self._raw_exciton_density.exciton_charge)
        grid = self._raw_exciton_density.exciton_charge.shape[1:]
        stoichiometry = StoichiometryHandler.from_data(
            self._raw_exciton_density.structure.stoichiometry
        )
        return f"""exciton charge density:
    structure: {str(stoichiometry)}
    grid: {grid[2]}, {grid[1]}, {grid[0]}
    excitons: {len(self._raw_exciton_density.exciton_charge)}"""

    def read(self) -> dict:
        return self.to_dict()

    def to_dict(self) -> dict:
        _raise_error_if_no_data(self._raw_exciton_density.exciton_charge)
        return {
            "structure": self._structure().read(),
            "charge": self.to_numpy(),
        }

    def to_numpy(self) -> np.ndarray:
        return np.moveaxis(self._raw_exciton_density.exciton_charge, 0, -1).T

    def to_view(
        self,
        selection: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
        center: bool = False,
        **user_options,
    ) -> view.View:
        _raise_error_if_no_data(self._raw_exciton_density.exciton_charge)
        map_ = self._create_map()
        selector = index.Selector({0: map_}, self._raw_exciton_density.exciton_charge)
        selection = selection or _DEFAULT_SELECTION
        tree = select.Tree.from_selection(selection)
        viewer = self._structure().to_view(supercell)
        viewer.grid_scalars = [
            view.GridQuantity(selector[sel].T[np.newaxis], label=selector.label(sel))
            for sel in tree.selections()
        ]
        if center:
            viewer.shift = np.array([0.5, 0.5, 0.5])
        for scalar in viewer.grid_scalars:
            scalar.isosurfaces = self._isosurfaces(**user_options)
        return viewer

    def _structure(self) -> StructureHandler:
        return StructureHandler.from_data(self._raw_exciton_density.structure)

    def _create_map(self) -> dict:
        num_excitons = self._raw_exciton_density.exciton_charge.shape[0]
        return {str(choice + 1): choice for choice in range(num_excitons)}

    def _isosurfaces(self, isolevel=0.8, color=None, opacity=0.6):
        color = color or _config.VASP_COLORS["cyan"]
        return [view.Isosurface(isolevel, color, opacity)]


@quantity("density", group="exciton")
class ExcitonDensity(view.Mixin):
    """This class accesses exciton charge densities of VASP."""

    def __init__(self, source, quantity_name: str = "exciton_density"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_exciton_density: raw.ExcitonDensity) -> "ExcitonDensity":
        return cls(source=DataSource(raw_exciton_density))

    def _handler_factory(self, raw):
        return ExcitonDensityHandler.from_data(raw)

    def __str__(self) -> str:
        return merge_strings(
            self._source, self._quantity_name, None,
            self._handler_factory, ExcitonDensityHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection: str | None = None) -> dict:
        return merge_default(
            self._source, self._quantity_name, selection,
            self._handler_factory, ExcitonDensityHandler.read,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Read the exciton density into a dictionary."""
        return self.read(selection=selection)

    def to_numpy(self, selection: str | None = None) -> np.ndarray:
        """Convert the exciton charge density to a numpy array."""
        return merge_default(
            self._source, self._quantity_name, selection,
            self._handler_factory, ExcitonDensityHandler.to_numpy,
        )

    def to_view(
        self,
        selection: str | None = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
        center: bool = False,
        **user_options,
    ) -> view.View:
        """Plot the selected exciton density as a 3d isosurface within the structure."""
        return merge_default(
            self._source, self._quantity_name, None,
            self._handler_factory, ExcitonDensityHandler.to_view,
            selection, supercell=supercell, center=center, **user_options,
        )


def _raise_error_if_no_data(data):
    if data.is_none():
        raise exception.NoData(
            "Exciton charge density was not found. Note that in order to calculate the"
            "exciton charge density the number of eigenvectors has to be selected with"
            "the tag NBSEEIG and the position of the hole or the electron has to be"
            "provided with the tag BSEHOLE or BSEELECTRON, correspondingly. The exciton"
            "density is written to vaspout.h5 if the tags LCHARGH5=T or LH5=T are set"
            "in the INCAR file, otherwise the charge density is written to CHG.XXX files."
        )
