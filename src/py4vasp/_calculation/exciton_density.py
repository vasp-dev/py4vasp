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
    def from_data(
        cls, raw_exciton_density: raw.ExcitonDensity
    ) -> "ExcitonDensityHandler":
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

    def to_dict(self) -> dict:
        _raise_error_if_no_data(self._raw_exciton_density.exciton_charge)
        return {
            "structure": self._structure().to_dict(),
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
    """This class accesses exciton charge densities of VASP.

    The exciton charge densities can be calculated via the BSE/TDHF algorithm in
    VASP. With this class you can extract these charge densities.

    Examples
    --------
    First, we create some example data so that you can follow along. Please define a
    variable `path` with the path to a directory that exists and does not contain any
    VASP calculation data. Alternatively, you can use your own data if you have run
    VASP and construct `calculation` from it.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    For your own postprocessing, you can read the exciton density data into a Python
    dictionary:

    >>> calculation.exciton.density.read()
    {'structure': {...}, 'charge': array([[[[...]]]]...)}

    Alternatively, obtain the density as a numpy array directly:

    >>> calculation.exciton.density.to_numpy()
    array([[[[...]]]]...)

    You can also visualize a 3d isosurface of the density:

    >>> calculation.exciton.density.plot()
    ...
    View(elements=array([[...]]...), lattice_vectors=array([[[...]]]...), positions=array([[[...]]]...), grid_scalars=[GridQuantity(quantity=array([[[[...]]]]...), label='1', isosurfaces=[Isosurface(...)])], ...)

    Finally, you can inspect possible selections with:

    >>> calculation.exciton.density.selections()
    {'exciton_density': ['default'...]...}

    Please check the documentation of these methods for more details on how to use
    them and which options they provide.
    """

    def __init__(self, source, quantity_name: str = "exciton_density"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_exciton_density: raw.ExcitonDensity) -> "ExcitonDensity":
        return cls(source=DataSource(raw_exciton_density))

    def _handler_factory(self, raw):
        return ExcitonDensityHandler.from_data(raw)

    def __str__(self, selection=None) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ExcitonDensityHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self, selection=None) -> dict:
        """Read the exciton density into a dictionary.

        Returns
        -------
        dict
            Contains the supercell structure information as well as the exciton
            charge density represented on a grid in the supercell.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ExcitonDensityHandler.to_dict,
        )

    def to_dict(self, selection=None) -> dict:
        """Convenient alias for :py:meth:`read`."""
        return self.read(selection=selection)

    def to_numpy(self, selection=None) -> np.ndarray:
        """Convert the exciton charge density to a numpy array.

        Returns
        -------
        np.ndarray
            Charge density of all excitons.
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ExcitonDensityHandler.to_numpy,
        )

    def to_view(
        self,
        selection: str | None = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
        center: bool = False,
        **user_options,
    ) -> view.View:
        """Plot the selected exciton density as a 3d isosurface within the structure.

        Parameters
        ----------
        selection : str | None = None
            Can be exciton index or a combination, i.e., "1" or "1+2+3"

        supercell : int | np.ndarray | None = None
            If present the data is replicated the specified number of times along each
            direction.

        center : bool = False
            Shift the origin of the unit cell to the center. This is helpful if
            the exciton is at the corner of the cell.

        user_options
            Further arguments with keyword that get directly passed on to the
            visualizer. Most importantly, you can set isolevel to adjust the
            value at which the isosurface is drawn.

        Returns
        -------
        View
            Visualize an isosurface of the exciton density within the 3d structure.

        Examples
        --------
        >>> calculation = py4vasp.Calculation.from_path(".")
        Plot an isosurface of the first exciton charge density
        >>> calculation.exciton.density.plot()
        Plot an isosurface of the third exciton charge density
        >>> calculation.exciton.density.plot("3")
        Plot an isosurface of the sum of first and second exciton charge densities
        >>> calculation.exciton.density.plot("1+2")
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ExcitonDensityHandler.to_view,
            supercell=supercell,
            center=center,
            **user_options,
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
