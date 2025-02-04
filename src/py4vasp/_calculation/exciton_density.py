# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _config, exception
from py4vasp._calculation import _stoichiometry, base, structure
from py4vasp._third_party import view
from py4vasp._util import import_, index, select

pretty = import_.optional("IPython.lib.pretty")


_DEFAULT_SELECTION = "1"


class ExcitonDensity(base.Refinery, structure.Mixin, view.Mixin):
    """This class accesses exciton charge densities of VASP.

    The exciton charge densities can be calculated via the BSE/TDHF algorithm in
    VASP. With this class you can extract these charge densities.
    """

    @base.data_access
    def __str__(self):
        _raise_error_if_no_data(self._raw_data.exciton_charge)
        grid = self._raw_data.exciton_charge.shape[1:]
        raw_stoichiometry = self._raw_data.structure.stoichiometry
        stoichiometry = _stoichiometry.Stoichiometry.from_data(raw_stoichiometry)
        return f"""exciton charge density:
    structure: {pretty.pretty(stoichiometry)}
    grid: {grid[2]}, {grid[1]}, {grid[0]}
    excitons: {len(self._raw_data.exciton_charge)}"""

    @base.data_access
    def to_dict(self):
        """Read the exciton density into a dictionary.

        Returns
        -------
        dict
            Contains the supercell structure information as well as the exciton
            charge density represented on a grid in the supercell.
        """
        _raise_error_if_no_data(self._raw_data.exciton_charge)
        result = {"structure": self._structure.read()}
        result.update({"charge": self.to_numpy()})
        return result

    @base.data_access
    def to_numpy(self):
        """Convert the exciton charge density to a numpy array.

        Returns
        -------
        np.ndarray
            Charge density of all excitons.
        """
        return np.moveaxis(self._raw_data.exciton_charge, 0, -1).T

    @base.data_access
    def to_view(self, selection=None, supercell=None, center=False, **user_options):
        """Plot the selected exciton density as a 3d isosurface within the structure.

        Parameters
        ----------
        selection : str
            Can be exciton index or a combination, i.e., "1" or "1+2+3"

        supercell : int or np.ndarray
            If present the data is replicated the specified number of times along each
            direction.

        center : bool
            Shift the origin of the unit cell to the center. This is helpful if you
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
        >>> calc = py4vasp.Calculation.from_path(".")
        Plot an isosurface of the first exciton charge density
        >>> calc.exciton.density.plot()
        Plot an isosurface of the third exciton charge density
        >>> calc.exciton.density.plot("3")
        Plot an isosurface of the sum of first and second exciton charge
        densities
        >>> calc.exciton.density.plot("1+2")
        """
        _raise_error_if_no_data(self._raw_data.exciton_charge)
        selection = selection or _DEFAULT_SELECTION
        viewer = self._structure.plot(supercell)
        map_ = self._create_map()
        selector = index.Selector({0: map_}, self._raw_data.exciton_charge)
        tree = select.Tree.from_selection(selection)
        viewer.grid_scalars = [
            self._grid_quantity(selector, selection, map_, user_options)
            for selection in tree.selections()
        ]
        if center:
            viewer.shift = (0.5, 0.5, 0.5)
        return viewer

    def _create_map(self):
        num_excitons = self._raw_data.exciton_charge.shape[0]
        return {str(choice + 1): choice for choice in range(num_excitons)}

    def _grid_quantity(self, selector, selection, map_, user_options):
        return view.GridQuantity(
            quantity=(selector[selection].T)[np.newaxis],
            label=selector.label(selection),
            isosurfaces=self._isosurfaces(**user_options),
        )

    def _isosurfaces(self, isolevel=0.8, color=None, opacity=0.6):
        color = color or _config.VASP_COLORS["cyan"]
        return [view.Isosurface(isolevel, color, opacity)]


def _raise_error_if_no_data(data):
    if data.is_none():
        raise exception.NoData(
            "Exciton charge density was not found. Note that the exciton density is"
            "written to vaspout.h5 if the tags LCHARGH5=T or LH5=T are set in"
            "the INCAR file"
        )
