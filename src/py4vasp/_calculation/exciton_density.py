# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from typing import Optional, Union

import numpy as np

from py4vasp import _config, exception
from py4vasp._calculation import _stoichiometry, base, structure
from py4vasp._raw import data as raw_data
from py4vasp._third_party import view
from py4vasp._util import import_, index, select
from py4vasp._util.density import Visualizer

pretty = import_.optional("IPython.lib.pretty")


_DEFAULT_SELECTION = "1"


class ExcitonDensity(base.Refinery, structure.Mixin, view.Mixin):
    """This class accesses exciton charge densities of VASP.

    The exciton charge densities can be calculated via the BSE/TDHF algorithm in
    VASP. With this class you can extract these charge densities.

    Examples
    --------

    First, we create some example data do that you can follow along. Please define a
    variable `path` with the path to a directory that exists and does not contain any
    VASP calculation data. Alternatively, you can use your own data if you have run
    VASP and construct `calculation` from it.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    For your own postprocessing, you can read the band data into a Python dictionary:

    >>> calculation.exciton.density.read()
    {'structure': {...}, 'charge': array([[[[...]]]]...)}

    Alternatively, obtain the density as a numpy array directly:

    >>> calculation.exciton.density.to_numpy()
    array([[[[...]]]]...)

    You can also visualize a 3d isosurface of the density:

    >>> calculation.exciton.density.plot()
    View(elements=array([[...]]...), lattice_vectors=array([[[...]]]...), positions=array([[[...]]]...), grid_scalars=[GridQuantity(quantity=array([[[[...]]]]...), label='1', isosurfaces=[Isosurface(...)])], ...)


    Finally, you can inspect possible selections with:

    >>> calculation.exciton.density.selections()
    {'exciton_density': ['default'...]...}

    Please check the documentation of these methods for more details on how to use them and which options they provide.
    """

    _raw_data: raw_data.ExcitonDensity

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
    def to_view(
        self,
        selection: Optional[str] = None,
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
        >>> calculation = py4vasp.Calculation.from_path(".")
        Plot an isosurface of the first exciton charge density
        >>> calculation.exciton.density.plot()
        Plot an isosurface of the third exciton charge density
        >>> calculation.exciton.density.plot("3")
        Plot an isosurface of the sum of first and second exciton charge
        densities
        >>> calculation.exciton.density.plot("1+2")
        """
        _raise_error_if_no_data(self._raw_data.exciton_charge)
        viewer = self._structure.plot(supercell)
        # build selector
        map_ = self._create_map()
        selector = index.Selector({0: map_}, self._raw_data.exciton_charge)

        # define selections
        selection = selection or _DEFAULT_SELECTION
        tree = select.Tree.from_selection(selection)

        # set up Visualizer
        visualizer = Visualizer(self._structure)
        dataDict = {selector.label(sel): (selector[sel].T) for sel in tree.selections()}
        viewer = visualizer.to_view(dataDict, supercell=supercell)

        # adjust viewer
        if center:
            viewer.shift = np.array([0.5, 0.5, 0.5])
        for scalar in viewer.grid_scalars:
            scalar.isosurfaces = self._isosurfaces(**user_options)
        return viewer

    def _create_map(self):
        num_excitons = self._raw_data.exciton_charge.shape[0]
        return {str(choice + 1): choice for choice in range(num_excitons)}

    def _isosurfaces(self, isolevel=0.8, color=None, opacity=0.6):
        color = color or _config.VASP_COLORS["cyan"]
        return [view.Isosurface(isolevel, color, opacity)]


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
