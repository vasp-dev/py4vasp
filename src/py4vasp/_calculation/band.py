# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np
from numpy.typing import ArrayLike

from py4vasp import exception
from py4vasp._calculation import _dispersion, base, kpoint, projector
from py4vasp._third_party import graph
from py4vasp._util import check, documentation, import_, index, select, slicing

pd = import_.optional("pandas")
pretty = import_.optional("IPython.lib.pretty")


class Band(base.Refinery, graph.Mixin):
    """The band structure contains the **k** point resolved eigenvalues.

    The most common use case of this class is to produce the electronic band
    structure along a path in the Brillouin zone used in a non self consistent
    VASP calculation. In some cases you may want to use the `to_dict` function
    just to obtain the eigenvalue and projection data though in that case the
    **k**-point distances that are calculated are meaningless.

    Examples
    --------

    First, we create some example data do that you can follow along. Please define a
    variable `path` with the path to a directory that exists and does not contain any
    VASP calculation data. Alternatively, you can use your own data if you have run
    VASP and construct `calculation` from it.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    To produce band structure plot use, please check the `to_graph` function for
    a more detailed documentation.

    >>> calculation.band.plot()
    Graph(series=[Series(x=array(...), y=array(...), label='bands', ...)],
        ..., xticks={...}, ..., ylabel='Energy (eV)', ...)

    For your own postprocessing, you can read the band data into a Python dictionary

    >>> calculation.band.read()
    {'kpoint_distances': array(...), 'fermi_energy': ..., 'bands': array(...),
        'occupations': array(...)}

    These methods take additional selections, if you used VASP with :tag:`LORBIT`.
    You can inspect possible choices with

    >>> calculation.band.selections()
    {'band': ['default', 'kpoints_opt', 'kpoints_wan'],
        'atom': [...], 'orbital': [...], 'spin': [...]}
    """

    @base.data_access
    def __str__(self) -> str:
        return f"""
{"spin polarized" if self._is_collinear() else ""} band data:
    {self._raw_data.dispersion.eigenvalues.shape[1]} k-points
    {self._raw_data.dispersion.eigenvalues.shape[2]} bands
{pretty.pretty(self._projector())}
    """.strip()

    @base.data_access
    @documentation.format(selection_doc=projector.selection_doc)
    def to_dict(
        self, selection: Optional[str] = None, fermi_energy: Optional[float] = None
    ) -> dict[str, Any]:
        """Read the data into a dictionary.

        You may use this data for your own postprocessing tools. Sometimes you may
        want to choose different representations of the electronic band structure or
        you want to use the electronic eigenvalues and occupations to compute integrals
        over the Brillouin zone.

        We create some example data do that you can follow along. Please define a
        variable `path` with the path to a directory that exists and does not contain any
        VASP calculation data. Alternatively, you can use your own data if you have run
        VASP and construct `calculation` from it.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> collinear_calculation = demo.calculation(path, selection="collinear")
        >>> noncollinear_calculation = demo.calculation(path, selection="noncollinear")

        Parameters
        ----------
        {selection_doc}
        fermi_energy : float
            Overwrite the Fermi energy of the band structure calculation with a more
            accurate one from a different calculation. This is recommended for metallic
            systems where the Fermi energy may be significantly different.

        Returns
        -------
        dict
            Contains the **k**-point path for plotting band structures with the
            eigenvalues shifted to bring the Fermi energy to 0. If available
            and a selection is passed, the projections of these bands on the
            selected projectors are included. If you specified '''k'''-point labels
            in the KPOINTS file, these are returned as well.

        Examples
        --------
        Return the **k** points, the electronic eigenvalues, and the Fermi energy as
        a Python dictionary

        >>> calculation.band.to_dict()
        {{'kpoint_distances': array(...), 'fermi_energy': ..., 'bands': array(...),
            'occupations': array(...)}}

        Select the p orbitals of the first atom in the POSCAR file:

        >>> calculation.band.to_dict(selection="1(p)")
        {{'kpoint_distances': array(...), 'fermi_energy': ..., 'bands': array(...),
            'occupations': array(...), 'Sr_1_p': array(...)}}

        Select the d orbitals of Sr and Ti:

        >>> calculation.band.to_dict("d(Sr, Ti)")
        {{'kpoint_distances': array(...), 'fermi_energy': ..., 'bands': array(...),
            'occupations': array(...), 'Sr_d': array(...), 'Ti_d': array(...)}}

        For collinear calculations, the spin channels are treated separately

        >>> collinear_calculation.band.to_dict()
        {{'kpoint_distances': array(...), 'fermi_energy': ..., 'bands_up': array(...),
            'bands_down': array(...), 'occupations_up': array(...),
            'occupations_down': array(...)}}

        You can also select particular spin channels, for example the spin-up contribution
        of the first three atoms combined

        >>> collinear_calculation.band.to_dict("up(1:3)")
        {{'kpoint_distances': array(...), 'fermi_energy': ..., 'bands_up': array(...),
            'bands_down': array(...), 'occupations_up': array(...),
            'occupations_down': array(...), '1:3_up': array(...)}}

        For noncollinear calculations, the resulting dictionary has the same structure
        as for the nonpolarized case
        >>> noncollinear_calculation.band.to_dict()
        {{'kpoint_distances': array(...), 'fermi_energy': ..., 'bands': array(...),
            'occupations': array(...)}}

        If you want to investigate the spin projection of the bands, you can select
        particular spin components. Here, we select the x and z components of the spin

        >>> noncollinear_calculation.band.to_dict("sigma_x, sigma_z")
        {{'kpoint_distances': array(...), 'fermi_energy': ..., 'bands': array(...),
            'occupations': array(...), 'sigma_x': array(...), 'sigma_z': array(...),
            'is_spin_projection': ['sigma_x', 'sigma_z']}}

        Add the contribution of three d orbitals

        >>> calculation.band.to_dict("dxy + dxz + dyz")
        {{'kpoint_distances': array(...), 'fermi_energy': ..., 'bands': array(...),
            'occupations': array(...), 'dxy + dxz + dyz': array(...)}}

        Read the density of states generated by the '''k'''-point mesh in the KPOINTS_OPT
        file

        >>> calculation.band.to_dict("kpoints_opt")
        {{'kpoint_distances': array(...), 'kpoint_labels': ..., 'fermi_energy': ...,
            'bands': array(...), 'occupations': array(...)}}
        """
        dispersion = self._dispersion().read()
        eigenvalues = dispersion.pop("eigenvalues")
        return {
            **dispersion,
            "fermi_energy": self._raw_data.fermi_energy,
            **self._shift_dispersion_by_fermi_energy(eigenvalues, fermi_energy),
            **self._read_occupations(),
            **self._read_projections(selection),
        }

    @base.data_access
    @documentation.format(selection_doc=projector.selection_doc)
    def to_graph(
        self,
        selection: Optional[str] = None,
        fermi_energy: Optional[float] = None,
        width: float = 0.5,
    ) -> graph.Graph:
        """Read the data and generate a graph.

        On the x axis, we show the **k** points as distances from the previous ones.
        This representation makes sense, if you selected a line mode in the KPOINTS
        file. When you provide labels for the **k** points those will be added in the
        plot. We show all bands included in the calculation :tag:`NBANDS`.

        If you used the code with :tag:`LORBIT`, you can also plot the projected band
        structure. Here, each band will have a linewidth proportional to the projection
        of the band on reference orbitals. The maximum width is adjustable with an
        argument.

        We create some example data do that you can follow along. Please define a
        variable `path` with the path to a directory that exists and does not contain any
        VASP calculation data. Alternatively, you can use your own data if you have run
        VASP and construct `calculation` from it.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> collinear_calculation = demo.calculation(path, selection="collinear")
        >>> noncollinear_calculation = demo.calculation(path, selection="noncollinear")

        Parameters
        ----------
        {selection_doc}
        fermi_energy : float
            Overwrite the Fermi energy of the band structure calculation with a more
            accurate one from a different calculation. This is recommended for metallic
            systems where the Fermi energy may be significantly different.
        width : float
            Specifies the width (in eV) of the fatbands if a selection of projections is
            specified. If the projection amounts to 100%, the line will be drawn with
            this width.

        Returns
        -------
        Graph
            Figure containing the spin-up and spin-down bands. If a selection
            is provided the width of the bands represents the projections of the
            bands onto the specified projectors.

        Examples
        --------
        Plot the band structure with possible **k** point labels if they have been
        provided in the KPOINTS file

        >>> calculation.band.to_graph()
        Graph(series=[Series(x=array(...), y=array(...), label='bands', ...)],
            ..., xticks={{...}}, ..., ylabel='Energy (eV)', ...)

        Select the p orbitals of the first atom in the POSCAR file:

        >>> calculation.band.to_graph(selection="1(p)")
        Graph(series=[Series(..., label='Sr_1_p', weight=array(...), ...)], ...)

        Select the d orbitals of Sr and Ti:

        >>> calculation.band.to_graph("d(Sr, Ti)")
        Graph(series=[Series(..., label='Sr_d', ...), Series(..., label='Ti_d', ...)], ...)

        For collinear calculations, the spin channels are treated separately

        >>> collinear_calculation.band.to_graph()
        Graph(series=[Series(..., label='up', ...), Series(..., label='down', ...)], ...)

        You can also select particular spin channels, for example the spin-up contribution
        of the first three atoms combined

        >>> collinear_calculation.band.to_graph("up(1:3)")
        Graph(series=[Series(..., label='1:3_up', ...)], ...)

        For noncollinear calculations, the resulting dictionary has the same structure
        as for the nonpolarized case
        >>> noncollinear_calculation.band.to_graph()
        Graph(series=[Series(..., label='bands', ...)], ...)

        If you want to investigate the spin projection of the bands, you can select
        particular spin components. Here, we select the x component of the spin

        >>> noncollinear_calculation.band.to_graph("sigma_x")
        Graph(series=[Series(..., label='sigma_x', ...)], ...)

        Add the contribution of three d orbitals

        >>> calculation.band.to_graph("dxy + dxz + dyz")
        Graph(series=[Series(..., label='dxy + dxz + dyz', ...)], ...)

        Read the density of states generated by the '''k'''-point mesh in the KPOINTS_OPT
        file

        >>> calculation.band.to_graph("kpoints_opt")
        Graph(series=[Series(..., label='bands', ...)], ...)

        If you use projections, you can also adjust the width of the lines. Passing
        the argument `width=1.0` increases the maximum linewidth to 1 eV

        >>> calculation.band.to_graph("d", width=1.0)
        Graph(series=[Series(..., label='d', weight=array(...), ...)], ...)
        """
        projections = self._projections(selection, width)
        graph = self._dispersion().plot(projections)
        graph = self._shift_series_by_fermi_energy(graph, fermi_energy)
        graph.ylabel = "Energy (eV)"
        return graph

    @base.data_access
    @documentation.format(selection_doc=projector.selection_doc)
    def to_frame(
        self, selection: Optional[str] = None, fermi_energy: Optional[float] = None
    ) -> pd.DataFrame:
        """Read the data into a DataFrame.

        We create some example data do that you can follow along. Please define a
        variable `path` with the path to a directory that exists and does not contain any
        VASP calculation data. Alternatively, you can use your own data if you have run
        VASP and construct `calculation` from it.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> collinear_calculation = demo.calculation(path, selection="collinear")
        >>> noncollinear_calculation = demo.calculation(path, selection="noncollinear")

        Parameters
        ----------
        {selection_doc}
        fermi_energy : float
            Overwrite the Fermi energy of the band structure calculation with a more
            accurate one from a different calculation. This is recommended for metallic
            systems where the Fermi energy may be significantly different.

        Returns
        -------
        pd.DataFrame
            Contains the eigenvalues and corresponding occupations for all k-points and
            bands. If a selection string is given, in addition the orbital projections
            on these bands are returned.

        Examples
        --------
        Get the band structure of all bands without projections

        >>> calculation.band.to_frame()
           kpoint_distances  bands  occupations
        0  ...

        Select the p orbitals of the first atom in the POSCAR file:

        >>> calculation.band.to_frame(selection="1(p)")
           kpoint_distances  bands  occupations  Sr_1_p
        0  ...

        Select the d orbitals of Sr and Ti:

        >>> calculation.band.to_frame("d(Sr, Ti)")
           kpoint_distances  bands  occupations  Sr_d  Ti_d
        0  ...

        For collinear calculations, the spin channels are treated separately

        >>> collinear_calculation.band.to_frame()
           kpoint_distances  bands_up  bands_down  occupations_up  occupations_down
        0  ...

        You can also select particular spin channels, for example the spin-up contribution
        of the first three atoms combined

        >>> collinear_calculation.band.to_frame("up(1:3)")
           kpoint_distances  bands_up  ...  occupations_down    1:3_up
        0  ...

        For noncollinear calculations, the resulting dictionary has the same structure
        as for the nonpolarized case
        >>> noncollinear_calculation.band.to_frame()
           kpoint_distances  bands  occupations
        0  ...

        If you want to investigate the spin projection of the bands, you can select
        particular spin components. Here, we select the x and z components of the spin

        >>> noncollinear_calculation.band.to_frame("sigma_x, sigma_z")
           kpoint_distances  bands  occupations  sigma_x  sigma_z
        0  ...

        Add the contribution of three d orbitals

        >>> calculation.band.to_frame("dxy + dxz + dyz")
           kpoint_distances  bands  occupations  dxy + dxz + dyz
        0  ...
        """
        return pd.DataFrame(self._extract_relevant_data(selection, fermi_energy))

    @base.data_access
    def to_quiver(
        self,
        selection: str,
        normal: Optional[str] = None,
        supercell: Optional[ArrayLike] = None,
    ) -> graph.Graph:
        """Generate a quiver plot of spin texture.

        Note that plotting the spin texture requires a special setup of the VASP calculation.
        You need to run a noncollinear calculation and a suitable **k**-point mesh. The
        **k**-point mesh must be a regular two-dimensional grid, i.e. one of the three
        reciprocal lattice directions must not be sampled (1 in that direction). py4vasp
        will check this condition and raise an error if it is not fulfilled.

        The spin texture is represented by arrows in a plane in reciprocal space. The
        plane is defined by the two reciprocal lattice vectors that are sampled by the
        **k**-point mesh. The normal of this plane can be rotated to align with a
        Cartesian axis if desired. The arrows represent the spin expectation value
        at each **k** point. You can select which components of the spin are shown
        and which bands are included. You can also select particular atoms and orbitals.

        Parameters
        ----------
        selection
            A string specifying the components of the spin and the bands to be included
            in the plot. This must be provided and py4vasp will raise an error if it is
            not in the correct format. The selection string has the following structure:

                <spin_component>~<spin_component>(band=<band_index>)

            where <spin_component> is one of "sigma_x", "sigma_y", or "sigma_z", and
            <band_index> is the index of the band to be included (1-based). For the
            spin component, you can also use the alias "x", "y", or "z" and "sigma_1",
            "sigma_2", or "sigma_3".

            In addition, you can project onto particular atoms and orbitals similar to
            the other methods in this class:

            -   To specify the **atom**, you can either use its element name (Si, Al, ...)
                or its index as given in the input file (1, 2, ...). For the latter
                option it is also possible to specify ranges (e.g. 1:4).
            -   To select a particular **orbital** you can give a string (s, px, dxz, ...)
                or select multiple orbitals by their angular momentum (s, p, d, f).
            -   If you used a different **k**-point mesh choose "kpoints_opt" or "kpoints_wan"
                to select them instead of the default mesh specified in the KPOINTS file.

            You separate multiple selections by commas or whitespace and can nest them using
            parenthesis, e.g. `Sr(s, p)` or `s(up), p(down)`. The order of the selections
            does not matter, but it is case sensitive to distinguish p (angular momentum
            l = 1) from P (phosphorus).

            It is possible to add or subtract different components, e.g., a selection of
            "Ti(d) - O(p)" would project onto the d orbitals of Ti and the p orbitals of O and
            then compute the difference of these two selections.

            If you are unsure about the specific projections that are available, you can use

            >>> calculation.projector.selections()
            {'atom': [...], 'orbital': [...], 'spin': [...]}

            to get a list of all available ones.
        normal
            Set the Cartesian direction "x", "y", or "z" parallel to which the normal of
            the plane is rotated. Alteratively, set it to "auto" to rotate to the closest
            Cartesian axis. If you set it to None, the normal will not be considered and
            the first remaining lattice vector will be aligned with the x axis instead.
        supercell
            Replicate the contour plot periodically a given number of times. If you
            provide two different numbers, the resulting cell will be the two remaining
            lattice vectors multiplied by the specific number.

        Returns
        -------
        A quiver plot for the spin texture in the plane spanned by the 2 remaining
        lattice vectors. The arrows represent the spin expectation value at each **k** point.
        The plot is replicated periodically according to the specified supercell.

        Examples
        --------
        Let us generate some example data do that you can follow along. Please define a
        variable `path` with the path to a directory that does not exist yet. Alternatively,
        you can use your own data if you have run VASP with an appropriate k-point mesh.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path, selection="spin_texture")

        Plot a projection of the spin texture in reciprocal space, summed over all atoms and orbitals, for the first band and the x and y components.

        >>> calculation.band.to_quiver("x~y(band=1)")
        Graph(series=[Contour(data=array([[[...

        >>> calculation.band.to_quiver()
        Graph(series=[Contour(data=array([[[...

        Select the Ba atom, the third band, the x and z spin components, then sum over all orbitals:

        >>> calculation.band.to_quiver("Ba(sigma_1~sigma_3(band[3]))")
        Graph(series=[Contour(data=array([[[...

        Select the Pb atom, s orbital, second band and the x and y spin components:

        >>> calculation.band.to_quiver("Pb(s(band[2](sigma_x~sigma_y)))")
        Graph(series=[Contour(data=array([[[...

        Select the 4th atom in the POSCAR file, d orbitals, the second band and the y and z spin components.
        The plot is shown for a 3x3 supercell:

        >>> calculation.band.to_quiver(selection="4(d(y~z(band[2])))", supercell=3)
        Graph(series=[Contour(data=array([[[...

        Select x & y spin components and the first band (default), sum over atoms and orbitals.
        Plot a 2x4 supercell, and rotate the plane normal to align with the nearest coordinate axis.

        >>> calculation.band.to_quiver(supercell=np.array([2, 4]), normal="auto") # doctest: +SKIP
        Graph(series=[Contour(data=array([[[...

        Select x & y spin components and the first band (default), sum over atoms and orbitals.
        Rotate the plane normal to align with the y coordinate axis.

        >>> calculation.band.to_quiver(normal="y")
        Graph(series=[Contour(data=array([[[...
        """
        reciprocal_lattice_vectors = self._kpoint()._reciprocal_lattice_vectors()
        nkp1, nkp2, cut = self._kmesh()
        # Plane is defined by KPOINTS file
        options = {
            "lattice": slicing.plane(
                reciprocal_lattice_vectors, cut, normal, axis_labels=("b1", "b2", "b3")
            )
        }
        if supercell is not None:
            options["supercell"] = np.ones(2, dtype=np.int_) * supercell
        #
        selector = self._make_selector(self._raw_data.projections)
        tree = select.Tree.from_selection(selection)
        quiver_plots = [
            graph.Contour(
                **self._quiver_plot(selector, selection, nkp1, nkp2), **options
            )
            for selection in tree.selections()
        ]
        return graph.Graph(quiver_plots, title="Spin Texture")

    def _quiver_plot(self, selector, selection, nkp1, nkp2):
        data = selector[selection]
        data = data.reshape(2, nkp1, nkp2)
        return {"data": data, "label": selector.label(selection)}

    def _make_selector(self, projections):
        maps = self._projector().to_dict()
        maps = {
            1: maps["atom"],
            2: maps["orbital"],
            0: self._spin_map(maps["spin"]),
            4: self._band_map(projections.shape[-1]),
        }
        return index.Selector(
            maps, projections, reduction=_ToQuiverReduction, use_number_labels=True
        )

    def _spin_map(self, spin_map):
        if "sigma_x" not in spin_map:
            # Spin Texture only makes sense for non-collinear systems
            raise exception.DataMismatch(
                "System is not noncollinear which is required to visualize spin texture."
            )
        return {
            "sigma_x~sigma_y": slice(1, 3),
            "sigma_x~sigma_z": slice(1, 4, 2),
            "sigma_y~sigma_z": slice(2, 4),
            "x~y": slice(1, 3),
            "x~z": slice(1, 4, 2),
            "y~z": slice(2, 4),
            "sigma_1~sigma_2": slice(1, 3),
            "sigma_1~sigma_3": slice(1, 4, 2),
            "sigma_2~sigma_3": slice(2, 4),
        }

    def _band_map(self, num_bands):
        return {"band": {i + 1: i for i in range(num_bands)}}

    @base.data_access
    def selections(self) -> dict[str, Any]:
        return {**super().selections(), **self._projector().selections()}

    def _scale(self):
        if isinstance(self._raw_data.dispersion.kpoints.cell.scale, np.float64):
            return self._raw_data.dispersion.kpoints.cell.scale
        if not self._raw_data.dispersion.kpoints.cell.scale.is_none():
            return self._raw_data.dispersion.kpoints.cell.scale[()]
        else:
            return 1.0

    def _is_collinear(self):
        return len(self._raw_data.dispersion.eigenvalues) == 2

    def _is_noncollinear(self):
        message = "If there are no projections, we cannot use them to check whether the system is noncollinear."
        assert not check.is_none(self._raw_data.projections), message
        return len(self._raw_data.projections) == 4

    def _dispersion(self):
        return _dispersion.Dispersion.from_data(self._raw_data.dispersion)

    def _projector(self):
        return projector.Projector.from_data(self._raw_data.projectors)

    def _kpoint(self):
        return kpoint.Kpoint.from_data(self._raw_data.dispersion.kpoints)

    def _projections(self, selection, width):
        if selection is None:
            return None
        error_message = "Width of fat band structure must be a number."
        check.raise_error_if_not_number(width, error_message)
        projections = self._read_projections(selection)
        spin_projections = projections.get(projector.SPIN_PROJECTION, [])
        for label, weight in projections.items():
            if label == projector.SPIN_PROJECTION or label in spin_projections:
                # do not scale spin projections
                continue
            weight *= width
        return projections

    def _read_projections(self, selection):
        return self._projector().project(selection, self._raw_data.projections)

    def _read_occupations(self):
        if self._is_collinear():
            return {
                "occupations_up": self._raw_data.occupations[0],
                "occupations_down": self._raw_data.occupations[1],
            }
        else:
            return {"occupations": self._raw_data.occupations[0]}

    def _shift_dispersion_by_fermi_energy(self, eigenvalues, fermi_energy):
        shifted = self._shift_array_by_fermi_energy(eigenvalues, fermi_energy)
        if len(shifted) == 2:
            return {"bands_up": shifted[0], "bands_down": shifted[1]}
        else:
            return {"bands": shifted[0]}

    def _shift_series_by_fermi_energy(self, graph, fermi_energy):
        for series in graph.series:
            series.y = self._shift_array_by_fermi_energy(series.y, fermi_energy)
        return graph

    def _shift_array_by_fermi_energy(self, array, fermi_energy):
        if fermi_energy is None:
            fermi_energy = self._raw_data.fermi_energy
        return array - fermi_energy

    def _extract_relevant_data(self, selection, fermi_energy):
        need_to_be_repeated = ("kpoint_distances", "kpoint_labels")
        relevant_keys = (
            "bands",
            "bands_up",
            "bands_down",
            "occupations",
            "occupations_up",
            "occupations_down",
        )
        data = {}
        for key, value in self.to_dict(selection, fermi_energy).items():
            if key in need_to_be_repeated:
                data[key] = np.repeat(value, self._raw_data.occupations[0].shape[-1])
            if key in relevant_keys:
                data[key] = _to_series(value)
        for key, value in self._read_projections(selection).items():
            if key == projector.SPIN_PROJECTION:
                # do not include spin projection in dataframe
                continue
            data[key] = _to_series(value)
        return data

    def _kmesh(self) -> tuple[int, int, str]:
        """
        Returns a tuple of number of k-points in two directions as per spin selection,
        and the corresponding cut direction in which the kpoint mesh is 1.
        """
        try:
            nkpx = self._raw_data.dispersion.kpoints.number_x
            nkpy = self._raw_data.dispersion.kpoints.number_y
            nkpz = self._raw_data.dispersion.kpoints.number_z

            if nkpx == 1:
                return (nkpy, nkpz, "a")
            elif nkpy == 1:
                return (nkpx, nkpz, "b")
            elif nkpz == 1:
                return (nkpx, nkpy, "c")
            else:
                raise exception.DataMismatch(
                    f"For spin texture visualisation, the plane normal (a,b,c) to the desired cutting plane must have exactly 1 k-point, but the k-point mesh is {nkpx},{nkpy},{nkpz}. Please adjust the KPOINTS file and re-run VASP."
                )
        except exception.NoData:
            raise exception.DataMismatch(
                f"For spin texture visualisation, a k-point grid is assumed, but could not be found for this VASP run."
            )


def _to_series(array):
    return array.T.flatten()


class _ToQuiverReduction(index.Reduction):
    def __init__(self, keys: List):
        # raise an IncorrectUsage Warning if:
        #   - no spin elements have been chosen
        #   - no band has been chosen
        if not (keys[0]):
            raise exception.IncorrectUsage(
                "Spin Elements must be chosen, but none are given. Please adapt your `selection` argument to include, e.g., `x~y`. You can combine arguments by `arg1(arg2(arg3(...)))`."
            )
        if not (keys[4]):
            raise exception.IncorrectUsage(
                "A band must be chosen, but none are given. Please adapt your `selection` argument to include, e.g., `band[1]`. You can combine arguments by `arg1(arg2(arg3(...)))`."
            )
        pass

    def __call__(self, array: ArrayLike, axis: Iterable):
        axis = tuple(filter(None, axis))  # prevents summation along 0-axis
        return np.sum(array, axis=axis)
