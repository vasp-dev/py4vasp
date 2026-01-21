# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base, projector
from py4vasp._third_party import graph
from py4vasp._util import documentation, import_

pd = import_.optional("pandas")
pretty = import_.optional("IPython.lib.pretty")


class Dos(base.Refinery, graph.Mixin):
    """The density of states (DOS) describes the number of states per energy.

    The DOS quantifies the distribution of electronic states within an energy range
    in a material. It provides information about the number of electronic states at
    each energy level and offers insights into the material's electronic structure.
    On-site projections near the atoms (projected DOS) offer a more detailed view.
    This analysis breaks down the DOS contributions by atom, orbital and spin.
    Investigating the projected DOS is often a useful step to understand the
    electronic properties because it shows how different orbitals and elements
    contribute and influence the material's properties.

    VASP writes the DOS after every calculation and the projected DOS if you set
    :tag:`LORBIT` in the INCAR file. You can use this class to extract this data.
    Typically you want to run a non self consistent calculation with a denser
    mesh for a smoother DOS but the class will work independent of it. If you
    generated a projected DOS, you can use this class to select which subset of
    these orbitals to read or plot.

    Examples
    --------

    First, we create some example data do that you can follow along. Please define a
    variable `path` with the path to a directory that exists and does not contain any
    VASP calculation data. Alternatively, you can use your own data if you have run
    VASP with :tag:`LORBIT`.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    If you want to visualize the total DOS, you can use the `plot` method. This will
    show the different spin components if :tag:`ISPIN` = 2

    >>> calculation.dos.plot()
    Graph(series=[Series(x=array(...), y=array(...), label='total', ...)],
        xlabel='Energy (eV)', ..., ylabel='DOS (1/eV)', ...)

    If you need the raw data, you can read the DOS into a Python dictionary

    >>> calculation.dos.read()
    {'energies': array(...), 'total': array(...), 'fermi_energy': ...}

    These methods also accept selections for specific orbitals if you used VASP with
    :tag:`LORBIT`. You can get a list of the allowed choices with

    >>> calculation.dos.selections()
    {'dos': ['default', 'kpoints_opt'], 'atom': [...], 'orbital': [...], 'spin': [...]}
    """

    _missing_data_message = "No DOS data found, please verify that LORBIT flag is set."

    @base.data_access
    def __str__(self):
        energies = self._raw_data.energies
        if self._is_collinear():
            label = "collinear Dos"
        elif self._is_noncollinear():
            label = "noncollinear Dos"
        else:
            label = "Dos"
        return f"""\
{label}:
    energies: [{energies[0]:0.2f}, {energies[-1]:0.2f}] {len(energies)} points
{pretty.pretty(self._projector())}"""

    @base.data_access
    @documentation.format(selection_doc=projector.selection_doc)
    def to_dict(self, selection=None):
        """Read the DOS into a dictionary.

        You will always get an "energies" component that describes the energy mesh for
        the density of states. The energies are shifted with respect to VASP such that
        the Fermi energy is at 0. py4vasp returns also the original "fermi_energy" so
        you can revert this if you want. If :tag:`ISPIN` = 2, you will get the total
        DOS spin resolved as "up" and "down" component. Otherwise, you will get just
        the "total" DOS. When you set :tag:`LORBIT` in the INCAR file and pass in a
        selection, you will obtain the projected DOS with a label corresponding to the
        projection.

        We create some example data do that you can follow along. Please define a
        variable `path` with the path to a directory that exists and does not contain any
        VASP calculation data. Alternatively, you can use your own data if you have run
        VASP with :tag:`LORBIT`.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> collinear_calculation = demo.calculation(path, selection="collinear")
        >>> noncollinear_calculation = demo.calculation(path, selection="noncollinear")

        Parameters
        ----------
        {selection_doc}

        Returns
        -------
        dict
            Contains the energies at which the DOS was evaluated aligned to the
            Fermi energy and the total DOS or the spin-resolved DOS for
            spin-polarized calculations. If available and a selection is passed,
            the orbital resolved DOS for the selected orbitals is included.

        Examples
        --------

        To obtain the total DOS along with the energy mesh and the Fermi energy you
        do not need any arguments. For :tag:`ISPIN` = 2, this will "up" and "down"
        DOS as two separate entries.

        >>> calculation.dos.to_dict()
        {{'energies': array(...), 'total': array(...), 'fermi_energy': ...}}

        Select the p orbitals of the first atom in the POSCAR file:

        >>> calculation.dos.to_dict(selection="1(p)")
        {{'energies': array(...), 'total': array(...), 'Sr_1_p': array(...),
            'fermi_energy': ...}}

        Select the d orbitals of Sr and Ti:

        >>> calculation.dos.to_dict("d(Sr, Ti)")
        {{'energies': array(...), 'total': array(...), 'Sr_d': array(...),
            'Ti_d': array(...), 'fermi_energy': ...}}

        Collinear calculations separate the DOS into two spin components

        >>> collinear_calculation.dos.read()
        {{'energies': array(...), 'up': array(...), 'down': array(...), ...}}

        You can also select spin contribution of specific atoms or orbitals, e.g. the
        spin-up contribution of the first three atoms combined

        >>> collinear_calculation.dos.to_dict("up(1:3)")
        {{'energies': array(...), 'up': array(...), 'down': array(...),
            '1:3_up': array(...), 'fermi_energy': ...}}

        Noncollinear calculations contain four spin components but by default only
        the total DOS is shown

        >>> noncollinear_calculation.dos.to_dict()
        {{'energies': array(...), 'total': array(...), 'fermi_energy': ...}}

        You can select specific spin components if you want, e.g. the spin projection
        along the z axis

        >>> noncollinear_calculation.dos.to_dict("sigma_z")
        {{'energies': array(...), 'total': array(...), 'sigma_z': array(...),
            'fermi_energy': ...}}

        You can also use simple addition and subtraction to combine the contributions of
        different orbitals, e.g. add the contribution of three d orbitals

        >>> calculation.dos.to_dict("dxy + dxz + dyz")
        {{'energies': array(...), 'total': array(...), 'dxy + dxz + dyz': array(...),
            'fermi_energy': ...}}

        Read the density of states generated by the '''k'''-point mesh in the KPOINTS_OPT
        file (analogously for KPOINTS_WAN)

        >>> calculation.dos.to_dict("kpoints_opt")
        {{'energies': array(...), 'total': array(...), 'fermi_energy': ...}}
        """
        data = self._read_data(selection)
        data.pop(projector.SPIN_PROJECTION, None)
        return {**data, "fermi_energy": self._raw_data.fermi_energy}

    @base.data_access
    def _to_database(self, *args, **kwargs):
        import numpy as np

        return {
            "dos": {
                "dos_at_fermi": None,  # TODO implement with up/down and total
                "min_energy": float(np.min(self._raw_data.energies[:])),
                "max_energy": float(np.max(self._raw_data.energies[:])),
            }
        }

    @base.data_access
    @documentation.format(selection_doc=projector.selection_doc)
    def to_graph(self, selection=None):
        """Read the DOS and convert it into a graph.

        The x axis is the energy mesh used in the calculation shifted such that the
        Fermi energy is at 0. On the y axis, we show the DOS. For :tag:`ISPIN` = 2, the
        different spin components are shown with opposite sign: "up" with a positive
        sign and "down" with a negative one. If you used :tag:`LORBIT` in your VASP
        calculation and you pass in a selection, py4vasp will add additional lines
        corresponding to the selected projections.

        We create some example data do that you can follow along. Please define a
        variable `path` with the path to a directory that exists and does not contain any
        VASP calculation data. Alternatively, you can use your own data if you have run
        VASP with :tag:`LORBIT`.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> collinear_calculation = demo.calculation(path, selection="collinear")
        >>> noncollinear_calculation = demo.calculation(path, selection="noncollinear")

        Parameters
        ----------
        {selection_doc}

        Returns
        -------
        Graph
            Graph containing the total DOS. If the calculation was spin polarized,
            the resulting DOS is spin resolved and the spin-down DOS is plotted
            towards negative values. If a selection is given the orbital-resolved
            DOS is given for the specified projectors.

        Examples
        --------
        For the total DOS, you do not need any arguments. py4vasp will automatically
        use two separate lines, if you used :tag:`ISPIN` = 2 in the VASP calculation

        >>> calculation.dos.to_graph()
        Graph(series=[Series(x=array(...), y=array(...), label='total', ...)],
            xlabel='Energy (eV)', ..., ylabel='DOS (1/eV)', ...)

        Select the p orbitals of the first atom in the POSCAR file:

        >>> calculation.dos.to_graph(selection="1(p)")
        Graph(series=[Series(..., label='total', ...), Series(..., label='Sr_1_p', ...)], ...)

        Select the d orbitals of Sr and Ti:

        >>> calculation.dos.to_graph("d(Sr, Ti)")
        Graph(series=[Series(..., label='total', ...), Series(..., label='Sr_d', ...),
            Series(..., label='Ti_d', ...)], ...)

        Collinear calculations separate the DOS into two spin components

        >>> collinear_calculation.dos.to_graph()
        Graph(series=[Series(..., label='up', ...), Series(..., label='down', ...)], ...)

        You can also select spin contribution of specific atoms or orbitals, e.g. the
        spin-up contribution of the first three atoms combined

        >>> collinear_calculation.dos.to_graph("up(1:3)")
        Graph(series=[Series(..., label='up', ...), Series(..., label='down', ...),
            Series(..., label='1:3_up', ...)], ...)

        Noncollinear calculations contain four spin components but by default only
        the total DOS is shown

        >>> noncollinear_calculation.dos.to_graph()
        Graph(series=[Series(..., label='total', ...)], ...)

        You can select specific spin components if you want, e.g. the spin projection
        along the z axis

        >>> noncollinear_calculation.dos.to_graph("sigma_z")
        Graph(series=[Series(..., label='total', ...), Series(..., label='sigma_z', ...)], ...)

        You can also use simple addition and subtraction to combine the contributions of
        different orbitals, e.g. add the contribution of three d orbitals

        >>> calculation.dos.to_graph("dxy + dxz + dyz")
        Graph(series=[Series(..., label='total', ...), Series(..., label='dxy + dxz + dyz', ...)], ...)

        Read the density of states generated by the '''k'''-point mesh in the KPOINTS_OPT
        file (analogously for KPOINTS_WAN)

        >>> calculation.dos.to_graph("kpoints_opt")
        Graph(series=[Series(..., label='total', ...)], ...)
        """
        data = self._read_data(selection)
        energies = data.pop("energies")
        data.pop(projector.SPIN_PROJECTION, None)
        return graph.Graph(
            series=list(_series(energies, data)),
            xlabel="Energy (eV)",
            ylabel="DOS (1/eV)",
        )

    @base.data_access
    @documentation.format(selection_doc=projector.selection_doc)
    def to_frame(self, selection=None):
        """Read the data into a pandas DataFrame.

        We create some example data do that you can follow along. Please define a
        variable `path` with the path to a directory that exists and does not contain any
        VASP calculation data. Alternatively, you can use your own data if you have run
        VASP with :tag:`LORBIT`.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)
        >>> collinear_calculation = demo.calculation(path, selection="collinear")
        >>> noncollinear_calculation = demo.calculation(path, selection="noncollinear")

        Parameters
        ----------
        {selection_doc}

        Returns
        -------
        pd.DataFrame
            Contains the energies at which the DOS was evaluated aligned to the
            Fermi energy and the total DOS or the spin-resolved DOS for
            spin-polarized calculations. If available and a selection is passed,
            the orbital resolved DOS for the selected orbitals is included.

        Examples
        --------
        >>> calculation.dos.to_frame()
           energies  total
        0  ...

        Select the p orbitals of the first atom in the POSCAR file:

        >>> calculation.dos.to_frame(selection="1(p)")
           energies  total  Sr_1_p
        0  ...

        Select the d orbitals of Sr and Ti:

        >>> calculation.dos.to_frame("d(Sr, Ti)")
           energies  total  Sr_d  Ti_d
        0  ...

        Collinear calculations separate the DOS into two spin components

        >>> collinear_calculation.dos.to_frame()
           energies  up  down
        0  ...

        You can also select spin contribution of specific atoms or orbitals, e.g. the
        spin-up contribution of the first three atoms combined

        >>> collinear_calculation.dos.to_frame("up(1:3)")
           energies  up  down  1:3_up
        0  ...

        Noncollinear calculations contain four spin components but by default only
        the total DOS is shown

        >>> noncollinear_calculation.dos.to_frame()
           energies  total
        0  ...

        You can select specific spin components if you want, e.g. the spin projection
        along the z axis

        >>> noncollinear_calculation.dos.to_frame("sigma_z")
           energies  total  sigma_z
        0  ...

        You can also use simple addition and subtraction to combine the contributions of
        different orbitals, e.g. add the contribution of three d orbitals

        Add the contribution of three d orbitals

        >>> calculation.dos.to_frame("dxy + dxz + dyz")
           energies  total  dxy + dxz + dyz
        0  ...

        Read the density of states generated by the '''k'''-point mesh in the KPOINTS_OPT
        file (analogously for KPOINTS_WAN)

        >>> calculation.dos.to_frame("kpoints_opt")
           energies  total
        0  ...
        """
        data = self._read_data(selection)
        data.pop(projector.SPIN_PROJECTION, None)
        df = pd.DataFrame(data)
        df.fermi_energy = self._raw_data.fermi_energy
        return df

    @base.data_access
    def selections(self):
        return {**super().selections(), **self._projector().selections()}

    def _is_collinear(self):
        return len(self._raw_data.dos) == 2

    def _is_noncollinear(self):
        return len(self._raw_data.dos) == 4

    def _projector(self):
        return projector.Projector.from_data(self._raw_data.projectors)

    def _read_data(self, selection):
        return {
            **self._read_energies(),
            **self._read_total_dos(),
            **self._projector().project(selection, self._raw_data.projections),
        }

    def _read_energies(self):
        return {"energies": self._raw_data.energies[:] - self._raw_data.fermi_energy}

    def _read_total_dos(self):
        if self._is_collinear():
            return {"up": self._raw_data.dos[0, :], "down": self._raw_data.dos[1, :]}
        else:
            return {"total": self._raw_data.dos[0, :]}


def _series(energies, data):
    for name, dos in data.items():
        spin_factor = -1 if _flip_down_component(name) else 1
        yield graph.Series(energies, spin_factor * dos, name)


def _flip_down_component(name):
    return "down" in name and "up" not in name and "total" not in name
