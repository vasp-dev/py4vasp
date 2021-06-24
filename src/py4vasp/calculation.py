"""Provide the tools to manage VASP calculations.

This is the main user interface if you want to simply investigate the results of VASP
calculations. Create a Calculation object associated with the VASP calculation that you
run. Then you can access the properties of that calculation via the attributes of the
object. For example you may use

.. code-block:: python

   calc = Calculation.from_path("path_to_your_calculation")
   calc.dos.plot()         # to plot the density of states
   calc.magnetism.read()   # to read the magnetic moments
   calc.structure.print()  # to print the structure in a POSCAR format
"""
import inspect
import py4vasp.data
import py4vasp.control
from pathlib import Path


class Calculation:
    """Manage access to input and output of VASP calculations.

    .. warning::
       Create new instances using the class method :meth:`from_path` and not the
       constructor. Otherwise some functionality will not work as intended.
    """

    @classmethod
    def from_path(cls, path_name):
        """Set up a Calculation for a particular path and so that all files are opened there.

        Parameters
        ----------
        path_name : str or Path
            Name of the path associated with the calculation.

        Returns
        -------
        Calculation
            A calculation associated with the given path.
        """
        calc = cls()
        calc._path = Path(path_name).expanduser().resolve()
        calc = _add_all_refinement_classes(calc)
        return _add_input_files(calc)

    def path(self):
        "Return the path in which the calculation is run."
        return self._path

    @property
    def INCAR(self):
        "The INCAR file of the VASP calculation."
        return self._INCAR

    @INCAR.setter
    def INCAR(self, incar):
        self._INCAR.write(str(incar))

    @property
    def KPOINTS(self):
        "The KPOINTS file of the VASP calculation."
        return self._KPOINTS

    @KPOINTS.setter
    def KPOINTS(self, kpoints):
        self._KPOINTS.write(str(kpoints))

    @property
    def POSCAR(self):
        "The POSCAR file of the VASP calculation."
        return self._POSCAR

    @POSCAR.setter
    def POSCAR(self, poscar):
        self._POSCAR.write(str(poscar))


def _add_all_refinement_classes(calc):
    for name, class_ in inspect.getmembers(py4vasp.data, inspect.isclass):
        if issubclass(class_, py4vasp.data._base.DataBase):
            calc = _add_refinement_class(calc, name, class_)
    return calc


def _add_refinement_class(calc, name, class_):
    instance = class_.from_file(calc.path())
    setattr(calc, name.lower(), instance)
    return calc


def _add_input_files(calc):
    calc._INCAR = py4vasp.control.INCAR(calc._path)
    calc._KPOINTS = py4vasp.control.KPOINTS(calc._path)
    calc._POSCAR = py4vasp.control.POSCAR(calc._path)
    return calc
