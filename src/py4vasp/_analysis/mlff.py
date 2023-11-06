# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from types import SimpleNamespace
from typing import Dict

import numpy as np

import py4vasp
from py4vasp import exception


class MLFFErrorAnalysis:
    """A class to handle the error analysis of MLFF calculations.

    This class is used to perform error analysis of MLFF calculations. It
    provides methods to calculate the error in energy, forces and stresses
    from MLFF and DFT calculations. See the documentation for the methods
    for more details on the type of error calculated.

    Notes
    -----
    The class is designed to be instantiated using the class methods
    :meth:`from_paths` and :meth:`from_files`. Please use these methods instead
    of directly calling the class.

    Examples
    --------
    >>> from py4vasp import MLFFErrorAnalysis
    >>> mlff_error_analysis = MLFFErrorAnalysis.from_paths(
    ...     dft_data="path/to/dft/data",
    ...     mlff_data="path/to/mlff/data",
    ... )
    >>> energy_error = mlff_error_analysis.get_energy_error_per_atom()
    >>> force_error = mlff_error_analysis.get_force_rmse()
    >>> stress_error = mlff_error_analysis.get_stress_rmse()
    >>> # If you want to normalize the error by the number of configurations
    >>> energy_error = mlff_error_analysis.get_energy_error_per_atom(
    ...     normalize_by_configurations=True
    ... )
    >>> force_error = mlff_error_analysis.get_force_rmse(
    ...     normalize_by_configurations=True
    ... )
    >>> stress_error = mlff_error_analysis.get_stress_rmse(
    ...     normalize_by_configurations=True
    ... )
    """

    def __init__(self, *args, **kwargs):
        self.mlff = SimpleNamespace()
        self.dft = SimpleNamespace()

    @classmethod
    def _from_data(cls, _calculations):
        mlff_error_analysis = cls(_internal=True)
        mlff_error_analysis._calculations = _calculations
        set_appropriate_attrs(mlff_error_analysis)
        return mlff_error_analysis

    @classmethod
    def from_paths(cls, dft_data, mlff_data):
        """Create an instance of MLFFErrorAnalysis from paths to the data.

        Starting from paths for DFT and MLFF data, this method creates an
        instance of MLFFErrorAnalysis. The paths are used to read the data
        from the files.

        Parameters
        ----------
        dft_data : str or pathlib.Path
            Path to the DFT data. Accepts wildcards.
        mlff_data : str or pathlib.Path
            Path to the MLFF data. Accepts wildcards.
        """
        mlff_error_analysis = cls(_internal=True)
        calculations = py4vasp.Calculations.from_paths(
            dft_data=dft_data, mlff_data=mlff_data
        )
        mlff_error_analysis._calculations = calculations
        set_appropriate_attrs(mlff_error_analysis)
        return mlff_error_analysis

    @classmethod
    def from_files(cls, dft_data, mlff_data):
        """Create an instance of MLFFErrorAnalysis from files.

        Starting from files for DFT and MLFF data, this method creates an
        instance of MLFFErrorAnalysis. The files are used to read the data
        from the files.

        Parameters
        ----------
        dft_data : str or pathlib.Path
            Path to the DFT data. Accepts wildcards.
        mlff_data : str or pathlib.Path
            Path to the MLFF data. Accepts wildcards.
        """
        mlff_error_analysis = cls(_internal=True)
        calculations = py4vasp.Calculations.from_files(
            dft_data=dft_data, mlff_data=mlff_data
        )
        mlff_error_analysis._calculations = calculations
        set_appropriate_attrs(mlff_error_analysis)
        return mlff_error_analysis

    def get_energy_error_per_atom(self, normalize_by_configurations=False):
        """Get the error in energy per atom.

        This method calculates the error in energy per atom between the MLFF
        and DFT calculations. The error is calculated as
        :math:`\\frac{E_{MLFF} - E_{DFT}}{N_{ions}}`, where :math:`E_{MLFF}`
        and :math:`E_{DFT}` are the energies from the MLFF and DFT calculations
        respectively, and :math:`N_{ions}` is the number of ions in the
        structure. If ``normalize_by_configurations`` is set to ``True``, the
        error is averaged over the number of configurations.

        Parameters
        ----------
        normalize_by_configurations : bool, optional
            If set to ``True``, the error is averaged over the number of
            configurations. Defaults to ``False``.
        """
        error = (self.mlff.energies - self.dft.energies) / self.dft.nions
        if normalize_by_configurations:
            error = np.sum(np.abs(error), axis=-1) / self.dft.nconfig
        return error

    def _get_rmse(self, dft_quantity, mlff_quantity, degrees_of_freedom):
        norm_error = np.linalg.norm(dft_quantity - mlff_quantity, axis=-1)
        error = np.sqrt(np.sum(norm_error**2, axis=-1) / degrees_of_freedom)
        return error

    def get_force_rmse(self, normalize_by_configurations=False):
        """Get the root mean square error in forces.

        This method calculates the root mean square error in forces between the
        MLFF and DFT calculations. The error is calculated as
        :math:`\\sqrt{\\frac{\\sum_{i=1}^{N_{ions}}{\\sum_{j=1}^{3}{(F_{MLFF} - F_{DFT})^2}}}{3N_{ions}}}`,
        where :math:`F_{MLFF}` and :math:`F_{DFT}` are the forces from the MLFF
        and DFT calculations respectively, and :math:`N_{ions}` is the number
        of ions in the structure. If ``normalize_by_configurations`` is set to
        ``True``, the error is averaged over the number of configurations.

        Parameters
        ----------
        normalize_by_configurations : bool, optional
            If set to ``True``, the error is averaged over the number of
            configurations. Defaults to ``False``.
        """
        deg_freedom = 3 * self.dft.nions
        error = self._get_rmse(self.dft.forces, self.mlff.forces, deg_freedom)
        if normalize_by_configurations:
            error = np.sum(error, axis=-1) / self.dft.nconfig
        return error

    def get_stress_rmse(self, normalize_by_configurations=False):
        """Get the root mean square error in stresses.

        This method calculates the root mean square error in stresses between
        the MLFF and DFT calculations. The error is calculated as
        :math:`\\sqrt{\\frac{\\sum_{i=1}^{6}{(\\sigma_{MLFF} - \\sigma_{DFT})^2}}{6}}`,
        where :math:`\\sigma_{MLFF}` and :math:`\\sigma_{DFT}` are the stresses
        from the MLFF and DFT calculations respectively. If
        ``normalize_by_configurations`` is set to ``True``, the error is
        averaged over the number of configurations.
        """
        deg_freedom = 6
        dft_stresses = np.triu(self.dft.stresses)
        mlff_stresses = np.triu(self.mlff.stresses)
        error = self._get_rmse(dft_stresses, mlff_stresses, deg_freedom)
        if normalize_by_configurations:
            error = np.sum(error, axis=-1) / self.dft.nconfig
        return error


def set_appropriate_attrs(cls):
    set_paths_and_files(cls)
    set_number_of_ions(cls)
    set_number_of_configurations(cls)
    set_energies(cls)
    set_force_related_attributes(cls)
    set_stresses(cls)
    validate_data(cls)


def validate_data(cls):
    """Validate the data passed to the class.

    This method validates the data passed to the class. It checks if the
    number of ions, lattice vectors and positions are consistent between the
    DFT and MLFF calculations. If not, it raises an exception.
    """
    try:
        np.testing.assert_almost_equal(cls.dft.positions, cls.mlff.positions)
        np.testing.assert_almost_equal(
            cls.dft.lattice_vectors, cls.mlff.lattice_vectors
        )
        np.testing.assert_almost_equal(cls.dft.nions, cls.mlff.nions)
    except AssertionError:
        raise exception.IncorrectUsage(
            """\
Please pass a consistent set of data between DFT and MLFF calculations."""
        )


def set_number_of_configurations(cls):
    """Set the number of configurations in the data.

    This method sets the number of configurations in the data. It uses the
    number of calculations performed to set the number of configurations.

    Parameters
    ----------
    cls : MLFFErrorAnalysis
        An instance of MLFFErrorAnalysis.
    """
    number_of_calculations = cls._calculations.number_of_calculations()
    cls.dft.nconfig = number_of_calculations["dft_data"]
    cls.mlff.nconfig = number_of_calculations["mlff_data"]


def set_number_of_ions(cls):
    """Set the number of ions in the data.

    This method sets the number of ions in the data. It uses the number of
    elements in the structures to set the number of ions.

    Parameters
    ----------
    cls : MLFFErrorAnalysis
        An instance of MLFFErrorAnalysis.
    """
    force_data = cls._calculations.forces.read()
    structures_dft = _dict_to_list(force_data["dft_data"], "structure")
    structures_mlff = _dict_to_list(force_data["mlff_data"], "structure")
    elements_dft = _dict_to_array(structures_dft, "elements")
    elements_mlff = _dict_to_array(structures_mlff, "elements")
    nions_dft = np.array([len(_elements) for _elements in elements_dft])
    nions_mlff = np.array([len(_elements) for _elements in elements_mlff])
    cls.dft.nions = nions_dft
    cls.mlff.nions = nions_mlff


def set_paths_and_files(cls):
    """Set the paths and files for the data.

    This method sets the paths and files for the data. It uses the
    :meth:`Calculations.paths` and :meth:`Calculations.files` methods to set
    the paths and files.

    Parameters
    ----------
    cls : MLFFErrorAnalysis
        An instance of MLFFErrorAnalysis.
    """
    paths = cls._calculations.paths()
    cls.dft.paths = paths["dft_data"]
    cls.mlff.paths = paths["mlff_data"]
    if hasattr(cls._calculations, "_files"):
        files = cls._calculations.files()
        cls.dft.files = files["dft_data"]
        cls.mlff.files = files["mlff_data"]


def set_energies(cls):
    """Set the energies for the data.

    This method sets the energies for the data. It uses the
    :meth:`Calculations.energies` method to set the energies.

    Parameters
    ----------
    cls : MLFFErrorAnalysis
        An instance of MLFFErrorAnalysis.
    """
    tag = "free energy    TOTEN"
    energies_data = cls._calculations.energies.read()
    cls.mlff.energies = _dict_to_array(energies_data["mlff_data"], tag)
    cls.dft.energies = _dict_to_array(energies_data["dft_data"], tag)


def _dict_to_array(data: Dict, key: str) -> np.ndarray:
    return np.array([_data[key] for _data in data])


def _dict_to_list(data: Dict, key: str) -> list:
    return [_data[key] for _data in data]


def set_force_related_attributes(cls):
    """Set the force related attributes for the data.

    This method sets the force related attributes for the data. It uses the
    :meth:`Calculations.forces` method to set the forces, lattice vectors and
    positions.

    Parameters
    ----------
    cls : MLFFErrorAnalysis
        An instance of MLFFErrorAnalysis.
    """
    force_data = cls._calculations.forces.read()
    cls.dft.forces = _dict_to_array(force_data["dft_data"], "forces")
    cls.mlff.forces = _dict_to_array(force_data["mlff_data"], "forces")
    dft_structures = _dict_to_list(force_data["dft_data"], "structure")
    mlff_structures = _dict_to_list(force_data["mlff_data"], "structure")
    cls.dft.lattice_vectors = _dict_to_array(dft_structures, "lattice_vectors")
    cls.mlff.lattice_vectors = _dict_to_array(mlff_structures, "lattice_vectors")
    cls.dft.positions = _dict_to_array(dft_structures, "positions")
    cls.mlff.positions = _dict_to_array(mlff_structures, "positions")


def set_stresses(cls):
    """Set the stresses for the data.

    This method sets the stresses for the data. It uses the
    :meth:`Calculations.stresses` method to set the stresses.

    Parameters
    ----------
    cls : MLFFErrorAnalysis
        An instance of MLFFErrorAnalysis.
    """
    stress_data = cls._calculations.stresses.read()
    cls.dft.stresses = _dict_to_array(stress_data["dft_data"], "stress")
    cls.mlff.stresses = _dict_to_array(stress_data["mlff_data"], "stress")
