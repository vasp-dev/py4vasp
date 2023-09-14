# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pytest

from py4vasp import Calculation
from py4vasp._analysis.mlff import MLFFErrorAnalysis


@patch("py4vasp._data.base.Refinery.from_path", autospec=True)
@patch("py4vasp.raw.access", autospec=True)
def test_read_inputs(mock_access, mock_from_path):
    absolute_path_dft = Path(__file__) / "dft"
    absolute_path_mlff = Path(__file__) / "mlff"
    error_analysis = MLFFErrorAnalysis.from_paths(
        dft_data=absolute_path_dft, mlff_data=absolute_path_mlff
    )
    assert isinstance(error_analysis.mlff_energies, np.ndarray)
    assert isinstance(error_analysis.dft_energies, np.ndarray)
    assert isinstance(error_analysis.mlff_forces, np.ndarray)
    assert isinstance(error_analysis.dft_forces, np.ndarray)
    assert isinstance(error_analysis.mlff_lattice_vectors, np.ndarray)
    assert isinstance(error_analysis.dft_lattice_vectors, np.ndarray)
    assert isinstance(error_analysis.mlff_positions, np.ndarray)
    assert isinstance(error_analysis.dft_positions, np.ndarray)
    assert isinstance(error_analysis.mlff_nions, int)
    assert isinstance(error_analysis.dft_nions, int)
