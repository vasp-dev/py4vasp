from unittest.mock import patch, mock_open
from pathlib import Path
import py4vasp.data
import py4vasp.control as ctrl
import inspect


@patch("py4vasp.raw.File", autospec=True)
def test_creation(MockFile):
    # note: in pytest __file__ defaults to absolute path
    absolute_path = Path(__file__)
    calc = py4vasp.Calculation.from_path(absolute_path)
    assert calc.path() == absolute_path
    relative_path = absolute_path.relative_to(Path.cwd())
    calc = py4vasp.Calculation.from_path(relative_path)
    assert calc.path() == absolute_path
    calc = py4vasp.Calculation.from_path("~")
    assert calc.path() == Path.home()
    MockFile.assert_not_called()
    MockFile.__enter__.assert_not_called()


@patch("py4vasp.raw.File", autospec=True)
def test_all_attributes(MockFile):
    calculation = py4vasp.Calculation.from_path("test_path")
    skipped = ["Viewer3d"]
    for name, _ in inspect.getmembers(py4vasp.data, inspect.isclass):
        if name in skipped:
            continue
        assert hasattr(calculation, name.lower())
    MockFile.assert_not_called()
    MockFile.__enter__.assert_not_called()


def test_input_files():
    text = "! comment line"
    calculation = py4vasp.Calculation.from_path("test_path")
    assert isinstance(calculation.INCAR, ctrl.INCAR)
    with patch("py4vasp.control._base.open", mock_open(read_data=text)) as mock:
        calculation.INCAR = text
        assert calculation.INCAR.read() == text
    #
    assert isinstance(calculation.KPOINTS, ctrl.KPOINTS)
    with patch("py4vasp.control._base.open", mock_open(read_data=text)) as mock:
        calculation.KPOINTS = text
        assert calculation.KPOINTS.read() == text
    #
    assert isinstance(calculation.POSCAR, ctrl.POSCAR)
    with patch("py4vasp.control._base.open", mock_open(read_data=text)) as mock:
        calculation.POSCAR = text
        assert calculation.POSCAR.read() == text
