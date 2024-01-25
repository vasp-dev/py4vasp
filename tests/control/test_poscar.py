# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from unittest.mock import patch

from py4vasp import calculation
from py4vasp._third_party.viewer import viewer3d
from py4vasp.control import POSCAR

from .test_base import AbstractTest


class TestPoscar(AbstractTest):
    tested_class = POSCAR


def test_plot_poscar(not_core):
    text = """! comment line
    5.43
    0.0 0.5 0.5
    0.5 0.0 0.5
    0.5 0.5 0.0
    Si
    2
    Direct
    0.00 0.00 0.00
    0.25 0.25 0.25
    """
    poscar = POSCAR.from_string(text)
    obj = viewer3d.Viewer3d
    cm_init = patch.object(obj, "__init__", autospec=True, return_value=None)
    cm_cell = patch.object(obj, "show_cell")
    with cm_init as init, cm_cell as cell:
        poscar.plot()
        init.assert_called_once()
        cell.assert_called_once()


def test_plot_argument_forwarding():
    text = "! comment line"
    poscar = POSCAR.from_string(text)
    with patch("py4vasp.calculation._structure.Structure.from_POSCAR") as struct:
        poscar.plot("argument", key="value")
        struct.return_value.plot.assert_called_once_with("argument", key="value")
