from unittest.mock import patch
from py4vasp.control import INCAR
import py4vasp.data as data
from .test_base import AbstractTest


class TestIncar(AbstractTest):
    tested_class = INCAR
