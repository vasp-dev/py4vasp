# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.control import KPOINTS

from .test_base import AbstractTest


class TestKpoints(AbstractTest):
    tested_class = KPOINTS
