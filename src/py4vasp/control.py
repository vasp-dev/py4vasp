# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Setup input data for VASP calculations.

VASP requires several input files to execute. We provide some simple helper classes and
routines to generate these files from python. You can also use the routines to extract
the input files from a path.
"""
from py4vasp._control.incar import INCAR
from py4vasp._control.kpoints import KPOINTS
from py4vasp._control.poscar import POSCAR
