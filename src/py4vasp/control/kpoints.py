from py4vasp.control._base import InputBase


class KPOINTS(InputBase):
    """The KPOINTS file defining the **k**-point grid for the VASP calculation.

    Parameters
    ----------
    path : str or Path
        Defines where the KPOINTS file is stored. If set to None, the file will be kept
        in memory.
    """
