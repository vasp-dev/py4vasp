from py4vasp._calculation import base, structure


class Nics(base.Refinery, structure.Mixin):
    """This class accesses information on the nucleus-independent chemical shift (NICS)."""

    @base.data_access
    def to_dict(self):
        """Read nics into a dictionary.

        Parameters
        ----------

        Returns
        -------
        dict
            Contains the structure information as well as the nucleus-independent chemical shift represented on a grid in the unit cell.
        """
        result = {"structure": self._structure.read()}
        # result.update(self._read_nics())
        return result
