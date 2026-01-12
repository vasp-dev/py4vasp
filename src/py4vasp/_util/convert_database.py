from typing import Any


def _to_database(
    level: int = None,
    primary_dict: dict[str, Any] = {},
    secondary_dict: dict[str, Any] = {},
    add_key_prefix: str = None,
) -> dict[str, Any]:
    """
    Wrapper function to create a dictionary with database properties for a dataclass.
    Dataclasses may or may not have a to_database method implemented.

    Parameters
    ----------
    level : int
        The allowed recursion level going forward.
        - level<=0: no properties are included.
        - level=1: only primary_dict is included.
        - level=2: primary_dict and direct properties of secondary_dict are included.
        - level>2: recursion continues for properties of secondary_dict.
    primary_dict : dict[str, Any]
        The primary dictionary of database properties.
        This holds key,value pairs for properties that are directly computed.
    secondary_dict : dict[str, Any]
        The secondary dictionary of database properties.
        This holds key,value pairs for properties that are dataclasses themselves
        and should have their to_database method invoked, if implemented.
        Values may also be tuples of (dataclass, level) to specify a different recursion level.
        Its keys will be added as prefixes to the properties returned by the to_database methods.
    add_key_prefix : str
        An optional prefix to add to all keys in the returned dictionary.

    Returns
    -------
    dict[str, Any]
        The combined dictionary of database properties.

    Examples
    --------
    Example definition:

    ```python
    def to_database(self, level=None) -> dict[str, Any]:
        return _to_database(level=level, primary_dict={
            "fermi_energy": self.fermi_energy,
        }, secondary_dict={
            "dispersion": self.dispersion,
        })
    ```

    Example usage:

    Obtain primary dict and secondary dict properties of Band dataclass:

    >>> from py4vasp import Calculation
    >>> calc = Calculation.from_path("path/to/calculation")
    >>> band = calc.band.read()
    >>> db_props = band.to_database(level=2)

    Obtain primary dict only:

    >>> db_props_primary = band.to_database(level=1)

    Obtain all properties recursively:

    >>> db_props_all = band.to_database()

    """
    if level is None:
        level = 10
    return_dict = {
        (f"{add_key_prefix}_" if add_key_prefix else "") + k: v
        for k, v in (primary_dict.copy() if (level > 0) else {}).items()
    }
    if level > 1:
        for key, value in secondary_dict.items():
            try:
                new_level = level - 1
                if isinstance(value, tuple):
                    value, new_level = value
                new_props = value.to_database(
                    level=new_level,
                    add_key_prefix=(f"{add_key_prefix}_" if add_key_prefix else "")
                    + key,
                )
                return_dict = return_dict | new_props
            except:
                pass
    return return_dict


def _shape_or_none(array):
    try:
        return array.shape
    except AttributeError:
        return None


def _length_or_none(array):
    try:
        return len(array)
    except TypeError:
        return None
