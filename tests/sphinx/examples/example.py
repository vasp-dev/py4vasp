class Example:
    """An example class for demonstration purposes."""

    def __init__(self, value: float):
        """
        Initialize the Example class with a value.

        - some list entry
        - some other list entry

        Parameters
        ----------
        value : float
            The value to be stored in the instance.
        """
        self.value = value

    def get_value(self) -> float:
        """
        Retrieve the stored value.

        Returns
        -------
        float
            The stored value.
        """
        return self.value

    def set_value(self, new_value: float):
        """
        Set a new value for the instance. The value can be obtained with the :py:meth:`get_value` method.

        Parameters
        ----------
        new_value : float
            The new value to be stored.
        """
        self.value = new_value

    def combined_returns(
        self, some_value: float, some_string: str | None = ""
    ) -> tuple[float, str | None]:
        """
        Combine a float and a string in a tuple.

        Parameters
        ----------
        some_value : float
            A value to be included in the tuple.
        some_string : str
            A string to be included in the tuple.

        Returns
        -------
        tuple[float, str | None]
            A tuple containing the float and a string representation.
        """
        return some_value, f"{some_string}: {some_value}" if some_string else None


def returns_type_without_desc_returns(value1: float, value2: float | str):
    """
    Return value 2.

    Parameters
    ----------
    value1 : float
        The first value.
    value2 : float | str
        The second value, which can be a float or a string.

    Returns
    -------
    float | str
        The second value.
    """
    return value2


def returns_type_without_returns_field(
    value1: float, value2: float | str
) -> float | str:
    """
    Return value 2.

    Parameters
    ----------
    value1 : float
        The first value.
    value2 : float | str
        The second value, which can be a float or a string.
    """
    return value2


def returns_type_without_returns_field_desc(
    value1: float, value2: float | str
) -> float | str:
    """
    Return value 2.

    Parameters
    ----------
    value1 : float
        The first value.
    value2 : float | str
        The second value, which can be a float or a string.

    Returns
    -------
    float | str
    """
    return value2


def returns_type_without_returns_field_type(
    value1: float, value2: float | str
) -> float | str:
    """
    Return value 2.

    Parameters
    ----------
    value1 : float
        The first value.
    value2 : float | str
        The second value, which can be a float or a string.

    Returns
    -------
    -
        The second value.
        With another line!
    """
    return value2


def params_types_only_in_signature(value1: float, value2: float | str = 0):
    """
    Example function with parameter types only in the signature.

    Parameters
    ----------
    value1
        The first value.
    value2
        The second value, which can be a float or a string.
    """
    value3 = value1 + value2
    return value3


def params_types_only_in_field(value1, value2):
    """
    Example function with parameter types only in the field.

    Parameters
    ----------
    value1 : float
        The first value.
    value2 : float | str, optional
        The second value, which can be a float or a string.
    """
    value3 = value1 + value2
    return value3


def params_types_in_signature_and_field(value1, value2: float | str = 0):
    """
    Example function with parameter types mixed in both field and signature.

    Parameters
    ----------
    value1 : float
        The first value.
    value2
        The second value, which can be a float or a string.
    """
    value3 = value1 + value2
    return value3


def params_types_mismatched(value1: float, value2=0):
    """
    Example function with parameter types mismatched.

    Parameters
    ----------
    value1 : float | None
        The first value.
    value2 : float | str, optional
        The second value, which can be a float or a string.
    """
    value3 = value1 + value2
