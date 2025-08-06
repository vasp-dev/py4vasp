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

    def combined_returns(self, some_value: float, some_string: str | None = "") -> tuple[float, str | None]:
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