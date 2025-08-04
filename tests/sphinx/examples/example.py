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
        Set a new value for the instance.

        Parameters
        ----------
        new_value : float
            The new value to be stored.
        """
        self.value = new_value