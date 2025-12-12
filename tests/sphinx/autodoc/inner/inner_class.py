# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


class InnerClass:
    """An inner class for testing nested class documentation."""

    def __init__(self, data):
        """
        Initialize the InnerClass with data.

        Parameters
        ----------
        data : str
            The data to be stored in the inner class.
        """
        self.data = data

    def get_data(self) -> str:
        """
        Retrieve the stored data.

        Returns
        -------
        str
            The data stored in the inner class.
        """
        return self.data
