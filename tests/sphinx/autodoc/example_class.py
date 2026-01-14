# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


class ExampleClass:
    """This is an example class for testing Sphinx autodoc.

    It demonstrates how to document a class with attributes and methods.
    """

    def __init__(self, value):
        """Initialize the ExampleClass with a value.

        Parameters
        ----------
        value : int
            An integer value to initialize the class instance.
        """
        self.value = value

    def get_value(self):
        """Return the stored value.

        Returns
        -------
        int
            The integer value stored in the instance.
        """
        return self.value
