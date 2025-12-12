# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Example module for Sphinx autodoc testing.
"""
from dataclasses import dataclass


class ExampleModule:
    """A simple example module to demonstrate Sphinx autodoc capabilities."""

    def example_method(self, param1, param2):
        """An example function that adds two parameters.

        Parameters
        ----------
        param1 : int
            The first parameter.
        param2 : int
            The second parameter.

        Returns
        -------
        int
            The sum of param1 and param2.
        """
        return param1 + param2


@dataclass
class ExampleDataClass:
    """An example data class for testing Sphinx autodoc."""

    attribute1: int
    """An integer attribute."""

    attribute2: str
    """A string attribute."""


def example_function(x):
    """An example function that squares its input.

    Parameters
    ----------
    x : int
        An integer to be squared.

    Returns
    -------
    int
        The square of the input integer.
    """
    return x * x
    return x * x
