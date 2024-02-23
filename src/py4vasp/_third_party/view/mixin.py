# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import abc

"""Use the Mixin for all quantities that define an option to produce a structure view.
This will automatically implement all the common functionality to visualize this data.turn this graphs into
different formats."""


class Mixin(abc.ABC):
    @abc.abstractmethod
    def to_view(self, *args, **kwargs):
        pass

    def plot(self, *args, **kwargs):
        """Wrapper around :meth:`to_view` method.

        This method will visualize the quantity in the structure. Please refer to
        the :meth:`to_view` method for a documentation of the allowed arguments.

        Returns
        -------
        View
            A visualization of the quantity within the crystal structure.
        """
        return self.to_view(*args, **kwargs)

    def to_ngl(self, *args, **kwargs):
        """Convert the view to an NGL widget.

        This method wraps the :meth:`to_view` method and converts the resulting View
        to an NGL widget. The :meth:`to_view` method documents all the possible
        arguments of this function.

        Returns
        -------
        NGLWidget
            A widget to display the structure and other quantities in the unit cell.
        """
        return self.to_view(*args, **kwargs).to_ngl()
