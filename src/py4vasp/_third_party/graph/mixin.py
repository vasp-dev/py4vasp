# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import abc
import os

from py4vasp._third_party.graph.graph import Graph
from py4vasp._util import convert

"""Use the Mixin for all quantities that define an option to produce an x-y graph. This
will automatically implement all the common functionality to turn this graphs into
different formats."""


class Mixin(abc.ABC):
    @abc.abstractmethod
    def to_graph(self, *args, **kwargs):
        pass

    def to_frame(self, *args, **kwargs):
        """Wrapper around the :py:meth:`to_frame` function.

        Generates dataframes from the graph object. For information about
        parameters that can be passed to this method, look at :py:meth:`to_graph`.

        Returns
        -------
        Dataframe
            Pandas dataframe corresponding to data in the graph
        """
        graph = self.to_graph(*args, **kwargs)
        return graph.to_frame()

    def to_csv(self, *args, filename=None, **kwargs):
        """Converts data to a csv file.

        Writes out a csv file for data stored in a dataframe generated with
        the :py:meth:`to_frame` method. Useful for creating external plots
        for further analysis.

        If no filename is provided a default filename is deduced from the
        name of the class.

        Note that the filename must be a keyword argument, i.e., you explicitly
        need to write *filename="name_of_file"* because the arguments are passed
        on to the :py:meth:`to_graph` function. Please check the documentation of that function
        to learn which arguments are allowed.

        Parameters
        ----------
        filename: str | Path
            Name of the csv file which the data is exported to.
        """
        classname = convert.quantity_name(self.__class__.__name__).strip("_")
        filename = filename if filename is not None else f"{classname}.csv"
        if os.path.isabs(filename):
            writeout_path = filename
        else:
            writeout_path = self._path / filename
        df = self.to_frame(*args, **kwargs)
        df.to_csv(writeout_path, index=False)

    def plot(self, *args, **kwargs):
        """Wrapper around the :py:meth:`to_graph` function.

        This will merge multiple graphs if you specify different sources with the
        selection arguments. All arguments are passed to the wrapped function."""
        graph_or_graphs = self.to_graph(*args, **kwargs)
        if isinstance(graph_or_graphs, Graph):
            return graph_or_graphs
        else:
            return _merge_graphs(graph_or_graphs)

    def to_plotly(self, *args, **kwargs):
        """Convert the graph of this quantity to a plotly figure.

        The arguments to this function are automatically passed on to the :py:meth:`to_graph`
        function. Please check the documentation of that function to learn which arguments
        are allowed."""
        return self.to_graph(*args, **kwargs).to_plotly()

    def to_image(self, *args, filename=None, **kwargs):
        """Read the data and generate an image writing to the given filename.

        The filetype is automatically deduced from the filename; possible
        are common raster (png, jpg) and vector (svg, pdf) formats.
        If no filename is provided a default filename is deduced from the
        name of the class and the picture has png format.

        Note that the filename must be a keyword argument, i.e., you explicitly
        need to write *filename="name_of_file"* because the arguments are passed
        on to the :py:meth:`to_graph` function. Please check the documentation of that function
        to learn which arguments are allowed."""
        fig = self.to_plotly(*args, **kwargs)
        classname = convert.quantity_name(self.__class__.__name__).strip("_")
        filename = filename if filename is not None else f"{classname}.png"
        if os.path.isabs(filename):
            writeout_path = filename
        else:
            writeout_path = self._path / filename
        fig.write_image(writeout_path)


def _merge_graphs(graphs):
    result = Graph([])
    for label, graph in graphs.items():
        result = result + graph.label(label)
    return result
