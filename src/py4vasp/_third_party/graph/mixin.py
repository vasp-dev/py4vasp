# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import abc
import os
from pathlib import Path
from typing import Optional

from py4vasp._third_party.graph.graph import Graph
from py4vasp._util import convert

"""Use the Mixin for all quantities that define an option to produce an x-y graph. This
will automatically implement all the common functionality to turn this graphs into
different formats."""


class Mixin(abc.ABC):
    @abc.abstractmethod
    def to_graph(self, *args, **kwargs):
        pass

    def plot(self, *args, **kwargs) -> Graph:
        """Plot the data by generating and optionally merging graphs.

        This method is almost identical to :py:meth:`to_graph`, but with one key difference:
        if :py:meth:`to_graph` would produce multiple graphs, this method will automatically
        merge them into a single graph.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to :py:meth:`to_graph`.
        **kwargs : dict
            Keyword arguments passed to :py:meth:`to_graph`.

        Returns
        -------
        -
            A single graph object. If :py:meth:`to_graph` produces multiple graphs,
            they are merged into one.
        """
        graph_or_graphs = self.to_graph(*args, **kwargs)
        if isinstance(graph_or_graphs, Graph):
            return graph_or_graphs
        else:
            return _merge_graphs(graph_or_graphs)

    def to_plotly(self, *args, **kwargs) -> "go.Figure":
        """Convert the data to a plotly figure for interactive plotting.

        This method calls :py:meth:`to_graph` with the provided arguments and converts
        the resulting graph to a plotly figure object that can be displayed in Jupyter
        notebooks or saved to HTML.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to :py:meth:`to_graph`.
        **kwargs : dict
            Keyword arguments passed to :py:meth:`to_graph`.

        Returns
        -------
        -
            Interactive plotly figure object.
        """
        return self.to_graph(*args, **kwargs).to_plotly()

    def to_image(self, *args, filename: Optional[str | Path] = None, **kwargs) -> None:
        """
        The filetype is automatically deduced from the filename; possible formats
        are common raster (png, jpg) and vector (svg, pdf) formats.
        If no filename is provided, a default filename is deduced from the
        name of the class and the picture has png format.

        Parameters
        ----------
        *args
            Positional arguments passed to the :py:meth:`to_plotly` method.
        filename
            Path where the image will be saved. Can be absolute or relative to
            the current working directory. If relative, the file will be saved
            relative to the internal path. If None, defaults to "{classname}.png"
            where classname is derived from the class name.
        **kwargs
            Keyword arguments passed to the :py:meth:`to_plotly` method.


        Notes
        -----
        This function has a side effect or writing the image to disk at the specified
        location. The filename must be a keyword argument, i.e., you explicitly need to
        write ``filename="name_of_file"`` because the positional arguments are passed
        on to the :py:meth:`to_plotly` method. Please check the documentation of
        that method to learn which arguments are allowed.
        """
        fig = self.to_plotly(*args, **kwargs)
        classname = convert.quantity_name(self.__class__.__name__).strip("_")
        filename = filename if filename is not None else f"{classname}.png"
        if os.path.isabs(filename):
            writeout_path = filename
        else:
            writeout_path = self._path / filename
        fig.write_image(writeout_path)

    def to_frame(self, *args, **kwargs) -> "pd.DataFrame":
        """Convert data to pandas DataFrame.

        This method first uses the :py:meth:`to_graph` method to convert the data to a
        Graph object, then converts the resulting graph to a pandas DataFrame.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to :py:meth:`to_graph`.
        **kwargs : dict
            Keyword arguments passed to :py:meth:`to_graph`.

        See Also
        --------
        to_graph : Convert data to Graph object.
        """
        graph = self.to_graph(*args, **kwargs)
        return graph.to_frame()

    def to_csv(self, *args, filename: Optional[str | Path] = None, **kwargs):
        """Convert data to CSV file and save to disk.

        This method calls :py:meth:`to_frame` with the provided arguments and saves
        the resulting DataFrame to a CSV file. The file format is comma-separated values.

        Parameters
        ----------
        *args
            Positional arguments passed to :py:meth:`to_frame`.
        filename
            Path where the CSV file will be saved. Can be absolute or relative to
            the current working directory. If relative, the file will be saved
            relative to the internal path. If None, defaults to "{classname}.csv"
            where classname is derived from the class name.
        **kwargs
            Keyword arguments passed to :py:meth:`to_frame`.

        Notes
        -----
        This function has a side effect of writing the CSV file to disk at the specified
        location. The filename must be a keyword argument, i.e., you explicitly need to
        write ``filename="name_of_file"`` because the positional arguments are passed
        on to the :py:meth:`to_frame` method. Please check the documentation of
        that method to learn which arguments are allowed.
        """
        classname = convert.quantity_name(self.__class__.__name__).strip("_")
        filename = filename if filename is not None else f"{classname}.csv"
        if os.path.isabs(filename):
            writeout_path = filename
        else:
            writeout_path = self._path / filename
        df = self.to_frame(*args, **kwargs)
        df.to_csv(writeout_path, index=False)


def _merge_graphs(graphs):
    result = Graph([])
    for label, graph in graphs.items():
        result = result + graph.label(label)
    return result
