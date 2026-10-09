# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import abc
from pathlib import Path
from typing import Optional

from py4vasp import exception
from py4vasp._third_party.graph.graph import (
    IMAGE_FORMATS,
    Graph,
    check_image_format,
    resolve_output_path,
)
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
        """Save the plot as an image file next to the calculation.

        The filetype is automatically deduced from the filename; possible formats
        are the raster formats png, jpg (or jpeg) and webp and the vector formats svg
        and pdf.
        If no filename is provided, a default filename is deduced from the
        name of the class and the picture has png format. To change the size of the
        image, set ``xsize`` and ``ysize`` (in pixels) of the graph that :py:meth:`plot`
        returns and save it with its own ``to_image`` method.

        Parameters
        ----------
        *args
            Positional arguments passed to the :py:meth:`plot` method.
        filename
            Path where the image will be saved. A relative path is relative to the
            directory of the calculation, not to the current working directory; pass
            an absolute path to save elsewhere. "~" is expanded to your home directory.
            For the example data of :func:`py4vasp.demo.calculation` without a path,
            the directory is temporary and saving there warns that it is removed.
            If None, defaults to "{classname}.png" where classname is derived from the
            class name.
        **kwargs
            Keyword arguments passed to the :py:meth:`plot` method.

        Raises
        ------
        py4vasp.exception.IncorrectUsage
            If the filename has no extension or one that is not a supported format.
        py4vasp.exception.FileAccessError
            If the directory the image should be saved to does not exist.

        Notes
        -----
        This function has a side effect of writing the image to disk at the specified
        location. The filename must be a keyword argument, i.e., you explicitly need to
        write ``filename="name_of_file"`` because the positional arguments are passed
        on to the :py:meth:`plot` method. Please check the documentation of
        that method to learn which arguments are allowed.
        """
        if filename is None:
            _raise_error_if_filename_is_positional("to_image", args, IMAGE_FORMATS)
        classname = convert.quantity_name(self.__class__.__name__).strip("_")
        filename = filename if filename is not None else f"{classname}.png"
        check_image_format(filename)
        path = self._output_path(filename)
        self.plot(*args, **kwargs).to_image(path)

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
            Path where the CSV file will be saved. A relative path is relative to the
            directory of the calculation, not to the current working directory; pass
            an absolute path to save elsewhere. "~" is expanded to your home directory.
            For the example data of :func:`py4vasp.demo.calculation` without a path,
            the directory is temporary and saving there warns that it is removed.
            If None, defaults to "{classname}.csv" where classname is derived from the
            class name.
        **kwargs
            Keyword arguments passed to :py:meth:`to_frame`.

        Raises
        ------
        py4vasp.exception.FileAccessError
            If the directory the file should be written to does not exist.

        Notes
        -----
        This function has a side effect of writing the CSV file to disk at the specified
        location. The filename must be a keyword argument, i.e., you explicitly need to
        write ``filename="name_of_file"`` because the positional arguments are passed
        on to the :py:meth:`to_frame` method. Please check the documentation of
        that method to learn which arguments are allowed.
        """
        if filename is None:
            _raise_error_if_filename_is_positional("to_csv", args, (".csv",))
        classname = convert.quantity_name(self.__class__.__name__).strip("_")
        filename = filename if filename is not None else f"{classname}.csv"
        path = self._output_path(filename)
        df = self.to_frame(*args, **kwargs)
        df.to_csv(path, index=False)

    def _output_path(self, filename):
        # ask for the directory only if it is used, because it may warn about it
        is_relative = not Path(filename).expanduser().is_absolute()
        directory = self._output_directory() if is_relative else None
        return resolve_output_path(filename, directory)

    def _output_directory(self):
        return self._path


def _raise_error_if_filename_is_positional(method, args, suffixes):
    for argument in args:
        if (
            isinstance(argument, (str, Path))
            and Path(argument).suffix.lower() in suffixes
        ):
            message = f"""\
The argument "{argument}" looks like a filename, but the positional arguments of
{method} select what is plotted. Please pass the file as a keyword argument, e.g.
{method}(filename="{argument}")."""
            raise exception.IncorrectUsage(message)


def _merge_graphs(graphs):
    result = Graph([])
    for label, graph in graphs.items():
        result = result + graph.label(label)
    return result
