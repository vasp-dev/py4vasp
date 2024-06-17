# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import contextlib
import dataclasses
import functools
import inspect
import pathlib

from py4vasp import exception, raw
from py4vasp._util import check, convert, select


def data_access(func):
    """Use this decorator for all public methods of Refinery children. It creates the
    necessary wrappers to load the data from the VASP output and makes it available
    via the _raw_data property. The decorator also selects the appropriate source from
    the HDF5 file based on the selection argument. If multiple selections are requested
    these will be handled iteratively and returned as a dictionary. Any remaining
    selection not matching a source is passed to the inner function."""

    @functools.wraps(func)
    def func_with_access(self, *args, **kwargs):
        wrapper = _FunctionWrapper(func, self)
        return wrapper.run(*args, **kwargs)

    return func_with_access


class Refinery:
    def __init__(self, data_context, **kwargs):
        self._data_context = data_context
        self._path = kwargs.get("path") or pathlib.Path.cwd()
        self._repr = kwargs.get("repr", f"({repr(data_context)})")
        self.__post_init__()

    def __post_init__(self):
        # overload this to do extra initialization
        pass

    @classmethod
    def from_data(cls, raw_data):
        """Create the instance directly from the raw data.

        Use this approach when the data is put into the correct format by other means
        than reading from the VASP output files. A typical use case is to read the
        data with `from_path` and then act on it with some postprocessing and pass
        the results to this method.

        Parameters
        ----------
        raw_data
            The raw data required to produce this Refinery.

        Returns
        -------
            A Refinery instance to handle the passed data.
        """
        repr_ = f".from_data({repr(raw_data)})"
        return cls(_DataWrapper(_quantity(cls), raw_data), repr=repr_)

    @classmethod
    def from_path(cls, path=None):
        """Read the quantities from the given path.

        The VASP schema determines the particular files accessed. The files will only
        be accessed when the data is required for a particular postprocessing call.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the directory with the outputs of the VASP calculation. If not
            set explicitly the current directory will be used.

        Returns
        -------
        Refinery
            The returned instance handles opening and closing the files for every
            function called on it.
        """
        repr_ = f".from_path({repr(path)})" if path is not None else ".from_path()"
        return cls(_DataAccess(_quantity(cls), path=path), repr=repr_, path=path)

    @classmethod
    def from_file(cls, file):
        """Read the quantities from the given file.

        You want to use this method if you want to avoid using the Calculation
        wrapper, for example because you renamed the output of the VASP calculation.

        Parameters
        ----------
        file : str or io.BufferedReader
            Filename from which the data is extracted. Alternatively, you can open the
            file yourself and pass the Reader object. In that case, you need to take
            care the file is properly closed again and be aware the generated instance
            of this class becomes unusable after the file is closed.

        Returns
        -------
        Refinery
            The returned instance handles opening and closing the file for every
            function called on it, unless a Reader object is passed in which case this
            is left to the user.

        Notes
        -----
        VASP produces multiple output files whereas this routine will only link to the
        single specified file. Prefer `from_path` routine over this method unless you
        renamed the VASP output files, because `from_path` can collate results from
        multiple files.
        """
        repr_ = f".from_file({repr(file)})"
        path = _get_path_to_file(file)
        return cls(_DataAccess(_quantity(cls), file=file), repr=repr_, path=path)

    @property
    def path(self):
        "Returns the path from which the output is obtained."
        return self._path

    @property
    def _raw_data(self):
        return self._data_context.data

    @property
    def _selection(self):
        return self._data_context.selection

    @data_access
    def print(self):
        "Print a string representation of this instance."
        print(str(self))

    def read(self, *args, **kwargs):
        "Convenient wrapper around to_dict. Check that function for examples and optional arguments."
        return self.to_dict(*args, **kwargs)

    @data_access
    def selections(self):
        """Returns possible alternatives for this particular quantity VASP can produce.

        The returned dictionary contains a single item with the name of the quantity
        mapping to all possible selections. Each of these selection may be passed to
        other functions of this quantity to select which output of VASP is used.

        Returns
        -------
        dict
            The key indicates this quantity and the values possible choices for arguments
            to other functions of this quantity.
        """
        sources = list(raw.selections(self._data_context.quantity))
        return {self._data_context.quantity: sources}

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __repr__(self):
        return f"{self.__class__.__name__}{self._repr}"


def _quantity(cls):
    return convert.quantity_name(cls.__name__)


def _get_path_to_file(file):
    try:
        file = pathlib.Path(file)
    except TypeError:
        file = pathlib.Path(file.name)
    return file.parent


class _FunctionWrapper:
    def __init__(self, func, refinery):
        self._data_context = refinery._data_context
        self._func = functools.partial(func, refinery)

    def run(self, *args, **kwargs):
        selection, bound_arguments = self._find_selection_in_arguments(*args, **kwargs)
        selections = self._parse_selection(selection)
        results = self._run_selections(bound_arguments, selections)
        return self._merge_results(results)

    def _find_selection_in_arguments(self, *args, **kwargs):
        signature = inspect.signature(self._func)
        if "selection" in signature.parameters:
            return self._get_selection_from_parameters(signature, *args, **kwargs)
        elif selection := kwargs.pop("selection", None):
            return selection, signature.bind(*args, **kwargs)
        else:
            return self._get_selection_from_args(signature, *args, **kwargs)

    def _get_selection_from_parameters(self, signature, *args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        return arguments.arguments["selection"], arguments

    def _get_selection_from_args(self, signature, *args, **kwargs):
        try:
            # if the signature works, there is no additional selection argument
            return None, signature.bind(*args, **kwargs)
        except TypeError as error:
            try:
                return args[-1], signature.bind(*(args[:-1]), **kwargs)
            except:
                raise error

    def _parse_selection(self, selection):
        tree = select.Tree.from_selection(selection)
        result = {}
        for selection in tree.selections():
            selected, remaining = self._find_selection_in_schema(selection)
            result.setdefault(selected, [])
            result[selected].append(remaining)
        return result

    def _find_selection_in_schema(self, selection):
        options = raw.selections(self._data_context.quantity)
        for option in options:
            if select.contains(selection, option, ignore_case=True):
                return self._remove_selected_option_if_possible(selection, option)
        return None, list(selection)

    def _remove_selected_option_if_possible(self, selection, option):
        is_option = lambda part: str(part).lower() == option.lower()
        remaining = [part for part in selection if not is_option(part)]
        _raise_error_if_option_was_not_removed(selection, option, remaining)
        return option.lower(), remaining

    def _run_selections(self, bound_arguments, selections):
        for selection in selections.items():
            yield self._run_selection(bound_arguments, *selection)

    def _run_selection(self, bound_arguments, selected, remaining):
        self._data_context.set_selection(selected)
        work = self._set_remaining_selection(bound_arguments, remaining)
        with self._data_context:
            check.raise_error_if_not_callable(self._func, *work.args, **work.kwargs)
            return selected or "default", self._func(*work.args, **work.kwargs)

    def _set_remaining_selection(self, bound_arguments, remaining):
        if "selection" not in bound_arguments.arguments:
            self._raise_error_if_any_remaining_selection(remaining)
            return bound_arguments
        if remaining == [[]]:
            selection = self._use_default_or_empty_string(bound_arguments.signature)
        else:
            selection = select.selections_to_string(remaining)
        bound_arguments.arguments["selection"] = selection
        return bound_arguments

    def _use_default_or_empty_string(self, signature):
        if signature.parameters["selection"].default == inspect.Parameter.empty:
            return ""
        else:
            return signature.parameters["selection"].default

    def _merge_results(self, results):
        results = dict(results)
        if all(value is None for value in results.values()):
            return None
        if len(results) == 1:
            return results.popitem()[1]  # unpack value of first element
        return results

    def _raise_error_if_any_remaining_selection(self, remaining):
        if remaining == [[]]:
            return
        selection = select.selections_to_string(remaining)
        sources = '", "'.join(raw.selections(self._data_context.quantity))
        message = f"""py4vasp found a selection "{selection}", but could not parse it.
            Please check for possible spelling errors. Possible sources for
            {self._data_context.quantity} are "{sources}"."""
        raise exception.IncorrectUsage(message)


def _raise_error_if_option_was_not_removed(selection, option, remaining):
    if len(remaining) == len(selection):
        message = f"""py4vasp identified the source "{option}" in your selection string
            "{select.selections_to_string((selection,))}". However, the source could not
            be extracted from the selection. A possible reason is that it is used in an
            addition or subtraction, which is not implemented."""
        raise exception.NotImplemented(message)


@dataclasses.dataclass
class _DataWrapper(contextlib.AbstractContextManager):
    quantity: str
    data: raw.VaspData
    selection: str = None

    def __enter__(self):
        pass

    def __exit__(self, *_):
        pass

    def set_selection(self, selection):
        if selection is None:
            return
        message = (
            f"Creating {self.quantity}.from_data does not allow to select a source."
        )
        raise exception.IncorrectUsage(message)


class _DataAccess(contextlib.AbstractContextManager):
    def __init__(self, quantity, **kwargs):
        self.selection = None
        self.quantity = quantity
        self._kwargs = kwargs
        self._counter = 0
        self._stack = contextlib.ExitStack()

    def __enter__(self):
        if self._counter == 0:
            context = raw.access(
                self.quantity, selection=self.selection, **self._kwargs
            )
            self.data = self._stack.enter_context(context)
        self._counter += 1
        return self.data

    def __exit__(self, *_):
        self._counter -= 1
        if self._counter == 0:
            self.selection = None
            self._stack.close()

    def set_selection(self, selection):
        if self._counter == 0:
            self.selection = selection
