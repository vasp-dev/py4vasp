from py4vasp.data import plot, read, to_dict, _util, _classes
from unittest.mock import MagicMock
from contextlib import contextmanager, nullcontext, redirect_stdout, redirect_stderr
import py4vasp.exceptions as exception
import py4vasp.config as config
import pytest
import io


def test_plot():
    plotable = MagicMock()
    context = plotable.from_file.return_value
    obj = context.__enter__.return_value
    # without arguments
    res = plot(plotable)
    plotable.from_file.assert_called_once()
    context.__enter__.assert_called_once()
    obj.plot.assert_called_once()
    assert res == obj.plot.return_value
    # with arguments
    res = plot(plotable, "arguments")
    obj.plot.assert_called_with("arguments")
    # with keyword arguments
    res = plot(plotable, key="value")


def test_read():
    readable = MagicMock()
    context = readable.from_file.return_value
    obj = context.__enter__.return_value
    # without arguments
    res = read(readable)
    readable.from_file.assert_called_once()
    context.__enter__.assert_called_once()
    obj.read.assert_called_once()
    assert res == obj.read.return_value
    # with arguments
    res = read(readable, "arguments")
    obj.read.assert_called_with("arguments")


def test_conversion():
    convertible = MagicMock()
    context = convertible.from_file.return_value
    obj = context.__enter__.return_value
    # without arguments
    res = to_dict(convertible)  # we test to_dict as generic for conversion
    convertible.from_file.assert_called_once()
    context.__enter__.assert_called_once()
    obj.to_dict.assert_called_once()
    assert res == obj.to_dict.return_value
    # with arguments
    res = to_dict(convertible, "arguments")
    obj.to_dict.assert_called_with("arguments")


def test_nonexisting_function_exception():
    class NoReadDefined:
        @contextmanager
        def from_file():
            yield NoReadDefined()

    with config.overwrite({"catch_exceptions_in_wrappers": False}):
        with pytest.raises(exception.NotImplemented):
            read(NoReadDefined)


def test_incorrect_argument():
    class ReadWithOneArgument:
        @contextmanager
        def from_file():
            yield ReadWithOneArgument()

        def read(self, expected):
            pass

    with config.overwrite({"catch_exceptions_in_wrappers": False}):
        with pytest.raises(exception.IncorrectUsage):
            read(ReadWithOneArgument)
        with pytest.raises(exception.IncorrectUsage):
            read(ReadWithOneArgument, "two", "arguments")
        with pytest.raises(exception.IncorrectUsage):
            read(ReadWithOneArgument, wrong="keyword")


def test_print():
    class DataImpl(_util.Data):
        def __init__(self, name="default"):
            self.name = name

        @classmethod
        def from_file(cls):
            return nullcontext(cls())

        def _repr_pretty_(self, p, cycle):
            p.text(f"{self.name} pretty")

        def _repr_html_(self):
            return f"{self.name} html"

    res, _ = _util.format_(DataImpl)
    assert {"text/plain": "default pretty", "text/html": "default html"}
    assert str(DataImpl) == "default pretty"
    data_impl = DataImpl("special")
    res, _ = _util.format_(data_impl)
    assert {"text/plain": "special pretty", "text/html": "special html"}
    assert str(data_impl) == "special pretty"
    for class_ in _classes:
        if hasattr(class_, "_repr_pretty_"):
            assert issubclass(class_, _util.Data)


def test_user_friendly_error_messages():
    class ErrorGenerator:
        @contextmanager
        def from_file():
            yield ErrorGenerator()

        def read(self, exception):
            raise exception(
                "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean "
                "commodo ligula eget dolor. Aenean massa.\n\nCum sociis natoque "
                "penatibus et magnis dis parturient montes, nascetur ridiculus mus. "
                "Donec quam felis, ultricies nec, pellentesque eu, pretium quis, sem. "
                "Nulla consequat massa quis enim. Donec pede justo, fringilla vel, "
                "aliquet nec, vulputate eget, arcu.\nend of message"
            )

    tests = {
        exception.Py4VaspError: """ -----------------------------------------------------------------------------
|                                                                             |
|     EEEEEEE  RRRRRR   RRRRRR   OOOOOOO  RRRRRR      ###     ###     ###     |
|     E        R     R  R     R  O     O  R     R     ###     ###     ###     |
|     E        R     R  R     R  O     O  R     R     ###     ###     ###     |
|     EEEEE    RRRRRR   RRRRRR   O     O  RRRRRR       #       #       #      |
|     E        R   R    R   R    O     O  R   R                               |
|     E        R    R   R    R   O     O  R    R      ###     ###     ###     |
|     EEEEEEE  R     R  R     R  OOOOOOO  R     R     ###     ###     ###     |
|                                                                             |
|     Py4VaspError: Lorem ipsum dolor sit amet, consectetuer adipiscing       |
|     elit. Aenean commodo ligula eget dolor. Aenean massa.                   |
|                                                                             |
|     Cum sociis natoque penatibus et magnis dis parturient montes,           |
|     nascetur ridiculus mus. Donec quam felis, ultricies nec,                |
|     pellentesque eu, pretium quis, sem. Nulla consequat massa quis          |
|     enim. Donec pede justo, fringilla vel, aliquet nec, vulputate eget,     |
|     arcu.                                                                   |
|     end of message                                                          |
|                                                                             |
|       ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----       |
|                                                                             |
 -----------------------------------------------------------------------------""",
        Exception: """ -----------------------------------------------------------------------------
|                     _     ____    _    _    _____     _                     |
|                    | |   |  _ \\  | |  | |  / ____|   | |                    |
|                    | |   | |_) | | |  | | | |  __    | |                    |
|                    |_|   |  _ <  | |  | | | | |_ |   |_|                    |
|                     _    | |_) | | |__| | | |__| |    _                     |
|                    (_)   |____/   \\____/   \\_____|   (_)                    |
|                                                                             |
|     Exception: Lorem ipsum dolor sit amet, consectetuer adipiscing          |
|     elit. Aenean commodo ligula eget dolor. Aenean massa.                   |
|                                                                             |
|     Cum sociis natoque penatibus et magnis dis parturient montes,           |
|     nascetur ridiculus mus. Donec quam felis, ultricies nec,                |
|     pellentesque eu, pretium quis, sem. Nulla consequat massa quis          |
|     enim. Donec pede justo, fringilla vel, aliquet nec, vulputate eget,     |
|     arcu.                                                                   |
|     end of message                                                          |
|                                                                             |
|     If you are not a developer, you should not encounter this problem.      |
|     Please submit a bug report.                                             |
|                                                                             |
 -----------------------------------------------------------------------------""",
    }
    for exception_, reference in tests.items():
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with pytest.raises(exception.StopExecution):
                read(ErrorGenerator, exception_)
        assert out.getvalue() == ""
        assert err.getvalue() == reference
