# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from pathlib import Path


class InputFile:
    def __init__(self, path):
        self._path = path

    @classmethod
    def from_string(cls, string, path=None):
        """Generate the file from a given string and store it.

        If no path is provided, the content of the file is stored in memory otherwise
        it is stored in the path.

        Parameters
        ----------
        string : str
            Content of the file.
        path : str or Path
            If provided should define where the file is stored.
        """
        obj = cls(path)
        obj.write(string)
        return obj

    def print(self):
        "Write the contents of the file to screen."
        print(self)

    def write(self, string):
        "Store the given string in the file."
        if self._path is not None:
            self._write_to_file(string)
        else:
            self._write_to_memory(string)

    def read(self):
        "Return the content of the file as a string."
        if self._path is not None:
            return self._read_from_file()
        else:
            return self._read_from_memory()

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def _write_to_file(self, string):
        path = Path(self._path) / self.__class__.__name__
        with open(path, "w", encoding="utf-8") as file:
            file.write(string)

    def _write_to_memory(self, string):
        self._content = string

    def _read_from_file(self):
        path = Path(self._path) / self.__class__.__name__
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    def _read_from_memory(self):
        return self._content

    def __str__(self):
        return self.read()
