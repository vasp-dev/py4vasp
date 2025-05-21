# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base, slice_
from py4vasp._third_party import graph
from py4vasp._util import import_, select


class ElectronPhononBandgapInstance:
    "Placeholder for electron phonon band gap"

    def __init__(self, parent, index):
        self.parent = parent
        self.index = index

    def __str__(self):
        return "electron phonon band gap %d" % self.index

    def _get_data(self, name):
        return self.parent._get_data(name, self.index)

    def _get_scalar(self, name):
        return self.parent._get_scalar(name, self.index)

    def to_graph(self, selection):
        tree = select.Tree.from_selection(selection)
        series = []
        for selection in tree.selections():
            y = self._get_data(selection[0])
            x = self._get_data("temperatures")
            series.append(graph.Series(x, y, label=selection[0]))
        return graph.Graph(series, ylabel="energy (eV)", xlabel="Temperature (K)")

    def to_dict(self):
        _dict = {
            "nbands_sum": self._get_scalar("nbands_sum"),
            "direct": self._get_data("direct"),
            "fundamental": self._get_data("fundamental"),
            "temperatures": self._get_data("temperatures"),
        }
        return _dict


class ElectronPhononBandgap(base.Refinery):
    @base.data_access
    def __str__(self):
        return "electron phonon bandgap"

    @base.data_access
    def _get_data(self, name, index):
        return getattr(self._raw_data, name)[index][:]

    @base.data_access
    def _get_scalar(self, name, index):
        return getattr(self._raw_data, name)[index][()]

    @base.data_access
    def __getitem__(self, key):
        # TODO add logic to select instances
        return ElectronPhononBandgapInstance(self, key)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @base.data_access
    def _get_data(self, name, index):
        return getattr(self._raw_data, name)[index][:]

    @base.data_access
    def _get_scalar(self, name, index):
        return getattr(self._raw_data, name)[index][()]

    @base.data_access
    def __len__(self):
        return len(self._raw_data.valid_indices)
