from py4vasp.data import _base, _trajectory
import py4vasp._third_party.graph as _graph


class PairCorrelation(_trajectory.DataTrajectory):
    "TODO"
    read = _base.RefinementDescriptor("_to_dict")
    plot = _base.RefinementDescriptor("_to_plotly")

    def _to_dict(self):
        return {
            "distances": self._raw_data.distances[:],
            **self._read_data(),
        }

    def _to_plotly(self, selection="total"):
        dict_ = self._to_dict()
        series = _graph.Series(x=dict_["distances"], y=dict_[selection], name=selection)
        return _graph.Graph(series, xlabel="Distance (Å)", ylabel="Pair correlation")

    def _read_data(self):
        return {
            label: self._raw_data.function[self._steps, index]
            for index, label in enumerate(self._raw_data.labels)
        }
