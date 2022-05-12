from py4vasp.data import _base, _trajectory


class PairCorrelation(_trajectory.DataTrajectory):
    "TODO"
    read = _base.RefinementDescriptor("_to_dict")

    def _to_dict(self):
        return {
            "distances": self._raw_data.distances[:],
            **self._read_data(),
        }

    def _read_data(self):
        return {
            label: self._raw_data.function[self._steps, index]
            for index, label in enumerate(self._raw_data.labels)
        }
