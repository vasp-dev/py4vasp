class Kpoints:
    def __init__(self, raw_kpoints):
        self._raw = raw_kpoints

    def read(self):
        return {
            "mode": self._raw.mode,
            "coordinates": self._raw.coordinates[:],
            "weights": self._raw.weights[:],
            "labels": None,
        }
