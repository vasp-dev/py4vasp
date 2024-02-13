# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import abc


class Mixin(abc.ABC):
    @abc.abstractmethod
    def to_view(self, *args, **kwargs):
        pass

    def plot(self):
        return self.to_view()

    def to_ngl(self):
        return self.to_view().to_ngl()
