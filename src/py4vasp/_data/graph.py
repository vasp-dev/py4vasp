import abc


class Mixin(abc.ABC):
    @abc.abstractmethod
    def to_graph(self):
        pass

    def plot(self):
        return self.to_graph()