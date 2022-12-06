import abc

from py4vasp._util import convert


class Mixin(abc.ABC):
    @abc.abstractmethod
    def to_graph(self):
        pass

    def plot(self):
        return self.to_graph()

    def to_plotly(self):
        return self.to_graph().to_plotly()

    def to_image(self, *, filename=None):
        fig = self.to_plotly()
        classname = convert.to_snakecase(self.__class__.__name__).strip("_")
        filename = filename if filename is not None else f"{classname}.png"
        fig.write_image(self._path / filename)
