import abc

from py4vasp._util import convert


class Mixin(abc.ABC):
    @abc.abstractmethod
    def to_graph(self, *args, **kwargs):
        pass

    def plot(self, *args, **kwargs):
        return self.to_graph(*args, **kwargs)

    def to_plotly(self, *args, **kwargs):
        return self.to_graph(*args, **kwargs).to_plotly()

    def to_image(self, *args, filename=None, **kwargs):
        fig = self.to_plotly(*args, **kwargs)
        classname = convert.to_snakecase(self.__class__.__name__).strip("_")
        filename = filename if filename is not None else f"{classname}.png"
        fig.write_image(self._path / filename)
