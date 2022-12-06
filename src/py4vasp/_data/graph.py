import abc


class Mixin(abc.ABC):
    @abc.abstractmethod
    def func(self):
        pass
