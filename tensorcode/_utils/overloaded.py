from tensorcode._utils.registry import Registry, HighestScore
from tensorcode._utils.namespaced import Namespaced


class Overloaded(Namespaced):
    registry = Registry(HighestScore(), self.__call__)
    def __additional__(self, condition, method):
        self.registry.register(condition, method)
    def __call__(self, *args, **kwargs):
        return self.registry.first(*args, **kwargs)
