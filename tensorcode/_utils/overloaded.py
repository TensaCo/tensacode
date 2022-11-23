import inspect

from tensorcode._utils.registry import Registry, HighestScore
from tensorcode._utils.namespaced import Namespaced


class Overloaded(Namespaced):
    """ Used for complex method overloading like so:

    ```
    class Foo:
        @Overloaded
        def default_handler

    ```
    """

    registry = Registry(HighestScore(), self.__call__)

    def __new__(cls: type[Self], name: str, *args, **kwargs) -> Self:
        if name is None and len(args) >= 1 and inspect.ismethod(args[0]):
            name = args[0].__qualname__
        return super().__new__(name=name, *args, **kwargs)
    def __post_init__(self, name, fn):
        self.registry.register(condition, method)
    overload = __additional__
    def __call__(self, *args, **kwargs):
        return self.registry.first(*args, **kwargs)
