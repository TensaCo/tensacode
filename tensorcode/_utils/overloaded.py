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
    def __additional__(self, condition, method):
        self.registry.register(condition, method)
    overload = __additional__
    def __call__(self, *args, **kwargs):
        return self.registry.first(*args, **kwargs)
