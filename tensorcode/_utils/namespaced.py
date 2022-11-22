from __future__ import annotations

from typing import Mapping
from typing_extensions import Self


class Namespaced:
    """Namespaced

    Seemlessly hijacks the `__new__` method of subclasses to ensure
    instances of `Namespaced` with identical `name`s are identical.

    Subclasses can implement desired behavior in
    ... __init__ for brand-new class instances
    ... __additional__ for another 'initialization' of the same instance
    """

    ALL: Mapping[str, object] = dict() # {name, obj}

    def __new__(cls: type[Self], *args, /, name: str = None, **kwargs) -> Self:
        if name is None and len(args) >= 1 and inspect.ismethod(args[-1]):
            # convenient when annotating a method 
            name = args[-1].__qualname__
        if name in Namespaced.ALL:
            self = Namespaced.ALL[name]
            self.__additional__(*args, **kwargs)
        else:
            self = super.__new__()
            Namespaced.ALL[name] = self
        return self

    def __additional__(self, *args, **kwargs):
        """Called when a class already exists but is being 'initialized' again"""
        pass
