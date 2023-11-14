from abc import ABC
from contextlib import contextmanager
from typing import Any, ClassVar, Generic, Self
from uuid import uuid4
from attr import attr
import glom

from tensacode.utils.internal_types import nested_dict
from tensacode.utils.user_types import R


class HasDefault(ABC):
    _current_stack: ClassVar[list[Self]] = []

    @classmethod
    def get_current(cls) -> Self:
        return cls._current_stack[-1]

    @contextmanager
    def as_default(self):
        self._current_stack.append(self)
        yield
        self._current_stack.pop()


@attr.s(auto_attribs=True)
class Namespace(Generic[R], ABC):
    namespace_stack: list[str] = ["tensacode"]

    @property
    def qualpath(self) -> str:
        return ".".join(self.namespace_stack)

    @contextmanager
    def namespace(self, name: str = None):
        name = name or uuid4().hex
        self.namespace_stack.append(name)
        yield
        _name = self.namespace_stack.pop()
        assert _name == name, f"Corrupted namespace stack: Expected {name}; got {_name}"
