from __future__ import annotations

import inspect
from typing import (
    Any,
    Union,
    Optional,
    TypeVar,
    Generic,
    Type,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
    Set,
    MutableSet,
    ByteString,
    MutableSequence,
)
from types import (
    NoneType,
    ModuleType,
    FunctionType,
    MethodType,
    GeneratorType,
    AsyncGeneratorType,
)
from types import FunctionType, MethodType, ModuleType
from abc import ABC, abstractmethod
from collections import namedtuple
import dataclasses
import typing

import typingx
import attr
import pydantic

K = str | int | None
T = TypeVar("T")


class enc(Generic[T], ABC):
    """
    Nonfunctional annotation.
    Indicates that a type is the encoded form of its generic parameter.
    Prefer annotating with the encoded type itself where possible
    since we can't enforce this constraint.
    """


R = TypeVar("R", bound=enc)


atomic_types = (
    bool
    | int
    | float
    | complex
    | str
    | bytes
    | bytearray
    | NoneType
    | None
    | Generic
    | TypeVar
)
container_types = (
    tuple
    | list
    | frozenset
    | set
    | Mapping
    | Sequence
    | Iterator
    | GeneratorType
    | AsyncGeneratorType
)
composite_types = (
    (namedtuple | tuple)  # see utils.is_namedtuple
    | dataclasses.dataclass
    | attr.s
    | pydantic.BaseModel
    | ModuleType
    | container_types
    | type
    | object
)
function_types = Callable | FunctionType | MethodType | classmethod | staticmethod

tree_types = atomic_types | container_types | composite_types | "tree"


class tree(Generic[T]):
    def __instancecheck__(self, __instance: Any) -> bool:
        return super().__instancecheck__(__instance) or typingx.isinstancex(
            __instance, tree_types
        )

    def __subclasscheck__(self, __subclass: type) -> bool:
        return super().__subclasscheck__(__subclass) or typingx.issubclassx(
            __subclass, tree_types
        )
