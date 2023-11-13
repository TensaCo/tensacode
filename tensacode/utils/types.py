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
from abc import ABC, abstractmethod
from collections import namedtuple

import dataclasses
import attr
import pydantic

Tkey = str | int | None
T = TypeVar("T")
R = TypeVar("R")
TKeyGroupPair = tuple[Tkey, tuple[T, ...]]


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

tree_types = atomic_types | container_types | composite_types | "tree"


class Tree:
    _T: tree_types

    def __class_getitem__(cls, item):
        assert item in tree_types
        return type(f"tree[{item.__name__}]", (Tree,), {"_T": item})

    def __new__(cls, val: tree_types, /):
        if cls._T:
            return cls._T(val)
        else:
            return val

    def __instancecheck__(self, __instance: Any) -> bool:
        return super().__instancecheck__(__instance) or isinstance(__instance, self._T)

    def __subclasscheck__(cls: Tree, subclass: type) -> bool:
        return super().__subclasscheck__(subclass) or issubclass(subclass, cls._T)
