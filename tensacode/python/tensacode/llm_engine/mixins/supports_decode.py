from __future__ import annotations
from abc import ABC

from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import _DataclassT, dataclass
import functools
from functools import singledispatchmethod
import inspect
from pathlib import Path
import pickle
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generator,
    Generic,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    TypeVar,
)
from box import Box
from uuid import uuid4
import attr
from jinja2 import Template
import loguru
from glom import glom
from pydantic import Field
import typingx
import pydantic, sqlalchemy, dataclasses, attr, typing


import tensacode as tc
from tensacode.utils.decorators import (
    Decorator,
    Default,
    dynamic_defaults,
    is_attrs_instance,
    is_attrs_type,
    is_dataclass_instance,
    is_dataclass_type,
    is_namedtuple_instance,
    is_namedtuple_type,
    is_object_instance,
    is_type,
    is_pydantic_model_instance,
    is_pydantic_model_type,
    is_sqlalchemy_instance,
    is_sqlalchemy_model_type,
    overloaded,
)
from tensacode.utils.oo import HasDefault, Namespace
from tensacode.utils.string0 import render_invocation, render_stacktrace
from tensacode.utils.types import (
    enc,
    T,
    R,
    atomic_types,
    container_types,
    composite_types,
    tree_types,
    tree,
    DataclassInstance,
    AttrsInstance,
)
from tensacode.utils.internal_types import nested_dict
from tensacode.base.base_engine import BaseEngine
from tensacode.llm_engine.base_llm_engine import BaseLLMEngine
import tensacode.base.mixins as mixins


class SupportsDecodeMixin(
    Generic[T, R], BaseLLMEngine[T, R], mixins.SupportsDecodeMixin[T, R], ABC
):
    @abstractmethod
    def _decode(
        self,
        object_enc: R,
        type: type[T],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @abstractmethod
    def _decode_to_object(
        self,
        object_enc: R,
        type: type[T],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, pydantic.BaseModel))
    def _decode_to_pydantic_model_instance(
        self,
        object_enc: R,
        type: type[pydantic.BaseModel],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, NamedTuple))
    def _decode_to_namedtuple_instance(
        self,
        object_enc: R,
        type: type[NamedTuple],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: is_dataclass_type(type))
    def _decode_to_dataclass_instance(
        self,
        object_enc: R,
        type: type[DataclassInstance],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, type))
    def _decode_to_type(
        self,
        object_enc: R,
        type: type,
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, type[pydantic.BaseModel]))
    def _decode_to_pydantic_model_type(
        self,
        object_enc: R,
        type: type[type[pydantic.BaseModel]],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, type[NamedTuple]))
    def _decode_to_namedtuple_type(
        self,
        object_enc: R,
        type: type[type[NamedTuple]],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, type))
    def _decode_to_dataclass_type(
        self,
        object_enc: R,
        type: type[type],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, None))
    def _decode_to_none(
        self,
        object_enc: R,
        type: type[None],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, bool))
    def _decode_to_bool(
        self,
        object_enc: R,
        type: type[bool],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, int))
    def _decode_to_int(
        self,
        object_enc: R,
        type: type[int],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, float))
    def _decode_to_float(
        self,
        object_enc: R,
        type: type[float],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, complex))
    def _decode_to_complex(
        self,
        object_enc: R,
        type: type[complex],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, str))
    def _decode_to_str(
        self,
        object_enc: R,
        type: type[str],
        /,
        depth_limit: int | None = None,
        instructions: R | None = None,
        visibility: Literal["public", "protected", "private"] = "public",
        inherited_members: bool = True,
        force_inline: bool = False,
        **kwargs,
    ) -> T:
        raise NotImplementedError()
