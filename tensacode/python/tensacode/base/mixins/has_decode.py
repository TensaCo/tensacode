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
from old.base_engine import Engine
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
from tensacode.utils.string import render_invocation, render_stacktrace
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
from tensacode.base.mixins.mixin_base import MixinBase


class HasDecode(Generic[T, R], MixinBase[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = MixinBase.trace
    DefaultParam = MixinBase.DefaultParam
    encoded_args = MixinBase.encoded_args

    @dynamic_defaults()
    @trace()
    def decode(
        self,
        object_enc: R,
        type: type[T],
        /,
        depth_limit: int = DefaultParam(qualname="hparams.decode.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.decode.instructions"),
        **kwargs,
    ) -> T:
        """
        Decodes an encoded representation of an object back into its original form or into a different form.

        One of the powerful features of this function is its ability to decode into a different type or modality than the original object. This is controlled by the `type` argument. For example, you could encode a text document into a vector representation, and then decode it into a different language or a summary.

        Args:
            object_enc (R): The encoded representation of the object to be decoded.
            type (type[T]): The expected type of the decoded object. This is used to guide the decoding process. It doesn't have to match the original type of the object before encoding.
            depth_limit (int): The maximum depth to which the decoding process should recurse. This is useful for controlling the complexity of the decoding, especially for deeply nested structures. Default is set in the engine's parameters.
            instructions (enc[str]): Additional instructions to the decoding algorithm. This could be used to customize the decoding process, for example by specifying certain features to focus on or ignore.
            **kwargs: Additional keyword arguments that might be needed for specific decoding algorithms.

        Returns:
            T: The decoded object. The exact type and structure of this depends on the decoding algorithm used and the `type` argument.

        Examples:
            >>> engine = Engine()
            >>> encoded_obj = <encoded representation of an object>
            >>> decoded_obj = engine.decode(encoded_obj, type=NewObjectType)
            >>> print(decoded_obj)
            # Output: <decoded representation of the object in the new type>
        """

        try:
            return type(object_enc).__tc_decode__(
                self,
                object_enc,
                type,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        self._decode(
            object_enc,
            type,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @abstractmethod
    def _decode(
        self,
        object_enc: R,
        type: type[T],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @abstractmethod
    def _decode_to_object(
        self,
        object_enc: R,
        type: type[T],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, pydantic.BaseModel))
    def _decode_to_pydantic_model_instance(
        self,
        object_enc: R,
        type: type[pydantic.BaseModel],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, NamedTuple))
    def _decode_to_namedtuple_instance(
        self,
        object_enc: R,
        type: type[NamedTuple],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: is_dataclass_type(type))
    def _decode_to_dataclass_instance(
        self,
        object_enc: R,
        type: type[DataclassInstance],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, type))
    def _decode_to_type(
        self,
        object_enc: R,
        type: type,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, type[pydantic.BaseModel]))
    def _decode_to_pydantic_model_type(
        self,
        object_enc: R,
        type: type[type[pydantic.BaseModel]],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, type[NamedTuple]))
    def _decode_to_namedtuple_type(
        self,
        object_enc: R,
        type: type[type[NamedTuple]],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, type))
    def _decode_to_dataclass_type(
        self,
        object_enc: R,
        type: type[type],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, None))
    def _decode_to_none(
        self,
        object_enc: R,
        type: type[None],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, bool))
    def _decode_to_bool(
        self,
        object_enc: R,
        type: type[bool],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, int))
    def _decode_to_int(
        self,
        object_enc: R,
        type: type[int],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, float))
    def _decode_to_float(
        self,
        object_enc: R,
        type: type[float],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, complex))
    def _decode_to_complex(
        self,
        object_enc: R,
        type: type[complex],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @_decode.overload(lambda type: issubclass(type, str))
    def _decode_to_str(
        self,
        object_enc: R,
        type: type[str],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()
