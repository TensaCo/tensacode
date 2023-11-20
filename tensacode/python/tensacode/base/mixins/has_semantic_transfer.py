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
from old.base_engine import FullEngine
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
from tensacode.base.engine_base import EngineBase


class SupportsSemanticTransferMixin(Generic[T, R], EngineBase[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = EngineBase.trace
    DefaultParam = EngineBase.DefaultParam
    encoded_args = EngineBase.encoded_args

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def semantic_transfer(
        self,
        object: T,
        semantics: enc[T] = None,
        exemplar: T = None,
        /,
        depth_limit: int = DefaultParam(
            qualname="hparams.semantic_transfer.depth_limit",
        ),
        instructions: enc[str] = DefaultParam(
            qualname="hparams.semantic_transfer.instructions",
        ),
        **kwargs,
    ) -> T:
        """
        Performs semantic transfer on the given object.

        Args:
            object (T): The object to perform semantic transfer on.
            semantics (enc[T], optional): The semantics to transfer. If not provided, an exemplar must be given. Defaults to None.
            exemplar (T, optional): An exemplar object to guide the semantic transfer. If not provided, a semantics must be given. Defaults to None.
            depth_limit (int, optional): The maximum depth to explore for semantic transfer. Defaults to engine.correct.depth_limit.
            instructions (enc[str], optional): Encoded instructions for the engine. Defaults to engine.correct.instructions.
            **kwargs: Additional keyword arguments.

        Returns:
            T: The object after semantic transfer.
        """
        try:
            return type(object).__tc_semantic_transfer__(
                self,
                object,
                semantics=semantics,
                exemplar=exemplar,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        return self._semantic_transfer(
            object,
            semantics=semantics,
            exemplar=exemplar,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )

    @abstractmethod
    def _semantic_transfer(
        self,
        object: T,
        semantics: R,
        exemplar: T,
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> T:
        raise NotImplementedError()
