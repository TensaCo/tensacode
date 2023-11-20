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


class SupportsSimilarityMixin(Generic[T, R], EngineBase[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = EngineBase.trace
    DefaultParam = EngineBase.DefaultParam
    encoded_args = EngineBase.encoded_args

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def similarity(
        self,
        objects: tuple[T],
        /,
        depth_limit: int = DefaultParam(qualname="hparams.similarity.depth_limit"),
        instructions: enc[str] = DefaultParam(
            qualname="hparams.similarity.instructions"
        ),
        **kwargs,
    ) -> float:
        """
        Calculates the similarity between the given objects.

        Args:
            objects (tuple[T]): The objects to compare.
            depth_limit (int, optional): The maximum depth to explore. Defaults to hparams.similarity.depth_limit.
            instructions (enc[str], optional): Encoded instructions for the engine. Defaults to hparams.similarity.instructions.

        Returns:
            float: The similarity score between the objects.
        """
        try:
            return type(objects[0]).__tc_similarity__(
                self,
                objects,
                depth_limit=depth_limit,
                instructions=instructions,
                **kwargs,
            )
        except (NotImplementedError, AttributeError):
            pass

        return self._similarity(
            objects, depth_limit=depth_limit, instructions=instructions, **kwargs
        )

    @abstractmethod
    def _similarity(
        self,
        objects: tuple[T],
        /,
        depth_limit: int | None,
        instructions: R | None,
        **kwargs,
    ) -> float:
        raise NotImplementedError()
