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
from tensacode.llm.llm_engine_base import LLMEngineBase
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
import tensacode.base.mixins as mixins


class SupportsCreateMixin(
    Generic[T, R], LLMEngineBase[T, R], mixins.SupportsCreateMixin[T, R], ABC
):
    # copied from MixinBase for aesthetic consistency
    trace = EngineBase.trace
    DefaultParam = EngineBase.DefaultParam
    encoded_args = EngineBase.encoded_args

    @dynamic_defaults()
    @trace()
    def create(
        self,
        object_enc: R,
        /,
        depth_limit: int = DefaultParam(qualname="hparams.create.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.create.instructions"),
        **kwargs,
    ) -> T:
        """
        Like decode, but also determines the type you want to create.
        """
