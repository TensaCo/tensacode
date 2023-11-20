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
from tensacode.base.base_engine import BaseEngine


class SupportsDecideMixin(Generic[T, R], BaseEngine[T, R], ABC):
    # copied from MixinBase for aesthetic consistency
    trace = BaseEngine.trace
    DefaultParam = BaseEngine.DefaultParam
    encoded_args = BaseEngine.encoded_args

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def decide(
        self,
        condition: T,
        if_true: Callable = lambda *a, **kw: True,
        if_false: Callable = lambda *a, **kw: False,
        *,
        threshold: float = DefaultParam("hparams.choice.threshold"),
        randomness: float = DefaultParam("hparams.choice.randomness"),
        depth_limit: int = DefaultParam(qualname="hparams.choice.depth_limit"),
        instructions: enc[str] = DefaultParam(qualname="hparams.choice.instructions"),
        **kwargs,
    ):
        """
        Makes a decision based on the provided condition. If the condition is met, the `if_true` function is called. Otherwise, the `if_false` function is called.

        Can be called in two ways:
        - `fn_result = engine.decide(condition, if_true=..., if_false=..., ...)`: in this case, you define the true and false branches as separate functions and pass them in as arguments. `engine.decide` then calls the appropriate function based on the condition.
        - `bool_result = engine.decide(condition, ...)`: in this case, you don't define the true and false branches, and `engine.decide` returns a boolean value based on the condition.

        Args:
            condition (T): The condition to evaluate.
            if_true (Callable): The function to call if the condition is met.
            if_false (Callable): The function to call if the condition is not met.
            threshold (float): The threshold for making the decision.
            randomness (float): The randomness factor in the decision making.
            depth_limit (int): The maximum depth to which the decision process should recurse.
            instructions (enc[str]): Additional instructions to the decision algorithm.
            **kwargs: Additional keyword arguments that might be needed for specific decision algorithms.

        Returns:
            The result of the `if_true` function if the condition is met, or the result of the `if_false` function if the condition is not met.

        Example:
            >>> engine = Engine()
            >>> class Person:
            ...    name: str
            ...    bio: str
            ...    thoughts: list[str]
            ...    friends: list[Person]
            >>> john, teyoni, huimin = ... # create people
            >>> person = engine.retrieve(john, instructions="find john's least favorite friend")
            ... # based on John's friends, decide if he is a jerk or not
            >>> engine.decide(person, if_true=lambda: print("John is a jerk"), if_false=lambda: print("John is a nice guy"))
            ... John is a jerk
        """
        return self.choice(
            [
                (condition, if_true),
                (lambda *a, **kw: not condition(*a, **kw), if_false),
            ],
            mode="last-winner",
            threshold=threshold,
            randomness=randomness,
            depth_limit=depth_limit,
            instructions=instructions,
            **kwargs,
        )
