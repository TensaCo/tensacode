from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from copy import deepcopy
import functools
from functools import singledispatchmethod
import inspect
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generator,
    Generic,
    Literal,
    Optional,
    Sequence,
    TypeVar,
)
from uuid import uuid4
import attr
import loguru
from pydantic import Field
from old.base_engine import FullEngine
from tensacode.base.old_engine import Engine
import typingx


import tensacode as tc
from tensacode.utils.decorators import Decorator, Default, dynamic_defaults
from tensacode.utils.oo import HasDefault, Namespace
from tensacode.utils.string0 import render_invocation, render_stacktrace
from tensacode.utils.types import (
    enc,
    atomic_types,
    container_types,
    composite_types,
    tree_types,
    tree,
)
from tensacode.utils.internal_types import nested_dict

YourObjectType = Literal["one", "two", "three"]  # probabbly `Any`
YourLatentType = Literal[1, 2, 3]


class ExampleEngineSubclass(FullEngine[YourObjectType, YourLatentType], ABC):
    #######################################
    ############### meta ##################
    #######################################

    T: ClassVar[type] = YourObjectType
    R: ClassVar[type] = YourLatentType

    # import or override these from the parent Engine class
    encoded_args = FullEngine.encoded_args
    trace = FullEngine.trace

    #######################################
    ############### config ################
    #######################################

    PARAM_DEFAULTS = {
        "forward_map": {"one": 1, "two": 2, "three": 3},
        "reverse_map": {1: "one", 2: "two", 3: "three"},
    }

    #######################################
    ######## intelligence methods #########
    #######################################

    @encoded_args()
    @trace()
    def chat(self, message: enc[YourObjectType]) -> enc[YourObjectType]:
        ...  # your implementation here

    @trace()
    def self_reflect(self):
        ...  # your implementation here

    @encoded_args()
    @trace()
    def reward(self, reward: enc[float]):
        ...  # your implementation here

    @trace()
    def train(self):
        ...  # your implementation here

    @trace()
    def save(self, path: str | Path):
        super().save(path)

    @trace()
    def load(self, path: str | Path):
        return super().load(path)

    #######################################
    ######## main operator methods ########
    #######################################

    # (Implemented in parent Engine class)

    #######################################
    ######## core operator methods ########
    ##### (subclasasaes override here) ####
    #######################################

    # use `singledispatchmethod` to overload on the type of the object
    # (only applies to methods with an `object` argument)

    @singledispatchmethod
    def _encode(
        self,
        object: YourObjectType,
        /,
        depth_limit: int,
        instructions: YourLatentType,
        **kwargs,
    ) -> YourLatentType:
        raise NotImplementedError("Sorry, this is only a toy example.")

    @_encode.register
    def _encode(
        self,
        object: str,
        /,
        depth_limit: int,
        instructions: YourLatentType,
        **kwargs,
    ) -> YourLatentType:
        return self.params["forward_map"][object]

    @singledispatchmethod
    def _decode(
        self,
        object_enc: YourLatentType,
        type: type[YourObjectType],
        /,
        depth_limit: int,
        instructions: YourLatentType,
        **kwargs,
    ) -> YourObjectType:
        raise NotImplementedError("Sorry, this is only a toy example.")

    @_decode.register
    def _decode(
        self,
        object_enc: YourLatentType,
        type: type[str],
        /,
        depth_limit: int,
        instructions: YourLatentType,
        **kwargs,
    ) -> str:
        return self.params["reverse_map"][object_enc]

    # @abstractmethod
    # def _retrieve(
    #     self,
    #     object: composite_types[T],
    #     /,
    #     count: int,
    #     allowed_glob: str,
    #     disallowed_glob: str,
    #     depth_limit: int,
    #     instructions: R,
    #     **kwargs,
    # ) -> T:
    #     ...

    # @abstractmethod
    # def _store(
    #     self,
    #     object: composite_types[T],
    #     /,
    #     values: list[T],
    #     allowed_glob: str,
    #     disallowed_glob: str,
    #     depth_limit: int,
    #     instructions: R,
    #     **kwargs,
    # ):
    #     ...

    # @abstractmethod
    # def _query(
    #     self,
    #     object: T,
    #     /,
    #     query: R,
    #     depth_limit: int,
    #     instructions: R,
    #     **kwargs,
    # ) -> R:
    #     raise NotImplementedError()

    # @abstractmethod
    # def _modify(
    #     self,
    #     object: T,
    #     /,
    #     depth_limit: int,
    #     instructions: R,
    #     **kwargs,
    # ) -> T:
    #     ...

    # @abstractmethod
    # def _combine(
    #     self,
    #     objects: Sequence[T],
    #     /,
    #     depth_limit: int,
    #     instructions: R,
    #     **kwargs,
    # ) -> T:
    #     ...

    # @abstractmethod
    # def _split(
    #     self,
    #     object: T,
    #     /,
    #     num_splits: int,
    #     depth_limit: int,
    #     instructions: R,
    #     **kwargs,
    # ) -> tuple[T]:
    #     ...

    # @abstractmethod
    # def _choice_first_winner(
    #     self,
    #     conditions_and_functions: tuple[Callable[..., bool], Callable[..., T]],
    #     /,
    #     default_case_idx: int | None,
    #     threshold: float,
    #     randomness: float,
    #     depth_limit: int,
    #     instructions: R,
    #     **kwargs,
    # ) -> T:
    #     ...

    # @abstractmethod
    # def _choice_last_winner(
    #     self,
    #     conditions_and_functions: tuple[Callable[..., bool], Callable[..., T]],
    #     /,
    #     default_case_idx: int | None,
    #     threshold: float,
    #     randomness: float,
    #     depth_limit: int,
    #     instructions: R,
    #     **kwargs,
    # ) -> T:
    #     ...

    # @abstractmethod
    # def _run(
    #     self,
    #     instructions: R,
    #     /,
    #     budget: Optional[float],
    #     **kwargs,
    # ) -> Any:
    #     ...

    # @abstractmethod
    # def _similarity(
    #     self,
    #     objects: tuple[T],
    #     /,
    #     depth_limit: int,
    #     instructions: R,
    #     **kwargs,
    # ) -> float:
    #     ...

    # @abstractmethod
    # def _predict(
    #     self,
    #     sequence: Sequence[T],
    #     /,
    #     steps: int,
    #     depth_limit: int,
    #     instructions: R,
    #     **kwargs,
    # ) -> Generator[T, None, None]:
    #     ...

    # @abstractmethod
    # def _correct(
    #     self,
    #     object: T,
    #     /,
    #     threshold: float,
    #     depth_limit: int,
    #     instructions: R,
    #     **kwargs,
    # ) -> T:
    #     ...

    # @abstractmethod
    # def _style_transfer(
    #     self,
    #     object: T,
    #     style: R,
    #     exemplar: T,
    #     /,
    #     depth_limit: int,
    #     instructions: R,
    #     **kwargs,
    # ) -> T:
    #     ...

    # @abstractmethod
    # def _semantic_transfer(
    #     self,
    #     object: T,
    #     semantics: R,
    #     exemplar: T,
    #     /,
    #     depth_limit: int,
    #     instructions: R,
    #     **kwargs,
    # ) -> T:
    #     ...
