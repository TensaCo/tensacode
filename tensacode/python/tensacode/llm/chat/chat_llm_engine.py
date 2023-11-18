from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import functools
from functools import singledispatchmethod
import inspect
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    ClassVar,
    Generator,
    Generic,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
)
from uuid import uuid4
import attr
import loguru
from pydantic import BaseModel, Field
from tensacode.base.base_engine import BaseEngine
from tensacode.base.engine import Engine
import typingx
from langchain.chat_models.base import BaseChatModel

import tensacode as tc
from tensacode.utils.decorators import Decorator, Default, dynamic_defaults, overloaded
from tensacode.utils.oo import HasDefault, Namespace
from tensacode.utils.string import render_invocation, render_stacktrace
from tensacode.utils.user_types import (
    enc,
    atomic_types,
    container_types,
    composite_types,
    function_types,
    tree_types,
    tree,
)
from tensacode.utils.internal_types import nested_dict

T = Any
R = str


class BaseChatLLMEngine(Engine[T, R], ABC):
    #######################################
    ############### meta ##################
    #######################################

    T: ClassVar[type] = T
    R: ClassVar[type] = R

    # import or override these from the parent Engine class
    encoded_args = Engine.encoded_args
    trace = Engine.trace

    #######################################
    ############### config ################
    #######################################

    # TODO: make params into a BaseModel
    PARAM_DEFAULTS = {
        "chat_model": BaseChatModel,
    }

    #######################################
    ######## intelligence methods #########
    #######################################

    @encoded_args()
    @trace()
    def chat(self, message: enc[T]) -> enc[T]:
        ...  # TODO: make a chatbot with the interaction data

    @trace()
    def self_reflect(self):
        ...  # TODO: make a reflexion agent with the interaction data

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

    @overloaded  # TODO: i need to maek it able to have its overloadeds subclasses and make it lookup by overload name and clsself key instead of exposing to the parent
    def _encode(
        self,
        object: T,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        raise NotImplementedError("Sorry, this is only a toy example.")

    @_encode.overload(lambda object: isinstance(object, atomic_types))
    def _encode(
        self,
        object: atomic_types,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return str(object)

    @_encode.overload(lambda object: isinstance(object, bool))
    def _encode_bool(
        self,
        object: bool,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return str(object)

    @_encode.overload(lambda object: isinstance(object, atomic_types))
    def _encode(
        self,
        object: atomic_types,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return str(object)

    @_encode.overload(lambda object: isinstance(object, atomic_types))
    def _encode(
        self,
        object: atomic_types,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return str(object)

    @_encode.overload(lambda object: isinstance(object, atomic_types))
    def _encode(
        self,
        object: atomic_types,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return str(object)

    @_encode.register
    def _encode(
        self,
        object: function_types,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return

    @_encode.register
    def _encode(
        self,
        object: Sequence[T],
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        individual_encodings = [
            self._encode(
                item,
                depth_limit=depth_limit - 1,
                instructions=instructions,
                **kwargs,
            )
            for item in object
        ]
        composite_encoding = ...  # make a chat completion
        return composite_encoding

    @_encode.register
    def _encode(
        self,
        object: Mapping[Any, T],
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        individual_encodings = {
            key: self._encode(
                value,
                depth_limit=depth_limit - 1,
                instructions=instructions,
                **kwargs,
            )
            for key, value in object.items()
        }
        composite_encoding = ...  # make a chat completion
        return composite_encoding

    @_encode.register
    def _encode(
        self,
        object: Iterator[T],
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return object

    @_encode.register
    def _encode(
        self,
        object: AsyncIterator[T],
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return object

    @_encode.register
    def _encode(
        self,
        object: namedtuple,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return object

    @_encode.register
    def _encode(
        self,
        object: dataclass,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return object

    @_encode.register
    def _encode(
        self,
        object: BaseModel,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return object

    @_encode.register
    def _encode(
        self,
        object: ModuleType,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return object

    @_encode.register
    def _encode(
        self,
        object: type,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return object

    @_encode.register
    def _encode(
        self,
        object: object,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        return object

    @singledispatchmethod
    def _decode(
        self,
        object_enc: R,
        type: type[T],
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> T:
        raise NotImplementedError("Sorry, this is only a toy example.")

    @_decode.register
    def _decode(
        self,
        object_enc: R,
        type: type[str],
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> str:
        return object_enc

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
