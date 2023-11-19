from abc import ABC
from functools import singledispatchmethod
from typing import Generator, Generic, Sequence
from old.base_engine import FullEngine

from tensacode.base.old_engine import Engine
from tensacode.utils.types import R, T, composite_types
import typingx


class ExampleOverload(Generic[T, R], ABC):
    """Example of how to overload the methods of the Engine class on specific types.

    Why overload?
    Overloading allows us to change the behavior of a function depending on the inputs. Engine supports overloading based on the *object*'s type as well as overloading based on the *engine* type. To overload based on the object's type, register your operator with the corresponding protected method of the Engine class. To overload based on the engine's type, implement a method like you see here and then implement and register a engine specific implementation below.

    Example:
    >>> class MyType:
            @singledispatchmethod
            @classmethod
            def __tc_is_encoded__(cls, engine: Engine, object: T | R) -> bool:
                return False

            @__tc_is_encoded__.register
            @classmethod
            def __tc_is_text_encoded__(cls, engine: LLMEngine, object: T | R) -> bool:
                return True
    >>> llm_engine.is_encoded(MyType())
    ... True
    >>> nn_engine.is_encoded(MyType())
    ... False
    """

    @singledispatchmethod
    @classmethod
    def __tc_is_encoded__(self, engine: FullEngine, object: T | R) -> bool:
        raise NotImplementedError()

    @singledispatchmethod
    @classmethod
    def __tc_encode__(
        self,
        engine: FullEngine,
        object: T,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        raise NotImplementedError()

    @singledispatchmethod
    @classmethod
    def __tc_decode__(
        self,
        engine: FullEngine,
        object_enc: R,
        type: type[T],
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @singledispatchmethod
    @classmethod
    def __tc_retrieve__(
        self,
        engine: FullEngine,
        object: composite_types[T],
        /,
        count: int,
        allowed_glob: str,
        disallowed_glob: str,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @singledispatchmethod
    @classmethod
    def __tc_store__(
        self,
        engine: FullEngine,
        object: composite_types[T],
        /,
        values: list[T],
        allowed_glob: str,
        disallowed_glob: str,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ):
        raise NotImplementedError()

    @singledispatchmethod
    @classmethod
    def __tc_query__(
        self,
        engine: FullEngine,
        object: T,
        /,
        query: R,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> R:
        raise NotImplementedError()

    @singledispatchmethod
    @classmethod
    def __tc_modify__(
        self,
        engine: FullEngine,
        object: T,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @singledispatchmethod
    @classmethod
    def __tc_combine__(
        self,
        engine: FullEngine,
        objects: Sequence[T],
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @singledispatchmethod
    @classmethod
    def __tc_split__(
        self,
        engine: FullEngine,
        object: T,
        /,
        num_splits: int,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> tuple[T]:
        raise NotImplementedError()

    @singledispatchmethod
    @classmethod
    def __tc_similarity__(
        self,
        engine: FullEngine,
        objects: tuple[T],
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> float:
        raise NotImplementedError()

    @singledispatchmethod
    @classmethod
    def __tc_predict__(
        self,
        engine: FullEngine,
        sequence: Sequence[T],
        /,
        steps: int,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> Generator[T, None, None]:
        raise NotImplementedError()

    @singledispatchmethod
    @classmethod
    def __tc_correct__(
        self,
        engine: FullEngine,
        object: T,
        /,
        threshold: float,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @singledispatchmethod
    @classmethod
    def __tc_style_transfer__(
        self,
        engine: FullEngine,
        object: T,
        style: R,
        exemplar: T,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> T:
        raise NotImplementedError()

    @singledispatchmethod
    @classmethod
    def __tc_semantic_transfer__(
        self,
        engine: FullEngine,
        object: T,
        semantics: R,
        exemplar: T,
        /,
        depth_limit: int,
        instructions: R,
        **kwargs,
    ) -> T:
        raise NotImplementedError()
