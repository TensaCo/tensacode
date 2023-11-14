from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
import functools
import inspect
from pathlib import Path
from typing import Any, Callable, ClassVar, Generic, Literal, Sequence, TypeVar
from uuid import uuid4
import attr
import loguru
from pydantic import Field
import typingx


import tensacode as tc
from tensacode.utils.decorators import Decorator, Default, dynamic_defaults
from tensacode.utils.oo import HasDefault, Namespace
from tensacode.utils.string import invokation
from tensacode.utils.user_types import (
    enc,
    T,
    R,
    atomic_types,
    container_types,
    composite_types,
    tree_types,
    tree,
)
from tensacode.utils.internal_types import nested_dict


class Engine(Generic[T, R], HasDefault, Namespace[R], ABC):
    #######################################
    ############# metadata ################
    #######################################

    T: ClassVar[type[T]] = T
    R: ClassVar[type[R]] = R

    class Object(Generic[T, R], ABC):
        def __tensorcode_encode__(
            self,
            depth_limit: int,
            engine: Engine,
        ) -> R:
            raise NotImplementedError

        @classmethod
        def __tensorcode_decode__(
            self,
            object_enc: R,
            depth_limit: int,
            engine: Engine,
        ) -> T:
            raise NotImplementedError

        def __tensorcode_get__(
            self,
            location: T,
            engine: Engine,
        ) -> T:
            raise NotImplementedError

        def __tensorcode_set__(
            self,
            location: T,
            value: T,
            engine: Engine,
        ) -> None:
            raise NotImplementedError

        # TODO: update these abstract methods to use the correct engine API

    class _HasEngine(ABC):
        _engine: ClassVar[Engine]

    class _EngineDecorator(Decorator, _HasEngine, ABC):
        pass

    @attr.s(auto_attribs=True)
    class trace(_EngineDecorator):
        trace_args = attr.ib(default=True)
        trace_result = attr.ib(default=True)

        def prologue(self, *a, **kw):
            if self.trace_args:
                self._engine.trace(...)  # TODO: implement this
            return super().prologue(*a, **kw)

        def epilogue(self, retval, *a, **kw):
            if self.trace_result:
                self._engine.trace(...)  # TODO: implement this
            return super().epilogue(retval, *a, **kw)

    @attr.s(auto_attribs=True)
    class encoded_args(_EngineDecorator):
        encode_args: bool = attr.ib(True)
        decode_retval: bool = attr.ib(True)

        def prologue(self, *a, **kw):
            if self.encode_args:
                # bind params to their values
                signature = inspect.signature(self.fn)
                bound_args = signature.bind(*a, **kw)
                bound_args.apply_defaults()
                bound_args = bound_args.arguments
                # encode the params that are annotated with `enc[...]`
                # TODO: implement this

            return super().prologue(*a, **kw)

        def epilogue(self, retval, *a, **kw):
            if self.decode_retval:
                # decode the retval if it is annotated with `enc[...]`
                ...  # TODO: implement this

            return super().epilogue(retval, *a, **kw)

    @attr.s(auto_attribs=True)
    class DefaultParam(Default, _HasEngine):
        initial_value: Any | None = attr.ib(default=None)
        initializer: Callable[[Engine], Any] | None = attr.ib(default=None)

        def __init__(self, initializer_or_initial_value: Any = None, /, **kw):
            if typingx.isinstance(self.default, Callable[[Engine], Any]):
                self.initializer = initializer_or_initial_value
            else:
                self.initial_value = initializer_or_initial_value
            self.kw = kw
            super().__init__(self.get)

        def get(self, *a, **kw):
            initial_val: Any
            if self.initial_value is not None:
                initial_val = self.initial_value
            elif self.initializer is not None:
                initial_val = self.initializer(self._engine)
            else:
                initial_val = None
            return self._engine.param(initial_val, **self.kw)

    #######################################
    ############### config ################
    #######################################

    params = {
        "hparams": {
            "defaults": (
                defaults := {
                    "depth_limit": 10,
                }
            ),
            "encode": {
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "decode": {
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "retrieve": {
                "count": 5,
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "store": {
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "query": {
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "modify": {
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "combine": {
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "split": {
                "num_splits": 2,
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "choice": {
                "threshold": 0.5,
                "randomness": 0.1,
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "decide": {
                "threshold": 0.5,
                "randomness": 0.1,
                "depth_limit": defaults["depth_limit"],
                "instructions": None,
            },
            "run": {
                "instructions": None,
            },
        },
    }

    #######################################
    ######## intelligence methods #########
    #######################################

    params: dict[str, Any] = {}

    def __init__(self):
        self._HasEngine._engine = self

    @encoded_args()
    def param(
        self, initial_value: enc[T] = None, name: str = None, qualname: str = None
    ) -> R:
        match name, qualname:
            case None, None:
                # TODO: I need to implement a use_state mechanism that tracks
                # the stack hierarchy and order of calling to make param calls idempotent
                name = ...
            case None, _:
                # keep qualname as is
                pass
            case _, None:
                qualname = self.qualpath + "." + name
            case _, _:
                # qualname overrides name
                pass
        self.params[qualname] = initial_value

    @encoded_args()
    @trace()
    def inform(self, message: enc[T]):
        pass

    @encoded_args()
    @trace()
    def chat(self, message: enc[T]) -> enc[T]:
        pass

    @trace()
    def self_reflect(self):
        pass

    @encoded_args()
    @trace()
    def reward(self, reward: enc[float]):
        pass

    @trace()
    def train(self):
        pass

    @abstractmethod
    @trace()
    def load(self, path: str | Path):
        pass

    @classmethod
    @abstractmethod
    def load(self, path: str | Path):
        pass

    #######################################
    ######## main operator methods ########
    #######################################

    def is_encoded(self, object: T | R) -> bool:
        return isinstance(object, R)

    def not_encoded(self, object: T | R) -> bool:
        return isinstance(object, T)

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def encode(
        self,
        object: T,
        /,
        depth_limit: int = DefaultParam(
            lambda engine: engine.encode.depth_limit,
            name="hparams.encode.depth_limit",
        ),
        instructions: enc[str] = DefaultParam(
            lambda engine: engine.encode.instructions,
            name="hparams.encode.instructions",
        ),
        **kwargs,
    ) -> R:
        try:
            return object.__tensorcode_encode__(object, self, **kwargs)
        except (NotImplementedError, AttributeError):
            pass
        self._encode(object)

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def decode(
        self,
        object_enc: R,
        /,
        depth_limit: int = DefaultParam(
            lambda engine: engine.decode.depth_limit,
            name="hparams.decode.depth_limit",
        ),
        instructions: enc[str] = DefaultParam(
            lambda engine: engine.decode.instructions,
            name="hparams.decode.instructions",
        ),
        **kwargs,
    ) -> T:
        try:
            return object.__tensorcode_decode__(object_enc, self)
        except (NotImplementedError, AttributeError):
            pass
        self._decode(object_enc)

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def retrieve(
        self,
        object: composite_types[T],
        /,
        location: T,
        count: int = DefaultParam(
            lambda engine: engine.retrieve.count,
            name="hparams.retrieve.count",
        ),
        allowed_glob: str = None,
        disallowed_glob: str = None,
        depth_limit: int = DefaultParam(
            lambda engine: engine.retrieve.depth_limit,
            name="hparams.retrieve.depth_limit",
        ),
        instructions: enc[str] = DefaultParam(
            lambda engine: engine.retrieve.instructions,
            name="hparams.retrieve.instructions",
        ),
        **kwargs,
    ) -> T:
        ...

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def store(
        self,
        object: composite_types[T],
        /,
        location: T,
        value: T = None,
        values: list[T] = None,
        allowed_glob: str = None,
        disallowed_glob: str = None,
        depth_limit: int = DefaultParam(
            lambda engine: engine.store.depth_limit,
            name="hparams.store.depth_limit",
        ),
        instructions: enc[str] = DefaultParam(
            lambda engine: engine.store.instructions,
            name="hparams.store.instructions",
        ),
        **kwargs,
    ):
        assert (
            value is None or values is None
        ), "Specify either value or values, not both"
        ...

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def query(
        self,
        object: T,
        /,
        query: T,
        depth_limit: int = DefaultParam(
            lambda engine: engine.query.depth_limit,
            name="hparams.query.depth_limit",
        ),
        instructions: enc[str] = DefaultParam(
            lambda engine: engine.query.instructions,
            name="hparams.query.instructions",
        ),
        **kwargs,
    ) -> T:
        ...

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def modify(
        self,
        object: T,
        /,
        depth_limit: int = DefaultParam(
            lambda engine: engine.modify.depth_limit,
            name="hparams.modify.depth_limit",
        ),
        instructions: enc[str] = DefaultParam(
            lambda engine: engine.modify.instructions,
            name="hparams.modify.instructions",
        ),
        **kwargs,
    ) -> T:
        pass

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def combine(
        self,
        objects: Sequence[T],
        /,
        depth_limit: int = DefaultParam(
            lambda engine: engine.combine.depth_limit,
            name="hparams.combine.depth_limit",
        ),
        instructions: enc[str] = DefaultParam(
            lambda engine: engine.combine.instructions,
            name="hparams.combine.instructions",
        ),
        **kwargs,
    ):
        ...

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def split(
        self,
        object: T,
        /,
        num_splits: int = DefaultParam(
            lambda engine: engine.split.num_splits,
            name="hparams.split.num_splits",
        ),
        depth_limit: int = DefaultParam(
            lambda engine: engine.split.depth_limit,
            name="hparams.split.depth_limit",
        ),
        instructions: enc[str] = DefaultParam(
            lambda engine: engine.split.instructions,
            name="hparams.split.instructions",
        ),
        **kwargs,
    ):
        pass

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def choice(
        self,
        conditions_and_functions: tuple[Callable[..., bool], Callable[..., T]],
        /,
        mode: Literal["first-winner", "last-winner"] = "first-winner",
        default_case_idx: int | None = None,
        threshold: float = DefaultParam(
            lambda engine: engine.hparams.choice.threshold,
            "hparams.choice.threshold",
        ),
        randomness: float = DefaultParam(
            lambda engine: engine.hparams.choice.randomness,
            "hparams.choice.randomness",
        ),
        depth_limit: int = DefaultParam(
            lambda engine: engine.hparams.choice.depth_limit,
            name="hparams.choice.depth_limit",
        ),
        instructions: enc[str] = DefaultParam(
            lambda engine: engine.hparams.choice.instructions,
            name="hparams.choice.instructions",
        ),
        **kwargs,
    ) -> T:
        match mode:
            case "first-winner":
                # evaluate the conditions in order
                # pick first one to surpass threshold
                # default to default_case or raise ValueError if no default_case
                pass
            case "last-winner":
                # evaluate all conditions
                # pick global max
                # default to default_case or raise ValueError if no default_case
                pass
            case _:
                raise ValueError(f"Invalid mode: {mode}")

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def decide(
        self,
        condition: T,
        if_true: Callable,
        if_false: Callable,
        *,
        threshold: float = DefaultParam(
            lambda engine: engine.hparams.choice.threshold,
            "hparams.choice.threshold",
        ),
        randomness: float = DefaultParam(
            lambda engine: engine.hparams.choice.randomness,
            "hparams.choice.randomness",
        ),
        depth_limit: int = DefaultParam(
            lambda engine: engine.hparams.choice.depth_limit,
            name="hparams.choice.depth_limit",
        ),
        instructions: enc[str] = DefaultParam(
            lambda engine: engine.hparams.choice.instructions,
            name="hparams.choice.instructions",
        ),
        **kwargs,
    ):
        # no if_true/False means we are decorating a function
        if if_true is None and if_false is None:

            def decorator(fn):
                @functools.wraps(fn)
                def wrapper(*args, **kwargs):
                    if_true = fn
                    if_false = lambda *a, **kw: None

                    return self.decide(
                        condition,
                        if_true=if_true,
                        if_false=if_false,
                        threshold=threshold,
                        randomness=randomness,
                        depth_limit=depth_limit,
                        **kwargs,
                    )

                return wrapper

            return decorator

        # otherwise, we are deciding a value
        return self.choice(
            [
                (condition, if_true),
                (lambda *a, **kw: not condition(*a, **kw), if_false),
            ],
            mode="last-winner",
            threshold=threshold,
            randomness=randomness,
            depth_limit=depth_limit,
            **kwargs,
        )

    @dynamic_defaults()
    @encoded_args()
    @trace()
    def run(
        self,
        instructions: enc[str] = DefaultParam(
            lambda engine: engine.run.instructions,
            name="hparams.run.instructions",
        ),
        /,
        **kwargs,
    ):
        pass

    #######################################
    ######## core operator methods ########
    ##### (subclasasaes override here) ####
    #######################################

    # TODO: make underscore protected version of some of the above methods (some are already complete by calling other methods)
    pass
