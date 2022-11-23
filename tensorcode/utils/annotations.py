from __future__ import annotations

import inspect
from typing import TypeVar, Generic, Protocol

T = TypeVar('T')

class like(Generic[T]):
    """Indicates that this types does not need to match exactly"""
    pass
class enc(Generic[T]):
    """Indicates that this type is an encoded form of T"""
    pass

class SupportsEncode(Protocol, Generic[T]):
    def encode(self, object: T, *args, **kwargs) -> enc[T]: ...

def encode_args(fn, *,
        ignore_arg_indeces: list[int],
        ignore_kwarg_names: list[str]):
    """
    Wraps with function that encodes arguments for params annotated with `rep[T]`
    if the arg is a `T` (raises error if arg not of T or enc[T]).
    """
    def _fn(self: SupportsEncode, *_args, **_kwargs):
        args = []
        kwargs = {}



        fn(self, *args, **kwargs)
    return _fn
