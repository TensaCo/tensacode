from __future__ import annotations

from abc import ABC
from typing import Generic, TypeVar

import tensacode as tc


T = TypeVar("T")
R = TypeVar("R")


class Object(Generic[T, R], ABC):
    def __tensorcode_encode__(self, engine: Engine, obj: T) -> R:
        raise NotImplementedError
    @classmethod
    def __tensorcode_decode__(self, engine: Engine, enc: R) -> T:
        raise NotImplementedError
    
class Engine(Generic[T, R]):
    def encode(self, obj: T) -> R:
        ...
    def decode(self, repr: R) -> T:
        ...
    def 