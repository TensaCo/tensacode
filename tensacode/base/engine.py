from abc import ABC
from typing import Generic, TypeVar


T = TypeVar("T")
R = TypeVar("R")


class CleanedFormat(Generic[T, R], ABC):
    def __tensorcode_render__(self, modality: tc.Modality) -> R:
        raise NotImplementedError
    
class Engine(Generic[T, R]):
    def encode(self, obj: T) -> R:
        ...
    def decode(self, repr: R) -> T:
        ...
    def 