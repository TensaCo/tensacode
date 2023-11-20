from abc import ABC
from typing import Any, Generic
from langchain.chains.base import Chain
import inflect
from tensacode.base.engine_base import EngineBase
from tensacode.utils.types import T, R


class MLLMEngineBase(Generic[T, R], EngineBase[T, R], ABC):
    pass
