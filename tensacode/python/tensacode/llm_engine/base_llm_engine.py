from abc import ABC
from typing import Any, Generic
from langchain.chains.base import Chain
import inflect
from tensacode.base.base_engine import BaseEngine
from tensacode.utils.types import T, R


class BaseLLMEngine(Generic[T, R], BaseEngine[T, R], ABC):
    kernel: Chain
    p = inflect.engine()
