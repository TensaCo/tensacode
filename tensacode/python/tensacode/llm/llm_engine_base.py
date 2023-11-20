from abc import ABC
from typing import Any
from langchain.chains.base import Chain
import inflect
from tensacode.base.engine_base import EngineBase
from tensacode.mllm.mllm_engine_base import MLLMEngineBase


class LLMEngineBase(MLLMEngineBase[Any, str], ABC):
    kernel: Chain
    p = inflect.engine()
