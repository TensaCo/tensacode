from abc import ABC
from langchain.chains.base import Chain
import inflect
from tensacode.base.engine_base import EngineBase


class LLMEngineBase(EngineBase, ABC):
    kernel: Chain
    p = inflect.engine()
