from abc import ABC
from tensacode.base.engine import Engine
from langchain.chains.base import Chain
import inflect


class LLMEngine(Engine, ABC):
    kernel: BaseChain
    p = inflect.engine()
