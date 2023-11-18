from abc import ABC
from tensacode.base.engine import Engine
from langchain.chains.base import Chain

class LLMEngine(Engine, ABC):
    chain: BaseChain
