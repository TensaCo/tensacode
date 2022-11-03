import code
from typing import Callable, Dict


class Operation:

  OP_NAME: str # affirmative command verb, eg, "encode", "select"
  REGISTRY: Dict[type, Callable]

  def __call__(self, *args, **kwargs):
    pass

  def add_exception(self, type, handler):
    pass

encoder.add_exception(Callable, str)
encoder.add_exception(Callable, (int,float))
encoder.add_exception(Callable, object)
encoder.add_exception(Callable, your_graph_code_bert)
encoder.add_exception(Callable, your_graph_code_bert)
encoder.add_exception(Callable, your_graph_code_bert)
