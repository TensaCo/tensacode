from enum import Enum
from typing import List

from tensorcode import pop_export, push_export
from tensorcode.operations.operation import Operation
from tensorcode.utils.inject_subclasses import inject_epilog_into_subclasses
from tensorcode.utils.unique import unique


class Model:

  DEFAULT_MODEL = "default_model"
  CURRENT_MODEL = "current_model"
  SAVE = "save"
  LOAD = "load"

  @property
  def operations(self) -> List[Operation]:
    return []

  def __init__(self, fn, input_spec, output_spec, context_name):

    self.fn = fn
    self.input_spec = input_spec
    self.output_spec = output_spec
    self.context_name = unique(context_name)

    # if I'm being subclassed, inject my context manager into all subclass methods
    @inject_epilog_into_subclasses(self.__name__, '__init__')
    def with_context(method, *args, **kwargs):
      with self:
        method(*args, **kwargs)

  def __enter__(self):
    for operation in self.operations:
      push_export(operation.OP_NAME, operation.handler)
    push_export(Model.CURRENT_MODEL, self)
    push_export(Model.SAVE, self.save)
    push_export(Model.LOAD, self.load)

    return self

  def __exit__(self, exc_type, exc_value, traceback):
    for operation in self.operations:
      pop_export(operation.OP_NAME, operation.handler)
    pop_export(Model.CURRENT_MODEL)
    pop_export(Model.SAVE)
    pop_export(Model.LOAD)

  # saving
  def save(self, path):
    pass

  # loading
  def load(self, path):
    pass