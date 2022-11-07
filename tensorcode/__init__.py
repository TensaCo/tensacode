"""TensorCode

Copyright (c) 2022 Jacob F Valdez. Released under the MIT License

TODO: introduction
"""

from tensorcode.models.model import Model
from tensorcode.operations.operation import Operation
from tensorcode.utils.export_helpers import make_export_helpers
from tensorcode.utils.stacked import stacked


encode: Encode
decode: Decoder
select: Select

current_model: Model









DEFAULT_MODEL = DefaultModel()
CURRENT_MODEL = DEFAULT_MODEL

register, deregister, _stack = make_export_helpers(__all__, __dict__)


# todo, somehow link to the current top-level model, but don't try to rewrite the signature
# save, load, train, reward, add_loss, encode, decode, select
# basically, whatever Model method have an @model_export annotation
save = _stack['CURRENT_MODEL'][-1].save




__all__ = ['CURRENT_MODEL', ]
