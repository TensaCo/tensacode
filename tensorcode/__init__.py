"""TensorCode

Copyright (c) 2022 Jacob F Valdez. Released under the MIT License

TODO: introduction
"""

from tensorcode.utils.registration_helpers import make_registration_helpers
from tensorcode.utils.stacked import stacked


DEFAULT_MODEL = DefaultModel()
CURRENT_MODEL = DEFAULT_MODEL

register, deregister, _stack = make_registration_helpers(__all__, __dict__)


# todo, somehow link to the current top-level model, but don't try to rewrite the signature
# save, load, train, reward, add_loss, encode, decode, select
# basically, whatever Model method have an @model_export annotation
save = _stack['CURRENT_MODEL'][-1].save




__all__ = ['CURRENT_MODEL', ]
