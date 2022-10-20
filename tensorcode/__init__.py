"""TensorCode

Copyright (c) 2022 Jacob F Valdez. Released under the MIT License

TODO: introduction
"""



# TODO: imports. The imports should export their members

from tensorcode.utils.registration_helpers import make_registration_helpers
from tensorcode.utils.stacked import stacked


DEFAULT_MODEL = DefaultModel()
CURRENT_MODEL = DEFAULT_MODEL

register, deregister = make_registration_helpers(__all__, __dict__)