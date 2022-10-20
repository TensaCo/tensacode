from tensorcode.utils.stacked import stacked

def make_registration_helpers(__all__, __dict__):

  _push, _pop = stacked(__dict__)

  def push(key, val):
    _push(key, val)
    __all__.append(key)
  def pop(key):
    _pop(key)
    __all__.remove(key)

  return push, pop