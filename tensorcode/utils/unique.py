SCOPE_DELIMINATOR = '/'
_NAMESPACE = {}
def unique(name: str, namespace=_NAMESPACE):
  if name not in _NAMESPACE:
    _NAMESPACE[name] = 0
    return name
  else:
    _NAMESPACE[name] += 1
    return f'{name}{SCOPE_DELIMINATOR}{_NAMESPACE[name]}'
