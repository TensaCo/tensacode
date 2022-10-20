def stacked(_dict):

  stack = dict()

  def push(key, val):
    if key not in stack: stack[key] = list()
    stack[key].push(val)
    _dict[key] = val

  def pop(key):
    val = stack[key].pop()
    if len(stack[key]) == 0: _dict.pop(key)
    else: _dict[key] = stack[key][-1]
    return val

  return push, pop