import tensorcode as tc

# the global module acts like a Model
# each Model provides most of the API that the global module does
# (Of course, the Model's API only uses it;
# The global API uses whatever Model happens to be on top)
with tc.Model(...) as m:
  m.encode(...)
  tc.encode(...)
  m.decode(...)
  tc.decode(...)
  m.select(...)
  tc.select(...)

################
class TensorCode_API:
  def encode(todo): pass
  def decode(todo): pass
  def select(todo): pass

class Model(TensorCode_API):
  ...

# in tensorcode.__init__.py:
...
for name, fn in TensorCode_API.__dict__.items():
  export(name, fn)

##################

# No

# `Model`s should implement their functions, and they should stack_export them to the namespace
