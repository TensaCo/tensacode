# TensorCode

TODO: update intro sections of README after more architecture has been done

Everyone should try incorperating deep learning into their software, and now anyone can do it. TensorCode turns your code into a heterogenous graph neural network with control flow programming, Y, and Z. It handles all the heavy lifting in either tensorflow, pytorch, or jax to help you transform ordinary Python objects and classes into data that neural networks can read, write, and understand.

```python
# class embedding
class A: ...
A_enc = deeppy.enc(A)


# function embedding
def b(): ...
b_enc = deeppy.enc(b)

# object embedding
c_enc = deeppy.enc(c)

# do your ML here
...

# decode to regular objects
y = tc.String.dec(c_enc)
print(y)

# train the model to produce the outputs you want
model = tc.Model()
model.fit(my_dataset)
```

TensorCode also support fully differentiable control flow enabling end-to-end learning on your increadibly complex software systems:

```python
x = ... # some scalar tensor
y = f(x)

@tc.If(x == 1)
def MyIf:
  return x
@MyIf.Elif(x == 2)
def ElifClause():
  return x ** 2
@MyIf.Else(x == 3)
def ElseClause():
  return -x

dx = y.grad(x)
```

Finally TensorCode introduces an 'intelligent' API. Just check out the code:

```python
a = tc.create("a dictionary obtained frmo post-order traversal of X")
b = tf.exec(a, *more_args, "TASK")
c = tf.select # TODO: add more examples here
```

And that's it.

Basic algorithm:

1. You define your problem in Python
2. Encode objects with `encode`
3. Add ML code to process the representations
4. Add decoders to get relevant objects back
5. Construct a `Model` to get all encoders and decoders in a dictionary for training
6. Connect your training pipeline

```text
deeppy/
|- __init__.py
|- base.py
|- backend.py
|- utils.py
|- tf/
|  |- __init__.py
|  |- encoders.py
|  |- decoders.py
|  |- ops.py
|  |- layers.py
|  |- programming.py
|- pytorch/
|  |- __init__.py
|  |- encoders.py
|  |- decoders.py
|  |- ops.py
|  |- layers.py
|  |- programming.py
|- jax/
|  |- __init__.py
|  |- encoders.py
|  |- decoders.py
|  |- ops.py
|  |- layers.py
|  |- programming.py
examples/
docs/
.github/
.gitignore
setup.py
README.md
LICENSE, CONTRIBUTING, CHANGELOG
```

TODO: Make a top level unification layer for supported data types, structures, etc.

- It is convenient for the developers to write framework-specific logic in separate files
- It is convenient for the users to `import List` and then use that same object for
  - encoding: `List.enc(my_list)`
  - decoding: `my_list_t.dec()`
  - tensor-programming primitives:

    ```python
    @my_list_t.map
    def map_fn(slice):
      return (slice - slice.mean())**2 / slice.std()
    ```
  
  - namescpace for intelligent operations: `List.add_items(list, options=[...], help=...)

```python
"""Common parameters:

- network: Either a string referencing a network or an actual network
  that takes in input, performs task, and returns output. The 'molecule'
  of ML computation for `deeppy`.

"""


# base
def encode(x, *args, **kwargs): ... # based on type, selects appropriate <submodule>.encoders.Encoder (that base encoder takes over routing to appropriate type specific encoder)
class Enc: ...
class Dec: ...

class Model:
  def __init__(self, scope=None):
    """
    Decorates either a function or a class.
    Extracts all inputs and outputs that were defined
    in the scope of the decorated function or class.
    
    OR

    Does nothing. Called from __enter__
    """
    self.inputs = ...
    self.outputs = ...

  def __enter__(self):
    """
    Used to track scope for a specific `with` statement like so:

    with Model() as model:
      i = Int()
      j = i + 1
    train(model.inputs, model.outputs)    

    """
    ...

  def __exit__(self, *args):
    """Finish tracking scope."""
    ...


# backend

# used for default LM, default CNN, default Int encoder, etc.
_PREFERENCES = {
  "copy_network_for": {
    "text_enc": True,
    "if": True,
  },
  "default_networks": {
    "text_enc": {"method": "hf_transformers", "name": "bert-base-uncased"},
    "if": {"method": "lstm", "hidden_size": 64},
  },
}
def set_preferences(updates): ...

"""`method` parsers. Several of them do not need full args
but will fill in the blanks if a match is found in the online hub
For example, if bert-uncased is found, it will also fill in the
"method" field with "hf_transformers".
"""
_PARSERS = [
  hardcoded_parser, # 2-mlp, rnn, lstm
  huggingface_parser, # bert, xlnet, xlm
  hub_parser, # gpt2, openai_gpt, bert_gpt2_joint_model (codex made this)
  tf_hub_parser,
  torch_hub_parser,
  ...
]

def parse_network(network, *args, **kwargs):
  for parser in _PARSERS:
    parsed_network = parser(network, *args, **kwargs)
    if parsed_network is not None:
      return parsed_network
  raise ValueError(f"Could not parse method {method}")

@Protocol
class Stateful
  """Use this class both as a protocol for defining your recurrent cells
  and for wrapping a state-management system around the object.
  """
  
  def __init__(self, func, hidden_state=None):
    self.func = func
    self.hidden_state = hidden_state

  def __call__(self, x, hidden_state=None, **kwargs):
    if hidden_state is not None:
      self.hidden_state = hidden_state
    y, self.hidden_state = self.func(x, hidden_state=self.hidden_state, **kwargs)
    return y

  def __getattr__(self, name):
    if name == 'func':
      return self.func
    elif name == 'hidden_state':
      return self.hidden_state
    else:
      return getattr(self.func, name)
  

# utils
@memoize
def get_tokenizer(model_name):
  return AutoTokenizer.from_pretrained(model_name)

@memoize
def get_transformer(model_name):
  return AutoModel.from_pretrained(model_name)


# tf/pytorch/jax
# encoders
class Encoder:
  registry = {} # subclasses add themselves to the registry on init. Order matters.
  def __call__(self, x, *args, **kwargs): ... # search all encoders for best match

class Int(Encoder): ... # linear, clipped emb lookup, knn
class Float(Encoder): ... # linear, clipped emb lookup, knn
class Bool(Encoder): ... # linear, emb
class String(Encoder): ... # emb pool, transformer LM (last-hidden-state, pooling, etc)

class Object(Encoder): ... # recursive embedding, key based on variable name
Dict = Object # synonym
class Class(Object): ... # key based on class name
Type = Class # synonym
class Module(Object): ... # key based on module name

class Iterable(Encoder): ... 
  def __init__(self, 
               iter: Iterable,
               length=None, 
               ordered=True, 
               architecture='lstm', 
               *args, **kwargs): ...

class List(Iterable):
  """List of objects. Variable length, ordered."""
  def __init__(l: list, architecture=None, *args, **kwargs):
    super().__init__(iter=l,
                     length=None, 
                     ordered=True,
                     architecture=architecture,
                     *args, **kwargs)
                       
class Tuple(Iterable):
  """Tuple of objects. Const length, ordered."""
  def __init__(t: list, architecture=None, *args, **kwargs):
    super().__init__(iter=t,
                     length=len(t), 
                     ordered=True,
                     architecture=architecture,
                     *args, **kwargs)

class Set(Iterable): 
  """Set of objects. Variable length, unordered."""
  def __init__(s: set, architecture=None, *args, **kwargs):
    super().__init__(iter=s,
                     length=None, 
                     ordered=False,
                     architecture=architecture,
                     *args, **kwargs)

class FrozenSet(Iterable):
  """Set of objects. Variable length, unordered."""
  def __init__(s: frozenset, architecture=None, *args, **kwargs):
    super().__init__(iter=s,
                     length=len(s), 
                     ordered=False,
                     architecture=architecture,
                     *args, **kwargs)

class Enum(Encoder): ... # emb

class Date(Encoder): ... # linear, emb
class Stream(Iterable): ... # default iterable params
File = Stream # synonym
class Code(String): ... # reads bytecode as list of tokens or reads AST as graph

class Image(Encoder): ... # transformer CNN, keras example, torchvision models, etc
class Audio(Encoder): ... # torchaudio, librosa, etc
class Video(Encoder): ...

# decoders
class Decoder:
  registry = {} # subclasses add themselves to the registry on init. Order matters.
  # no call func

# same as Encoders, but for decoding


# Layers

tf.keras.layers


# Programming

## control flow
COND = Callable[[], bool]
BLOCK = Callable[[], NoReturn]
def If(*args, *): ... # Parsed as `(cond: COND, true_block: BLOCK)+, else_block: BLOCK`
IfElse = IfElifElse = If  # synonym
class Switch(Callable[[], int|str], *ordered_cases, **keyword_cases): ... # if int, use ordered args, else use kwargs
class While(cond: COND, network, loop: BLOCK): ...
class Until(cond: COND, network, loop: BLOCK): ...
class DoWhile(cond: COND, network, loop: BLOCK): ...
class Repeat(count: Tensor[], loop: BLOCK): ...
class For(initializer: BLOCK, cond: COND, inc: BLOCK, body: BLOCK)

## iterables
TimeSeries = Tensor['b...']
TimeSlice = Tensor['bt...']
class Iterable(iterable)
  def __init__(self): ...
  def __next__(self): ...
  def foreach(self, fn: Callable[[TensorSlice], NoReturn])
  def map(self, fn: Callable[[TensorSlice], NoReturn])
class List(Iterable, list)
  def __init__(self, tensor): ... # decorate regular tensor
  def index(self, fn: Callable[[TensorSlice], NoReturn])
  def remove(self, fn: Callable[[TensorSlice], NoReturn])
class Set(Iterable, set): ...

## logic
def select(
  options: list[Any] | dict[str, Any] = None, 
  *, 
  help: dict[str, Any] = None, 
  certainty=0.9, 
  network='default', 
  default=None): 
  """Intelligently select item from list or dict.

  Options default to locals and globals. 
  Help is anything that might be useful to the network 
    like {"description": "...", "examples": [ObjA, ObjB], ...}
  Certainty is a threshold for selection.
  Network is the network or string identifier for the network
    that will be used to select the item. 
  """
  ...

def call(
  functype: Callable, 
  *, 
  help: dict[str, Any] = None, 
  certainty=0.9, 
  task='default', 
  default=()): 
  """Selects parameters for and calls a function using `select` -- possibly recursively for complex data types."""
  ...

def make(
  options: list[type] = None,
  *,
  help: dict[str, Any], 
  certainty=0.9, 
  task='default', 
  default=None): 
  """Select type and initialize object using `select` and `call`."""
  ...

def make_list(
  options: list[type] = None,
  *,
  help: dict[str, Any], 
  certainty=0.9, 
  task='default', 
  default=None): 
  """Selects items and initializes list with them using `select`."""
  ...

def extend_list(
  l: list,
  options: list[type] = None,
  *,
  append: bool = True,
  insert: bool = False,
  remove: bool = False,
  reorder: bool = False,
  help: dict[str, Any], 
  certainty=0.9, 
  task='default', 
  default=None): 
  """Selects items, initializes them, and inserts or appends them to l,
  removes items in l that should not be in l, and/or reorders l using 
  `select` and `make`."""
  ...
```
