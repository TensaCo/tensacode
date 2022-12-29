# LM Prompting Algorithms

## Encode

Encodes arbitrary objects into strings, recursively

```python
@overload(encode, (str, int, float))
def encode_atomic_type(input, name=None, context=None, max_depth=3):
  return encode_object(input, name=f'the {type(input)}', context=context, max_depth=max_depth)

@overload(encode, (list, set, tuple, dict, frozenset, namedtuple))
def encode_complex_type(input, name=None, context=None, max_depth=3):
  return encode_object(input, name=f'the {type(input)}', context=context, max_depth=max_depth)

@overload(encode, object)
def encode_object(input, name=None, context=None, max_depth=3):
  # direct encoding (base case)
  if max_depth=0:
    output = str(input)
    if len(output) > MAX_LENGTH:
      output = output[:MAX_LENGTH]
    return output

  # smart encoding, using members for context, recursively
  name = name or input.__name__
  member_triples = [
    (
      key, # key
      get_annotation(input, key), # annotation
      encode(
        input=getattr(input, key),
        name=key
        context=context,
        max_depth=max_depth-1
      ) # encoded_value
    )
    for key in dir(input)
  ]
  def stringify(obj):
    return Template(dedent('''\
      {{name}}:
        ({% for key, annotation, encoded_value in items -%}
        {{key}}{{% if annotation %}}: {{annotation}}{{% endif %}} = {{encoded_value}}
        {%- endfor %})'
      ''')).render(name=name, items=member_triples)
  # if there isn't enough space, stochastically decide whether to summarize large members or just remove small ones
  while len(stringify(input)) > MAX_INPUT_LENGTH:
    # 1. generate unnormalized probabbility weights
    # values are less likely to be important if there are many of them and if they are small
    w_trim = {key: len(encoded_values)/(len(encoded_value)) for key, _, encoded_value in encoded_values}
    # values are more likely to be important as they grow longer
    w_summarize = {key: len(encoded_value) for key, _, encoded_value in encoded_values}
    # 2. Combine and normalize
    # TODO
    # 3. Sample
    # TODO
    # 4. Perform appropriate action
    # TODO
  # prompt to encode object
  prompt = Template(dedent('''\
    {% if context -%}
    {{context}}
    {%- endif %}

    {{stringified_input}}

    Provide a concise description of {{name}}:
    ''')).render(
      context=context,
      stringified_input=stringify_input(input),
      name=name
    )
  return completion(prompt)
```

### Example

```python
class Statement: pass
class Action(Statement):
  description: str
class Dialog(Statement):
  speaker: str
  statement: str
  tone: str
class Narration(Statement):
  statement: str
Script = List[Statement]

Environment = str
class Character:
  name: str
  personality: str
  thoughts: str
  dialogue: List[Dialogue]
  actions: List[Action]
  script: Script

class Scene:
  script: Script
  characters: List[Character]
  environment: Environment
class Act:
  subplots: List[Plot]
  characters: List[Character]
  scenes: List[Scene]
class Story:
  plots: List[Plot]
  characters: List[Character]
  acts: List[Act]

# quickly ask questions about the script
story = ...
question = input('Question: ').strip()

# In most cases, the story will be too large and it is too structured for GPT-3 to directly process it
encoded_story = encode(story, context=f"We're try to find the answer to the question '{Question}'.")
answer = completion(dedent(f'''\
  {encoded_story}

  {question}'''))

print(f'Answer: {answer}')
```

## Summarize

Compresses strings longer than the context length

```python
def summarize(input: str):
  segments = segment(input)
  output = segments[0]
  for segment in segments[1:]:
    output = completion(f'''\
      Summary: {output}

      Additional info: {segment}

      Modify the summary based on the additional information: ''')
  return output
```

## Segment

Breaks up long pieces of text into shorter ones which, ideally, represent independant pieces of information.

```python
def segment(input):
  pass # TODO
```

## decide

Decide is used for making programs 'smart'.

```python
def decide(prompt, context):
  return create(prompt, bool, context)
```

### Example

```python
def health_checker():
  status = aws.get_ec2_status()
  if tc.decide('Should we notify IT?', context=status):
    notify_IT(status)
```

## Choice

Similar to decide, but multiple choice

```python
def choice(prompt, options, context):
  ...
```

## Call

`call` is for invoking a callable object.
- Generate an output for each parameter in the callable signature
- if the parameter has no annotation and is just a 'normal' arg (positional, keyword, or mixed), use the `create` method to generate a value, but set the `create` type parameter as None
- if the parameter has a type annotation, use the `create` method for finer-grained control of the object produced
- if the parameter is a vararg or varkwarg param, recursively generate values until a done signal is emitted

Invoke these outputs against the callable and return

```python
def call(fn, context):
  params = inspect.get_signature(fn)
  argvals = []
  kwargvals = {}

  def _render_fn_invokation():
    # TODO show the function invokation partially filled out, based on the values determined thus far for `argvals` and `kwargsvals`
    # Example: foo(a=21, b=_, c, d, *args, e: str = 4, **kwargs) - Docstring for foo
    return (
      f'{fn.__name__}(TODO)' +
      ' - {fn.__doc__}'
        if hasattr(fn, '__doc__') and fn.__doc__ != ''
        else '')

  def _get_single_value(name, type):
    context=[
      context,
      _render_fn_invokation(),
      f'Determine value for {name}',
    ]
    selection_mode = choice(
      f'Instantiate new object or reference existing variable for {name} parameter?',
      options=['instantiate new object', 'reference existing variable'],
      context=context)
    if selection_mode == 'instantiate new object':
      return create(prompt=name, type=type, context={**context})
    elif selection_mode == 'reference existing variable':
      # TODO: we need to be passing the calling frame around, because I need to get the locals from that frame (running `locals()` here just gives me the locals in *this* function, which is not useful for users)
      return get('Get a variable to supply for {name}')
    else:
      raise Error('Invalid selection mode {selection_mode}')

  # get values for each of its params
  for param in params:
    if param.kind == inspect.parameter.POSARG:
      argvals.append(_get_single_value(param.name, (param.annotation
        if param.annotation is not inspect.Parameter.EMPTY_ANNOTATION
        else None))
    elif param.kind == inspect.parameter.KWARG:
      argvals.append(_get_single_value(param.name, (param.annotation
        if param.annotation is not inspect.Parameter.EMPTY_ANNOTATION
        else None))
    elif param.kind == inspect.parameter.VARARGS:
      single_type=somehow_get_type_of_indv_list_elem(param.annotation)
        if param.annotation is not inspect.Parameter.EMPTY_ANNOTATION \
        and can_somehow_get_type_of_indv_list_elem(param.annotation)
        else None
      while decide('Add another vararg?', context=[context, _render_fn_invokation()]):
        argvals.append(_get_single_value(param.name, single_type)
    elif param.kind == inspect.parameter.KWONLYARG:
      argvals.append(_get_single_value(param.name, (param.annotation
        if param.annotation is not inspect.Parameter.EMPTY_ANNOTATION
        else None))
    elif param.kind == inspect.parameter.VARKWARGS:
      single_type=somehow_get_type_of_indv_list_elem(param.annotation)
        if param.annotation is not inspect.Parameter.EMPTY_ANNOTATION \
        and can_somehow_get_type_of_indv_list_elem(param.annotation)
        else None
      while decide('Add another vararg?', context=[context, _render_fn_invokation()]):
        argvals.append(_get_single_value(param.name, single_type)
    else:
      raise Error('Invalid param type')

  return fn(*argvals, **kwargsvals)
```

### Examples

```python
y = x**3 - 2*x**2 + 5
call(plt.plot, 'Plot my data with a modern style')
```

## Create

`create` is for instantiating classes as well as producing literals (int's, str's, lists, etc.).

```python
@overload(create)
def create_float(prompt: str, type: float, context: str = None):
  return completion(
    prompt=Template(dedent(f'''\
      {% if context %}{context}

      {% endif %}Q: {prompt} (decimal):

      A: ''')).render(
        context=context,
        prompt=prompt
      ),
    is_valid=(lambda answer: answer.isnumeric()),
    parse=float
  )

@overload(create)
def create_int(prompt: str, type: int, context: str = None):
  return completion(
    prompt=Template(dedent(f'''\
      {% if context %}{context}

      {% endif %}Q: {prompt} (integer):

      A: ''')).render(
        context=context,
        prompt=prompt
      ),
    is_valid=(lambda answer: answer.isnumeric()),
    parse=int
  )

@overload(create)
def create_str(prompt: str, type: str, context: str = None):
  return completion(
    prompt=Template(dedent(f'''\
      {% if context %}{context}

      {% endif %}Q: {prompt} (text):

      A: ''')).render(
        context=context,
        prompt=prompt
      )
  )

@overload(create)
def create_bool(prompt: str, type: bool, context: str = None):
  return completion(
    prompt=Template(dedent(f'''\
      {% if context %}{context}

      {% endif %}Q: {prompt} (true or false):

      A: ''')),
    is_valid=lambda answer: (
      False if len(answer) == 0 else
      True if answer[0].lower() == 't' else
      True if answer[0].lower() == 'f' else
      False
    )
    parse=lambda answer: True if answer[0].lower() == 't' else False
  )


@overload(create)
def create_list(
  prompt: str,
  type: (list, set, tuple, dict, frozenset, namedtuple),
  context: str = None,
  elem_types: list[type] = None):

  if elem_types is None:
    elem_types = [object]

  # Iteratively add items to the list until its ready
  items = []
  while decide('Add another item?', context={**context, 'items': items}):
    next_item = create(
      prompt=f'Create another item',
      context={**context, 'items': items},
      type=elem_types[0] if len(elem_types) == 1 else choice(elem_types)
    )
    items.append(next_item)

  # convert items to the appropriate data structure
  # here `type` is the function parameter, not the python `type` special function
  return type(items)


@overload(create, {'type': object})
def create_object(prompt: str, type: type = type, context: str = None):
  return create_object(prompt, type, )

@overload(create, {'type': object})
def create_object(prompt: str, type: type = None, context: str = None):
  if not type:
    type = create_type(
      f"Create a type definition for instantiating an object that satisfies the prompt: '{prompt}'",
      context=context
    )
  return call(
    prompt=f'Instantiate this {type.__name__}',
    fn=type.__new__,
    context={
      **context,
      f'{type.__name__}.__doc__': type.__doc__,
      f'{type.__name__}.__name__': type.__name__,
      f'{type.__name__}.__new__': inspect.signature(type.__new__),
      f'{type.__name__}.__init__': inspect.signature(type.__init__),
    }
  )
```

### Examples:

- atomic-types
  ```python
  next_num = tc.create(float, 'based on these results, what alpha value should we try next?', context=results)
  ```

- complex-type instantiation
  ```python
  todos = tc.create(list, 'top 3 most urgent tasks', context=items)
  ```

- composite-type instantiation
  ```python
  model = tc.create(tf.keras.Model, 'a residual AlexNet')
  ```


## Similarity

```python
def similarity(a, b):
  ratings = (
    'identical',
    'very similar',
    'similar',
    'vaguely similar',
    'completely different'
  )

  a_name = 'A' if not hasattr(a, '__name__') else a.__name__
  b_name = 'B' if not hasattr(b, '__name__') else b.__name__

  rating = choice(
    f'How similar would you say {a_name} is to {b_name}?',
    options=ratings,
    context={
      **context,
      a_name: encode(a),
      b_name: encode(b),
    }
  )

  return rating / len(ratings)
```

## Get

Grabs variables from the scope (even if they are nested inside other objects).

```python


```

## put

## sort

## nearest

## anomalies

## patterns

## group

## predict

## average

## combine

## fix

## filter

## do

## test
