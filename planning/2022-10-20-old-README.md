![](assets/img/logo-color.png)

# TensorCode: Machine Learning + Software Engineering + Runtime Introspection and Code Generation = Programming 2.0

TensorCode is a framework that introduces simple abstractions and functions for encoding and decoding arbitrary python objects, differentiable programming (including differentiable control flow), and intelligent object creation / selection / other runtime code generation features. Rather than just think about the underlying mathematical objects, TensorCode's Programming 2.0 paradigm gives your brain the abstractions it needs to apply software engineering patterns including of encapsulation, abstraction, composition, design patterns, and derived concepts to your underlying problem. If you’re developing a multi-modal agent, building the next generation of end-to-end differentiable cognitive architectures, or just adding magic to an existing program, TensorCode may fit right into your stack!

## TensorCode allows you to do the following

1. arbitrary object encoding, decoding, and selection
1. overriding the encoder, decoder, or selector to prefer when processing a given type, object, or parameter
2. overriding the encoder, decoder, or selector to prefer inside a `with` scope (eg, different percieve vs. act scopes use different parameters)
2. various code execution primitives (`if`, `while`, etc.) made differentiable
3. runtime code-generation and execution for a given objective

## Basics

- The Python scope can be viewed as a heterogenous directed graph. `object`'s are vertices, attributes are edges, and the `__dict__` is the adjacency list.
- The `encode` operation recieves a Python object to encode. First, it initializes each vertex of the object to-be-encoded and its connected objects with an embedding computed from their `__name__` attribute. Then a graph neural network recursively updates those embeddings for a constant, computed, or learned number of steps. Finally, the embedding of the object to-be-encoded is returned.
- The `select` operation recieves a query object and a list of options. First, these inputs are encoded. Then an attention-based classifier selects the top `n` options from the list of options. Finally, those top `n` options are returned.
- The `select_call` operation wraps the `select` operation with a small modification: callable options are passed along to `select` as both a callable object and a callable invocation. `select_call` assists `select` by attaching the signature, return type, and syntax graph (if parsable) directly to the callables before passing them along.

- The `call` operation recursively selects/creates args for a function, but doesn't necesarily return a newly created object

tc.call(fn=tc.select(opts=(f, g, h)), call_args=True)

- The `call` operation recieves a query and a list of functions to choose from. It inspects and attaches each function's signature information is attached  Each functions is considered its own subgraph. The `select` operation uses this information to select a constructor to use. Then,   If the constructor is a primitive type, a special decoder overrides the `create` method. Finally, the resulting object is returned.
- determines the top-match constructor is selected and executed with the query as its argument. The result is returned.
- Decoding works by recursively creating or selecting appropriate arguments to pass into an initializer.
- Operations may be overriden, eg, `float` and `Tensor` are allready considered to be in encoded form, so their values are not changed by `encode` and `decode`.
- explain the model at any scope
  - many tensorcode top-level endpoints redirect to the model's corresponding method
    - eg, tc.add_loss() redirects to tc.MODEL.add_loss()
- explain functional and object-oriented training.
  - Mention flax @compact annotation
- learn via manual fitting (SL), self-supervision on normal execution, and reinforcement learning.

## Details

1. Encoding, decoding, selection, and future tensorcode operations are all an `Operation`. The `Operation` base class defines a `__call__` method which checks the operand's `__tensorcode__.<operation_name>` attribute for a `Callable` object. If a `Callable` exists, it is called with the operand as the first argument. Otherwise, the `Operation` subclass's `forward` method is called with its operand as the first argument (with args and kwargs forwarded). In addition to manually passed arguments, `Operation` inspects and passes the `globals`, `locals`, `args`, `kwargs`, code, and comments accessible in the operation's calling frame, and this information may be used to inform behavior (eg, `encode(description=<string>)` may query variables with similar, but not necesarily identical, names as the description argument). "Operand" refers to the object, type, or argument being operated on. For argument operands, this information is stored in its annotation's `__tensorcode__...` attribute. Operations can also be overriden by subclassing and implementing the appropriate method.

2. TensorCode supports two paradigms for differentiable programming:

   1. Treat the decision as a reinforcement learning action using `action, estimated_reward = decide(...)`. The policy's `reward` can be automatically connected to the objective function (default), updated programmatically, or recieved via backpropagation from the `action`'s gradients. Python 3.10+ supports inline assignment (`:=`), which simplifies the latter two cases.

   2. Run all branches and merge the results through a condition-parametrized gate using `tensorcode.If`, `tensorcode.Elif`, `tensorcode.Else`, `tensorcode.Switch`, or `tensorcode.While` functions. Since they are end-to-end differentiable, these primitives support supervised learning. Branches can be mutating, however, they should not assume any particular order of execution relative to other branches.

3. Open-ended execution is approached as a (constrained) selection problem where the search space include all `locals`, `globals`, and operators. To prevent infinite recursion, fundemental types and their constructors (eg, `int.__new__`, `list`, `type`) are overriden at import-time to support the `select` operation, so when the selection operation recurses down to `int.__new__`, a neural network takes over when deciding what integer to pass as the value. However, the search space can also be manually defined using the `options` argument. Additionally, most code shares software design patterns which are made explicit in the `tensorcode.patterns` submodule.

### Running, training, and exporting

There are two basic use cases for tensorcode:

- adding handwavium to your code. When the codebase is large or integrates with other systems it is impossible to convert it to an ML model. For many practical purposes, TensorCode doesn't know what parameters the code will need before it runs. In this case, we just have to 1) load parameters (if any), 2) run the code (with as little involvement as possible), and 3) train parameters (which may have been created in step 2), and 4) save the parameters. The machine learning 'model' is the codebase + parameters. A positive aspect of this use-case is that there are generally few parameters to learn.

- building advanced ML models to integrate with existing tools, trainers, and frameworks. In this case, our models are probabbly non-mutating and can be parallelized. Still it may not be possible to compile the model to a static graph as many tensorcode operations inherently require a full Python REPL (eg, branch decision, runtime code generation). Models can be wrapped in a `MultiProcessingModel` to parallelize execution.

TensorCode provides 5 approaches to convert differentiable codebases into `tensorcode.Model`'s:

1. by passing a `Callable` into the `Model` constructor's `fn` argument along with input and output annotations on either the Callable's signature or the `Model` constructor's `inputs` and `output` arguments. The constructor will then convert this into a `Model` by using the second approach.
2. by writing a custom function that uses `Model` (with optional `context_name`) as a context manager. In addition to automatic saving, this approach is useful for isolating separate functionalities, as the operation-specific parameters are stored in that `Model` rather than the default model.
3. by writing a custom function that calls `tensorcode.load` and `tensorcode.save` functions to load and save parameters at the begging and end of TensorCode-managed code respectively. (These functions call the default model's `load` and `save` methods.)
4. by subclassing or annotating a function or class you want to isolate with `tensorcode.Model`. This approach automatically enters and exits contexts when the function or class is called. (It injects itself at the end of the class's __init__ method so it can wrap each method.)
5. by initializing using keras-style primitives, which are the CapitalCase versions of their corresponding lowercase functions. These are useful for building models with a functional API.

I need to clarify this: TensorCode supports two training paradigms:

- object-oriented style: this is the keras-style
- functional style:

`tensorcode.learn` is a convenience function for building a model, immediantly training, and saving the updates based on reward (or loss) supplied during program execution using `tensorcode.reward(<reward>)` and this can be extended to support supervised learning using user-supplied loss functions.

Under the hood, all models are treated this way, and `tensorcode.DEFAULT_MODEL` enters scope at import-time and is identified by the `__FILE__` that it was first imported from. `Operation`'s are not always public citizens: they are attatched to or removed from their corresponding package-level endpoints whenever the model that defines them enters or exits the context. However, most models do not override the default model's `operations` property.

## Code Organization

Several TensorCode endpoints are not located where their name would indicate for convenience (eg, `tensorcode.If` is located under `programming._if`).

```
tensorcode
- operations
  - __init__.py
  - operation.py
  - encode.py
  - decode.py
  - select.py
- programming
  - __init__.py
  - if.py
  - while.py
  - switch.py
- patterns
- models
  - __init__.py
  - model.py
  - default.py
- utils
  - __init__.py
  - ...
- __init__.py
- __main__.py
```
