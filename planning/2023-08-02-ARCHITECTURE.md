# TensaCode: Machine Learning + Software Engineering + Runtime Introspection and Code Generation = Programming 2.0

TensaCode is a framework that introduces simple abstractions and functions for encoding, decoding, querying, and manipulating arbitrary python objects, differentiable programming (including differentiable control flow), and intelligent runtime code generation and execution. Rather than just think about the underlying mathematical objects, TensaCode's abstractions enable you to draw on decades of software engineering patterns, paradigms, and derived concepts to concisely approach your underlying problem. If you’re building models that require highly structured inputs or outputs (ie, embarassingly multi-modal architectures), researching the next generation of end-to-end differentiable cognitive architectures, or just adding learning capability on top of an existing program, TensaCode may fit right into your stack!

## Introduction

### TensaCode allows you to do the following

1. encode, decode, query, and manipulate arbitrary objects
2. differentiable programming (including differentiable control flow)
3. intelligently generate and execute code at runtime for a given objective

### What's the Problem?

- ML is a math-heavy field, but the abstract problems we're trying to solve are not.
- Most general ML frameworks are verbose and require a lot of boilerplate
- Differentiable programs currently must be written from scratch in raw math using a framework like PyTorch, TensorFlow, or JAX
- These all make it difficult to prototype and experiment with new architectures

### How does TensaCode solve it?

- We introduce simple abstractions and functions for encoding, decoding, querying, and manipulating arbitrary python objects,
- we also provide a pythonic interface for making regular programs differentiable (including differentiable control flow),
- and we have an introspective runtime code generation and execution engine that can be used to solve a variety of problems.

### How does TensaCode work?

- Under the hood, objects are converted to an intermediate representation, which can be a natural language description, a 0, 1, or 2D-vector embedding, or a graph of embeddings.
- The operation is dynamically dispatched to the appropriate implementation depending on the python type, and intermediate representation type.
- Both gating and RL-based approaches are used for differentiable control flow.
- The program submits feedback to the TensaCode engine, which it uses to improve its performance over time via a combination of RL and synthetic self-supervision.
- Past examples also considtion prompt context, filters, and sampling strategies for runtime code generation and execution.

### Where would I use TensaCode?

- TensaCode is useful for semi-structured problems that require a combination of machine learning, software engineering, and/or runtime introspection and code generation.
- If you can write a complete and precise set of rules for your problem, you're probably better off with traditional software engineering. If your problem is totally unstructured, you're probably better off using a traditional ML framework.
- Examples use cases:
  - Model that require highly structured inputs or outputs (embarassingly multi-modal architectures)
  - End-to-end differentiable cognitive architectures
  - Adding learning capability on top of an existing program

## Getting Started

### Installation

```bash
pip install tensacode
```

### Usage

```python
import tensacode as tc

# encode, decode, query, and manipulate arbitrary objects
...

# differentiable programming (including differentiable control flow)
...

# intelligently generate and execute code at runtime for a given objective
...
```

### Examples

You can find more examples in the [examples](examples) directory.

### A game that learns to maximize user engagement

```python

```

### A GUI assistant (vision + language -> keyboard + mouse)

```python

```

### An AGI

```python
# the prompt should be "increase understanding, minimize suffering, maximize peace and prosperity"
# vision, language, STM, LTM, and motor control
```


## Contributing

### Code Organization

```
tensacode/
    __init__.py
    __main__.py
    __version__.py
    _base/
        engine.py
        ops/
            ...
    _text/
        engine.py
        ops/
            op_category/
                op_name.py (contains all op_name overloads):
                    def op(engine, *args, **kwargs):
                        ... # default implementation
                    @op.overload(type1, ...) # just use the python overloading library
                    def op(engine, *args, **kwargs):
                        ... # implementation for type1, ...
                    ...
                ...
            ...
    _vector/
        engine.py
        ops/
            ...
    _graph/
        engine.py
        ops/
            ...
    _utils/
        ...
    _external/
        inspect_mate.py # useful for code introspection
README
LICENSE
CONTRIBUTING
CHANGELOG
poetry.lock
pyproject.toml
.gitignore
```
