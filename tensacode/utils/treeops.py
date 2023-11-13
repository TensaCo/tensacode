from __future__ import annotations

from typing import (
    Interator,
    Iterator,
    TypeVar,
    Generic,
    Union,
    Any,
    Callable,
    Iterable,
    Set,
    NamedTuple,
    Type,
    get_type_hints,
    Literal,
    Generator,
    FrozenSet,
    runtime_checkable,
    Protocol,
)
from types import (
    NoneType,
    ModuleType,
    FunctionType,
)
from copy import copy
import dataclasses

from frozendict import frozendict
import attr
import pydantic

from treeops._types import (
    T,
    R,
    Tkey,
    TKeyGroupPair,
    tree,
    predicate,
    atomic_types,
    container_types,
    composite_types,
    tree_types,
)
from treeops._utils import (
    try_all_subclasses,
    next_or_default,
    next_with_done,
    zip_with_fill,
    zip_with_matching_keys,
    is_namedtuple_class,
    is_namedtuple_instance,
    is_dataclass_type,
    is_dataclass_instance,
)


def to_pytree(obj: T, /) -> Generator[tuple[Tkey, T], None, None]:
    # None
    if obj is None:
        pass

    # namedtuple
    elif is_namedtuple_instance(obj):
        yield from ((field, getattr(obj, field)) for field in obj._fields)

    # dataclass
    elif is_dataclass_instance(obj):
        yield from (
            (field.name, getattr(obj, field.name)) for field in dataclasses.fields(obj)
        )

    # attrs
    elif attr.has(obj):
        yield from (
            (field.name, getattr(obj, field.name)) for field in attr.fields(obj)
        )

    # pydantic BaseModel
    elif isinstance(obj, pydantic.BaseModel):
        yield from ((field, getattr(obj, field)) for field in obj.__fields__)

    # set
    elif isinstance(obj, (set, frozenset)):
        yield from ((None, value) for value in obj)

    # dict, frozendict
    elif isinstance(obj, (dict, frozendict)):
        yield from obj.items()

    # list, tuple
    elif isinstance(obj, (list, tuple)):
        yield from enumerate(obj)

    # iterator
    elif isinstance(obj, Iterator):
        for next_yield in obj:
            if len(next_yield) == 2 and isinstance(next_yield[0], Tkey):
                yield next_yield
            else:
                yield (None, next_yield)

    # atomic types
    elif isinstance(obj, atomic_types):
        pass

    # class / object
    elif isinstance(obj, object):
        yield from {key: getattr(obj, key, None) for key in get_type_hints(obj)}

    # default
    else:
        raise ValueError(f"Invalid node type: {type(obj)}")


def from_pytree(
    leaves: Iterable[T], /, proto: tree[Any], max_depth: int or None = None
) -> tree[T]:
    """
    This function takes a list of leaves and a prototype tree structure, and unflattens them into a tree.
    The 'max_depth' parameter can be used to limit the depth to which the tree is unflattened.

    Args:
        leaves (Iterable[T]): The list of leaves to unflatten.
        proto (tree[Any]): The prototype tree structure to use for unflattening.
        max_depth (int | None): The maximum depth to unflatten to. If None, no limit is applied.

    Returns:
        tree[T]: An unflattened tree.
    """

    if not isinstance(leaves, Iterator):
        leaves = iter(leaves)

    if max_depth is not None and max_depth <= 0:
        return next(leaves)

    def _dict_unflatten(dict_):
        return {
            key: from_pytree(leaves, proto=proto_item, max_depth=max_depth - 1)
            for key, proto_item in dict_
        }

    # namedtuple
    if is_namedtuple_class(proto):
        proto_dict = {
            **{k: Any for k in proto._fields},  # the defaults
            **proto.__annotations__,  # the overrides
        }
        unflattened_members = _dict_unflatten(proto_dict)
        return proto(**unflattened_members)
    elif is_namedtuple_instance(proto):
        unflattened_members = _dict_unflatten(proto_dict)
        return proto.__class__(**unflattened_members)

    # dataclass
    elif is_dataclass_type(proto):
        proto_dict = {
            field.name: field.default or field.type
            for field in dataclasses.fields(proto)
        }
        unflattened_members = _dict_unflatten(proto_dict)
        return proto(**unflattened_members)
    elif is_dataclass_instance(proto):
        proto_dict = {
            field.name: getattr(proto_dict, field.name)
            for field in dataclasses.fields(proto)
        }
        unflattened_members = _dict_unflatten(proto_dict)
        return proto.__class__(**unflattened_members)

    # attrs
    elif isinstance(proto, type) and attr.has(proto):
        proto_dict = {field.name: field.type for field in attr.fields(proto)}
        unflattened_members = _dict_unflatten(proto_dict)
        return proto(**unflattened_members)
    elif isinstance(proto, object) and attr.has(proto.__class__):
        proto_dict = {
            field.name: getattr(proto, field.name)
            for field in attr.fields(proto.__class__)
        }
        unflattened_members = _dict_unflatten(proto_dict)
        return proto.__class__(**unflattened_members)

    # pydantic BaseModel
    elif isinstance(proto, type) and issubclass(proto, pydantic.BaseModel):
        proto_dict = {field: Any for field in proto.__fields__}
        unflattened_members = _dict_unflatten(proto_dict)
        return proto(**unflattened_members)
    elif isinstance(proto, pydantic.BaseModel):
        proto_dict = {field: getattr(proto, field) for field in proto.__fields__}
        unflattened_members = _dict_unflatten(proto_dict)
        return proto.__class__(**unflattened_members)

    # (dict, frozendict)
    elif isinstance(proto, type) and issubclass(proto, (dict, frozendict)):
        raise ValueError(
            "Can't unflatten a dict by type because the length is variable. "
            "Try using a namedtuple or else specify the keys of the dict by example."
        )
    elif isinstance(proto, (dict, frozendict)):
        proto_dict = proto
        unflattened_members = _dict_unflatten(proto_dict)
        return proto.__class__(**unflattened_members)

    # list, tuple, set
    # tuple, in this case, we actually can parse the tuple by type because it has a deifnite number of args
    elif isinstance(proto, type) and issubclass(proto, tuple):
        if hasattr(proto, "__args__"):
            proto_tuple = proto.__args__
            if len(proto_tuple) == 0:
                raise ValueError(
                    "Can't unflatten a tuple with an unknown number of args by type. "
                    "Try specifying the generic args or else specify them by example."
                )
            proto_dict = {i: proto_tuple[i] for i in range(len(proto_tuple))}
            unflattened_members = _dict_unflatten(proto_dict)
            return tuple(v for _, v in unflattened_members)
    # in general, we can't parse a list by type because it has a variable number of args
    elif isinstance(proto, type) and issubclass(proto, list):
        raise ValueError(
            "Can't unflatten a list by type because the length is variable. "
            "Try using a tuple or else specify the length of the list by example."
        )
    elif isinstance(proto, (list, tuple, set)):
        proto_dict = {i: elem for i, elem in enumerate(proto)}
        unflattened_members = _dict_unflatten(proto_dict)
        return proto.__class__(*(v for _, v in unflattened_members))

    # iterator
    elif isinstance(proto, type) and issubclass(proto, Iterator):
        raise ValueError(
            "Can't unflatten an iterator by type because the length is variable. "
            "Try using a tuple or else specify the length of the iterator by example."
        )
    elif isinstance(proto, Iterator):
        for next_ in proto:
            if len(next_) == 2 and isinstance(next_[0], Tkey):
                next_yield_key, next_yield_val = next_
            else:
                next_yield_key, next_yield_val = None, next_
            yield (
                next_yield_key,
                from_pytree(leaves, proto=next_yield_val, max_depth=max_depth - 1),
            )

    # atomic types
    elif isinstance(proto, type) and issubclass(proto, atomic_types):
        return next(leaves)  # ?
    elif isinstance(proto, atomic_types):
        return next(leaves)

    # class (if the proto explicitly specifies a class, we take that as a sign to instantiate a tree of that class)
    elif isinstance(proto, type):
        proto_dict = get_type_hints(proto)
        unflattened_members = _dict_unflatten(proto_dict)
        return type(proto.__name__, (proto,), unflattened_members)

    # object
    elif isinstance(proto, object):
        proto_dict = get_type_hints(proto)
        unflattened_members = _dict_unflatten(proto_dict)
        copy_ = copy(proto)
        copy_.__dict__.update(unflattened_members)
        return copy_

    # default
    else:
        raise ValueError(f"Invalid node type: {type(proto)}")
