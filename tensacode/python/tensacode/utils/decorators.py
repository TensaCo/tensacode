import functools
import inspect
from typing import Annotated, Any, Callable, get_type_hints


class Default:
    get: Callable[..., Any]  # should take in all the args supplied to the function

    def __init__(self, get: Callable[..., Any]):
        self.get = get


class Decorator(callable):
    prologue: Callable[..., None] = lambda *a, **kw: (a, kw)
    epilogue: Callable[..., None] = lambda retval, *a, **kw: retval
    fn: Callable[..., Any] = None

    def __call__(self, fn):
        self.fn = fn

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            args, kwargs = self.prologue(*args, **kwargs)
            retval = fn(*args, **kwargs)
            retval = self.epilogue(retval, *args, **kwargs)
            return retval

        return wrapper


class dynamic_defaults(Decorator):
    default_type: type[Default] = Default

    def prologue(self, *a, **kw):
        # if any args are depends, replace them with their invocation result
        args = [arg() if isinstance(arg, self.default_type) else arg for arg in a]
        kwargs = {
            k: v() if isinstance(v, self.default_type) else v for k, v in kw.items()
        }
        return args, kwargs

    def epilogue(self, retval, *a, **kw):
        """if retval is annotated with a depends, replace it with its invocation result"""
        if retval:
            return retval  # this returns the retval if it is not annotated

        return_annotation = get_type_hints(self.fn).get("return")

        if isinstance(return_annotation, self.default_type):
            return retval()

        if isinstance(return_annotation, Annotated):
            for arg in return_annotation.__args__:
                if isinstance(arg, self.default_type):
                    return_annotation = arg()
                    return retval()

        return retval  # this returns None if retval is None and not annotated
