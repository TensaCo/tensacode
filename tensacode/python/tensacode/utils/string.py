def invokation(fn, /, args=None, kwargs=None, result=None):
    args = args or []
    kwargs = kwargs or {}

    args_str = ", ".join(args)
    kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())

    result_str = f" -> {result}" if result is not None else ""

    match args_str, kwargs_str:
        case "", "":
            return f"{fn.__name__}(){result_str}"
        case "", _:
            return f"{fn.__name__}({kwargs_str}){result_str}"
        case _, "":
            return f"{fn.__name__}({args_str}){result_str}"
        case _, _:
            return f"{fn.__name__}({args_str}, {kwargs_str}){result_str}"
