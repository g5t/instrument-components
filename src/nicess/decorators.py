def needs(module):
    import functools
    from importlib.util import find_spec
    missing = find_spec(module) is None

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if missing:
                raise ModuleNotFoundError(f"Install {module} to use this functionality")
            return func(*args, **kwargs)

        return wrapper

    return decorator
