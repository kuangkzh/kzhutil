import functools


def try_for(n):
    """
    try a function for n times
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            for _ in range(n):
                try:
                    return f(*args, **kwargs)
                except:
                    pass
        return wrapped_f
    return decorator
