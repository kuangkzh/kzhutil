import functools
import time
from typing import Callable, Iterable


def try_for(n):
    """
    try a function for n times till success
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            return try_for_(n, f, *args, **kwargs)
        return wrapped_f
    return decorator


def try_for_(n, f: Callable, *args, **kwargs):
    """
    try call f(*args, **kwargs) for n times till success
    """
    exception = None
    for _ in range(n):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            exception = e
    raise exception


def try_until(f: Callable, args: Iterable):
    """
    try f(*args[0]), f(*args[1]), f(*args[2]), ... until success then return
    """
    exception = None
    for arg in args:
        try:
            return f(*arg) if isinstance(arg, Iterable) else f(arg)
        except Exception as e:
            exception = e
    raise exception


def repeat_for(n):
    """
    repeat a function for n times
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapped_f(*args, **kwargs):
            return repeat_for_(n, f, *args, **kwargs)
        return wrapped_f
    return decorator


def repeat_for_(n, f: Callable, *args, **kwargs):
    return [f(*args, **kwargs) for _ in range(n)]


def benchmark(n, f: Callable, *args, **kwargs):
    """
    repeat f(*args, **kwargs) n times and get the time
    """
    t0 = time.time()
    for _ in range(n):
        f(*args, **kwargs)
    return time.time()-t0


def execute_once(f):
    """
    execute the function for only once in one life cycle of the program
    """
    @functools.wraps(f)
    def wrapped_f(*args, **kwargs):
        return execute_once_(f, *args, **kwargs)
    return wrapped_f


def execute_once_(f: Callable, *args, attach_id: str = None, **kwargs):
    call_id = f if attach_id is None else (f, attach_id)
    execute_once_.stat = getattr(execute_once_, "stat", set())
    if call_id not in execute_once_.stat:
        execute_once_.stat.add(call_id)
        return f(*args, **kwargs)


class ExecuteOnce:
    """
    execute the function for only once in one life cycle of the program
    Example:
        >>> for _ in ExecuteOnce("xxx"):    # the code under here only execute once
        >>>     pass
    """
    stat = set()

    def __init__(self, attach_id: str):
        self.repeat = attach_id in self.stat
        self.stat.add(attach_id)

    def __iter__(self):
        return self

    def __next__(self):
        if self.repeat:
            raise StopIteration
        self.repeat = True
