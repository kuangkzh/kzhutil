from typing import Iterable, Dict


def flatten(arr: Iterable):
    """
    [a, b, [c, [d, e], [f], g] -> [a, b, c, d, e, f, g]
    """
    return [i for item in arr for i in (flatten(item) if isinstance(item, Iterable) else [item])]


def flatten_dict(d: Dict, join=None) -> Dict:
    """
    {'a': 1, 'b': {'c': {'d': 2, 'e': 3}, 4: 5}, 'f': 6}
     -> {('a',): 1, ('b', 'c', 'd'): 2, ('b', 'c', 'e'): 3, ('b', 4): 5, 'f': 6}
    """
    res = {}
    for k, v in d.items():
        if isinstance(v, Dict):
            for ki, vi in flatten_dict(v, join).items():
                res.__setitem__((k,)+ki if join is None else str(k)+join+ki, vi)
        else:
            res.__setitem__((k,) if join is None else str(k), v)
    return res
