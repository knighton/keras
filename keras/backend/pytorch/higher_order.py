from functools import reduce

from .type import cast, normalize_dtype


def map_fn(fn, elems, name=None, dtype=None):
    dtype = normalize_dtype(dtype)
    x = list(map(fn, elems))
    x = torch.cat(x)
    return cast(x, dtype)


def foldl(fn, elems, initializer=None, name=None):
    if initializer is None:
        initializer = elems[0]
        elems = elems[1:]
    return reduce(fn, elems, initializer)


def foldr(fn, elems, initializer=None, name=None):
    if initializer is None:
        initializer = elems[-1]
        elems = elems[:-1]
    x = initializer
    for elem in reversed(elems):
        x = fn(elem, x)
    return x
