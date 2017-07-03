from collections import Counter
from contextlib import contextmanager
import numpy as np
import torch
from torch.autograd import Variable

from .type import normalize_dtype


_UID_PREFIXES = Counter()

_NAME_SCOPE_STACK = []


def get_uid(prefix=''):
    _UID_PREFIXES[prefix] += 1
    return _UID_PREFIXES[prefix]


def reset_uids():
    global _UID_PREFIXES
    _UID_PREFIXES = Counter()


def is_sparse(tensor):
    return tensor.is_sparse


def to_dense(tensor):
    return tensor.to_dense()


@contextmanager
def name_scope(name):
    global _NAME_SCOPE_STACK
    _NAME_SCOPE_STACK.append(name)
    yield
    _NAME_SCOPE_STACK.pop()


def _prepare_name(name, default):
    prefix = '/'.join(_NAME_SCOPE_STACK)
    if name is None:
        return prefix + '/' + default
    return prefix + '/' + name


def variable(value, dtype=None, name=None):
    dtype = normalize_dtype(dtype)
    name = _prepare_name(name, 'variable')
    if isinstance(value, Variable):
        tensor = value.data.cpu()
    elif isinstance(value, torch._TensorBase):
        tensor = value
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np_value)
    else:
        np_value = np.asarray(value, dtype=dtype)
        tensor = torch.from_numpy(np_value)
    v = Variable(tensor, requires_grad=True)
    v._keras_shape = tuple(tensor.size())
    v._uses_learning_phase = False
    return v


def constant(value, dtype=None, shape=None, name=None):
    dtype = normalize_dtype(dtype)
    if shape is None:
        shape = ()
    name = _prepare_name(name, 'constant')
    np_value = np.full(shape, value, dtype)
    tensor = torch.from_numpy(np_value)
    v = Variable(tensor, requires_grad=False)
    v._keras_shape = shape
    v._uses_learning_phase = False
    return v


def is_keras_tensor(x):
    if not isinstance(x, torch.tensor._TensorBase):
        return False
    return hasattr(x, '_keras_history')


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    assert shape
    fake_shape = map(lambda dim: 1 if dim is None else dim, shape)
    dtype = normalize_dtype(dtype)
    assert not sparse
    name = _prepare_name(name, 'placeholder')
    np_value = np.zeros(fake_shape, dtype)
    tensor = torch.from_numpy(np_value)
    v = Variable(tensor, requires_grad=False)
    v._keras_shape = shape
    v._uses_learning_phase = False
    return v


def shape(x):
    return x.size()


def int_shape(x):
    return tuple(x.size())


def ndim(x):
    return x.dim()


def eval(x):
    return x.data.cpu()


def zeros(shape, dtype=None, name=None):
    dtype = normalize_dtype(dtype)
    value = np.zeros(shape, dtype)
    return variable(value, dtype, name)


def ones(shape, dtype=None, name=None):
    dtype = normalize_dtype(dtype)
    value = np.ones(shape, dtype)
    return variable(value, dtype, name)


def eye(size, dtype=None, name=None):
    dtype = normalize_dtype(dtype)
    value = np.eye(size, dtype=dtype)
    return variable(value, dtype, name)


def zeros_like(x, dtype=None, name=None):
    dtype = normalize_dtype(dtype)
    value = np.zeros(x.size(), dtype)
    return variable(value, dtype, name)


def ones_like(x, dtype=None, name=None):
    dtype = normalize_dtype(dtype)
    value = np.ones(x.size(), dtype)
    return variable(value, dtype, name)


def identity(x):
    return x.clone()


def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
    dtype = normalize_dtype(dtype)
    value = random_uniform(shape, low, high, dtype, seed)
    return variable(value, dtype, name)


def random_normal_variable(shape, mean, scale, dtype=None, name=None,
                           seed=None):
    dtype = normalize_dtype(dtype)
    value = random_normal(shape, mean, scale, dtype, seed)
    return variable(value, dtype, name)


def count_params(x):
    return x.numel()


def list_contains(items, x):
    for item in items:
        if item.size() != x.size():
            continue
        if (item == x).sum().data.cpu().numpy()[0] == x.numel():
            return True
    return False
