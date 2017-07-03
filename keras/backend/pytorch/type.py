import numpy as np
import torch
from torch.autograd import Variable

from ..common import floatx


def _make_data():
    data = [
        ('float16', None,                 'torch.cuda.HalfTensor'),
        ('float32', 'torch.FloatTensor',  'torch.cuda.FloatTensor'),
        ('float64', 'torch.DoubleTensor', 'torch.cuda.DoubleTensor'),
        ('uint8',   'torch.ByteTensor',   'torch.cuda.ByteTensor'),
        ('int8',    'torch.CharTensor',   'torch.cuda.CharTensor'),
        ('int16',   'torch.ShortTensor',  'torch.cuda.ShortTensor'),
        ('int32',   'torch.IntTensor',    'torch.cuda.IntTensor'),
        ('int64',   'torch.LongTensor',   'torch.cuda.LongTensor'),
    ]

    tensor2numpy = {}
    numpy2tensor = {}
    for dtype, cpu, gpu in data:
        assert dtype
        if cpu:
            tensor2numpy[cpu] = dtype
            numpy2tensor[dtype] = cpu
        if gpu:
            tensor2numpy[gpu] = dtype
            numpy2tensor[dtype] = gpu

    return tensor2numpy, numpy2tensor


_TENSOR2NUMPY, _NUMPY2TENSOR = _make_data()


def normalize_dtype(dtype):
    if dtype is None:
        return floatx()
    elif isinstance(dtype, str):
        return dtype
    elif isinstance(dtype, np.dtype):
        return dtype.name
    else:
        assert False


def dtype(x):
    if isinstance(x, Variable):
        tensor_type = x.data.type()
    else:
        tensor_type = x.type()
    return _TENSOR2NUMPY[tensor_type]


def cast(x, dtype):
    dtype = normalize_dtype(dtype)
    tensor_type = _NUMPY2TENSOR[dtype]
    assert tensor_type
    return x.type(tensor_type)
