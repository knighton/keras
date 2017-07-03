import numpy as np
import torch


def get_value(x):
    return x.data.cpu().numpy()


def batch_get_value(ops):
    return list(map(get_value, ops))


def set_value(x, value):
    dt = dtype(x)
    if isinstance(value, (int, float)):
        value = [value]
    if not isinstance(value, np.ndarray):
        value = np.array(value, dt)
    x.data = torch.from_numpy(value)


def batch_set_value(tuples):
    for x, value in tuples:
        set_value(x, value)


def get_variable_shape(x):
    return tuple(x.size())


def print_tensor(x, message=''):
    print(x)
