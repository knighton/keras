import torch

from .type import cast, normalize_dtype


def concatenate(tensors, axis=-1):
    return torch.cat(tensors, axis)


def reshape(x, shape):
    return x.view(*shape)


def permute_dimensions(x, pattern):
    return x.permute(pattern)


def resize_images(x, height_factor, width_factor, data_format):
    raise NotImplementedError


def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
    raise NotImplementedError


def repeat_elements(x, rep, axis):
    sizes = [1] * x.dim()
    sizes[axis] = rep
    return x.repeat(*sizes)


def repeat(x, n):
    assert x.dim() == 2
    x = x.unsqueeze(1)
    return repeat_elements(x, n, 1)


def repeat_to_shape(x, new_shape):
    old_shape = x.size()
    assert len(old_shape) == len(new_shape)
    multiples = []
    for i in range(len(old_shape)):
        assert not new_shape[i] % old_shape[i]
        mul = new_shape[i] // old_shape[i]
        multiples.append(mul)
    return x.repeat(*multiples)


def arange(start, stop=None, step=1, dtype='int32'):
    dtype = normalize_dtype(dtype)
    if stop is None and start < 0:
        start = 0
    result = torch.arange(start, stop, step)
    return cast(result, dtype)


def tile(x, n):
    if isinstance(n, int):
        n = [n]
    return x.repeat(*n)


def flatten(x):
    return x.view(-1)


def batch_flatten(x):
    return x.view(x.size(0), -1)


def expand_dims(x, axis=-1):
    return x.unsqueeze(axis)


def squeeze(x, axis):
    if isinstance(axis, int):
        return x.squeeze(axis)

    for dim in reversed(sorted(axis)):
        x = x.squeeze(dim)
    return x


def stack(x, axis=0):
    return torch.stack(x, axis)


def one_hot(indices, num_classes):
    size = indices.size()[0], num_classes
    mask = torch.LongTensor(*size).fill_(0)
    ones = 1
    if isinstance(indices, Variable):
        ones = Variable(torch.LongTensor(indices.size()).fill_(1))
        mask = Variable(mask, volatile=indices.volatile)
    ret = mask.scatter_(1, indices, ones)
    return ret


def reverse(x, axes):
    raise NotImplementedError
