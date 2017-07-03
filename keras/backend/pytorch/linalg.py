def dot(x, y):
    return x @ y


def batch_dot(x, y, axes=None):
    raise NotImplementedError


def transpose(x):
    n = x.dim()
    if n == 1:
        x = x.unsqueeze(1)
        x = x.transpose(0, 1)
    elif n == 2:
        x = x.transpose(0, 1)
    else:
        assert False
    return x


def gather(reference, indices):
    raise NotImplementedError
