from torch.nn import functional as F


def elu(x, alpha=1.0):
    return F.elu(x, alpha)


def hard_sigmoid(x):
    x = (0.2 * x) + 0.5
    return x.clamp(0, 1)


def relu(x, alpha=0., max_value=None):
    if alpha != 0.:
        negative_part = F.relu(-x)
    x = F.relu(x)
    if max_value is not None:
        x.clamp(0., max_value)
    if alpha != 0.:
        x -= alpha * negative_part
    return x


def sigmoid(x):
    return F.sigmoid(x)


def softmax(x):
    assert len(x.size()) == 2
    return F.softmax(x)


def softplus(x):
    return F.softplus(x)


def softsign(x):
    return F.softsign(x)


def tanh(x):
    return F.tanh(x)
