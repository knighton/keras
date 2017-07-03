import torch

from .data import constant
from .elementwise import mul
from .random import maybe_seed, random_binomial
from .type import dtype


def dropout(x, level, noise_shape=None, seed=None):
    assert 0. <= level <= 1.
    maybe_seed(seed)
    retain_prob = 1. - level
    dt = dtype(x)
    if noise_shape is None:
        tensor = random_binomial(x.size(), p=retain_prob, dtype=dt)
    else:
        tensor = random_binomial(noise_shape, p=retain_prob, dtype=dt)
        tensor = tensor.expand(x.size())
    tensor = constant(tensor)
    x = mul(x, tensor)
    x /= retain_prob
    return x


def in_top_k(predictions, targets, k):
    if k < 1:
        return torch.zeros(targets.size()).byte()

    if predictions.size()[1] <= k:
        return torch.ones(targets.size()).byte()

    predictions_k = torch.sort(predictions)[0][:, -k]
    targets_values = predictions[torch.arange(targets.size()[0]), targets]
    return predictions_k <= targets_values
