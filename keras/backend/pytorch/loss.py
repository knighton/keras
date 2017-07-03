import torch

from ..common import epsilon
from .elementwise import sum
from .nonlin import softmax
from .shape import repeat_to_shape


def binary_crossentropy(output, target, from_logits=False):
    if not from_logits:
        output = output.clamp(epsilon(), 1 - epsilon())
        output = (output / (1 - output)).log()
    return y_true @ y_pred.log() - (1 - y_true) @ (1 - y_pred).log()


def categorical_crossentropy(output, target, from_logits=False):
    if from_logits:
        denom = output.sum(-1)
        denom = repeat_to_shape(denom, output.size())
        output = torch.div(output, denom)
    else:
        output = softmax(output)
    output = output.clamp(epsilon(), 1 - epsilon())
    left = sum(output.exp(), axis=[1]).log()
    right = sum(torch.mul(output, target), axis=[1])
    return left - right


def sparse_categorical_crossentropy(output, target, from_logits=False):
    raise NotImplementedError


def l2_normalize(x, axis, epsilon=1e-12):
    square_sum = x.pow(2).sum(axis)
    norm = torch.max(square_sum, epsilon).pow(0.5)
    return x / norm
