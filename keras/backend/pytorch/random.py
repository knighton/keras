import numpy as np
import torch

from .type import normalize_dtype


def maybe_seed(seed):
    if seed is not None:
        np.random.seed(seed)


def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = normalize_dtype(dtype)
    maybe_seed(seed)
    x = np.random.normal(mean, stddev, shape).astype(dtype)
    return torch.from_numpy(x)


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = normalize_dtype(dtype)
    maybe_seed(seed)
    x = np.random.uniform(minval, maxval, shape).astype(dtype)
    return torch.from_numpy(x)


def random_binomial(shape, p=0.0, dtype=None, seed=None):
    dtype = normalize_dtype(dtype)
    maybe_seed(seed)
    x =  np.random.choice([0, 1], size=shape, p=[1 - p, p]).astype(dtype)
    return torch.from_numpy(x)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = normalize_dtype(dtype)
    maybe_seed(seed)
    x = np.random.normal(mean, stddev, shape).astype(dtype)
    min_value = mean - 2 * stddev
    max_value = mean + 2 * stddev
    x = x.clip(min_value, max_value)
    return torch.from_numpy(x)
