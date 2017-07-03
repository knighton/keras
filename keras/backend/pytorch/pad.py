import torch


def _padded(x, pattern):
    assert len(pattern) == x.dim() - 1
    for pair in pattern:
        a, b = pair
        assert 0 <= a
        assert 0 <= b
    for i, pair in enumerate(pattern):
        if pair == (0, 0):
            continue
        dim = i + 1
        top, bottom = pair
        to_cat = []
        if top:
            shape = list(x.size())
            shape[dim] = top
            to_cat.append(torch.zeros(shape))
        to_cat.append(x)
        if bottom:
            shape = list(x.size())
            shape[dim] = bottom
            to_cat.append(torch.zeros(shape))
        x = torch.cat(to_cat, dim)
    return x


def temporal_padding(x, padding=(1, 1)):
    assert len(padding) == 2
    pattern = [
        (0, 0),
        padding,
        (0, 0),
    ]
    return _padded(x, pattern)


def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
    assert len(padding) == 2
    for pair in padding:
        assert len(pair) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format == 'channels_first':
        pattern = [
            (0, 0),
            (0, 0),
            padding[0],
            padding[1],
        ]
    elif data_format == 'channels_last':
        pattern = [
            (0, 0),
            padding[0],
            padding[1],
            (0, 0),
        ]
    else:
        assert False
    return _padded(x, pattern)


def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
    assert len(padding) == 2
    for pair in padding:
        assert len(pair) == 2
    if data_format is None:
        data_format = image_data_format()
    if data_format == 'channels_first':
        pattern = [
            (0, 0),
            (0, 0),
            padding[0],
            padding[1],
            padding[2],
        ]
    elif data_format == 'channels_last':
        pattern = [
            (0, 0),
            padding[0],
            padding[1],
            padding[2],
            (0, 0),
        ]
    else:
        assert False
    return _padded(x, pattern)
