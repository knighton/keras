from math import floor
from torch.nn import functional as F

from ..common import normalize_image_data_format
from .data import ndim
from .shape import repeat_to_shape


def _preprocess_conv1d_input(x, data_format):
    if data_format == 'channels_first':
        pass
    elif data_format == 'channels_last':
        x = x.permute(0, 2, 1)
    else:
        assert False
    return x


def _postprocess_conv1d_output(x, data_format):
    if data_format == 'channels_first':
        pass
    elif data_format == 'channels_last':
        x = x.permute(0, 2, 1)
    else:
        assert False
    return x


def conv1d(x, kernel, strides=1, padding='valid', data_format=None,
           dilation_rate=1):
    """
    channels_last:
        x: (batch_size, length, in_channels)
        kernel: (kernel_width, in_channels, out_channels)

    channels_first:
        x: (batch_size, in_channels, length)
        kernel: (kernel_width, in_channels, out_channels)
    """
    assert x.dim() == 3
    assert kernel.dim() == 3

    data_format = normalize_image_data_format(data_format)
    x = _preprocess_conv1d_input(x, data_format)

    if padding == 'same':
        kernel_width = kernel.size(0)
        padding = kernel_width // 2
    elif padding == 'valid':
        padding = 0
    else:
        assert False

    bias = None
    groups = 1
    x = F.conv1d(x, kernel, bias, strides, padding, dilation_rate, groups)

    return _postprocess_conv1d_output(x, data_format)


def _preprocess_conv2d_input(x, data_format):
    if data_format == 'channels_first':
        pass
    elif data_format == 'channels_last':
        x = x.permute(0, 3, 1, 2)
    else:
        assert False
    return x


def _postprocess_conv2d_output(x, data_format):
    if data_format == 'channels_first':
        pass
    elif data_format == 'channels_last':
        x = x.permute(0, 2, 3, 1)
    else:
        assert False
    return x


def _preprocess_deconv2d_output_shape(x, shape, data_format):
    if data_format == 'channels_first':
        pass
    elif data_format == 'channels_last':
        shape = shape[0], shape[3], shape[1], shape[2]
    else:
        assert False
    return shape


def conv2d(x, kernel, strides=(1, 1), padding='valid', data_format=None,
           dilation_rate=(1, 1)):
    """
    channels_last:
        x: (batch_size, height, width, in_channels)
        kernel: (kernel_height, kernel_width, in_channels, out_channels)

    channels_first:
        x: (batch_size, in_channels, height, width)
        kernel: (kernel_height, kernel_width, in_channels, out_channels)
    """
    assert x.dim() == 4
    assert kernel.dim() == 4

    data_format = normalize_image_data_format(data_format)
    x = _preprocess_conv2d_input(x, data_format)

    if padding == 'same':
        k_height, k_width = kernel.size()[:2]
        padding = floor(k_height / 2), floor(k_width / 2)
    elif padding == 'valid':
        padding = 0, 0
    else:
        assert False

    bias = None
    groups = 1
    x = F.conv2d(x, kernel, bias, strides, padding, dilation_rate, groups)

    return _postprocess_conv2d_output(x, data_format)


def conv2d_transpose(x, kernel, output_shape, strides=(1, 1), padding='valid',
                     data_format=None):
    """
    channels_last:
        x: (batch_size, height, width, in_channels)
        kernel: (kernel_height, kernel_width, in_channels, out_channels)

    channels_first:
        x: (batch_size, in_channels, height, width)
        kernel: (kernel_height, kernel_width, in_channels, out_channels)
    """
    assert x.dim() == 4
    assert kernel.dim() == 4

    data_format = normalize_image_data_format(data_format)
    x = _preprocess_conv2d_input(x, data_format)
    output_shape = _preprocess_deconv2d_output_shape(
        x, output_shape, data_format)

    if padding == 'same':
        k_height, k_width = kernel.size()[:2]
        padding = floor(k_height / 2), floor(k_width / 2)
    elif padding == 'valid':
        padding = 0, 0
    else:
        assert False

    bias = None
    output_padding = 0
    groups = 1
    dilation_rate = 1, 1
    x = F.conv_transpose2d(x, kernel, bias, strides, padding, output_padding,
                           groups, dilation_rate)

    return _postprocess_conv2d_output(x, data_format)


def separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1),
                     padding='valid', data_format=None, dilation_rate=(1, 1)):
    raise NotImplementedError


def depthwise_conv2d(x, depthwise_kernel, strides=(1, 1), padding='valid',
                     data_format=None, dilation_rate=(1, 1)):
    raise NotImplementedError


def _preprocess_conv3d_input(x, data_format):
    if data_format == 'channels_first':
        pass
    elif data_format == 'channels_last':
        x = x.permute(0, 2, 3, 4, 1)
    else:
        assert False
    return x


def _postprocess_conv3d_output(x, data_format):
    if data_format == 'channels_first':
        pass
    elif data_format == 'channels_last':
        x = x.permute(0, 4, 1, 2, 3)
    else:
        assert False
    return x


def _preprocess_deconv3d_output_shape(x, shape, data_format):
    if data_format == 'channels_first':
        pass
    elif data_format == 'channels_last':
        shape = shape[0], shape[4], shape[1], shape[2], shape[3]
    else:
        assert False
    return shape


def conv3d(x, kernel, strides=(1, 1, 1), padding='valid', data_format=None,
           dilation_rate=(1, 1, 1)):
    """
    channels_last:
        x: (batch_size, depth, height, width, in_channels)
        kernel: (kernel_depth, kernel_height, kernel_width, in_channels,
                 out_channels)

    channels_first:
        x: (batch_size, in_channels, depth, height, width)
        kernel: (kernel_depth, kernel_height, kernel_width, in_channels,
                 out_channels)
    """
    assert x.dim() == 5
    assert kernel.dim() == 5

    data_format = normalize_image_data_format(data_format)
    x = _preprocess_conv3d_input(x, data_format)

    if padding == 'same':
        k_depth, k_height, k_width = kernel.size()[:3]
        padding = floor(k_depth / 2), floor(k_height / 2), floor(k_width / 2)
    elif padding == 'valid':
        padding = 0, 0, 0
    else:
        assert False

    bias = None
    groups = 1
    x = F.conv3d(x, kernel, bias, strides, padding, dilation_rate, groups)

    return _postprocess_conv3d_output(x, data_format)


def conv3d_transpose(x, kernel, output_shape, strides=(1, 1, 1),
                     padding='valid', data_format=None):
    """
    channels_last:
        x: (batch_size, depth, height, width, in_channels)
        kernel: (kernel_depth, kernel_height, kernel_width, in_channels,
                 out_channels)

    channels_first:
        x: (batch_size, in_channels, depth, height, width)
        kernel: (kernel_depth, kernel_height, kernel_width, in_channels,
                 out_channels)
    """
    assert x.dim() == 5
    assert kernel.dim() == 5

    data_format = normalize_image_data_format(data_format)
    x = _preprocess_conv3d_input(x, data_format)
    output_shape = _preprocess_deconv3d_output_shape(
        x, output_shape, data_format)

    if padding == 'same':
        k_depth, k_height, k_width = kernel.size()[:3]
        padding = floor(k_depth / 2), floor(k_height / 2), floor(k_width / 2)
    elif padding == 'valid':
        padding = 0, 0, 0
    else:
        assert False

    bias = None
    output_padding = 0
    groups = 1
    dilation_rate = 1, 1, 1
    x = F.conv_transpose3d(x, kernel, bias, strides, padding, output_padding,
                           groups, dilation_rate)

    return _postprocess_conv3d_output(x, data_format)


def pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None,
           pool_mode='max'):
    data_format = normalize_image_data_format(data_format)
    x = _preprocess_conv2d_input(x, data_format)

    if padding == 'same':
        k_height, k_width = kernel.size()[:2]
        padding = floor(k_height / 2), floor(k_width / 2)
    elif padding == 'valid':
        padding = 0, 0
    else:
        assert False

    if pool_mode == 'avg':
        x = F.avg_pool2d(x, pool_size, strides, padding)
    elif pool_mode == 'max':
        x = F.max_pool2d(x, pool_size, strides, padding)
    else:
        assert False

    return _postprocess_conv2d_output(x, data_format)


def pool3d(x, pool_size, strides=(1, 1, 1), padding='valid', data_format=None,
           pool_mode='max'):
    data_format = normalize_image_data_format(data_format)
    x = _preprocess_conv3d_input(x, data_format)

    if padding == 'same':
        k_depth, k_height, k_width = kernel.size()[:3]
        padding = floor(k_depth / 2), floor(k_height / 2), floor(k_width / 2)
    elif padding == 'valid':
        padding = 0, 0, 0
    else:
        assert False

    if pool_mode == 'avg':
        x = F.avg_pool3d(x, pool_size, strides, padding)
    elif pool_mode == 'max':
        x = F.max_pool3d(x, pool_size, strides, padding)
    else:
        assert False

    return _postprocess_conv3d_output(x, data_format)


def bias_add(x, bias, data_format=None):
    data_format = normalize_image_data_format(data_format)

    bias_ndim = ndim(bias)
    x_ndim = ndim(x)

    assert not (bias_ndim != 1 and bias_ndim != x_ndim - 1)
    bias_shape = bias.size()
    if x_ndim == 5:
        if data_format == 'channels_first':
            if bias_ndim == 1:
                x += reshape(bias, (1, bias_shape[0], 1, 1, 1))
            else:
                x += reshape(bias, (1, bias_shape[3]) + bias_shape[:3])
        elif data_format == 'channels_last':
            if bias_ndim == 1:
                x += reshape(bias, (1, 1, 1, 1, bias_shape[0]))
            else:
                x += reshape(bias, (1,) + bias_shape)
        else:
            assert False
    elif x_ndim == 4:
        if data_format == 'channels_first':
            if bias_ndim == 1:
                x += reshape(bias, (1, bias_shape[0], 1, 1))
            else:
                x += reshape(bias, (1, bias_shape[2]) + bias_shape[:2])
        elif data_format == 'channels_last':
            if bias_ndim == 1:
                x += reshape(bias, (1, 1, 1, bias_shape[0]))
            else:
                x += reshape(bias, (1,) + bias_shape)
        else:
            assert False
    elif x_ndim == 3:
        if data_format == 'channels_first':
            if bias_ndim == 1:
                x += reshape(bias, (1, bias_shape[0], 1))
            else:
                x += reshape(bias, (1, bias_shape[1], bias_shape[0]))
        elif data_format == 'channels_last':
            if bias_ndim == 1:
                x += reshape(bias, (1, 1, bias_shape[0]))
            else:
                x += reshape(bias, (1,) + bias_shape)
        else:
            assert False
    elif x_ndim == 2:
        bias = bias.unsqueeze(0)
        bias = repeat_to_shape(bias, x.size())
        x += bias
    elif x_ndim == 1:
        x += bias
    else:
        assert False
    return x
