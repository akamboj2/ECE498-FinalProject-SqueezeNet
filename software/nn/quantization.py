"""
Author: Rex Geng

quantization API for FX training

todo: you can use this template to write all of your quantization code
"""

import functools

import torch
import torch.nn as nn


def quantize_model(model, args):
    """
    generate a quantized model
    :param model:
    :param args:
    :return:
    """
    return model


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def quantize_uniform(data, n_bits, clip, device='cuda'):
    w_c = data.clamp(-clip, clip)
    b = torch.pow(torch.tensor(2.0), 1 - n_bits).to(device)
    w_q = clip * torch.min(b * torch.round(w_c / (b * clip)), 1 - b)

    return w_q


def quantize_act(data, n_bits, clip, device='cuda'):
    d_c = data.clamp(0, clip)
    b = torch.pow(torch.tensor(2.0), -n_bits).to(device)
    d_q = clip * torch.min(b * torch.round(d_c / (b * clip)), 1 - b)

    return d_q


class QSGD(torch.optim.SGD):
    def __init__(self, *kargs, **kwargs):
        super(QSGD, self).__init__(*kargs, **kwargs)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
        super(QSGD, self).step()
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'org'):
                    p.org.copy_(p.data)


class QAdam(torch.optim.Adam):
    def __init__(self, *kargs, **kwargs):
        super(QAdam, self).__init__(*kargs, **kwargs)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
        super(QAdam, self).step()
        for group in self.param_groups:
            for p in group['params']:
                if hasattr(p, 'org'):
                    p.org.copy_(p.data)


class QConv2d(nn.Conv2d):
    def __init__(self, quant_scheme='TWN', quant_args=None, init_args=None, *kargs, **kwargs):
        super(QConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, inputs):
        return inputs

    def quantize_params(self):
        return


class QLinear(nn.Linear):
    def __init__(self, quant_scheme, quant_args=None, init_args=None, *kargs, **kwargs):
        super(QLinear, self).__init__(*kargs, **kwargs)

    def forward(self, inputs):
        return inputs

    def quantize_params(self):
        return
