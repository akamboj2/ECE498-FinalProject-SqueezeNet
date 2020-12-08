"""
Author: Rex Geng

quantization API for FX training

todo: you can use this template to write all of your quantization code
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.modules import Fire
import numpy as np


def quantize_model(model, args):
    """
    generate a quantized model
    :param model:
    :param args:
    :return:
    """
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            bias = False
            if m.bias is not None:
                bias = True
            # init_args = {'weight_data': m.weight.data, 'bias_data': m.bias.data if bias else None}
            # quant_args = {'quant_act': args.quant_act, 'num_bits_weight': args.num_bits_weight,
            #               'num_bits_act': args.num_bits_act}
            conv_args = {'in_channels': m.in_channels, 'out_channels': m.out_channels, 'kernel_size': m.kernel_size,
                         'stride': m.stride, 'padding': m.padding, 'groups': m.groups, 'bias': bias}
            conv = QConv2d(**conv_args)
            rsetattr(model, n, conv)
            print('CONV layer ' + n + ' quantized' )

        if isinstance(m, nn.Linear):
            bias = False
            if m.bias is not None:
                bias = True
            # quant_args = {'quant_act': args.quant_act, 'num_bits_weight': args.num_bits_weight,
            #               'num_bits_act': args.num_bits_act}
            fc_args = {'in_features': m.in_features, 'out_features': m.out_features, 'bias': bias}
            init_args = {'weight_data': m.weight.data, 'bias_data': m.bias.data if bias else None}
            lin = QLinear(**fc_args)
            print('FC layer ' + n + 'quantized')
            rsetattr(model, n, lin)

        # if isinstance(m, nn.BatchNorm2d) and args.quant_act:
        #     quant_args = {'num_bits_weight': args.num_bits_weight, 'num_bits_act': args.num_bits_act}
        #     bn_args = {'num_features': m.num_features, 'eps': m.eps, 'momentum': m.momentum, 'affine': m.affine,
        #                'track_running_stats': m.track_running_stats}
        #     init_args = {'weight_data': m.weight.data, 'bias_data': m.bias.data}
        #     bn = QBatchNorm2d(quant_args=quant_args, init_args=init_args, **bn_args)
        #     rsetattr(model, n, bn)
        #     print('BN layer ' + n + ' quantized')

        # if isinstance(m, Fire):
        #     fire = QFire(m.inplanes, m.squeeze_planes, m.expand1x1_planes, m.expand3x3_planes)
        #     print('Fire Layer' + n + 'quantized')
        #     rsetattr(model, n, fire)
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
    # d_c = data.clamp(0, clip)
    # b = torch.pow(torch.tensor(2.0), -n_bits).to(device)
    # d_q = clip * torch.min(b * torch.round(d_c / (b * clip)), 1 - b)
    # BX = n_bits
    # input = torch.clamp(data,0,clip) / clip
    # d_q = 6 * torch.min(torch.round(data*(2**BX))*(2**(-BX)) ,(1.0-(2**(-BX)))*torch.ones_like(data))
    data = data.numpy()
    data = data.clip(0,clip)
    d_q = np.minimum(np.round(data*np.power(2.0,n_bits))*np.power(2.0,-n_bits) ,1.0-np.power(2.0,-n_bits))
    d_q = torch.Tensor(d_q)
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
    # def __init__(self, quant_scheme='TWN', quant_args=None, init_args=None, *kargs, **kwargs):
    #     super(QConv2d, self).__init__(*kargs, **kwargs)
    #     self.weight.data = init_args['weight_data']
    #     if kwargs['bias']:
    #         self.bias.data = init_args['bias_data']
    #     self.quant_scheme = quant_scheme
    #     self.clip_val = 0
    #     self.num_bits_weight = quant_args['num_bits_weight']
    #     self.num_bits_act = quant_args['num_bits_act']
    #     self.quant_act = quant_args['quant_act']

    def forward(self, inputs):
        # if not hasattr(self.weight, 'org'):
        #     self.weight.org = self.weight.data.clone()
        # if self.bias is not None:
        #     if not hasattr(self.bias, 'org'):
        #         self.bias.org = self.bias.data.clone()
        self.quantize_params()
        inputs = quantize_uniform(inputs, 8 , 4, device='cpu')
        out = F.conv2d(inputs, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        out = quantize_uniform(out, 8 , 8, device='cpu')
        return out

    def quantize_params(self):
        self.weight = nn.Parameter(quantize_uniform(self.weight, 8, 1, device='cpu'))
        self.bias = nn.Parameter(quantize_uniform(self.bias, 8, 1, device='cpu'))



class QLinear(nn.Linear):
    # def __init__(self, quant_scheme, quant_args=None, init_args=None, *kargs, **kwargs):
    #     super(QLinear, self).__init__(*kargs, **kwargs)

    def forward(self, inputs):
        inputs = quantize_act(inputs, 8, 4, device='cpu')
        self.quantize_params()
        return quantize_uniform(F.linear(inputs, self.weight, self.bias), 8, 16, device='cpu')

    def quantize_params(self, inputs):
        self.weight = nn.Parameter(quantize_uniform(self.weight, 8, 1 , device='cpu'))
        self.bias = nn.Parameter(quantize_uniform(self.bias, 8, 1, device='cpu'))
    
    
class QFire(nn.Module):
    def __init__(
        self,
        inplanes: int,
        squeeze_planes: int,
        expand1x1_planes: int,
        expand3x3_planes: int
    ) -> None:
        super(QFire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = QConv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = QConv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = QConv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        x = quantize_act(x, 13, 4, device='cpu')
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)
