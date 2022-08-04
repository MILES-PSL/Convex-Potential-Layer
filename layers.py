import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from util import SpectralNormPowerMethod
import torch
import math
import numpy as np

MAX_ITER = 1
EVAL_MAX_ITER = 100


class ConvexPotentialLayerConv(nn.Module):
    def __init__(self, input_size, cin, cout, kernel_size=3, stride=1, epsilon=1e-4):
        super(ConvexPotentialLayerConv, self).__init__()

        self.activation = nn.ReLU(inplace=False)
        self.stride = stride
        self.register_buffer('eval_sv_max', torch.Tensor([0]))

        self.kernel = torch.zeros(cout, cin, kernel_size, kernel_size)
        self.bias = torch.zeros(cout)
        self.kernel = nn.Parameter(self.kernel)
        self.bias = nn.Parameter(self.bias)

        self.pm = SpectralNormPowerMethod(input_size)
        self.train_max_iter = MAX_ITER
        self.eval_max_iter = EVAL_MAX_ITER

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.kernel, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.kernel)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon

    def forward(self, x):
        res = F.conv2d(x, self.kernel, bias=self.bias, stride=self.stride, padding=1)
        res = self.activation(res)
        res = F.conv_transpose2d(res, self.kernel, stride=self.stride, padding=1)
        if self.training == True:
            self.eval_sv_max -= self.eval_sv_max
            sv_max = self.pm(self.kernel, self.train_max_iter)
            h = 2 / (sv_max ** 2 + self.epsilon)
        else:
            if self.eval_sv_max == 0:
                self.eval_sv_max += self.pm(self.kernel, self.eval_max_iter)
            h = 2 / (self.eval_sv_max ** 2 + self.epsilon)
        out = x - h * res
        return out


class ConvexPotentialLayerLinear(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-4):
        super(ConvexPotentialLayerLinear, self).__init__()
        self.activation = nn.ReLU(inplace=False)
        self.register_buffer('eval_sv_max', torch.Tensor([0]))

        self.weights = torch.zeros(cout, cin)
        self.bias = torch.zeros(cout)

        self.weights = nn.Parameter(self.weights)
        self.bias = nn.Parameter(self.bias)

        self.pm = SpectralNormPowerMethod((1, cin))
        self.train_max_iter = MAX_ITER
        self.eval_max_iter = EVAL_MAX_ITER

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon
        self.alpha = torch.zeros(1)
        self.alpha = nn.Parameter(self.alpha)

    def forward(self, x):
        res = F.linear(x, self.weights, self.bias)
        res = self.activation(res)
        res = F.linear(res, self.weights.t())
        if self.training == True:
            self.eval_sv_max -= self.eval_sv_max
            sv_max = self.pm(self.weights, self.train_max_iter)
            h = 2 / (sv_max ** 2 + self.epsilon)
        else:
            if self.eval_sv_max == 0:
                self.eval_sv_max += self.pm(self.weights, self.eval_max_iter)
            h = 2 / (self.eval_sv_max ** 2 + self.epsilon)

        out = x - h * res
        return out


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.normalize = Normalize(mean, std)

    def forward(self, x):
        return self.model(self.normalize(x))


class PaddingChannels(nn.Module):
    def __init__(self, ncout, ncin=3, mode="zero"):
        super(PaddingChannels, self).__init__()
        self.ncout = ncout
        self.ncin = ncin
        self.mode = mode

    def forward(self, x):
        if self.mode == "clone":
            return x.repeat(1, int(self.ncout / self.ncin), 1, 1) / np.sqrt(int(self.ncout / self.ncin))
        elif self.mode == "zero":
            bs, _, size1, size2 = x.shape
            out = torch.zeros(bs, self.ncout, size1, size2, device=x.device)
            out[:, :self.ncin] = x
            return out


class PoolingLinear(nn.Module):
    def __init__(self, ncin, ncout, agg="mean"):
        super(PoolingLinear, self).__init__()
        self.ncout = ncout
        self.ncin = ncin
        self.agg = agg

    def forward(self, x):
        if self.agg == "trunc":
            return x[:, :self.ncout]

        k = 1. * self.ncin / self.ncout
        out = x[:, :self.ncout * int(k)]
        out = out.view(x.shape[0], self.ncout, -1)
        if self.agg == "mean":
            out = np.sqrt(k) * out.mean(axis=2)
        elif self.agg == "max":
            out, _ = out.max(axis=2)

        return out


class LinearNormalized(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

    def forward(self, x):
        self.Q = F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, self.Q, self.bias)
