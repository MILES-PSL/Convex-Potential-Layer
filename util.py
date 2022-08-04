import numpy as np
import os
import torch.nn.functional as F
import torch
from torch import nn
from advertorch.attacks import L2PGDAttack
import pandas as pd
from autoattack import AutoAttack


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def calc_l2distsq(x, y):
    d = (x - y) ** 2
    return d.view(d.shape[0], -1).sum(dim=1)


def margin_loss(y_pred, y, eps):
    return F.multi_margin_loss(y_pred, y, margin=eps, p=1)


def certified_accuracy(test_loader, model, lip_cst=1, eps=36 / 255):
    model.eval()
    cert_right = 0.
    cert_wrong = 0.
    insc_right = 0.
    insc_wrong = 0.
    acc = 0.
    n = 0

    for batch in test_loader:
        with torch.no_grad():
            X, y = batch['input'], batch['target']
            yhat = model(X)
        correct = yhat.max(1)[1] == y
        margins = torch.sort(yhat, 1)[0]
        certified = (margins[:, -1] - margins[:, -2]) > np.sqrt(2.) * lip_cst * eps
        n += len(y)
        cert_right += torch.sum(correct & certified).item()
        cert_wrong += torch.sum(~correct & certified).item()
        insc_right += torch.sum(correct & ~certified).item()
        insc_wrong += torch.sum(~correct & ~certified).item()
        acc += torch.sum(correct).item()

    cert_right /= n
    cert_wrong /= n
    insc_right /= n
    insc_wrong /= n
    acc /= n

    return acc, cert_right, cert_wrong, insc_right, insc_wrong


def certified_accuracy_LLN(test_loader, model, lip_cst=3, eps=36 / 255):
    model.eval()
    cert_right = 0.
    cert_wrong = 0.
    insc_right = 0.
    insc_wrong = 0.
    acc = 0.
    n = 0
    normalized_weight = F.normalize(model.module.model.last_last.weight, p=2, dim=1)
    for batch in test_loader:
        with torch.no_grad():
            X, y = batch['input'], batch['target']
            yhat = model(X)
        correct = yhat.max(1)[1] == y
        margins, indices = torch.sort(yhat, 1)
        margins = (margins[:, -1][:, None] - margins[:, 0:-1])
        for idx in range(margins.shape[0]):
            margins[idx] /= torch.norm(normalized_weight[indices[idx, -1]] - normalized_weight[indices[idx, 0:-1]],
                                       dim=1, p=2)
        margins, _ = torch.sort(margins, 1)
        certified = margins[:, 0] > eps * lip_cst
        n += len(y)
        cert_right += torch.sum(correct & certified).item()
        cert_wrong += torch.sum(~correct & certified).item()
        insc_right += torch.sum(correct & ~certified).item()
        insc_wrong += torch.sum(~correct & ~certified).item()
        acc += torch.sum(correct).item()

    cert_right /= n
    cert_wrong /= n
    insc_right /= n
    insc_wrong /= n
    acc /= n
    return acc, cert_right, cert_wrong, insc_right, insc_wrong


def test_pgd_l2(model, test_batches, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=36. / 255,
                nb_iter=10, eps_iter=0.2, rand_init=True, clip_min=0.0,
                clip_max=1.0, targeted=False):
    adversary = L2PGDAttack(model, loss_fn=loss_fn, eps=eps,
                            nb_iter=nb_iter, eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
                            clip_max=clip_max, targeted=targeted)
    model.eval()
    correct = 0
    id = 0
    for batch in test_batches:
        images, target = batch['input'], batch['target']
        images_adv = adversary.perturb(images, target)
        prediction_ad = model(images_adv)

        prediction_ad = prediction_ad.argmax(dim=1, keepdim=True)
        correct += prediction_ad.eq(target.view_as(prediction_ad)).sum().item()
        id += len(target)

    accuracy = 100. * correct / id

    print('\nTest set adverserial: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, id, accuracy))

    return accuracy


def test_auto_attack(model, test_batches, eps=36. / 255):
    adversary = AutoAttack(model, norm='L2', eps=eps, version='standard')
    model.eval()
    correct = 0
    id = 0
    for batch in test_batches:
        images, target = batch['input'], batch['target']
        images_adv = adversary.run_standard_evaluation(images, target, bs=256)
        prediction_ad = model(images_adv)
        prediction_ad = prediction_ad.argmax(dim=1, keepdim=True)
        correct += prediction_ad.eq(target.view_as(prediction_ad)).sum().item()
        id += len(target)

    accuracy = 100. * correct / id

    print('\nTest set adverserial: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, id, accuracy))

    return accuracy


class TriangularLRScheduler:
    def __init__(self, optimizer, lr_steps, lr):
        self.optimizer = optimizer
        self.epochs = lr_steps
        self.lr = lr

    def step(self, t):
        lr = np.interp([t],
                       [0, self.epochs * 2 // 5, self.epochs * 4 // 5, self.epochs],
                       [0, self.lr, self.lr / 20.0, 0])[0]
        self.optimizer.param_groups[0].update(lr=lr)


class SpectralNormPowerMethod(nn.Module):

    def __init__(self, input_size, eps=1e-8):
        super(SpectralNormPowerMethod, self).__init__()
        self.input_size = input_size
        self.eps = eps
        self.u = torch.randn(input_size)
        self.u = self.u / self.u.norm(p=2)
        self.u = nn.Parameter(self.u, requires_grad=False)

    def normalize(self, arr):
        norm = torch.sqrt((arr ** 2).sum())
        return arr / (norm + 1e-12)

    def _compute_dense(self, M, max_iter):
        """Compute the largest singular value with a small number of
        iteration for training"""
        for _ in range(max_iter):
            v = self.normalize(F.linear(self.u, M))
            self.u.data = self.normalize(F.linear(v, M.T))
        z = F.linear(self.u, M)
        sigma = torch.mul(z, v).sum()
        return sigma

    def _compute_conv(self, kernel, max_iter):
        """Compute the largest singular value with a small number of
        iteration for training"""
        pad = (1, 1, 1, 1)
        pad_ = (-1, -1, -1, -1)
        for i in range(max_iter):
            v = self.normalize(F.conv2d(F.pad(self.u, pad), kernel))
            self.u.data = self.normalize(F.pad(F.conv_transpose2d(v, kernel), pad_))
        u_hat, v_hat = self.u, v

        z = F.conv2d(F.pad(u_hat, pad), kernel)
        sigma = torch.mul(z, v_hat).sum()
        return sigma

    def forward(self, M, max_iter):
        """ Return the highest singular value of a matrix
        """
        if len(M.shape) == 4:
            return self._compute_conv(M, max_iter)
        elif len(M.shape) == 2:
            return self._compute_dense(M, max_iter)
