# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""
import torch
import math
from torch.nn import Module
from . import resnet_ibn_mbn as resnet_ibn


class M2Teacher(Module):
    def __init__(self, param_momentum, total_iters):
        super(M2Teacher, self).__init__()
        self.total_iters = total_iters
        self.param_momentum = param_momentum
        self.current_train_iter = 0
        self.student_encoder = resnet_ibn.resnet50_ibn_a(
            low_dim=256, width=1, hidden_dim=4096, MLP="byol", CLS=False, bn="customized", predictor=True
        )
        self.teacher_encoder = resnet_ibn.resnet50_ibn_a(
            low_dim=256, width=1, hidden_dim=4096, MLP="byol", CLS=False, bn="mbn", predictor=False
        )
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        self.momentum_update(m=0)
        for m in self.teacher_encoder.modules():
            if isinstance(m, resnet_ibn.MomentumBatchNorm1d) or isinstance(m, resnet_ibn.MomentumBatchNorm2d):
                m.total_iters = self.total_iters

    @torch.no_grad()
    def momentum_update(self, m):
        for p1, p2 in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            # p2.data.mul_(m).add_(1 - m, p1.detach().data)
            p2.data = m * p2.data + (1.0 - m) * p1.detach().data

    def get_param_momentum(self):
        return 1.0 - (1.0 - self.param_momentum) * (
            (math.cos(math.pi * self.current_train_iter / self.total_iters) + 1) * 0.5
        )

    def forward(self, inps, update_param=True):
        if update_param:
            current_param_momentum = self.get_param_momentum()
            self.momentum_update(current_param_momentum)

        x1, x2 = inps[0], inps[1]
        q1 = self.student_encoder(x1)
        q2 = self.student_encoder(x2)

        with torch.no_grad():
            k1 = self.teacher_encoder(x2)
            k2 = self.teacher_encoder(x1)
        con_loss = (4 - 2 * ((q1 * k1).sum(dim=-1, keepdim=True) + (q2 * k2).sum(dim=-1, keepdim=True))).mean()

        self.current_train_iter += 1
        if self.training:
            return con_loss
