# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

import os
from torch import nn
from momentum_teacher.exps.arxiv.momentum2_teacher_exp import Exp as BaseExp
from momentum_teacher.layers.optimizer import LARS_SGD


class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.max_epoch = 100
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ----------------------- setting for 4096 batch-size -------------------------- #
        self.param_momentum = 0.99
        self.basic_lr_per_img = 0.45 / 256.0
        self.weight_decay = 1e-6

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            params_lars = []
            params_exclude = []
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                    params_exclude.append(m.weight)
                    params_exclude.append(m.bias)
                elif isinstance(m, nn.Linear):
                    params_lars.append(m.weight)
                    params_exclude.append(m.bias)
                elif isinstance(m, nn.Conv2d):
                    params_lars.extend(list(m.parameters()))

            assert len(params_lars) + len(params_exclude) == len(list(self.model.parameters()))

            self.optimizer = LARS_SGD(
                [{"params": params_lars, "lars_exclude": False}, {"params": params_exclude, "lars_exclude": True}],
                lr=lr,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        return self.optimizer
