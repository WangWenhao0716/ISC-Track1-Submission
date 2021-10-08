# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

import os
import torch
from momentum_teacher.models.m2_teacher import M2Teacher
import torch.distributed as dist

from momentum_teacher.exps.arxiv import base_exp


class Exp(base_exp.BaseExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ------------------------------------- model config ------------------------------ #
        self.param_momentum = 0.99

        # ------------------------------------ data loader config ------------------------- #
        self.data_num_workers = 6

        # ------------------------------------  training config --------------------------- #
        self.warmup_epochs = 10
        self.max_epoch = 40
        self.warmup_lr = 1e-6
        self.basic_lr_per_img = 0.05 / 256.0

        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_model(self):
        if "model" not in self.__dict__:
            self.model = M2Teacher(self.param_momentum, len(self.data_loader["train"]) * self.max_epoch)
        return self.model

    def get_data_loader(self, batch_size, is_distributed):
        if "data_loader" not in self.__dict__:
            from momentum_teacher.data.transforms import byol_transform
            from momentum_teacher.data.dataset import SSL_Dataset

            transform = byol_transform()
            train_set = SSL_Dataset(transform)
            sampler = None

            if is_distributed:
                batch_size = batch_size // dist.get_world_size()
                sampler = torch.utils.data.distributed.DistributedSampler(train_set)

            dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": False}
            dataloader_kwargs["sampler"] = sampler
            dataloader_kwargs["batch_size"] = batch_size
            dataloader_kwargs["shuffle"] = False
            dataloader_kwargs["drop_last"] = True
            train_loader = torch.utils.data.DataLoader(train_set, **dataloader_kwargs)
            self.data_loader = {"train": train_loader, "eval": None}

        return self.data_loader

    def get_optimizer(self, batch_size):
        # Noticing hear we only optimize student_encoder
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            self.optimizer = torch.optim.SGD(
                self.model.student_encoder.parameters(), lr=lr, weight_decay=self.weight_decay, momentum=self.momentum
            )
        return self.optimizer
