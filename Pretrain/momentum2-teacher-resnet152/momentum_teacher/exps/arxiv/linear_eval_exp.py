# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

import torch
from torch import nn
from momentum_teacher.models import resnet_mbn as resnet
import torch.distributed as dist
from momentum_teacher.exps.arxiv.base_exp import BaseExp


class ResNetWithLinear(nn.Module):
    def __init__(self):
        super(ResNetWithLinear, self).__init__()

        self.encoder = resnet.resnet50(width=1, bn="vanilla")
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(nn.Linear(2048, 1000), nn.BatchNorm1d(1000))
        self.criterion = nn.CrossEntropyLoss()
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def train(self, mode: bool = True):
        self.training = mode
        self.encoder.eval()
        self.classifier.train(mode)

    def forward(self, x, target=None):
        with torch.no_grad():
            feat = self.encoder(x, res5=True).detach()
        logits = self.classifier(feat)
        if self.training:
            loss = self.criterion(logits, target)
            return logits, loss
        else:
            return logits


class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ----------------------------- moco setting ------------------------------- #
        self.basic_lr_per_img = 30.0 / 256.0
        self.max_epochs = 100
        self.scheduler = "multistep"  # "cos"
        self.epoch_of_stage = [60, 80]
        self.save_folder_prefix = ""

    def get_model(self):
        if "model" not in self.__dict__:
            self.model = ResNetWithLinear()
        return self.model

    def get_data_loader(self, batch_size, is_distributed):
        if "data_loader" not in self.__dict__:

            from momentum_teacher.data.dataset import ImageNet
            from momentum_teacher.data.transforms import typical_imagenet_transform

            train_set = ImageNet(True, typical_imagenet_transform(True))
            eval_set = ImageNet(False, typical_imagenet_transform(False))

            if is_distributed:
                batch_size = batch_size // dist.get_world_size()

            train_dataloader_kwargs = {
                "num_workers": 10,
                "pin_memory": False,
                "batch_size": batch_size,
                "shuffle": False,
                "drop_last": True,
                "sampler": torch.utils.data.distributed.DistributedSampler(train_set) if is_distributed else None,
            }
            train_loader = torch.utils.data.DataLoader(train_set, **train_dataloader_kwargs)

            eval_loader = torch.utils.data.DataLoader(
                eval_set,
                batch_size=100,
                shuffle=False,
                num_workers=2,
                pin_memory=False,
                drop_last=False,
                sampler=torch.utils.data.distributed.DistributedSampler(eval_set) if is_distributed else None,
            )
            self.data_loader = {"train": train_loader, "eval": eval_loader}
        return self.data_loader

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            lr = self.basic_lr_per_img * batch_size
            self.optimizer = torch.optim.SGD(
                self.model.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0, nesterov=False
            )
        return self.optimizer
