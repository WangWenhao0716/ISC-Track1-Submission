# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

import os
import math
import torch
import shutil


def adjust_learning_rate_iter(optimizer, iters, args, ITERS_PER_EPOCH=5004):
    """Decay the learning rate based on schedule"""
    total_iters = ITERS_PER_EPOCH * args.total_epochs

    lr = args.lr
    if args.scheduler == "cos":  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * iters / total_iters))
    elif args.scheduler == "warmcos":
        warmup_total_iters = ITERS_PER_EPOCH * args.warmup_epochs
        if iters <= warmup_total_iters:
            lr_start = 1e-6
            lr = (lr - lr_start) * iters / float(warmup_total_iters) + lr_start
        else:
            lr *= 0.5 * (1.0 + math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters)))
    elif args.scheduler == "multistep":  # stepwise lr schedule
        milestones = [int(total_iters * milestone / args.total_epochs) for milestone in args.milestones]
        for milestone in milestones:
            lr *= 0.1 if iters >= milestone else 1.0
    else:
        raise ValueError("Scheduler version {} not supported.".format(args.scheduler))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(state, is_best, save, model_name=""):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, model_name + "_ckpt.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, model_name + "_best_ckpt.pth.tar")
        shutil.copyfile(filename, best_filename)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def parse_devices(gpu_ids):
    if "-" in gpu_ids:
        gpus = gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        parsed_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))
        return parsed_ids
    else:
        return gpu_ids


class AvgMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.reset()
        self.val = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
