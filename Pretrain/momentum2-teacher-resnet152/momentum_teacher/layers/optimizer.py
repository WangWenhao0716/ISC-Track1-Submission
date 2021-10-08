# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

import torch
from torch.optim.optimizer import Optimizer, required


class LARS_SGD(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.
    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eta (float, optional): LARS coefficient

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required, momentum=0.9, dampening=0, weight_decay=0.0001, eta=0.001, eps=1e-8):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, eta=eta)
        super(LARS_SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            eta = group["eta"]
            lr = group["lr"]
            lars_exclude = group["lars_exclude"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                d_p = p.grad.data

                # ------------------------------ grad_norm without grad of weight-decay ---------------- #
                if not lars_exclude:
                    weight_norm = torch.norm(p.data)
                    grad_norm = torch.norm(d_p)
                    if weight_norm * grad_norm > 1e-8:
                        local_lr = eta * weight_norm / (grad_norm + weight_decay * weight_norm)
                        actual_lr = local_lr * lr
                        # local_lr = eta * weight_norm / (grad_norm + weight_decay * weight_norm + 1e-8)
                    else:
                        actual_lr = lr
                else:
                    actual_lr = lr

                # Update the momentum term
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.zeros_like(p.data)
                else:
                    buf = param_state["momentum_buffer"]
                buf.mul_(momentum).add_(d_p + weight_decay * p.data, alpha=actual_lr)
                p.data.add_(-buf)
        return loss
