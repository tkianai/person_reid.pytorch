from __future__ import absolute_import
from __future__ import print_function

import torch
from bisect import bisect_right

AVAI_SCH = ['single_step', 'multi_step', 'cosine', 'warmup_multi_step', 'warmup_cosine']


def build_lr_schedulers(optimizers, *args):

    lr_schedulers = []
    for i, optimizer in enumerate(optimizers):
        if args[i].get('lr_scheduler') is None:
            lr_schedulers.append(None)
        else:
            lr_schedulers.append(build_lr_scheduler(optimizer, **args[i]))
    return lr_schedulers


def build_lr_scheduler(optimizer, lr_scheduler='single_step', step_size=1, gamma=0.1, max_epoch=1, warmup_factor=0.01, warmup_method="linear", warmup_epoch=10):
    """A function wrapper for building a learning rate scheduler.

    Args:
        optimizer (Optimizer): an Optimizer.
        lr_scheduler (str, optional): learning rate scheduler method. Default is single_step.
        step_size (int or list, optional): step size to decay learning rate. When ``lr_scheduler``
            is "single_step", ``step_size`` should be an integer. When ``lr_scheduler`` is
            "multi_step", ``step_size`` is a list. Default is 1.
        gamma (float, optional): decay rate. Default is 0.1.
        max_epoch (int, optional): maximum epoch (for cosine annealing). Default is 1.

    Examples::
        >>> # Decay learning rate by every 20 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='single_step', step_size=20
        >>> )
        >>> # Decay learning rate at 30, 50 and 55 epochs.
        >>> scheduler = torchreid.optim.build_lr_scheduler(
        >>>     optimizer, lr_scheduler='multi_step', step_size=[30, 50, 55]
        >>> )
    """
    if lr_scheduler not in AVAI_SCH:
        raise ValueError('Unsupported scheduler: {}. Must be one of {}'.format(lr_scheduler, AVAI_SCH))
    
    if lr_scheduler == 'single_step':
        if isinstance(step_size, list):
            step_size = step_size[-1]
        
        if not isinstance(step_size, int):
            raise TypeError(
                'For single_step lr_scheduler, step_size must '
                'be an integer, but got {}'.format(type(step_size))
            )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

    elif lr_scheduler == 'multi_step':
        if not isinstance(step_size, list):
            raise TypeError(
                'For multi_step lr_scheduler, step_size must '
                'be a list, but got {}'.format(type(step_size))
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=step_size, gamma=gamma
        )

    elif lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(max_epoch)
        )
    elif lr_scheduler == 'warmup_multi_step':
        scheduler = WarmupMultiStepLR(optimizer, milestones=step_size, gamma=gamma, warmup_factor=warmup_factor, warmup_epoch=warmup_epoch, warmup_method=warmup_method)

    elif lr_scheduler == 'warmup_cosine':
        scheduler = WarmupCosineAnnealingLR(optimizer, float(
            max_epoch), warmup_factor=warmup_factor, warmup_epoch=warmup_epoch, warmup_method=warmup_method)

    return scheduler


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=0.1,
        warmup_epoch=10,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epoch = warmup_epoch
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epoch:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_epoch
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=0,
        warmup_factor=0.1,
        warmup_epoch=10,
        warmup_method="linear",
        last_epoch=-1,
    ):

        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_factor = warmup_factor
        self.warmup_epoch = warmup_epoch
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epoch:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_epoch
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        return [warmup_factor * (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2) for base_lr in self.base_lrs]
        
