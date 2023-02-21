from torch import optim
import math


def SelectOptim(param, kind, lr):
    if kind == "adamw":
        optimizer = optim.AdamW(param, lr=lr, weight_decay=1e-3)
        return optimizer
    if kind == "adam":
        optimizer = optim.Adam(param.parameters(), lr=lr)
        return optimizer


def SelectScheduler(optimizer, kind, t=100, eta=1e-5, exp_g=0.98, linear_g=1.3, warmEpoch=5, ):
    if kind == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t, eta_min=eta)
        return scheduler
    elif kind == "exponent":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_g)
        return scheduler
    elif kind == "linear":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=linear_g)
        return scheduler
    elif kind == "warmup":
        lbd = lambda epoch: (epoch + 1) / warmEpoch if epoch < warmEpoch else \
            0.5 * (math.cos(epoch / t * math.pi) + 1) + 0.01
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lbd)
        return scheduler
    else:
        raise (Exception("there is no {} scheduler in the config, please check".format(kind)))
