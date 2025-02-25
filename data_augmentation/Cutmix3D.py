import numpy as np
import torch


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    D = size[4]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cut_d = int(D * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    cz = np.random.randint(D)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbz1 = np.clip(cz - cut_d // 2, 0, D)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    bbz2 = np.clip(cz + cut_d // 2, 0, D)

    return bbx1, bby1, bbz1, bbx2, bby2, bbz2


def cutmix(input, target, beta):
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(input.size()[0])
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbz1, bbx2, bby2, bbz2 = rand_bbox(input.size(), lam)

    input[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = input[rand_index, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) * (bbz2 - bbz1) / (input.size()[-1] * input.size()[-2] * input.size()[-3]))

    return input, target_a, target_b, lam
