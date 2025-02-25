import numpy as np
def rand_bbox_3d(size, lam):
    D, H, W = size[2], size[3], size[4]
    cut_rat = np.sqrt(1. - lam)
    cut_d = int(D * cut_rat)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    cz = np.random.randint(D)
    cy = np.random.randint(H)
    cx = np.random.randint(W)

    bdz1 = np.clip(cz - cut_d // 2, 0, D)
    bdy1 = np.clip(cy - cut_h // 2, 0, H)
    bdx1 = np.clip(cx - cut_w // 2, 0, W)
    bdz2 = np.clip(cz + cut_d // 2, 0, D)
    bdy2 = np.clip(cy + cut_h // 2, 0, H)
    bdx2 = np.clip(cx + cut_w // 2, 0, W)

    return bdx1, bdy1, bdz1, bdx2, bdy2, bdz2


def cutout(input, beta):
    lam = np.random.beta(beta, beta)
    bdx1, bdy1, bdz1, bdx2, bdy2, bdz2 = rand_bbox_3d(input.size(), lam)

    input[:, :, bdz1:bdz2, bdy1:bdy2, bdx1:bdx2] = 0

    lam = 1 - ((bdx2 - bdx1) * (bdy2 - bdy1) * (bdz2 - bdz1) / (input.size()[-1] * input.size()[-2] * input.size()[-3]))

    return input, lam
