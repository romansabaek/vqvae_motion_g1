import torch
import torch.nn.functional as F

# --- losses.py의 함수들 ---

def cal_loss(a, b, args):
    """ Reconstruction Loss """
    if args.recons_loss == 'l1':
        return torch.mean(torch.abs(a - b))
    elif args.recons_loss == 'l2':
        return torch.mean((a - b) ** 2)
    elif args.recons_loss == 'l1_smooth':
        return F.smooth_l1_loss(a, b)
    else:
        raise NotImplementedError(f"Reconstruction loss {args.recons_loss} is not implemented")

def cal_vel_loss(a, b, args):
    """ Velocity Loss """
    vel_a = a[:, :, 1:] - a[:, :, :-1]
    vel_b = b[:, :, 1:] - b[:, :, :-1]
    return cal_loss(vel_a, vel_b, args)