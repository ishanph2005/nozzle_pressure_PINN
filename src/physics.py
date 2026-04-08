"""
Physics-Inspired Regularization

Provides soft physical constraints for training the shock-aware PINN.
These are not full PDE residuals (the 8-station data is too coarse for
autograd-based PDE enforcement), but physics-motivated penalties that
guide the model toward physically plausible solutions.
"""

import torch
import torch.nn.functional as F


def total_variation_penalty(model, x_eval, device='cpu'):
    """
    Penalize non-smooth oscillations in each branch separately.

    Evaluates each branch on a sorted set of x-points and computes the
    mean absolute first-difference. Keeps individual branches smooth
    while allowing the sigmoid blend to create a sharp transition.
    """
    x_sorted = x_eval[x_eval[:, 0].argsort()].to(device)

    P_pre = model.pre_net(x_sorted)
    P_post = model.post_net(x_sorted)

    tv_pre = torch.mean(torch.abs(P_pre[1:] - P_pre[:-1]))
    tv_post = torch.mean(torch.abs(P_post[1:] - P_post[:-1]))

    return tv_pre + tv_post


def boundary_loss(model, norm_config, device='cpu'):
    """
    Soft constraint: at the nozzle exit (x=1.0), pressure should roughly
    scale with back pressure. Higher P_back → higher exit pressure.

    Evaluates the model at x=1.0 for several P_back values and penalizes
    deviation from a monotonically increasing trend.
    """
    max_pb = norm_config['max_p_back']
    pb_values = [0.0, 200.0, 400.0, 600.0, 650.0]
    x_exit = 1.0
    A_exit = 1.94  # area at exit from dataset

    penalties = []
    prev_P = None
    for pb in pb_values:
        pb_s = pb / max_pb
        inp = torch.tensor([[x_exit, A_exit, pb_s]], dtype=torch.float32, device=device)
        P_exit = model(inp)
        if prev_P is not None:
            # Penalize if exit pressure decreases when P_back increases
            penalties.append(F.relu(prev_P - P_exit))
        prev_P = P_exit

    if not penalties:
        return torch.tensor(0.0, device=device)
    return torch.mean(torch.stack(penalties))


def ordering_penalty(model, x_eval, device='cpu'):
    """
    Encourage branch specialization: the pre-shock branch should generally
    predict lower pressure than the post-shock branch (since pre-shock
    corresponds to supersonic flow with lower static pressure).

    This is a soft penalty — violated cases are common in the converging
    section where both branches should agree.
    """
    x_sub = x_eval[x_eval[:, 0] > 0.3].to(device)  # diverging section only
    if len(x_sub) == 0:
        return torch.tensor(0.0, device=device)

    P_pre = model.pre_net(x_sub)
    P_post = model.post_net(x_sub)

    # Penalize P_pre > P_post (wrong ordering)
    return torch.mean(F.relu(P_pre - P_post + 0.05))
