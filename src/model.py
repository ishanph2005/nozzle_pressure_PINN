"""
Shock-Aware PINN for Nozzle Pressure Prediction

Dual-branch architecture that blends pre-shock (supersonic expansion) and
post-shock (subsonic recovery) pressure predictions via a learned sigmoid
transition, enabling near-discontinuous shock representation.

    P(x) = (1 - S) * P_pre(x) + S * P_post(x)
    S(x) = sigmoid(k * (x - x_shock))

where x_shock and k are learned functions of P_back.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Feed-forward network with Tanh activation and Xavier initialization."""

    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ShockAwarePINN(nn.Module):
    """
    Dual-branch neural network for shock-aware pressure prediction.

    Architecture
    ------------
    Pre-shock branch:  MLP [3 → 64 → 64 → 64 → 1]  (isentropic expansion)
    Post-shock branch: MLP [3 → 64 → 64 → 64 → 1]  (subsonic recovery)
    Shock parameters:  MLP [1 → 32 → 32 → 2]        (x_shock, steepness from P_back)

    The sigmoid transition S(x) smoothly blends the two branches,
    with learned shock location and sharpness that depend on back pressure.
    """

    def __init__(self, hidden=64, shock_hidden=32):
        super().__init__()

        # Pre-shock branch: learns smooth supersonic pressure profile
        self.pre_net = MLP([3, hidden, hidden, hidden, 1])

        # Post-shock branch: learns subsonic recovery profile
        self.post_net = MLP([3, hidden, hidden, hidden, 1])

        # Shock parameter network: P_back → (x_shock_logit, log_steepness)
        self.shock_net = MLP([1, shock_hidden, shock_hidden, 2])

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs : Tensor [N, 3] — columns are [x, A(x), P_back_scaled].

        Returns
        -------
        Tensor [N, 1] — predicted normalized pressure P/P0.
        """
        x = inputs[:, 0:1]
        p_back = inputs[:, 2:3]

        # Each branch predicts a raw pressure value
        P_pre = self.pre_net(inputs)
        P_post = self.post_net(inputs)

        # Learn shock location and steepness from back pressure
        shock_params = self.shock_net(p_back)
        # x_shock in (-0.25, 1.25) — allows "no shock" placement outside domain
        x_shock = -0.25 + 1.5 * torch.sigmoid(shock_params[:, 0:1])
        # Steepness k in [10, 100] — controls transition sharpness
        k = 10.0 + 90.0 * torch.sigmoid(shock_params[:, 1:2])

        # Sigmoid blend: 0 before shock (pre-shock), 1 after (post-shock)
        S = torch.sigmoid(k * (x - x_shock))

        # Final pressure is a blend of the two branches
        P = (1.0 - S) * P_pre + S * P_post

        # Softplus ensures strictly positive pressure
        P = F.softplus(P, beta=5.0)

        return P

    @torch.no_grad()
    def get_shock_info(self, p_back_scaled):
        """
        Extract the learned shock location for a given back pressure.

        Returns dict with x_shock, steepness, and whether the shock is
        within the physical domain [0.125, 1.0].
        """
        device = next(self.parameters()).device
        pb = torch.tensor([[p_back_scaled]], dtype=torch.float32, device=device)
        params = self.shock_net(pb)
        x_s = (-0.25 + 1.5 * torch.sigmoid(params[:, 0:1])).item()
        k = (10.0 + 90.0 * torch.sigmoid(params[:, 1:2])).item()
        in_domain = 0.125 < x_s < 1.0
        return {'x_shock': x_s, 'steepness': k, 'in_domain': in_domain}
