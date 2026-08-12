import torch
import torch.nn as nn
import numpy as np

class RealNVPStep(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 64):
        """
        Single RealNVP coupling layer.
        """
        super(RealNVPStep, self).__init__()
        self.dim = dim
        self.split_dim = dim // 2
        
        # Scale and translation networks
        self.s_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - self.split_dim),
            nn.Tanh()
        )
        
        self.t_net = nn.Sequential(
            nn.Linear(self.split_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim - self.split_dim)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward transformation: x -> z, and computes log determinant Jacobian.
        """
        x1 = x[..., :self.split_dim]
        x2 = x[..., self.split_dim:]
        
        s = self.s_net(x1)
        t = self.t_net(x1)
        
        # Transform x2
        z1 = x1
        z2 = x2 * torch.exp(s) + t
        
        z = torch.cat([z1, z2], dim=-1)
        log_det_j = torch.sum(s, dim=-1)
        
        return z, log_det_j

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inverse transformation: z -> x.
        """
        z1 = z[..., :self.split_dim]
        z2 = z[..., self.split_dim:]
        
        s = self.s_net(z1)
        t = self.t_net(z1)
        
        x1 = z1
        x2 = (z2 - t) * torch.exp(-s)
        
        return torch.cat([x1, x2], dim=-1)


class NormalizingFlow(nn.Module):
    def __init__(self, dim: int, num_layers: int = 4, hidden_dim: int = 64):
        """
        Normalizing Flow mapping a simple base distribution (Gaussian) to complex target space.
        """
        super(NormalizingFlow, self).__init__()
        self.dim = dim
        self.layers = nn.ModuleList([RealNVPStep(dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Transform x -> z and calculate total log Jacobian.
        """
        log_det_total = torch.zeros(x.shape[:-1], device=x.device)
        curr = x
        for layer in self.layers:
            curr, log_det = layer(curr)
            log_det_total += log_det
            # Permute dimensions to mix information
            curr = torch.flip(curr, dims=[-1])
            
        return curr, log_det_total

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Transform z -> x.
        """
        curr = z
        for layer in reversed(self.layers):
            curr = torch.flip(curr, dims=[-1])
            curr = layer.inverse(curr)
        return curr
