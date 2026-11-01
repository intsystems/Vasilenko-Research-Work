import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Row-wise gamma * x / sqrt(mean(x**2) + eps)."""

    def __init__(self, dim: int, eps: float = 1e-8, gamma: float = 1.0):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim) * gamma)

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms

    def get_gamma_max(self):
        return torch.max(torch.abs(self.gamma)).item()


class IterativeTransformerBlock(nn.Module):
    """
    Iterative Transformer Block.
    """

    def __init__(self, dim: int, num_heads: int, ffn_hidden_dim: int, nu: float = 0.5):
        super().__init__()
        self.nu = nu
        self.dim = dim

        self.msa = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden_dim), nn.GELU(), nn.Linear(ffn_hidden_dim, dim)
        )

        self.rms1 = RMSNorm(dim)
        self.rms2 = RMSNorm(dim)
        self.rms_final = RMSNorm(dim)

    def forward_subblocks(self, Y):
        """
        Executes the TB(Y) logic:
        1. Y' = RMS(Y + MSA(Y))
        2. TB(Y) = RMS(Y' + FFN(Y'))
        """
        attn_out, _ = self.msa(Y, Y, Y)
        Y_prime = self.rms1(Y + attn_out)

        ffn_out = self.ffn(Y_prime)
        TB_Y = self.rms2(Y_prime + ffn_out)

        return TB_Y

    def forward_step(self, Y, C):
        """
        Single recurrent step:
        Y(t+1) = RMS(Y(t) + nu * (C + TB(Y(t))))
        """
        if C.dim() == 2:
            C = C.unsqueeze(0).expand(Y.shape[0], -1, -1)

        TB_Y = self.forward_subblocks(Y)
        U = Y + self.nu * (C + TB_Y)
        Y_next = self.rms_final(U)
        return Y_next
