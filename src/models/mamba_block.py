"""Bidirectional Mamba interface with an explicit PyTorch selective-scan fallback.

The fallback implements an input-selective diagonal SSM with depthwise causal
convolution and gating. It is a correctness/reference path, not a speed-equivalent
replacement for the official fused mamba-ssm CUDA kernels.
"""
import torch
from torch import nn
from torch.nn import functional as F

try:
    from mamba_ssm import Mamba as _CudaMamba
    MAMBA_SSM_AVAILABLE = True
except Exception:
    _CudaMamba = None
    MAMBA_SSM_AVAILABLE = False

class SelectiveScanFallback(nn.Module):
    def __init__(self, dim: int, state_dim: int = 32, conv_kernel: int = 4, expand: int = 2):
        super().__init__()
        self.inner = dim * expand
        self.state_dim = state_dim
        self.in_proj = nn.Linear(dim, 2 * self.inner, bias=False)
        self.conv = nn.Conv1d(self.inner, self.inner, conv_kernel, groups=self.inner, padding=conv_kernel-1)
        self.x_proj = nn.Linear(self.inner, self.inner + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(self.inner, self.inner, bias=True)
        # Stable negative diagonal A, initialized as in common Mamba implementations.
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(self.inner, 1)))
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        # Keep exp(delta*A) and recurrent state in fp32 for numerical stability.
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = self._forward_fp32(x.float())
        return y.to(dtype)

    def _forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        b, length, _ = x.shape
        xz = self.in_proj(x); u, z = xz.chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :length].transpose(1, 2)
        u = F.silu(u)
        dt_raw, B, C = torch.split(self.x_proj(u), [self.inner, self.state_dim, self.state_dim], dim=-1)
        delta = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log).to(x.device)  # [inner,state]
        dA = torch.exp(delta.unsqueeze(-1) * A[None, None, :, :])
        dBx = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        state = x.new_zeros((b, self.inner, self.state_dim))
        outputs = []
        for t in range(length):
            state = dA[:, t] * state + dBx[:, t]
            outputs.append((state * C[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(outputs, dim=1)
        y = (y + u * self.D) * F.silu(z)
        return self.out_proj(y)

def make_mamba_layer(dim: int, state_dim: int = 32, prefer_cuda: bool = True) -> nn.Module:
    if prefer_cuda and MAMBA_SSM_AVAILABLE and torch.cuda.is_available():
        return _CudaMamba(d_model=dim, d_state=state_dim)
    return SelectiveScanFallback(dim, state_dim)

def masked_flip(x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    """Reverse only the valid (right-aligned padding) prefix of each sequence.

    torch.flip on a right-padded batch would move the padding to the START of the backward
    scan, so padded steps (conv bias, dt bias) would change the backward state of real clips
    and a video's output would depend on the other videos in its batch. Involution.
    """
    if padding_mask is None:
        return torch.flip(x, dims=[1])
    b, length = padding_mask.shape
    lengths = padding_mask.long().sum(1, keepdim=True)
    ar = torch.arange(length, device=x.device).unsqueeze(0).expand(b, length)
    idx = torch.where(ar < lengths, lengths - 1 - ar, ar)
    return x.gather(1, idx.unsqueeze(-1).expand_as(x))

class BidirectionalMamba(nn.Module):
    def __init__(self, dim: int, state_dim: int = 32, prefer_cuda: bool = True):
        super().__init__()
        self.forward_mamba = make_mamba_layer(dim, state_dim, prefer_cuda)
        self.backward_mamba = make_mamba_layer(dim, state_dim, prefer_cuda)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Mamba has no native padding mask. Zero invalid positions before/after each scan and
        # reverse only the valid part for the backward direction (see masked_flip).
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).to(x.dtype)
        fwd = self.forward_mamba(x)
        bwd = masked_flip(self.backward_mamba(masked_flip(x, padding_mask)), padding_mask)
        out = fwd + bwd
        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).to(out.dtype)
        return out
