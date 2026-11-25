"""Hybrid Modulated Bi-Mamba block following HM-Net Eqs. (5)-(11)."""
import torch
from torch import nn
from torch.nn import functional as F
from .mamba_block import BidirectionalMamba

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__(); self.p = float(p)
    def forward(self, x):
        if not self.training or self.p <= 0: return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

class HMBBlock(nn.Module):
    def __init__(self, dim=256, heads=8, state_dim=32, max_seq_len=200, dropout=0.1, drop_path=0.0, gate_init=1e-3):
        super().__init__()
        self.main_norm = nn.LayerNorm(dim); self.ctx_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mod_proj = nn.Linear(dim, dim)
        self.gamma_mod = nn.Parameter(torch.tensor(float(gate_init)))
        self.pos_bias = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        nn.init.trunc_normal_(self.pos_bias, std=0.02)
        self.gamma_bias = nn.Parameter(torch.tensor(float(gate_init)))
        self.max_seq_len = max_seq_len
        self.mamba_norm = nn.LayerNorm(dim)
        self.mamba = BidirectionalMamba(dim, state_dim)
        self.gamma_mamba = nn.Parameter(torch.tensor(float(gate_init)))
        self.drop_path = DropPath(drop_path)

    @staticmethod
    def masked_mean(x, mask):
        w = mask.unsqueeze(-1).to(x.dtype)
        return (x * w).sum(1) / w.sum(1).clamp_min(1.0)

    def forward(self, x_main, x_ctx, main_mask=None, ctx_mask=None):
        b, lm, d = x_main.shape
        if main_mask is None: main_mask = torch.ones((b, lm), dtype=torch.bool, device=x_main.device)
        if ctx_mask is None: ctx_mask = torch.ones((b, x_ctx.shape[1]), dtype=torch.bool, device=x_ctx.device)
        xm = self.main_norm(x_main); xc = self.ctx_norm(x_ctx)
        # MultiheadAttention produces NaNs if every context token is masked.
        # Keep one zero sentinel unmasked for empty rows, then mask pooled context to zero.
        empty_ctx = ~ctx_mask.any(dim=1)
        safe_ctx_mask = ctx_mask.clone()
        if empty_ctx.any():
            safe_ctx_mask[empty_ctx, 0] = True
            xc = xc.clone(); xc[empty_ctx, 0] = 0
        attn, _ = self.cross_attn(xm, xc, xc, key_padding_mask=~safe_ctx_mask, need_weights=False)
        global_ctx = self.masked_mean(xc, ctx_mask)
        fused = xm + attn + torch.tanh(self.gamma_mod) * self.mod_proj(global_ctx).unsqueeze(1)
        if lm < self.max_seq_len:
            pb = F.interpolate(self.pos_bias.transpose(1, 2), size=lm, mode='linear', align_corners=False).transpose(1, 2)
            fused = fused + torch.tanh(self.gamma_bias) * pb
        seq = self.mamba(self.mamba_norm(fused), padding_mask=main_mask)
        out = x_main + self.drop_path(fused + torch.tanh(self.gamma_mamba) * seq)
        return out * main_mask.unsqueeze(-1).to(out.dtype)
