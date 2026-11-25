"""Operations for normalized 1-D temporal spans."""
import torch

def cw_to_se(spans: torch.Tensor) -> torch.Tensor:
    center, width = spans.unbind(-1)
    return torch.stack((center - width * 0.5, center + width * 0.5), dim=-1)

def se_to_cw(spans: torch.Tensor) -> torch.Tensor:
    start, end = spans.unbind(-1)
    return torch.stack(((start + end) * 0.5, (end - start).clamp_min(0)), dim=-1)

def pairwise_iou(spans_a: torch.Tensor, spans_b: torch.Tensor) -> torch.Tensor:
    a0, a1 = spans_a.unbind(-1); b0, b1 = spans_b.unbind(-1)
    inter = (torch.minimum(a1[:, None], b1[None, :]) - torch.maximum(a0[:, None], b0[None, :])).clamp_min(0)
    la = (a1-a0).clamp_min(0); lb = (b1-b0).clamp_min(0)
    union = la[:, None] + lb[None, :] - inter
    return torch.where(union > 0, inter / union.clamp_min(1e-8), torch.zeros_like(union))

def pairwise_giou(spans_a: torch.Tensor, spans_b: torch.Tensor) -> torch.Tensor:
    iou = pairwise_iou(spans_a, spans_b)
    a0, a1 = spans_a.unbind(-1); b0, b1 = spans_b.unbind(-1)
    la = (a1-a0).clamp_min(0); lb = (b1-b0).clamp_min(0)
    inter = (torch.minimum(a1[:, None], b1[None, :]) - torch.maximum(a0[:, None], b0[None, :])).clamp_min(0)
    union = la[:, None] + lb[None, :] - inter
    enclosing = (torch.maximum(a1[:, None], b1[None, :]) - torch.minimum(a0[:, None], b0[None, :])).clamp_min(1e-8)
    return iou - (enclosing - union) / enclosing

def elementwise_iou(spans_a: torch.Tensor, spans_b: torch.Tensor) -> torch.Tensor:
    inter = (torch.minimum(spans_a[..., 1], spans_b[..., 1]) - torch.maximum(spans_a[..., 0], spans_b[..., 0])).clamp_min(0)
    union = (spans_a[..., 1]-spans_a[..., 0]).clamp_min(0) + (spans_b[..., 1]-spans_b[..., 0]).clamp_min(0) - inter
    return torch.where(union > 0, inter / union.clamp_min(1e-8), torch.zeros_like(union))
