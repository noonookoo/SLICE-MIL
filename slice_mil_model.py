from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


__all__ = ["SLICEMILHead", "GatedAttentionPool"]


def _as_bool_mask(instance_mask: Optional[torch.Tensor], batch: int, length: int, device) -> torch.Tensor:
    if instance_mask is None:
        return torch.ones(batch, length, device=device, dtype=torch.bool)
    mask = instance_mask.to(device=device)
    if mask.dim() == 3:
        if mask.shape[-1] != 1:
            raise ValueError(f"instance_mask with 3 dims must be [B, N, 1], got {tuple(mask.shape)}")
        mask = mask[..., 0]
    if mask.dim() != 2 or tuple(mask.shape) != (batch, length):
        raise ValueError(f"instance_mask must be [B, N] or [B, N, 1], got {tuple(instance_mask.shape)}")
    if mask.dtype != torch.bool:
        mask = mask > 0.5
    if not mask.any(dim=1).all():
        bad = (~mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"each bag must contain at least one valid instance; empty bag indices: {bad}")
    return mask


def _masked_softmax(scores: torch.Tensor, valid: torch.Tensor, dim: int = 1) -> torch.Tensor:
    v = valid
    while v.dim() < scores.dim():
        v = v.unsqueeze(-1)
    if not v.any(dim=dim, keepdim=True).all():
        raise ValueError("masked softmax received an all-invalid slice along the softmax dimension")
    scores_fp32 = scores.float().masked_fill(~v, -1e9)
    return torch.softmax(scores_fp32, dim=dim).to(dtype=scores.dtype)


class GatedAttentionPool(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 128, temperature: float = 0.8, dropout: float = 0.0):
        super().__init__()
        self.temperature = float(temperature)
        self.drop = nn.Dropout(float(dropout))
        self.v = nn.Linear(int(dim), int(hidden_dim))
        self.u = nn.Linear(int(dim), int(hidden_dim))
        self.w = nn.Linear(int(hidden_dim), 1, bias=False)

    def forward(self, h: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.drop(h)
        scores = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x)))
        scores = scores / max(self.temperature, 1e-6)
        attn = _masked_softmax(scores, valid, dim=1)
        z = torch.sum(attn.float() * h.float(), dim=1).to(dtype=h.dtype)
        return z, attn, scores


class ZeroInitResidualAdapter(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, int(dim) // int(reduction))
        self.net = nn.Sequential(
            nn.Linear(int(dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, int(dim)),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class NullSpaceBasisDecomposition(nn.Module):
    def __init__(self, dim: int, rank: int = 64):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(min(int(rank), int(dim)))
        self.basis = nn.Parameter(torch.empty(self.dim, self.rank))
        nn.init.orthogonal_(self.basis)

    def forward(self, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if f.shape[-1] != self.dim:
            raise ValueError(f"expected feature dim {self.dim}, got {f.shape[-1]}")
        u, _ = torch.linalg.qr(self.basis.float(), mode="reduced")
        u = u[:, : self.rank]
        f_fp32 = f.float()
        coeff = torch.einsum("bnd,dr->bnr", f_fp32, u)
        h_s0 = torch.einsum("bnr,dr->bnd", coeff, u).to(dtype=f.dtype)
        h_res = (f_fp32 - h_s0.float()).to(dtype=f.dtype)
        basis_coeff_l1 = coeff.abs().sum(dim=-1)
        return h_s0, h_res, basis_coeff_l1


class ContextModulatedResidualRouter(nn.Module):
    def __init__(self, dim: int, hidden_ratio: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        hidden = max(1, self.dim // int(hidden_ratio))
        dropout = float(dropout)
        self.norm = nn.LayerNorm(self.dim)
        self.context_to_film = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * self.dim),
        )
        self.gate_e1 = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.dim),
        )
        self.gate_e2 = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.dim),
        )
        nn.init.zeros_(self.context_to_film[-1].weight)
        nn.init.zeros_(self.context_to_film[-1].bias)
        nn.init.zeros_(self.gate_e1[-1].weight)
        nn.init.zeros_(self.gate_e1[-1].bias)
        nn.init.zeros_(self.gate_e2[-1].weight)
        nn.init.zeros_(self.gate_e2[-1].bias)

    def forward(
        self,
        h_res: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma, beta = self.context_to_film(context.float()).chunk(2, dim=-1)
        h_tilde = self.norm(h_res).float() * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        gate_logit_e1 = self.gate_e1(h_tilde)
        gate_logit_e2 = self.gate_e2(h_tilde)
        gate_e1 = torch.sigmoid(gate_logit_e1)
        gate_e2 = torch.sigmoid(gate_logit_e2)
        return (
            h_tilde.to(dtype=h_res.dtype),
            gate_e1.to(dtype=h_res.dtype),
            gate_e2.to(dtype=h_res.dtype),
            gate_logit_e1.to(dtype=h_res.dtype),
            gate_logit_e2.to(dtype=h_res.dtype),
        )


class SLICEMILHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        basis_rank: int = 64,
        attn_hidden_dim: int = 128,
        attn_temperature: float = 0.8,
        attn_dropout: float = 0.0,
        router_hidden_ratio: int = 8,
        router_dropout: float = 0.0,
        adapter_reduction: int = 4,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.adapter = ZeroInitResidualAdapter(dim=self.input_dim, reduction=int(adapter_reduction))
        self.nsbd = NullSpaceBasisDecomposition(dim=self.input_dim, rank=int(basis_rank))
        self.res_context_pool = GatedAttentionPool(
            dim=self.input_dim,
            hidden_dim=int(attn_hidden_dim),
            temperature=float(attn_temperature),
            dropout=float(attn_dropout),
        )
        self.router = ContextModulatedResidualRouter(
            dim=self.input_dim,
            hidden_ratio=int(router_hidden_ratio),
            dropout=float(router_dropout),
        )
        self.final_pool = GatedAttentionPool(
            dim=self.input_dim,
            hidden_dim=int(attn_hidden_dim),
            temperature=float(attn_temperature),
            dropout=float(attn_dropout),
        )
        self.scale_e1 = nn.Parameter(torch.ones(1, 1, self.input_dim) * 0.1)
        self.scale_e2 = nn.Parameter(torch.ones(1, 1, self.input_dim) * 0.1)

    def forward(
        self,
        h: torch.Tensor,
        instance_mask: Optional[torch.Tensor] = None,
        zero_e1: bool = False,
        zero_e2: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if h.dim() != 3:
            raise ValueError(f"h must be [B, N, D], got {tuple(h.shape)}")
        batch, length, dim = h.shape
        if dim != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {dim}")

        valid = _as_bool_mask(instance_mask, batch, length, h.device)
        f = self.adapter(h)
        h_s0, h_res, basis_coeff_l1 = self.nsbd(f)
        c_res, attn_context, scores_context = self.res_context_pool(h_res, valid=valid)
        h_tilde_res, gate_e1, gate_e2, gate_logit_e1, gate_logit_e2 = self.router(h_res, c_res)

        if zero_e1:
            gate_e1 = torch.zeros_like(gate_e1)
        if zero_e2:
            gate_e2 = torch.zeros_like(gate_e2)

        h_e1 = (h_tilde_res * gate_e1) * self.scale_e1
        h_e2 = (h_tilde_res * gate_e2) * self.scale_e2

        z_full, attn_full, scores_full = self.final_pool(h_s0 + h_e1 + h_e2, valid=valid)
        z_no_e2, attn_no_e2, scores_no_e2 = self.final_pool(h_s0 + h_e1, valid=valid)
        z_no_e1, attn_no_e1, scores_no_e1 = self.final_pool(h_s0 + h_e2, valid=valid)
        z_base, attn_base, scores_base = self.final_pool(h_s0, valid=valid)

        aux = {
            "z_stage2": z_full,
            "z_stage1": z_no_e2,
            "z_e1_off": z_no_e1,
            "z_base": z_base,
            "z_no_e2": z_no_e2,
            "z_no_e1": z_no_e1,
            "h_s0": h_s0,
            "h_res": h_res,
            "h_tilde_res": h_tilde_res,
            "h_e1": h_e1,
            "h_e2": h_e2,
            "gate_e1": gate_e1,
            "gate_e2": gate_e2,
            "g_early": gate_e1,
            "g_sev": gate_e2,
            "gate_logit_e1": gate_logit_e1,
            "gate_logit_e2": gate_logit_e2,
            "logit_early": gate_logit_e1,
            "logit_sev": gate_logit_e2,
            "attn": attn_full,
            "scores": scores_full,
            "attn_full": attn_full,
            "scores_full": scores_full,
            "attn_no_e2": attn_no_e2,
            "scores_no_e2": scores_no_e2,
            "attn_no_e1": attn_no_e1,
            "scores_no_e1": scores_no_e1,
            "attn_base": attn_base,
            "scores_base": scores_base,
            "attn_context": attn_context,
            "scores_context": scores_context,
            "h_path_norm": basis_coeff_l1,
            "valid": valid,
        }
        return z_full, aux
