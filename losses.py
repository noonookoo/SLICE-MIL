from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "CounterfactualTargets",
    "SLICEMILCounterfactualLoss",
]


@dataclass(frozen=True)
class CounterfactualTargets:
    factual: torch.Tensor
    no_e2: torch.Tensor
    no_e1: torch.Tensor
    base: torch.Tensor


def _make_counterfactual_targets(y: torch.Tensor) -> CounterfactualTargets:
    if y.dim() != 1:
        raise ValueError(f"y must be a 1D class-index tensor, got {tuple(y.shape)}")
    if not torch.is_floating_point(y) and y.dtype != torch.long:
        y = y.long()
    else:
        y = y.to(dtype=torch.long)

    no_e2 = torch.minimum(y, torch.ones_like(y))
    no_e1 = torch.where(y == 1, torch.zeros_like(y), y)
    base = torch.zeros_like(y)
    return CounterfactualTargets(factual=y, no_e2=no_e2, no_e1=no_e1, base=base)


class SLICEMILCounterfactualLoss(nn.Module):
    def __init__(
        self,
        lambda_cf: float = 0.5,
        factual_key: str = "stage2",
        no_e2_key: str = "no_e2",
        no_e1_key: str = "no_e1",
        base_key: str = "base",
    ):
        super().__init__()
        self.lambda_cf = float(lambda_cf)
        self.factual_key = str(factual_key)
        self.no_e2_key = str(no_e2_key)
        self.no_e1_key = str(no_e1_key)
        self.base_key = str(base_key)

    def forward(
        self,
        logits: Mapping[str, torch.Tensor],
        y: torch.Tensor,
        class_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        targets = _make_counterfactual_targets(y)
        required = (self.factual_key, self.no_e2_key, self.no_e1_key, self.base_key)
        missing = [key for key in required if key not in logits]
        if missing:
            raise KeyError(f"missing logits for branches: {missing}")

        loss_factual = F.cross_entropy(logits[self.factual_key], targets.factual, weight=class_weight)
        loss_no_e2 = F.cross_entropy(logits[self.no_e2_key], targets.no_e2, weight=class_weight)
        loss_no_e1 = F.cross_entropy(logits[self.no_e1_key], targets.no_e1, weight=class_weight)
        loss_base = F.cross_entropy(logits[self.base_key], targets.base, weight=class_weight)

        loss_cf = loss_no_e2 + loss_no_e1 + loss_base
        loss = loss_factual + self.lambda_cf * loss_cf

        return {
            "loss": loss,
            "loss_factual": loss_factual.detach(),
            "loss_no_e2": loss_no_e2.detach(),
            "loss_no_e1": loss_no_e1.detach(),
            "loss_base": loss_base.detach(),
            "loss_cf": loss_cf.detach(),
            "target_factual": targets.factual,
            "target_no_e2": targets.no_e2,
            "target_no_e1": targets.no_e1,
            "target_base": targets.base,
        }
