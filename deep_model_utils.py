from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.autograd import Function


@dataclass
class EncodedLabels:
    name: str
    values: np.ndarray
    classes: list[str]
    valid_mask: np.ndarray


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class SparseFeatureGate(nn.Module):
    def __init__(self, input_dim: int, use_l1_gate: bool = True, use_hard_concrete_gate: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.use_l1_gate = use_l1_gate or not use_hard_concrete_gate
        self.use_hard_concrete_gate = use_hard_concrete_gate
        self.logits = nn.Parameter(torch.zeros(input_dim))

    def gate_values(self) -> torch.Tensor:
        if self.use_hard_concrete_gate:
            return torch.sigmoid(self.logits)
        return torch.sigmoid(self.logits)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.gate_values()
        return x * gates.unsqueeze(0), gates

    def sparsity_loss(self) -> torch.Tensor:
        gates = self.gate_values()
        return gates.abs().mean() if self.use_l1_gate else gates.mean()


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SparseInvariantAdversarialAutoencoderClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        adversary_specs: dict[str, int],
        grl_lambda: float = 1.0,
        use_l1_gate: bool = True,
        use_hard_concrete_gate: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.gate = SparseFeatureGate(input_dim, use_l1_gate=use_l1_gate, use_hard_concrete_gate=use_hard_concrete_gate)

        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(MLPBlock(prev_dim, hidden_dim, dropout=dropout))
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        classifier_hidden = max(latent_dim // 2, 8)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(classifier_hidden, 1),
        )

        decoder_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for hidden_dim in reversed(tuple(hidden_dims)):
            decoder_layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout / 2)])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.gradient_reversal = GradientReversal(grl_lambda)
        self.adversaries = nn.ModuleDict()
        for name, n_classes in adversary_specs.items():
            self.adversaries[name] = nn.Sequential(
                nn.Linear(latent_dim, max(latent_dim // 2, 8)),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(max(latent_dim // 2, 8), n_classes),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        gated_x, gate_values = self.gate(x)
        z = self.encoder(gated_x)
        logits = self.classifier(z).squeeze(-1)
        reconstruction = self.decoder(z)
        adversary_outputs = {name: head(self.gradient_reversal(z)) for name, head in self.adversaries.items()}
        return {
            "gated_x": gated_x,
            "gate_values": gate_values,
            "z": z,
            "logits": logits,
            "reconstruction": reconstruction,
            "adversary_outputs": adversary_outputs,
        }


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_hidden_dims(input_dim: int, latent_dim: int) -> list[int]:
    first = int(min(512, max(128, round(input_dim / 8))))
    second = int(min(128, max(latent_dim * 2, 64)))
    if second >= first:
        second = max(32, first // 2)
    return [first, second]


def encode_label_series(series, name: str) -> EncodedLabels:
    normalized = series.fillna("Unknown").astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    valid_mask = ~normalized.str.lower().eq("unknown")
    classes = sorted(normalized[valid_mask].unique().tolist())
    mapping = {label: idx for idx, label in enumerate(classes)}
    values = np.full(len(series), -1, dtype=int)
    for idx, label in enumerate(normalized):
        if label in mapping:
            values[idx] = mapping[label]
    return EncodedLabels(name=name, values=values, classes=classes, valid_mask=valid_mask.to_numpy())


def transform_with_existing_classes(series, name: str, classes: Sequence[str]) -> EncodedLabels:
    normalized = series.fillna("Unknown").astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
    mapping = {label: idx for idx, label in enumerate(classes)}
    valid_mask = normalized.isin(classes)
    values = np.full(len(series), -1, dtype=int)
    for idx, label in enumerate(normalized):
        if label in mapping:
            values[idx] = mapping[label]
    return EncodedLabels(name=name, values=values, classes=list(classes), valid_mask=valid_mask.to_numpy())


def environment_risk_variance(logits: torch.Tensor, labels: torch.Tensor, environment_matrix: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    per_sample_loss = criterion(logits, labels.float())
    penalties: list[torch.Tensor] = []
    details: dict[str, float] = {}
    for name, env_values in environment_matrix.items():
        valid_mask = env_values.ge(0)
        if int(valid_mask.sum().item()) < 4:
            details[name] = math.nan
            continue
        usable_values = env_values[valid_mask]
        usable_loss = per_sample_loss[valid_mask]
        levels = torch.unique(usable_values)
        level_means = []
        for level in levels:
            level_mask = usable_values.eq(level)
            if int(level_mask.sum().item()) < 2:
                continue
            level_means.append(usable_loss[level_mask].mean())
        if len(level_means) < 2:
            details[name] = math.nan
            continue
        stacked = torch.stack(level_means)
        penalties.append(stacked.var(unbiased=False))
        details[name] = float(stacked.var(unbiased=False).detach().cpu())
    if not penalties:
        return logits.new_tensor(0.0), details
    return torch.stack(penalties).mean(), details
