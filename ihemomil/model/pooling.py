import math
from abc import ABC
from typing import Dict, Optional

import torch
from torch import nn


# ---------------------------------------------------------------------------
# To protect the associated intellectual property, source code of ranking-based pooling is not included.
# We provide a callable interface backed by a pre-compiled implementation, so the methods can still be invoked and evaluated.
# Install the optional pre-compiled backend package `ihemomil_ranking` (distributed as a binary wheel).
# For full access, please contact the authors for authorization: <daomiao.wang@live.com>.
# ---------------------------------------------------------------------------


class RankingPoolingUnavailableError(NotImplementedError):
    """Signals that a ranking-based pooling was requested while the compiled
    `ihemomil_ranking` backend is not importable."""


def _load_ranking_impl(key: str):
    try:
        import ihemomil_ranking
    except Exception:
        return None
    getter = getattr(ihemomil_ranking, "get_pooling", None)
    if getter is None:
        return None
    try:
        return getter(key)
    except Exception:
        return None


class _ProtectedRankingPooling:
    _RANKING_KEY: Optional[str] = None

    def __new__(cls, *args, **kwargs):
        impl_cls = _load_ranking_impl(cls._RANKING_KEY)
        if impl_cls is not None:
            return impl_cls(*args, **kwargs)
        raise RankingPoolingUnavailableError(
            "install the licensed `ihemomil_ranking` backend to enable it.".format(cls.__name__)
        )


def _build_attention_head(d_in: int, d_attn: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_attn),
        nn.Tanh(),
        nn.Linear(d_attn, 1),
        nn.Sigmoid(),
    )


class MILPooling(nn.Module, ABC):
    def __init__(
        self,
        d_in: int,
        n_clz: int,
        dropout: float = 0.1,
        p_rank: float = 0.2,
        p_alpha: float = 0.05,
        apply_positional_encoding: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_clz = n_clz
        self.dropout_p = dropout
        self.p_rank = p_rank
        self.p_alpha = p_alpha
        self.apply_positional_encoding = apply_positional_encoding
        if apply_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_in)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)

    def _prepare_instances(
        self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor]
    ) -> torch.Tensor:
        prepared = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            prepared = self.positional_encoding(prepared, pos)
        if self.dropout_p > 0:
            prepared = self.dropout(prepared)
        return prepared


# ============================================================================
# Non-ranking poolings (gap, ins, atte, addi, conj)
# ============================================================================

class GlobalAveragePooling(MILPooling):
    """Simple mean-pooling across all instances."""

    def __init__(
        self,
        d_in: int,
        n_clz: int,
        dropout: float = 0.1,
        p_rank: float = 0.2,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(d_in, n_clz, dropout, p_rank, 0.0, apply_positional_encoding)
        self.bag_classifier = nn.Linear(d_in, n_clz)

    def forward(
        self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        raw_features = instance_embeddings
        instances = self._prepare_instances(instance_embeddings, pos)
        pooled = instances.mean(dim=1)
        logits = self.bag_classifier(pooled)
        return {
            "bag_logits": logits,
            "interpretation": logits.unsqueeze(2).expand(-1, -1, instances.size(1)),
            "instance_logits": logits.unsqueeze(2).expand(-1, -1, instances.size(1)),
            "bag_embeddings": pooled,
            "feature": raw_features,
        }


class MILInstancePooling(MILPooling):
    """Max-pooling across instance logits."""

    def __init__(
        self,
        d_in: int,
        n_clz: int,
        dropout: float = 0.1,
        p_rank: float = 0.2,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(d_in, n_clz, dropout, p_rank, 0.0, apply_positional_encoding)
        self.instance_classifier = nn.Linear(d_in, n_clz)

    def forward(
        self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        raw_features = instance_embeddings
        instances = self._prepare_instances(instance_embeddings, pos)
        per_instance = self.instance_classifier(instances)
        bag_logits, _ = per_instance.max(dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": per_instance.transpose(1, 2),
            "instance_logits": per_instance.transpose(1, 2),
            "bag_embeddings": bag_logits,
            "feature": raw_features,
        }


class MILAttentionPooling(MILPooling):
    """Attention-weighted instance embeddings aggregated into bag embedding, then classified."""

    def __init__(
        self,
        d_in: int,
        n_clz: int,
        d_attn: int = 8,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(d_in, n_clz, dropout, 0.0, 0.0, apply_positional_encoding)
        self.attention_head = _build_attention_head(d_in, d_attn)
        self.bag_classifier = nn.Linear(d_in, n_clz)

    def forward(
        self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        raw_features = instance_embeddings
        instances = self._prepare_instances(instance_embeddings, pos)
        attention_weights = self.attention_head(instances)
        weighted_emb = (instances * attention_weights).sum(dim=1)
        logits = self.bag_classifier(weighted_emb)
        interpretation = logits.unsqueeze(2) * attention_weights.transpose(1, 2)
        return {
            "bag_logits": logits,
            "interpretation": interpretation,
            "instance_logits": logits.unsqueeze(2).expand(-1, -1, instances.size(1)),
            "attn": attention_weights,
            "bag_embeddings": weighted_emb,
            "feature": raw_features,
        }


class MILAdditivePooling(MILPooling):
    """Attention-weighted sum of instance logits."""

    def __init__(
        self,
        d_in: int,
        n_clz: int,
        d_attn: int = 8,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(d_in, n_clz, dropout, 0.0, 0.0, apply_positional_encoding)
        self.attention_head = _build_attention_head(d_in, d_attn)
        self.instance_classifier = nn.Linear(d_in, n_clz)

    def forward(
        self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        raw_features = instance_embeddings
        instances = self._prepare_instances(instance_embeddings, pos)
        attention_weights = self.attention_head(instances)
        per_instance_logits = self.instance_classifier(instances)
        weighted_logits = per_instance_logits * attention_weights
        bag_logits = weighted_logits.sum(dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": weighted_logits.transpose(1, 2),
            "instance_logits": per_instance_logits.transpose(1, 2),
            "attn": attention_weights,
            "bag_embeddings": bag_logits,
            "feature": raw_features,
        }


class MILConjunctivePooling(MILPooling):
    """Attention-gated mean of instance logits (COP baseline)."""

    def __init__(
        self,
        d_in: int,
        n_clz: int,
        d_attn: int = 8,
        dropout: float = 0.1,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.attention_head = _build_attention_head(d_in, d_attn)
        self.instance_classifier = nn.Linear(d_in, n_clz)

    def forward(
        self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        raw_features = instance_embeddings
        prepared = self._prepare_instances(instance_embeddings, pos)

        attn = self.attention_head(prepared)
        per_instance_logits = self.instance_classifier(prepared)
        gated_logits = per_instance_logits * attn
        bag_logits = gated_logits.mean(dim=1)

        return {
            "bag_logits": bag_logits,
            "interpretation": gated_logits.transpose(1, 2),
            "instance_logits": per_instance_logits.transpose(1, 2),
            "bag_embeddings": bag_logits,
            "attn": attn,
            "feature": raw_features,
        }


# ============================================================================
# Ranking-based poolings (rap, ratte, raddi, rconj) — delegate to compiled backend
# ============================================================================
class RankAveragePooling(_ProtectedRankingPooling, MILPooling):
    _RANKING_KEY = "rap"

class RankWeightPooling(_ProtectedRankingPooling, MILPooling):
    _RANKING_KEY = "ratte"

class MILRankAdditivePooling(_ProtectedRankingPooling, MILPooling):
    _RANKING_KEY = "raddi"

class MILRankConjunctivePooling(_ProtectedRankingPooling, MILPooling):
    _RANKING_KEY = "rconj"


# ============================================================================
# Helper: Positional Encoding
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor, x_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x_pos is None:
            x_pe = self.pe[:, : x.size(1)]
        else:
            x_pe = self.pe[0, x_pos]
        return x + x_pe
