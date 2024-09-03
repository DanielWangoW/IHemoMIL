import math
from abc import ABC
from typing import Dict, Optional

import torch
from torch import nn


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


class GlobalAveragePooling(MILPooling):
    """FMIL==>GAP: Global Average Pooling."""
    def __init__(
        self,
        d_in: int,
        n_clz: int,
        dropout: float = 0,
        p_rank: float = 0.2,
        apply_positional_encoding: bool = False,
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            p_rank=p_rank,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.bag_classifier = nn.Linear(d_in, n_clz)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = instance_embeddings
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        instance_embeddings = instance_embeddings.transpose(2, 1)
        cam = self.bag_classifier.weight @ instance_embeddings
        bag_embeddings = instance_embeddings.mean(dim=-1)
        bag_logits = self.bag_classifier(bag_embeddings)
        return {
            "bag_logits": bag_logits,
            "interpretation": cam,
            "bag_embeddings": bag_embeddings,
            "feature": x,
        }


class RankAveragePooling(MILPooling):
    """FMIL==>RAP: Ranking-based Average Pooling under RAMP."""
    def __init__(
        self,
        d_in: int,
        n_clz: int,
        dropout: float = 0.1,
        p_rank: float = 0.2,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout,
            p_rank=p_rank,
            apply_positional_encoding = apply_positional_encoding,
        )

        self.instance_classifier = nn.Linear(d_in, n_clz)
        self.p_rank = p_rank

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = instance_embeddings
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)

        d_rank = int(self.d_in * self.p_rank)
        rap_instance_embeddings, _ = torch.sort(instance_embeddings, dim=1, descending=True)

        rap_cam = self.instance_classifier.weight @ instance_embeddings.transpose(2, 1)
        _, indices = torch.sort(rap_cam, dim=1, descending=True)
        rap_cam[indices>=d_rank] = 0
        bag_embedding = rap_instance_embeddings[:, :d_rank, :].mean(dim=1)
        bag_logits = self.instance_classifier(bag_embedding)
        return {
            "bag_logits": bag_logits,
            "interpretation": rap_cam,
            "bag_embeddings": bag_embedding,
            "feature": x,
        }
    

class MILAttentionPooling(MILPooling):
    """FMIL==>ATP: Attention MIL pooling."""
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
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        
        self.bag_classifier = nn.Linear(d_in, n_clz)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        attn = self.attention_head(instance_embeddings)
        
        instance_embeddings = instance_embeddings * attn
        bag_embedding = torch.mean(instance_embeddings, dim=1)
        
        bag_logits = self.bag_classifier(bag_embedding)
        return {
            "bag_logits": bag_logits,
            "interpretation": attn.repeat(1, 1, self.n_clz).transpose(1, 2),
        }


class RankWeightPooling(MILPooling):
    """FMIL==>RATP: Ranking-based Attention MIL pooling."""
    def __init__(
        self,
        d_in: int,
        n_clz: int,
        d_attn: int = 8,
        dropout: float = 0.1,
        p_alpha: float = 0.05,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            p_alpha=p_alpha,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        
        self.bag_classifier = nn.Linear(d_in, n_clz)
        self.p_alpha = p_alpha

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)

        _, indices = torch.sort(instance_embeddings, dim=1, descending=True)
        exp_prob = self.p_alpha ** ((1-self.p_alpha) ** indices)
        instance_embeddings = instance_embeddings * exp_prob
        w_attn = self.attention_head(instance_embeddings)
        instance_embeddings = instance_embeddings * w_attn
        bag_embedding = torch.mean(instance_embeddings, dim=1)
        bag_logits = self.bag_classifier(bag_embedding)
        return {
            "bag_logits": bag_logits,
            # Attention is not class wise, so repeat for each class
            "interpretation": w_attn.repeat(1, 1, self.n_clz).transpose(1, 2),
        }


class MILInstancePooling(MILPooling):
    """PMIL==>INP: Instance MIL pooling."""
    def __init__(
        self,
        d_in: int,
        n_clz: int,
        dropout: float = 0.1,
        p_rank: float = 0.2,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            p_rank=p_rank,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)

        instance_logits = self.instance_classifier(instance_embeddings)
        bag_logits = instance_logits.mean(dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": instance_logits.transpose(1, 2),
        }


class MILAdditivePooling(MILPooling):
    """PMIL==>ADP: Additive MIL pooling."""
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
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        attn = self.attention_head(instance_embeddings)
        instance_embeddings = instance_embeddings * attn
        instance_logits = self.instance_classifier(instance_embeddings)
        bag_logits = instance_logits.mean(dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": (instance_logits * attn).transpose(1, 2),
            "instance_logits": instance_logits.transpose(1, 2),
            "attn": attn,
        }


class MILRankAdditivePooling(MILPooling):
    """PMIL==>RADP: Ranking-based Additive MIL pooling."""
    def __init__(
        self,
        d_in: int,
        n_clz: int,
        d_attn: int = 8,
        dropout: float = 0.1,
        p_alpha: float = 0.05,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            p_alpha=p_alpha,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)
        self.p_alpha = p_alpha

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        _, indices = torch.sort(instance_embeddings, dim=1, descending=True)
        exp_prob = self.p_alpha ** ((1-self.p_alpha) ** indices)
        instance_embeddings = instance_embeddings * exp_prob
        w_attn = self.attention_head(instance_embeddings)
        instance_embeddings = instance_embeddings * w_attn
        instance_logits = self.instance_classifier(instance_embeddings)
        bag_logits = instance_logits.mean(dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": (instance_logits * w_attn).transpose(1, 2),
            # Also return additional outputs
            "instance_logits": instance_logits.transpose(1, 2),
            "attn": w_attn,
        }
    

class MILConjunctivePooling(MILPooling):
    """PMIL==>COP: Conjunctive MIL pooling."""
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
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = instance_embeddings
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        attn = self.attention_head(instance_embeddings)
        instance_logits = self.instance_classifier(instance_embeddings)
        weighted_instance_logits = instance_logits * attn
        bag_logits = torch.mean(weighted_instance_logits, dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": weighted_instance_logits.transpose(1, 2),
            # Also return additional outputs
            "instance_logits": instance_logits.transpose(1, 2),
            "bag_embeddings": bag_logits,
            "attn": attn,
            "feature": x,
        }


class MILRankConjunctivePooling(MILPooling):
    """PMIL==>RCOP: Conjunctive MIL pooling."""
    def __init__(
        self,
        d_in: int,
        n_clz: int,
        d_attn: int = 8,
        dropout: float = 0.1,
        p_alpha: float = 0.5,
        apply_positional_encoding: bool = True,
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            p_alpha=p_alpha,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        self.instance_classifier = nn.Linear(d_in, n_clz)
        self.p_alpha = p_alpha

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        x = instance_embeddings
        instance_embeddings = instance_embeddings.transpose(1, 2)
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        _, indices = torch.sort(instance_embeddings, dim=1, descending=True)
        exp_prob = self.p_alpha ** ((1-self.p_alpha) ** indices)
        instance_embeddings = instance_embeddings * exp_prob   
        w_attn = self.attention_head(instance_embeddings)
        instance_logits = self.instance_classifier(instance_embeddings)
        weighted_instance_logits = instance_logits * w_attn
        bag_logits = torch.mean(weighted_instance_logits, dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": weighted_instance_logits.transpose(1, 2),
            # Also return additional outputs
            "instance_logits": instance_logits.transpose(1, 2),
            "attn": w_attn,
            "bag_embeddings": bag_logits,
            "feature": x,
        }
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Batch, ts len, d_model
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
        x = x + x_pe
        return x
