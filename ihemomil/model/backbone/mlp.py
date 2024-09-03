import torch
from torch import nn

from ihemomil.model.backbone.common import ConvBlock, manual_pad

class MLPFeatureExtractor(nn.Module):
    def __init__(self, n_in_size: int, padding_mode: str = "replicate"):
        super(MLPFeatureExtractor, self).__init__()
        self.n_in_size = n_in_size
        self.n_out_size = n_in_size * 128
        self.mlp = nn.Sequential(
            nn.Linear(self.n_in_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_out_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_len = 5
        if x.shape[-1] >= min_len:
            x = x.view(x.size(0), -1)
            x = self.mlp(x)
            x = x.view(x.size(0), 128, -1)
            return x
        else:
            padded_x = manual_pad(x, min_len)
            return self.mlp(padded_x)
    