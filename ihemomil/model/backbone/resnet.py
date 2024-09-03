import torch
import torch.nn.functional as F
from torch import nn
from ihemomil.model.backbone.common import ConvBlock, manual_pad


class ResNetFeatureExtractor(nn.Module):

    def __init__(self, n_in_channels: int, padding_mode: str = "replicate"):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.instance_encoder = nn.Sequential(
            ResNetBlock(n_in_channels, 64, padding_mode=padding_mode),
            ResNetBlock(64, 128, padding_mode=padding_mode),
            ResNetBlock(128, 128, padding_mode=padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_len = 5
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            return self.instance_encoder(padded_x)


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, padding_mode: str = "replicate") -> None:
        super().__init__()

        layers = []
        for block_idx, kernel_size in enumerate([8, 5, 3]):
            in_c = in_channels if block_idx == 0 else out_channels
            include_relu = block_idx == 2
            conv_block = ConvBlock(in_c, out_channels, kernel_size, padding_mode, include_relu)
            layers.append(conv_block)
        self.layers = nn.Sequential(*layers)

        self.residual: nn.Module
        if in_channels != out_channels:
            residual_layers = [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding="same",
                    padding_mode=padding_mode,
                ),
                nn.BatchNorm1d(num_features=out_channels),
            ]
            self.residual = nn.Sequential(*residual_layers)
        else:
            self.residual = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.layers(x) + self.residual(x))
