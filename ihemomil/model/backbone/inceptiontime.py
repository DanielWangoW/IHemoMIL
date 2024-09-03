import torch
import torch.nn.functional as F
from torch import nn
from ihemomil.model.backbone.common import manual_pad

class InceptionTimeFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        out_channels: int = 32,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.instance_encoder = nn.Sequential(
            InceptionBlock(n_in_channels, out_channels=out_channels, padding_mode=padding_mode),
            InceptionBlock(out_channels * 4, out_channels, padding_mode=padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_len = 21
        if x.shape[-1] >= min_len:
            return self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            return self.instance_encoder(padded_x)


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        bottleneck_channels: int = 32,
        padding_mode: str = "replicate",
        n_modules: int = 3,
    ) -> None:
        super().__init__()
        inception_modules = []
        for i in range(n_modules):
            inception_modules.append(
                InceptionModule(
                    in_channels=in_channels if i == 0 else out_channels * 4,
                    out_channels=out_channels,
                    bottleneck_channels=bottleneck_channels,
                    padding_mode=padding_mode,
                ),
            )
        self.inception_modules = nn.Sequential(*inception_modules)
        self.residual = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=4 * out_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(num_features=4 * out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_modules = self.inception_modules(x)
        x_residual = self.residual(x)
        return F.relu(x_modules + x_residual)


class InceptionModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        bottleneck_channels: int = 32,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()

        self.bottleneck: nn.Module
        if in_channels > 1:
            self.bottleneck = nn.Conv1d(
                in_channels=in_channels,
                out_channels=bottleneck_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            )
        else:
            self.bottleneck = nn.Identity()
            bottleneck_channels = 1

        self.conv_layers = nn.ModuleList()
        for kernel_size in [10, 20, 40]:
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=bottleneck_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                    padding_mode=padding_mode,
                )
            )

        self.max_pooling_w_bottleneck = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
                padding_mode=padding_mode,
            ),
        )

        self.activation = nn.Sequential(nn.BatchNorm1d(num_features=4 * out_channels), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bottleneck = self.bottleneck(x)
        z0 = self.conv_layers[0](x_bottleneck)
        z1 = self.conv_layers[1](x_bottleneck)
        z2 = self.conv_layers[2](x_bottleneck)
        z3 = self.max_pooling_w_bottleneck(x)
        z = torch.cat([z0, z1, z2, z3], dim=1)
        z = self.activation(z)
        return z
