import torch
import torch.nn.functional as F
from torch import nn

_MIN_SEQ_LEN = 21
_BRANCH_KERNELS = (10, 20, 40)


def _same_conv1d(in_channels: int, out_channels: int, kernel_size: int, padding_mode: str) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding="same",
        padding_mode=padding_mode,
    )


def _pad_to_min_length(x: torch.Tensor, min_length: int) -> torch.Tensor:
    deficit = min_length - x.shape[-1]
    left = deficit // 2
    right = deficit - left
    padded = F.pad(x, [left, 0], mode="constant", value=x[:, :, 0].item())
    padded = F.pad(padded, [0, right], mode="constant", value=x[:, :, -1].item())
    return padded


class InceptionTimeFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        out_channels: int = 32,
        padding_mode: str = "replicate",
    ):
        super().__init__()
        self.n_in_channels = n_in_channels
        block_in_channels = (n_in_channels, out_channels * 4)
        self.instance_encoder = nn.Sequential(
            *[
                InceptionBlock(c_in, out_channels=out_channels, padding_mode=padding_mode)
                for c_in in block_in_channels
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < _MIN_SEQ_LEN:
            x = _pad_to_min_length(x, _MIN_SEQ_LEN)
        return self.instance_encoder(x)


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
        modules = []
        for idx in range(n_modules):
            module_in = in_channels if idx == 0 else out_channels * 4
            modules.append(
                InceptionModule(
                    in_channels=module_in,
                    out_channels=out_channels,
                    bottleneck_channels=bottleneck_channels,
                    padding_mode=padding_mode,
                )
            )
        self.inception_modules = nn.Sequential(*modules)

        self.residual = nn.Sequential(
            _same_conv1d(in_channels, 4 * out_channels, 1, padding_mode),
            nn.BatchNorm1d(num_features=4 * out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.inception_modules(x) + self.residual(x))


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
            self.bottleneck = _same_conv1d(in_channels, bottleneck_channels, 1, padding_mode)
        else:
            self.bottleneck = nn.Identity()
            bottleneck_channels = 1

        self.conv_layers = nn.ModuleList(
            [_same_conv1d(bottleneck_channels, out_channels, k, padding_mode) for k in _BRANCH_KERNELS]
        )

        self.max_pooling_w_bottleneck = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            _same_conv1d(in_channels, out_channels, 1, padding_mode),
        )

        self.activation = nn.Sequential(nn.BatchNorm1d(num_features=4 * out_channels), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bottleneck = self.bottleneck(x)
        branches = [conv(x_bottleneck) for conv in self.conv_layers]
        branches.append(self.max_pooling_w_bottleneck(x))
        return self.activation(torch.cat(branches, dim=1))
