import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        n_out_channels: int,
        kernel_size: int,
        padding_mode: str = "replicate",
        include_relu: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv1d(
                in_channels=n_in_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding="same",
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(num_features=n_out_channels),
        ]
        if include_relu:
            layers.append(nn.ReLU())
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        return out


def manual_pad(x: torch.Tensor, min_length: int) -> torch.Tensor:
    pad_amount = min_length - x.shape[-1]
    pad_left = pad_amount // 2
    pad_right = pad_amount - pad_left
    pad_x = F.pad(x, [pad_left, 0], mode="constant", value=x[:, :, 0].item())
    pad_x = F.pad(pad_x, [0, pad_right], mode="constant", value=x[:, :, -1].item())
    return pad_x
