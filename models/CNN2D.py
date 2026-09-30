"""A configurable CNN that predicts a single 2D streamfunction field.

Input:  [batch, input_channels, height, width]
Output: [batch, 1, output_height, output_width]

Only the network is implemented here. The physics loss, training loop, and
spatial differentiation of psi into u and v belong in separate components.
Requires Python 3.10+ and PyTorch.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import nn

Size2D = int | tuple[int, int]
ModuleFactory = Callable[[], nn.Module]
NormFactory = Callable[[int], nn.Module]


def optional_layer(factory: ModuleFactory | None) -> nn.Module:
    return nn.Identity() if factory is None else factory()


@dataclass(frozen=True)
class ConvBlockConfig:
    """One convolution, followed by optional norm/activation/pool/dropout.

    Optional layers are factories, not module instances. For example:
        normalization=nn.BatchNorm2d
        activation=partial(nn.LeakyReLU, negative_slope=0.1)
        pooling=partial(nn.MaxPool2d, kernel_size=2)
        dropout=partial(nn.Dropout2d, p=0.1)
    Normalization receives out_channels; other factories take no arguments.
    Optional layers must preserve channels and return a tensor.
    """

    out_channels: int
    kernel_size: Size2D = 3
    stride: Size2D = 1
    padding: Size2D | str = "same"
    dilation: Size2D = 1
    groups: int = 1
    bias: bool = True
    padding_mode: str = "zeros"
    normalization: NormFactory | None = None
    activation: ModuleFactory | None = nn.GELU
    pooling: ModuleFactory | None = None
    dropout: ModuleFactory | None = None

    def build(self, in_channels: int) -> nn.Module:
        if in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("Channel counts must be positive.")
        # PyTorch validates groups, padding, stride, etc.
        return nn.Sequential(
            nn.Conv2d(
                in_channels, self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                bias=self.bias,
                padding_mode=self.padding_mode,
            ),
            nn.Identity() if self.normalization is None
            else self.normalization(self.out_channels),
            optional_layer(self.activation),
            optional_layer(self.pooling),
            optional_layer(self.dropout),
        )


class StreamfunctionCNN(nn.Module):
    """Input fields -> optional resize -> convolution blocks -> psi.

    The last convolution always produces ONE channel, with no activation.
    No sigmoid/ReLU clips the streamfunction's signed output values.

    upsample runs BEFORE the convolution stack, allowing convolutions to
    learn at the target resolution. None keeps the input resolution.
    With stride=1, padding='same', and no pooling, blocks preserve H and W.
    Pooling/strides and output_conv_kwargs can change the final resolution.
    Padding is a numerical network choice, not a physical boundary condition.

    Use model.to(device=..., dtype=...) to select hardware and precision.
    PyTorch initialization is used unless init_fn is supplied.
    """

    def __init__(
        self,
        in_channels: int,
        blocks: Sequence[ConvBlockConfig],
        *,
        upsample: ModuleFactory | None = None,
        output_conv_kwargs: Mapping[str, Any] | None = None,
        init_fn: Callable[[nn.Module], None] | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or len(blocks) == 0:
            raise ValueError("Use positive in_channels and at least one block.")
        self.in_channels = in_channels
        self.upsample = optional_layer(upsample)

        layers = []
        channels = in_channels
        for block in blocks:
            layers.append(block.build(channels))
            channels = block.out_channels
        self.features = nn.Sequential(*layers)

        # One scalar streamfunction value per output grid point.
        options = {"kernel_size": 1, "padding": "same", "bias": True}
        options.update(output_conv_kwargs or {})
        if {"in_channels", "out_channels"} & options.keys():
            raise ValueError("Output channel counts are managed by the model.")
        self.psi_head = nn.Conv2d(channels, 1, **options)
        if init_fn is not None:
            self.apply(init_fn)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [N, {self.in_channels}, H, W], got {tuple(inputs.shape)}."
            )
        features = self.features(self.upsample(inputs))
        psi = self.psi_head(features)
        return psi


if __name__ == "__main__":
    # Example input channels: coarse u and v. These are inputs, not outputs.
    # Change in_channels to match the fields supplied by your dataset.
    config = {
        "in_channels": 2,
        "blocks": [
            ConvBlockConfig(
                out_channels=channels,
                kernel_size=kernel,
                stride=1,
                padding="same",
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
                normalization=None,
                activation=nn.GELU,
                pooling=None,
                dropout=None,
            )
            for channels, kernel in [(32, 5), (64, 3), (64, 3), (32, 3)]
        ],
        "upsample": None,
        # For 2x resolution, replace None above with:
        # partial(nn.Upsample, scale_factor=2,
        #         mode="bilinear", align_corners=False)
        "output_conv_kwargs": {"kernel_size": 1, "padding": 0, "bias": True},
        "init_fn": None,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StreamfunctionCNN(**config).to(device=device, dtype=torch.float32)

    # Random data only demonstrates tensor shapes; the model is untrained.
    inputs = torch.randn(4, config["in_channels"], 32, 48, device=device)
    psi = model(inputs)
    print("Input shape:", tuple(inputs.shape))   # (4, 2, 32, 48)
    print("Psi shape:  ", tuple(psi.shape))      # (4, 1, 32, 48)
