from typing import Optional

import torch
import torch.nn as nn


class CQTProcess(nn.Module):

    def __init__(
        self,
        power: float = 0.5,
        downsample_factor: Optional[int] = None,
        add_noise: Optional[float] = None,
        affine: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.power = float(power)
        self.downsample_factor = downsample_factor
        if add_noise is not None:
            add_noise = float(add_noise)
            assert add_noise >= 0, "If provided noise should be non-negative."
        self.add_noise = add_noise
        self.affine = affine
        self.eps = float(eps)

        self.downsample = None
        if self.downsample_factor is not None:
            if self.downsample_factor > 1:
                self.downsample = nn.AvgPool1d(
                    int(self.downsample_factor), stride=int(self.downsample_factor)
                )
            else:
                raise ValueError

        if affine:
            self.gain = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """x: (B,F,L)"""

        with torch.no_grad():
            if self.downsample is not None:
                # Downsample each sample's each frequency bin in time
                x = self.downsample(x)

            # Just in case
            x = x.clamp(min=0)

            # Power compression / expansion
            if self.power != 1.0:
                x = x.pow(self.power)

            # Min-Max Scaling per sample
            x = self._min_max_scale(x)

            # You can add a small noise to avoid fully silent segments
            if self.add_noise:
                # NOTE: with a float16 x, rand_like will return a float16 tensor
                # and a badly chosen eps can flush to zero.
                x = x + self.add_noise * torch.rand_like(x)
                x = self._min_max_scale(x)

        # Constant gain and bias across minmax scaled CQT bins
        if self.affine:
            x = self.gain * x + self.bias

        return x

    def _min_max_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Min-Max Scaling per sample"""
        x_min = torch.amin(x, dim=(1, 2), keepdim=True)
        x_max = torch.amax(x, dim=(1, 2), keepdim=True)
        x = (x - x_min) / (x_max - x_min + self.eps)
        return x
