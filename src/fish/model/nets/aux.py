from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: Tuple[int, int] = (1, 1),
    groups: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


def conv1x1(
    in_planes: int,
    out_planes: int,
    stride: Tuple[int, int] = (1, 1),
) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
    """

    def __init__(self, planes: int):
        super(IBN, self).__init__()

        self.ndim = planes // 2
        self.IN = nn.InstanceNorm2d(self.ndim, affine=True)
        self.BN = nn.BatchNorm2d(self.ndim)

    def forward(self, x: torch.Tensor):
        split = torch.split(x, self.ndim, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class GeM(nn.Module):
    def __init__(self, num_ch: int, p=torch.log(torch.tensor(3)), eps=1e-6):
        """If you want to L2-normalize the GeM output use encoding normalization."""

        super(GeM, self).__init__()
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        self.p = nn.Parameter(torch.ones(1, num_ch, 1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        x = x.clamp(min=self.eps)
        p = torch.exp(self.p)
        x = x.pow(p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.pow(1.0 / p)
        return x

    def __repr__(self):
        p = self.p.detach().cpu()
        if p.numel() == 1:
            p_str = f"{p.item():.6f}"
        else:
            p_str = "[" + ", ".join(f"{v:.6f}" for v in p.flatten().tolist()) + "]"
        return f"{self.__class__.__name__}(p={p_str}, eps={self.eps})"
