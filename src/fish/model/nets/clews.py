"""Model components adapted from the official CLEWS implementation.

The ``CLEWS`` encoder and its building blocks (``MyIBNResBlock``, ``Head``,
``InstanceBatchNorm2d``, ``GeMPool``, ``PadConv2d``, ``SqueezeExcitation2d``)
are adapted from https://github.com/sony/clews, the reference implementation
accompanying:

    J. Serrà, R. O. Araz, D. Bogdanov, and Y. Mitsufuji, "Supervised
    Contrastive Learning from Weakly-Labeled Audio Segments for Musical
    Version Matching," in Proc. of the Int. Conf. on Machine Learning (ICML),
    2025, pp. 53923-53939.

The original code is released under the MIT license:

    Copyright (c) 2025 Sony Research Inc.

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.

Modifications made here (integration with the ``VINet`` base class, the
configurable projection head, and the pooling options) are part of this
project and are released under the MIT license as well; see the LICENSE file
at the repository root.
"""

import torch

from .VINet import VINet


class CLEWS(VINet):

    def __init__(
        self,
        ncha0: int,
        ncha1: int,
        blocks: list[int],
        channels: list[int],
        down: list[int],
        gem_pool: dict,
        out_dim: int,
        **kwargs,
    ):

        super().__init__(**kwargs)

        assert len(blocks) == len(channels) == len(down)

        self.front_end = torch.nn.Sequential(
            torch.nn.Conv2d(1, ncha0, (12, 3), stride=(1, 2), bias=False),
            torch.nn.BatchNorm2d(ncha0),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(ncha0, ncha1, (12, 3), stride=(1, 2), bias=False),
        )

        self.back_end = []
        for nb, nc, st in zip(blocks, channels, down):
            self.back_end += [MyIBNResBlock(ncha1, nc, stride=st)]
            for _ in range(nb - 1):
                self.back_end += [MyIBNResBlock(nc, nc)]
            ncha1 = nc

        gem_pool = dict(gem_pool)
        mode = gem_pool.pop("mode", "single")
        if mode.lower() == "multi":
            gem_pool["ncha"] = ncha1
        self.back_end += [GeMPool(eps=self.eps, **gem_pool)]

        self.back_end = torch.nn.Sequential(*self.back_end)

        if out_dim != ncha1:
            self.projection = torch.nn.Sequential(
                torch.nn.BatchNorm1d(ncha1),
                torch.nn.Linear(ncha1, out_dim, bias=False),
            )
        else:
            self.projection = torch.nn.Identity()


class MyIBNResBlock(torch.nn.Module):

    def __init__(
        self,
        ncin: int,
        ncout: int,
        factor: float = 0.5,
        kern: int = 3,
        stride: int = 1,
        ibn: str = "pre",
        se: str = "none",
    ):
        super().__init__()
        ncmid = max(1, int(max(ncin, ncout) * factor))
        ncmid += ncmid % 2
        tmp = []
        if ibn == "pre":
            tmp += [InstanceBatchNorm2d(ncin)]
        else:
            tmp += [torch.nn.BatchNorm2d(ncin)]
        if se == "pre":
            tmp += [SqueezeExcitation2d(ncin)]
        tmp += [
            torch.nn.ReLU(inplace=True),
            PadConv2d(ncin, ncmid, kern, stride=stride),
        ]
        if ibn == "post":
            tmp += [InstanceBatchNorm2d(ncmid)]
        else:
            tmp += [torch.nn.BatchNorm2d(ncmid)]
        tmp += [
            torch.nn.ReLU(inplace=True),
            PadConv2d(ncmid, ncout, kern),
        ]
        if se == "post":
            tmp += [SqueezeExcitation2d(ncout)]
        self.convs = torch.nn.Sequential(*tmp)
        if ncin != ncout or stride != 1:
            self.skip = torch.nn.Sequential(
                torch.nn.BatchNorm2d(ncin),
                torch.nn.ReLU(inplace=True),
                PadConv2d(ncin, ncout, kern, stride=stride),
            )
        else:
            self.skip = torch.nn.Identity()
        self.gain = torch.nn.Parameter(torch.zeros(1))

    def forward(self, h):
        return self.gain * self.convs(h) + self.skip(h)


class Head(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)
        torch.nn.init.orthogonal_(self.proj.weight)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        z = self.proj(f)
        return z


class InstanceBatchNorm2d(torch.nn.Module):

    def __init__(self, ncha, affine=True):
        super().__init__()
        assert ncha % 2 == 0
        self.bn = torch.nn.BatchNorm2d(ncha // 2, affine=affine)
        self.inst = torch.nn.InstanceNorm2d(ncha // 2, affine=affine)

    def forward(self, h):
        h1, h2 = torch.chunk(h, 2, dim=1)
        h1 = self.bn(h1)
        h2 = self.inst(h2)
        h = torch.cat([h1, h2], dim=1)
        return h


class GeMPool(torch.nn.Module):

    def __init__(
        self, ncha: int = 1, p: float = 3, learnable: bool = True, eps: float = 1e-6
    ):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=2, end_dim=3)
        self.p = torch.nn.Parameter(
            float(p) * torch.ones(1, ncha, 1), requires_grad=learnable
        )
        self.eps = float(eps)

    def forward(self, h):
        h = self.flatten(h)
        # NOTE: Rational powers of negative numbers is NaN
        # so we clamp to eps before powering. However, this means
        # that the outputs are confined to the positive orthant (R > eps)
        h = h.clamp(min=self.eps)
        h = h.pow(self.p).mean(-1).pow(1 / self.p.squeeze(-1))
        return h


class PadConv2d(torch.nn.Module):

    def __init__(self, nin, nout, kern, stride=1, bias=False):
        super().__init__()
        assert kern % 2 == 1
        pad = kern // 2
        self.conv = torch.nn.Conv2d(
            nin, nout, kern, stride=stride, padding=pad, bias=bias
        )

    def forward(self, h):
        return self.conv(h)


class SqueezeExcitation2d(torch.nn.Module):

    def __init__(self, ncha, r=2):
        super().__init__()
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        nmid = max(1, int(ncha / r))
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(ncha, nmid, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(nmid, ncha, bias=False),
            torch.nn.Sigmoid(),
        )

    def forward(self, h):
        s = self.pooling(h).transpose(1, -1)
        s = self.mlp(s).transpose(-1, 1)
        return h * s
