from typing import List, Tuple, Callable, Union
import math

import torch.nn as nn

from .parts import BasicBlock_IBN, Bottleneck_IBN
from ..aux import IBN, GeM, conv1x1
from ..VINet import VINet


class ResCQTNet(VINet):

    def __init__(
        self,
        block: Union[BasicBlock_IBN, Bottleneck_IBN],
        layers: List[int],
        front_end: str,
        norm_layer: str = "bn",
        pool: str = "gem-multi",
        projection: str = "linear",
        zero_init_residual: bool = True,
        **kwargs,
    ):

        super().__init__(**kwargs)

        assert len(layers) == 4, "layers should be a list of 4 ints"
        self.inplanes = 64
        self.out_planes = 512 * block.expansion
        self.embedding_size = self.out_planes
        self.zero_init_residual = zero_init_residual

        self.front_end = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(12, 3), stride=(1, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(12, 3), stride=(1, 2), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        if front_end.lower() == "ct_ct":
            pass
        elif front_end.lower() == "ct_ct_ct":
            self.front_end.append(
                nn.Conv2d(
                    64, self.inplanes, kernel_size=(12, 3), stride=(1, 2), bias=False
                )
            )
            self.front_end.append(nn.BatchNorm2d(self.inplanes))
            self.front_end.append(nn.ReLU(inplace=True))
        elif front_end.lower() == "ct_ct_mt":
            self.front_end.append(nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)))
        else:
            raise NotImplementedError

        if norm_layer.lower() == "bn":
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer.lower() == "ibn":
            self.norm_layer = IBN
        elif norm_layer.lower() == "in":
            self.norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError
        self.back_end = nn.Sequential(
            self._make_layer(
                block,
                planes=64,
                blocks=layers[0],
                norm_layer=self.norm_layer,
            ),
            self._make_layer(
                block,
                planes=128,
                blocks=layers[1],
                norm_layer=self.norm_layer,
                stride=(1, 2),
            ),
            self._make_layer(
                block,
                planes=256,
                blocks=layers[2],
                norm_layer=self.norm_layer,
                stride=(1, 2),
            ),
            self._make_layer(
                block,
                planes=512,
                blocks=layers[3],
                norm_layer=self.norm_layer,
                stride=(1, 2),
            ),
        )
        if pool.lower() == "gem-single":
            self.back_end.append(GeM(1, eps=self.eps))
        elif pool.lower() == "gem-multi":
            self.back_end.append(GeM(self.out_planes, eps=self.eps))
        elif pool.lower() == "adaptive_max":
            self.back_end.append(nn.AdaptiveMaxPool2d((1, 1)))
        elif pool.lower() == "adaptive_avg":
            self.back_end.append(nn.AdaptiveAvgPool2d((1, 1)))
        else:
            raise NotImplementedError

        if projection.lower() == "none":
            self.projection = nn.Identity()
        else:
            if projection.lower() == "linear":
                self.projection = nn.Linear(
                    self.out_planes,
                    self.embedding_size,
                    bias=not self.l2_normalize_projection,
                )
            elif projection.lower() == "bn-linear":
                self.projection = nn.Sequential(
                    nn.BatchNorm1d(self.out_planes),
                    nn.Linear(
                        self.out_planes,
                        self.embedding_size,
                        bias=not self.l2_normalize_projection,
                    ),
                )
            else:
                raise NotImplementedError
        self.init_weights()

    def _make_layer(
        self,
        block: Union[BasicBlock_IBN, Bottleneck_IBN],
        planes: int,
        blocks: int,
        norm_layer: Callable[..., nn.Module],
        stride: Tuple[int, int] = (1, 1),
    ) -> nn.Sequential:
        downsample = None
        if stride != (1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def init_weights(self) -> None:

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_IBN) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock_IBN) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)


class ResCQTNet18(ResCQTNet):
    def __init__(
        self,
        front_end: str = "ct_ct",
        norm_layer: str = "ibn",
        pool: str = "gem-single",
        projection: str = "linear",
        **kwargs,
    ):
        super().__init__(
            block=BasicBlock_IBN,
            front_end=front_end,
            layers=[2, 2, 2, 2],
            norm_layer=norm_layer,
            pool=pool,
            projection=projection,
            **kwargs,
        )


class ResCQTNet34(ResCQTNet):
    def __init__(
        self,
        front_end: str = "ct_ct",
        norm_layer: str = "ibn",
        pool: str = "gem-single",
        projection: str = "linear",
        **kwargs,
    ):
        super().__init__(
            front_end=front_end,
            block=BasicBlock_IBN,
            layers=[3, 4, 6, 3],
            norm_layer=norm_layer,
            pool=pool,
            projection=projection,
            **kwargs,
        )


class ResCQTNet50(ResCQTNet):
    def __init__(
        self,
        front_end: str = "ct_ct",
        norm_layer: str = "ibn",
        pool: str = "gem-single",
        projection: str = "linear",
        **kwargs,
    ):
        super().__init__(
            front_end=front_end,
            block=Bottleneck_IBN,
            layers=[3, 4, 6, 3],
            norm_layer=norm_layer,
            pool=pool,
            projection=projection,
            **kwargs,
        )
