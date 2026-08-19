import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform


class IdentityTransform(BaseWaveformTransform):
    """"""

    supports_multichannel = True

    def __init__(
        self,
    ):

        super().__init__(1.0)

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        return samples
