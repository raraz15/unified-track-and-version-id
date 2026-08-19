import random

import librosa
import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform


class PlaybackSpeed(BaseWaveformTransform):
    """Simulate playback speed change for a signal, which will also change the pitch perception."""

    supports_multichannel = True

    def __init__(
        self,
        min_rate: float = 0.8,
        max_rate: float = 1.25,
        p: float = 0.5,
    ):
        """
        :param min_rate: Minimum time-stretch rate. Values less than 1.0 slow down the audio (reduce the playback speed).
        :param max_rate: Maximum time-stretch rate. Values greater than 1.0 speed up the audio (increase the playback speed).
        :param p: The probability of applying this transform.
        """
        super().__init__(p)
        if min_rate > max_rate:
            raise ValueError("min_rate must not be greater than max_rate")
        # TODO set limits for min_rate and max_rate?

        self.min_rate = min_rate
        self.max_rate = max_rate

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            """
            If rate > 1, then the signal is sped up.
            If rate < 1, then the signal is slowed down.
            """
            self.parameters["rate"] = random.uniform(self.min_rate, self.max_rate)

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:

        target_sr = int(sample_rate / self.parameters["rate"])

        if samples.ndim == 1:

            # NOTE We use the best available resampler to avoid aliasing
            # TODO energy matching?
            # TODO length fix?
            samples_sp = librosa.resample(
                samples, orig_sr=sample_rate, target_sr=target_sr, res_type="soxr_vhq"
            )

        # If multi-channel, resample each channel independently
        else:

            samples_sp = np.vstack(
                [
                    librosa.resample(
                        y,
                        orig_sr=sample_rate,
                        target_sr=target_sr,
                        res_type="soxr_vhq",
                    )
                    for y in samples
                ]
            )

        return samples_sp
