import functools
import random
import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve

from audiomentations.core.transforms_interface import BaseWaveformTransform

# OGUZ Use out backend for audio loading
from src.common.audio import load_audio


class ApplyImpulseResponse(BaseWaveformTransform):
    """"""

    supports_multichannel = True

    def __init__(
        self,
        ir_path: Union[List[Path], List[str], str, Path],
        p=0.5,
        lru_cache_size=128,
        leave_length_unchanged: bool = True,
    ):
        """
        :param ir_path: A path or list of paths to audio file(s) and/or folder(s) with
            audio files. Can be str or Path instance(s). The audio files given here are
            supposed to be impulse responses.
        :param p: The probability of applying this transform
        :param lru_cache_size: Maximum size of the LRU cache for storing impulse response files
        in memory.
        :param leave_length_unchanged: When set to True, the tail of the sound (e.g. reverb at
            the end) will be chopped off so that the length of the output is equal to the
            length of the input.
        """
        super().__init__(p)
        self.ir_path = ir_path
        self.ir_files = list(Path(self.ir_path).rglob("*.wav"))
        assert self.ir_files, "No impulse response files found at the specified path."
        self.lru_cache_size = lru_cache_size
        self.__load_ir = functools.lru_cache(maxsize=self.lru_cache_size)(
            self.__load_ir
        )
        self.leave_length_unchanged = leave_length_unchanged

    @staticmethod
    def __load_ir(file_path, sample_rate):
        return load_audio(file_path, sample_rate).numpy()

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["ir_file_path"] = random.choice(self.ir_files)

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        ir = self.__load_ir(
            self.parameters["ir_file_path"],
            sample_rate,
        )

        assert samples.shape[0] == 1
        assert ir.shape[0] == 1

        # Expand dimensions to match
        samples_original_dim = samples.ndim
        samples, ir = np.atleast_2d(samples), np.atleast_2d(ir)

        # Preallocate the output array
        output_shape = (samples.shape[0], samples.shape[1] + ir.shape[1] - 1)
        signal_ir = np.empty(output_shape, dtype=samples.dtype)

        # NOTE OGUZ Since we always have (1,T) arrays
        signal_ir[0, :] = convolve(samples[0], ir[0])

        # Peak normalize the signals
        max_value = max(np.amax(signal_ir), -np.amin(signal_ir))
        if max_value > 0.0:
            scale = 1.0 / max_value
            signal_ir *= scale

        if self.leave_length_unchanged:
            signal_ir = signal_ir[..., : samples.shape[-1]]

        # reshape if mono input
        if samples_original_dim == 1:
            signal_ir = signal_ir[0]

        return signal_ir

    def __getstate__(self):
        state = self.__dict__.copy()
        warnings.warn(
            "Warning: the LRU cache of ApplyImpulseResponse gets discarded when pickling it."
            " E.g. this means the cache will be not be used when using ApplyImpulseResponse"
            " together with multiprocessing on Windows"
        )
        del state["_ApplyImpulseResponse__load_ir"]
        return state
