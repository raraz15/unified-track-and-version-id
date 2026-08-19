from audiomentations import (
    OneOf,
    Compose,
    PitchShift,
    TimeStretch,
    SevenBandParametricEQ,
    Gain,
    Clip,
    Normalize,
)

from .varispeed import PlaybackSpeed
from ..lib import IdentityTransform


class Manipulator(Compose):

    def __init__(self, params: dict):

        # TODO more on the post chain?
        # TODO: DAFX separate from pre-chain?
        # TODO I hope that my playback speed implementation is good enough
        super().__init__(
            transforms=[
                OneOf(
                    transforms=[
                        PitchShift(**params["Manipulation"]["PitchShift"], p=1.0),
                        TimeStretch(
                            leave_length_unchanged=False,
                            **params["Manipulation"]["TimeStretch"],
                            p=1.0,
                        ),
                        PlaybackSpeed(**params["Manipulation"]["PlaybackSpeed"], p=1.0),
                        IdentityTransform(),
                    ],
                    p=1.0,
                ),
                SevenBandParametricEQ(**params["Manipulation"]["EQ"]),
                # A digital signal's samples must be bounded between ±1
                # This will peak normalize 50 % of the time and clip 50 % of the time
                # both will happen only to samples outside ±1
                # TODO in the future: loudness norm + limiter can be the third option
                OneOf(
                    transforms=[
                        Normalize(apply_to="only_too_loud_sounds", p=1.0),
                        Clip(p=1.0),
                    ],
                    p=1.0,
                ),
                # This to have some gain variety at the manipulated signal level relative to 0dBFS
                Gain(max_gain_db=0, **params["Manipulation"]["Gain"]),
            ],
            p=params["Manipulation"]["p"],
            shuffle=False,
        )
        self.params = params
