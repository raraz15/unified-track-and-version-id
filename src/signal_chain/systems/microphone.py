from audiomentations import Compose, ClippingDistortion, Gain

from ..lib import ApplyImpulseResponse


class Microphone(Compose):

    def __init__(self, params: dict):

        super().__init__(
            transforms=[
                ApplyImpulseResponse(
                    **params["Degradation"]["Microphone"]["MicrophoneImpulseResponse"],
                    leave_length_unchanged=True,
                    p=1.0,
                ),
                # NOTE A mic + ADC will return samples in the -1,1 range. Weather this is by
                # clipping or not depends. We already peak_normalize all signals inside the mic
                # Therefore we added some clipping here first. And then added some negative gain
                # So that we create gain variety.
                ClippingDistortion(**params["Degradation"]["Microphone"]["Clipping"]),
                # This is to deal with the peak normalizations happening with IRs
                Gain(
                    max_gain_db=0,
                    **params["Degradation"]["Microphone"]["Gain"],
                ),
            ],
            p=params["Degradation"]["Microphone"]["p"],
            shuffle=False,
        )
        self.params = params
