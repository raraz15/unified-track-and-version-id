from audiomentations import (
    OneOf,
    Compose,
    ClippingDistortion,
    HighPassFilter,
    Limiter,
    TanhDistortion,
)


class PlaybackDevice(Compose):

    def __init__(self, params: dict):

        super().__init__(
            transforms=[
                Limiter(
                    threshold_mode="absolute",
                    **params["Degradation"]["PlaybackDevice"]["Limiter"],
                ),
                HighPassFilter(**params["Degradation"]["PlaybackDevice"]["HPF"]),
                OneOf(
                    transforms=[
                        TanhDistortion(
                            **params["Degradation"]["PlaybackDevice"]["Saturation"][
                                "SoftClipping"
                            ],
                            p=1.0,
                        ),
                        ClippingDistortion(
                            **params["Degradation"]["PlaybackDevice"]["Saturation"][
                                "HardClipping"
                            ],
                            p=1.0,
                        ),
                    ],
                    p=params["Degradation"]["PlaybackDevice"]["Saturation"]["p"],
                ),
            ],
            p=params["Degradation"]["PlaybackDevice"]["p"],
            shuffle=False,
        )
        self.params = params
