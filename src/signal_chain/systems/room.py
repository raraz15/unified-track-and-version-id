from audiomentations import Compose

from ..lib import ApplyImpulseResponse, AddBackgroundNoise


class Room(Compose):

    def __init__(self, params: dict):

        super().__init__(
            transforms=[
                AddBackgroundNoise(
                    **params["Degradation"]["Room"]["BackgroundNoise"], p=1.0
                ),
                # TODO: dry/wet?
                ApplyImpulseResponse(
                    **params["Degradation"]["Room"]["RoomImpulseResponse"],
                    leave_length_unchanged=True,
                    p=1.0,
                ),
            ],
            p=params["Degradation"]["Room"]["p"],
            shuffle=False,
        )
        self.params = params
