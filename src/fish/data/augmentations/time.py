import torch
from nnAudio.utils import nextpow2


class AugmenterTime:
    def __init__(self, cfg: dict, eps: float = 1e-6):

        self.cfg = cfg
        self.eps = eps

        # Room impulse response parameters
        rir_cfg = cfg["room_impulse_response"]
        if rir_cfg is not None:
            self.rir_p = rir_cfg["p"]
            rir_gain_low, rir_gain_high = rir_cfg["gain_db_range"]
            assert rir_gain_low <= rir_gain_high, "RIR gain_db low must be <= high"
            self.rir_gain_db_low = rir_gain_low
            self.rir_gain_db_high = rir_gain_high
        else:
            self.rir_p = 0.0
            self.rir_gain_db_low = None
            self.rir_gain_db_high = None

        # Microphone impulse response parameters
        mir_cfg = cfg["microphone_impulse_response"]
        if mir_cfg is not None:
            self.mir_p = mir_cfg["p"]
            mir_gain_low, mir_gain_high = mir_cfg["gain_db_range"]
            assert (
                mir_gain_low <= mir_gain_high
            ), "Microphone IR gain_db low must be <= high"
            self.mir_gain_db_low = mir_gain_low
            self.mir_gain_db_high = mir_gain_high
        else:
            self.mir_p = 0.0
            self.mir_gain_db_low = None
            self.mir_gain_db_high = None

        # Additive noise parameters
        noise_cfg = cfg["additive_noise"]
        if noise_cfg is not None:
            self.noise_p = noise_cfg["p"]
            snr_low, snr_high = noise_cfg["SNR"]
            assert snr_low <= snr_high, "SNR low must be <= high"
            self.noise_snr_low = snr_low
            self.noise_snr_high = snr_high

            noise_gain_low, noise_gain_high = noise_cfg["gain_db_range"]
            assert (
                noise_gain_low <= noise_gain_high
            ), "Noise gain_db low must be <= high"
            self.noise_gain_db_low = noise_gain_low
            self.noise_gain_db_high = noise_gain_high
        else:
            self.noise_p = 0.0
            self.noise_snr_low = None
            self.noise_snr_high = None
            self.noise_gain_db_low = None
            self.noise_gain_db_high = None

    def __call__(
        self,
        x: torch.Tensor,
        noise: torch.Tensor | None = None,
        rir: torch.Tensor | None = None,
        mir: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = x.squeeze(1)  # (B,T)

        # NOTE: Additive noise first to make it harder, even though losing realism.
        if noise is not None:
            noise = noise.squeeze(1)  # (B,T)
            x = self.additive_noise(x, noise)  # (B,T)

        # Room impulse response
        if rir is not None:
            rir = rir.squeeze(1)  # (B,T_ir)
            x = self.room_impulse_response(x, rir)  # (B,T)

        # Microphone impulse response
        if mir is not None:
            mir = mir.squeeze(1)  # (B,T_mir)
            x = self.microphone_impulse_response(x, mir)  # (B,T)

        return x.unsqueeze(1)  # (B,1,T)

    @torch.no_grad()
    def additive_noise(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:

        if self.noise_p == 0.0:
            return x

        idx = self._coin_flip(x.shape[0], self.noise_p, x.device)
        if idx.numel() == 0:
            return x

        x[idx] = self._add_noise(
            x[idx],
            noise[idx],
            self.noise_snr_low,  # type: ignore
            self.noise_snr_high,  # type: ignore
            self.noise_gain_db_low,  # type: ignore
            self.noise_gain_db_high,  # type: ignore
        )

        return x

    @torch.no_grad()
    def room_impulse_response(self, x: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:

        if self.rir_p == 0.0:
            return x

        idx = self._coin_flip(x.shape[0], self.rir_p, x.device)
        if idx.numel() == 0:
            return x

        x[idx] = self._convolve(
            x[idx], ir[idx], self.rir_gain_db_low, self.rir_gain_db_high  # type: ignore
        )

        return x

    @torch.no_grad()
    def microphone_impulse_response(
        self, x: torch.Tensor, ir: torch.Tensor
    ) -> torch.Tensor:

        if self.mir_p == 0.0:
            return x

        idx = self._coin_flip(x.shape[0], self.mir_p, x.device)
        if idx.numel() == 0:
            return x

        x[idx] = self._convolve(
            x[idx], ir[idx], self.mir_gain_db_low, self.mir_gain_db_high  # type: ignore
        )

        return x

    @staticmethod
    def _coin_flip(B: int, p: float, device: torch.device) -> torch.Tensor:
        return torch.where(torch.rand(B, device=device) < p)[0]

    def _add_noise(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        snr_db_low: float,
        snr_db_high: float,
        gain_db_low: float,
        gain_db_high: float,
    ) -> torch.Tensor:
        """x: (B,T), noise: (B,T)"""

        # Sample random SNR value(s) in dB
        snr_db = torch.empty((x.shape[0], 1), device=x.device).uniform_(
            snr_db_low, snr_db_high
        )

        # NOTE: I do not apply DC removal here.
        power_signal = x.pow(2).mean(dim=-1, keepdim=True)
        power_noise = noise.pow(2).mean(dim=-1, keepdim=True)

        # Scale the noise to match the needed power:
        power_noise_des = power_signal / (10 ** (snr_db / 10))
        gain_required = torch.sqrt(power_noise_des / power_noise.clamp(min=self.eps))
        noise = noise * gain_required

        # Add the noise to the signal
        x_noise = x + noise

        return self._clip_protect(x_noise, gain_db_low, gain_db_high)

    def _convolve(
        self, x: torch.Tensor, ir: torch.Tensor, gain_db_low: float, gain_db_high: float
    ) -> torch.Tensor:
        """x: (B,T), ir: (B,T_ir). Same convolution (start-aligned)."""

        T = x.shape[-1]
        L = ir.shape[-1]
        fast_N = 2 ** nextpow2(T + L - 1)  # full conv length is T+L-1

        # Perform convolution in the frequency domain
        x_fft = torch.fft.rfft(x, n=fast_N, dim=-1)
        ir_fft = torch.fft.rfft(ir, n=fast_N, dim=-1)
        x_ir_fft = x_fft * ir_fft
        x_ir = torch.fft.irfft(x_ir_fft, n=fast_N, dim=-1)

        # Trim to original length T (same convolution, start-aligned)
        x_ir = x_ir[:, :T]

        return self._clip_protect(x_ir, gain_db_low, gain_db_high)

    def _clip_protect(
        self, x: torch.Tensor, gain_db_low: float, gain_db_high: float
    ) -> torch.Tensor:
        """Peak normalize + random gain only for samples that clip."""

        peak = x.abs().amax(dim=-1, keepdim=True).clamp(min=self.eps)
        clipped = peak > 1.0
        x_norm = x / peak
        x_norm = self._random_gain(x_norm, gain_db_low, gain_db_high)

        return torch.where(clipped, x_norm, x)

    @staticmethod
    def _random_gain(
        x: torch.Tensor, gain_db_low: float, gain_db_high: float
    ) -> torch.Tensor:

        # Sample random gain value(s) in dB
        gain_db = torch.empty((x.shape[0], 1), device=x.device).uniform_(
            gain_db_low, gain_db_high
        )
        # Convert gain from dB to linear scale
        gain_linear = 10 ** (gain_db / 20)
        # Apply the gain
        x = x * gain_linear

        return x
