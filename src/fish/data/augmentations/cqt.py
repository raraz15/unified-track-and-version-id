import math

import torch
import torch.nn.functional as F


class AugmenterCQT:
    def __init__(self, cfg: dict, context_length_cqt: int, n_bins_cqt: int):

        self.cfg = cfg
        # 0 for magnitude -x for log-magnitude, dB
        self.lowest_value = float(cfg["lowest_value"])
        # We need to know the context length in CQT frames to remove
        # the extra frames added by time-stretching.
        self.context_length_cqt = context_length_cqt

        # To remove the extra frequency bins after pitch transposition
        self.n_bins_cqt = n_bins_cqt

        self.params_spectral_masking = cfg["spectral_masking"]
        self.params_time_stretching = cfg["time_stretching"]
        self.params_pitch_transpose = cfg["pitch_transpose"]

        self.ops = []
        if self.params_time_stretching is not None:
            assert (
                1 >= self.params_time_stretching["rmin"] > 0.0
            ), "rmin must be in (0.0, 1.0]"
            assert self.params_time_stretching["rmax"] >= 1.0, "rmax must be >= 1.0"
            self.ops.append(self.time_stretching)
        if self.params_pitch_transpose is not None:
            self.ops.append(self.pitch_transpose)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:

        if self.ops:
            for i in torch.randperm(len(self.ops)).tolist():
                y = self.ops[i](y)

        # Remove the extra context frames added for time stretching
        y = y[:, :, : self.context_length_cqt]

        # Remove the extra frequency bins added for pitch transposition
        y = y[:, : self.n_bins_cqt, :]

        # Spectral masking is always last
        y = self.spectral_masking(y)

        return y

    @torch.no_grad()
    def spectral_masking(self, y: torch.Tensor) -> torch.Tensor:

        if (
            self.params_spectral_masking is None
            or self.params_spectral_masking["p"] == 0.0
        ):
            return y

        max_n = self.params_spectral_masking["max_times"]
        n = int(torch.randint(1, max_n + 1, (1,)).item())
        f_pc = self.params_spectral_masking["freq_ratio"]
        t_pc = self.params_spectral_masking["time_ratio"]
        p = self.params_spectral_masking["p"]
        apply_f = f_pc > 0
        apply_t = t_pc > 0

        mask_val = torch.as_tensor(
            self.lowest_value,
            device=y.device,
            dtype=y.dtype,
        )

        y_masked = y
        for _ in range(n):
            # NOTE: For each n, we flip a coin per item in the batch to decide whether to apply the masking or not.
            # So at max, an item can get masked n times, but it can also get masked 0 times.
            y_masked = self._spectral_masking(
                y_masked,
                apply_f,
                apply_t,
                f_pc,
                t_pc,
                p,
                mask_val,
            )

        return y_masked

    @torch.no_grad()
    def time_stretching(self, y: torch.Tensor) -> torch.Tensor:

        if (
            self.params_time_stretching is None
            or self.params_time_stretching["p"] == 0.0
        ):
            return y

        rmin = self.params_time_stretching["rmin"]
        rmax = self.params_time_stretching["rmax"]
        p = self.params_time_stretching["p"]

        y_s = []
        for i in range(y.size(0)):
            if torch.rand(1) < p:
                r = rmin + (rmax - rmin) * torch.rand(1).item()
                y_s_i = self._time_stretching(y[i], r)  # (F, T)
            else:
                y_s_i = y[i]  # (F, T)
            # Discard the extra frames for training to make a batch
            # We always load more than context_length_cqt frames
            if y_s_i.size(1) > self.context_length_cqt:
                y_s_i = y_s_i[:, : self.context_length_cqt]
            elif y_s_i.size(1) < self.context_length_cqt:
                padding = self.context_length_cqt - y_s_i.size(1)
                print(
                    f"Time-stretched length {y_s_i.size(1)} is shorter than context_length_cqt {self.context_length_cqt}",
                    flush=True,
                )
                # Pad with lowest value
                y_s_i = F.pad(y_s_i, (0, padding), value=self.lowest_value)
            y_s.append(y_s_i)
        y_s = torch.stack(y_s, dim=0)

        return y_s

    @torch.no_grad()
    def pitch_transpose(self, y: torch.Tensor) -> torch.Tensor:

        if (
            self.params_pitch_transpose is None
            or self.params_pitch_transpose["p"] == 0.0
        ):
            return y

        p = self.params_pitch_transpose["p"]

        B, F, T = y.shape
        delta = F - self.n_bins_cqt
        assert (
            delta >= 0
        ), "n_bins_cqt should be less than or equal to the number of frequency bins"

        apply = torch.rand(B, device=y.device) < p
        r = torch.randint(0, delta + 1, (B,), device=y.device) * apply
        idx = r.unsqueeze(1) + torch.arange(self.n_bins_cqt, device=y.device)
        return torch.gather(y, 1, idx.unsqueeze(2).expand(-1, -1, T))

    def _time_stretching(
        self,
        y: torch.Tensor,
        r: float,
    ) -> torch.Tensor:
        y = y.unsqueeze(0)  # (1, F, T)
        length_orig = y.size(2)

        # ceil to avoid dropping a frame when compressing
        length_new = math.ceil(length_orig * r)
        if length_orig != length_new:
            # Linear interpolation to change the length without mixing frequencies
            y_s = F.interpolate(
                y,
                size=length_new,
                mode="linear",  # interpolation in time not frequency
                align_corners=False,  # uniform stretching
            )  # (1, F, T_new)
        else:
            y_s = y
        y_s = y_s.squeeze(0)  # (F, T)

        return y_s

    def _spectral_masking(
        self,
        y: torch.Tensor,
        apply_f: bool,
        apply_t: bool,
        f_pc: float,
        t_pc: float,
        p: float,
        mask_value: torch.Tensor,
    ) -> torch.Tensor:

        # If both axes are disabled, keep the input unchanged
        if not (apply_f or apply_t):
            return y

        B, F, T = y.size()

        freq_cond = torch.zeros((B, F, 1), device=y.device, dtype=torch.bool)
        if apply_f:
            fpc = torch.rand(B, 1, 1, device=y.device) * f_pc
            flen = (fpc * F).clamp(min=1).long()
            fmax = F - flen
            f0 = (torch.rand_like(fpc) * fmax).long()
            fids = torch.arange(0, F, device=y.device).view(1, -1, 1)
            freq_cond = (fids >= f0) & (fids < f0 + flen)

        time_cond = torch.zeros((B, 1, T), device=y.device, dtype=torch.bool)
        if apply_t:
            tpc = torch.rand(B, 1, 1, device=y.device) * t_pc
            tlen = (tpc * T).clamp(min=1).long()
            tmax = T - tlen
            t0 = (torch.rand_like(tpc) * tmax).long()
            tids = torch.arange(0, T, device=y.device).view(1, 1, -1)
            time_cond = (tids >= t0) & (tids < t0 + tlen)

        if apply_f and apply_t:
            if self.params_spectral_masking["union"]:
                cond = freq_cond | time_cond
            else:
                cond = freq_cond & time_cond
        elif apply_f:
            cond = freq_cond
        else:
            cond = time_cond
        y_masked = torch.where(cond, mask_value, y)

        # For each item in the batch, apply with probability p
        mask = torch.rand(B, 1, 1, device=y.device) < p
        y_masked = torch.where(mask, y_masked, y)

        return y_masked
