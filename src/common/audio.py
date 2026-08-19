from pathlib import Path
from typing import Optional
import math

import numpy as np
import soxr
import torchaudio
import soundfile as sf
import torch
from nnAudio.utils import nextpow2

SAMPLE_RATE = 16000


def get_backend(file_path: Path | str) -> str:
    if Path(file_path).suffix in {".mp3", ".aac", ".mp4", ".m4a"}:
        backend = "ffmpeg"
    else:
        backend = "soundfile"
    return backend


def resample(
    audio: torch.Tensor | np.ndarray,
    in_sr: int,
    out_sr: int,
    prevent_clip: bool = True,
) -> torch.Tensor | np.ndarray:
    """Keeps the data type of the input audio."""
    # audio is (C,T) or (B,T)

    # 3 dB headrom to avoid clipping
    audio *= 0.5
    if isinstance(audio, np.ndarray):
        npy = True
        audio = soxr.resample(audio.T, in_sr, out_sr).T
        audio = torch.FloatTensor(audio)
    elif isinstance(audio, torch.Tensor):
        npy = False
        device = audio.device
        audio = audio.cpu().numpy()
        audio = soxr.resample(audio.T, in_sr, out_sr).T
        audio = torch.FloatTensor(audio).to(device)
    else:
        raise NotImplementedError
    audio *= 2
    if prevent_clip:
        mx = audio.abs().amax(-1, keepdim=True)
        audio /= torch.clamp(mx, min=1)
    else:
        audio = torch.clamp(audio, -1, 1)
    if npy:
        audio = audio.numpy()
    return audio


def load_audio(
    file_path: Path | str,
    sample_rate: Optional[int] = None,
    n_channels: Optional[int] = None,
    start: int = 0,  # in samples
    length: Optional[int] = None,  # in samples
    backend: Optional[str] = None,
    pad: bool = False,
    pad_mode: str = "zeros",
) -> torch.Tensor | None:

    try:
        x, sr = torchaudio.load(
            str(file_path),
            frame_offset=start,
            num_frames=length if length is not None else -1,
            normalize=True,  # to float32
            channels_first=True,
            backend=get_backend(file_path) if backend is None else backend,
        )
    except Exception:
        print(f"\nnERROR: Could not load {file_path}", flush=True)
        print(start, length, flush=True)
        return None

    if x.size(0) not in {1, 2}:
        raise NotImplementedError("We only support mono and stereo inputs!")
    # Adjust channels
    if n_channels is None or n_channels == x.size(0):
        pass
    elif n_channels == 1:
        x = x.mean(0, keepdim=True)
    elif n_channels == 2:
        if x.size(0) == 1:
            x = torch.cat([x, x], dim=0)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    # Adjust sample rate
    if (sample_rate is not None) and sr != sample_rate:
        x = resample(x, sr, sample_rate)
    # Pad length
    if pad and (length is not None) and length > x.size(1):
        if pad_mode == "zeros":
            x = torch.nn.functional.pad(
                x, (0, length - x.size(1)), mode="constant", value=0
            )
        elif pad_mode == "repeat":
            aux = torch.cat([x, x], dim=-1)
            while aux.size(-1) < length:
                aux = torch.cat([aux, x], dim=-1)
            x = aux[:, :length]
        else:
            raise NotImplementedError
    return x


def write_audio(
    file_path: Path | str,
    audio: torch.Tensor | np.ndarray,
    sample_rate: int,
    prevent_clip: bool = False,
):
    """Write audio to disk as 16-bit PCM WAV."""

    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    elif not isinstance(audio, np.ndarray):
        raise TypeError("audio must be a torch.Tensor or np.ndarray")

    if not np.issubdtype(audio.dtype, np.floating):
        raise TypeError("Audio data must be of floating-point type.")

    # If channels-first (C, T), transpose to (T, C) for soundfile
    if audio.ndim == 1:
        data = audio[:, None]  # (T,) → (T,1)
    elif audio.ndim == 2:
        if audio.shape[0] == 1 and audio.shape[1] != 1:
            data = audio.T
        elif audio.shape[1] == 1 and audio.shape[0] != 1:
            data = audio
        else:
            raise ValueError(f"Audio must be mono!")
    else:
        raise ValueError("Audio must be 1D or 2D")

    if prevent_clip:
        max_val = np.max(np.abs(data))
        # Bypass scaling if it does not clip
        data /= np.clip(max_val, a_min=1.0, a_max=None)
    else:
        data = np.clip(data, -1.0, 1.0)

    # Convert to 16-bit PCM
    max_amplitude = np.iinfo(np.int16).max  # 32767
    data = (data * max_amplitude).round().astype(np.int16)

    sf.write(str(file_path), data, sample_rate, format="WAV", subtype="PCM_16")


def force_length(
    x: torch.Tensor,
    length: int,
    pad_mode: str = "repeat",
    cut_mode: str = "start",
) -> torch.Tensor:
    """Improved from https://github.com/sony/clews/blob/main/lib/tensor_ops.py

    Makes sure that x has a MINIMUM length with padding. Expects the input to
    have (C, T) or (B, C, T) shape!. I only tested for C=1 (mono).

    Returns:
    --------
    x': torch.Tensor (num_segments, num_ch, length) or,
        (x.shape[0], num_segments, num_ch, length)
    """

    assert pad_mode in ("repeat", "zeros")
    assert cut_mode in ("start", "end", "random")

    size = x.size(-1)
    if size < length:
        reps = math.ceil(length / size)
        if pad_mode == "repeat":
            x = torch.cat([x] * reps, dim=-1)
        else:
            zeros_reps = [1] * x.ndim
            zeros_reps[-1] = max(1, reps - 1)
            zeros = torch.zeros_like(x).repeat(*zeros_reps)
            x = torch.cat([x, zeros], dim=-1)

    size = x.size(-1)
    if size > length:
        start = 0
        if cut_mode == "end":
            start = size - length
        elif cut_mode == "random":
            start = int(torch.randint(0, size - length + 1, (1,)).item())
        x = x.narrow(-1, start, length)

    return x


def segmentation(
    x: torch.Tensor,
    segment_length: int,
    hop_length: int,
    pad_mode: Optional[str] = None,
    pad_end: bool = True,  # left or right
    pad_value: float = 0.0,
    verbose: bool = False,
) -> torch.Tensor:
    """Segments the input signal. If the last segment is not landing on the
    end of the signal, can apply padding. Expects the input to have (C, T)
    or (B, C, T) shape!. I only tested for C=1 (mono)

    Returns:
    --------
    segments: torch.Tensor (num_segments, num_ch, segment_length) or,
        (x.shape[0], num_segments, num_ch, segment_length)
    """

    assert 0 < hop_length <= segment_length
    assert pad_mode in {"constant", "repeat", None}
    assert x.ndim in {2, 3}, "Input must have shape (C, T) or (B, C, T)"

    T = x.size(-1)

    if pad_mode is not None:
        if T < segment_length:
            need = segment_length - T
            if verbose:
                print(f"[Warning] Signal {T} shorter than {segment_length}")
        else:
            rem = (T - segment_length) % hop_length
            need = 0 if rem == 0 else (hop_length - rem)
        if need > 0:  # TODO if need is more than half the segment_length just cut it
            if pad_mode == "constant":
                pad_tuple = torch.zeros(x.ndim, 2, dtype=torch.long)
                pad_tuple[-1, int(pad_end)] = need
                pad_tuple = tuple(pad_tuple.flip(0).view(-1).tolist())
                x = torch.nn.functional.pad(x, pad_tuple, "constant", pad_value)
            else:
                reps = math.ceil((T + need) / T)
                # NOTE: if the signal is long, this will be slow
                x = torch.cat([x] * reps, dim=-1)
                start = 0
                if not pad_end:
                    start = x.size(-1) - (T + need)
                x = x.narrow(-1, start, T + need)
    else:
        assert T >= segment_length, "Signal too short for framing, allow padding."
    # NOTE: I keep contiguous just in case, maybe it slows down?
    frames = x.unfold(-1, segment_length, hop_length).transpose(-2, -3).contiguous()
    return frames


@torch.no_grad()
def decide_silence(
    x: torch.Tensor,
    frame_length: int = 1600,  # 100ms at 16kHz
    hop_length: Optional[int] = None,
    threshold_db: float = -60.0,
    silence_min_fraction: float = 0.5,
) -> torch.Tensor:

    db = calculate_frame_rms_dBFS(
        x,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    silent_frames = (db <= threshold_db).float()
    frac = silent_frames.mean(dim=-1)
    return frac >= silence_min_fraction


@torch.no_grad()
def calculate_frame_rms_dBFS(
    x: torch.Tensor,
    frame_length: int = 1600,  # 100ms at 16kHz
    hop_length: Optional[int] = None,
    center: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:

    ndim = x.ndim
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim == 2:
        assert x.shape[0] == 1, "Only channel-first mono audio supported (1,T)"
    else:
        assert x.shape[1] == 1, "Only channel-first mono audio supported (B,1,T)"
        x = x.squeeze(1)

    B, T = x.shape
    x = x.to(torch.float32)  # TODO is this necessary?
    if center:
        x = x - x.mean(dim=1, keepdim=True)

    assert frame_length > 0
    if hop_length is None:
        hop_length = int(frame_length // 2)
    else:
        assert frame_length >= hop_length > 0

    if T < frame_length:
        ms = x.pow(2).mean(dim=1, keepdim=True).pow(0.5)
    else:
        x2 = x.pow(2)
        pad_right = (-(T - frame_length)) % hop_length
        if pad_right:
            x2 = torch.nn.functional.pad(x2, (0, pad_right))
        ms = torch.nn.functional.avg_pool1d(
            x2, kernel_size=frame_length, stride=hop_length
        ).pow(0.5)

    db = 20.0 * torch.log10(ms.clamp_min(eps))

    if ndim == 1:
        db = db.squeeze(0)
    elif ndim == 3:
        db = db.unsqueeze(1)

    return db


def rms_normalize(
    x: torch.Tensor, target_rms: float = 0.2, eps: float = 1e-6
) -> torch.Tensor:
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
    x = x * (target_rms / rms.clamp(min=eps))
    return x


def generate_colored_noise(
    size: tuple[int, int],
    exponent: float = 1.0,
) -> torch.Tensor:

    B, T = size
    fast_T = 2 ** nextpow2(T)

    # Generate white noise
    white = torch.randn(B, fast_T)
    white_fft = torch.fft.rfft(white, dim=-1)

    # Create a spectral filter for the desired exponent
    freqs = torch.fft.rfftfreq(fast_T)
    freqs[0] = 1.0  # avoid exponenting zero
    spectral_filter = freqs ** (-exponent / 2)
    spectral_filter[0] = 0.0  # zero-mean: remove DC component

    # Apply the spectral filter to the white noise in the frequency domain
    colored_fft = white_fft * spectral_filter
    colored = torch.fft.irfft(colored_fft, n=fast_T, dim=-1)[:, :T]

    # RMS normalize to 0.2
    colored = rms_normalize(colored)

    return colored


def peak_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    peak = x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    x = x / peak
    return x
