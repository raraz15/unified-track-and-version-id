import os
from pathlib import Path
from typing import Union, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.common.audio import load_audio, SAMPLE_RATE

SHORT_POLICIES = ("discard", "zero_pad")
LONG_POLICIES = ("discard", "cut")


class InferenceDataset(Dataset):
    """Audio files to fingerprint, with a policy for out-of-range durations.

    `short_policy` decides what happens to files shorter than `min_duration`:
    "discard" drops them, "zero_pad" right-pads them with silence up to
    `min_duration`. `long_policy` decides what happens to files longer than
    `max_duration`: "discard" drops them, "cut" keeps the first `max_duration`
    seconds. Both are ignored if the corresponding duration bound is None.
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        short_policy: str = "discard",
        long_policy: str = "cut",
        sample_rate: int = SAMPLE_RATE,
        verbose: bool = True,
    ) -> None:

        self.input_path = Path(input_path)
        self.sample_rate = sample_rate
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        if short_policy not in SHORT_POLICIES:
            raise ValueError(
                f"short_policy must be one of {SHORT_POLICIES}, got: {short_policy}"
            )
        if long_policy not in LONG_POLICIES:
            raise ValueError(
                f"long_policy must be one of {LONG_POLICIES}, got: {long_policy}"
            )
        self.short_policy = short_policy
        self.long_policy = long_policy

        if (
            min_duration is not None
            and max_duration is not None
            and min_duration > max_duration
        ):
            raise ValueError(
                f"min_duration ({min_duration}) must not exceed max_duration "
                f"({max_duration}); no file can satisfy both bounds."
            )

        self.min_duration = min_duration
        self.min_length = None
        if min_duration is not None:
            self.min_length = int(min_duration * sample_rate)
            self._log(
                f"Minimum audio length set to {self.min_length:,} samples "
                f"({min_duration:.2f} s), shorter files: {self.short_policy}."
            )

        self.max_duration = max_duration
        self.max_length = None
        if max_duration is not None:
            self.max_length = int(max_duration * sample_rate)
            self._log(
                f"Maximum audio length set to {self.max_length:,} samples "
                f"({max_duration:.2f} s), longer files: {self.long_policy}."
            )

        self.items = []
        if self.input_path.is_file() and self.input_path.suffix == ".wav":
            output_path = self.output_dir / self.input_path.with_suffix(".npy").name
            self._add_item(self.input_path, output_path)
        elif self.input_path.is_file() and self.input_path.suffix == ".txt":
            with open(self.input_path) as f:
                audio_paths = [Path(line.strip()) for line in f if line.strip()]
            if audio_paths:
                common_parent = (
                    Path(os.path.commonpath(audio_paths))
                    if len(audio_paths) > 1
                    else audio_paths[0].parent
                )
                for audio_path in audio_paths:
                    if not audio_path.exists():
                        print(
                            f"[InferenceDataset] File not found, skipping: {audio_path}",
                            flush=True,
                        )
                        continue
                    relative_path = audio_path.relative_to(common_parent)
                    output_path = self.output_dir / relative_path.with_suffix(".npy")
                    self._add_item(audio_path, output_path)
        elif self.input_path.is_dir():
            audio_paths = sorted(self.input_path.rglob("*.wav"))
            for audio_path in audio_paths:
                relative_path = audio_path.relative_to(self.input_path)
                output_path = self.output_dir / relative_path.with_suffix(".npy")
                self._add_item(audio_path, output_path)
        else:
            raise ValueError(f"Input path {self.input_path} is not valid.")

        self._log(f"{len(self.items):,} audio files will be processed.")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[InferenceDataset] {message}", flush=True)

    def _add_item(self, audio_path: Path, output_path: Path) -> None:
        if output_path.exists():
            self._log(f"Output already exists, skipping: {output_path}")
            return
        self.items.append((audio_path, output_path))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index) -> Tuple[torch.Tensor | None, Path, Path]:

        path_input, path_output = self.items[index]

        # NOTE: the model expects mono audio at self.sample_rate. The whole file
        # is decoded before the duration bounds are applied, so they are exact
        # regardless of the file's native sample rate.
        audio = load_audio(
            path_input, sample_rate=self.sample_rate, n_channels=1
        )  # (1,T)
        if audio is None:
            # load_audio has already printed the underlying error
            self._log(f"Discarding {path_input.name}: could not be loaded.")
            return None, path_input, path_output

        length = audio.shape[-1]
        if self.min_length is not None and length < self.min_length:
            self._log(
                f"{path_input.name} is too short: {length / self.sample_rate:.2f} s "
                f"< {self.min_duration:.2f} s ({length:,} < {self.min_length:,} samples)."
            )
            if self.short_policy == "discard":
                self._log(f"Discarding {path_input.name}.")
                return None, path_input, path_output
            self._log(f"Zero-padding {path_input.name} to {self.min_duration:.2f} s")
            audio = F.pad(audio, (0, self.min_length - length))

        elif self.max_length is not None and length > self.max_length:
            self._log(
                f"{path_input.name} is too long: {length / self.sample_rate:.2f} s "
                f"> {self.max_duration:.2f} s ({length:,} > {self.max_length:,} samples)."
            )
            if self.long_policy == "discard":
                self._log(f"Discarding {path_input.name}.")
                return None, path_input, path_output
            self._log(
                f"Cutting {path_input.name} to the first {self.max_duration:.2f} s "
            )
            audio = audio[:, : self.max_length]

        return audio, path_input, path_output

    @staticmethod
    def collate_fn(
        batch,
    ) -> Tuple[torch.Tensor | None, list[Path], list[Path], list[int]]:

        audio_clips, input_paths, output_paths, lengths = [], [], [], []
        for audio, path_input, path_output in batch:
            if audio is None:
                continue
            audio_clips.append(audio)
            input_paths.append(path_input)
            output_paths.append(path_output)
            lengths.append(audio.shape[1])  # return the original length

        if not audio_clips:
            return None, input_paths, output_paths, lengths

        # Zero pad all audio to the maximum length
        max_len = max(lengths)
        for i, (audio_clip, true_len) in enumerate(zip(audio_clips, lengths)):
            pad_right = max_len - true_len
            # NOTE: we remove these paddings later, so the mode is irrelevant
            audio_clips[i] = F.pad(
                audio_clip, (0, pad_right), mode="constant", value=0.0
            )
        audio_clips = torch.stack(audio_clips, 0)  # (B,1,T_max)

        return audio_clips, input_paths, output_paths, lengths
