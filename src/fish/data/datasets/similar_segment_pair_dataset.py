from pathlib import Path
from typing import Union, Optional
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from audiomentations import Normalize, Gain, Compose

from src.common.audio import load_audio, SAMPLE_RATE, force_length

COLUMN_DTYPES = {
    "clique_id": "object",
    "version_id0": "object",
    "version_id1": "object",
    "youtube_id0": "object",
    "youtube_id1": "object",
    "duration0": "float32",
    "duration1": "float32",
    "segment_duration": "float32",
    "hole_v0_start": "float32",
    "hole_v1_start": "float32",
    "k": "int32",
}

# NOTE: THE MIC IR RECORDINGS USED ALL HAVE 10 SECONDS DURATION BUT
# They have T60 much lower than 0.5 SECONDS (THEY GIVE np.allclose(mir, 0))
MIR_DUR = 0.5

# NOTE: Most RIRs have much shorter T60 anyways
RIR_DUR = 5.0


class SimilarSegmentPairDataset(Dataset):
    def __init__(
        self,
        csv_path: Union[str, Path],
        audio_dir: Union[str, Path],
        context_duration: float,
        top_k: Optional[int] = None,
        augmentation_dict: Optional[dict] = None,
        additional_context_duration: float = 0.0,
        sample_rate: int = SAMPLE_RATE,
        mode: str = "train",
        shuffle: bool = False,
        verbose: bool = False,
        noise_dir: Optional[Union[str, Path]] = None,
        rir_dir: Optional[Union[str, Path]] = None,
        mir_dir: Optional[Union[str, Path]] = None,
    ) -> None:

        self.csv_path = Path(csv_path)
        self.audio_dir = Path(audio_dir)
        self.context_duration = context_duration
        self.top_k = top_k
        self.augmentation_dict = augmentation_dict
        self.additional_context_duration = additional_context_duration
        self.sample_rate = sample_rate
        self.mode = mode
        self.shuffle = shuffle
        self.verbose = verbose

        # Pre-compute lengths in samples
        self.context_length = int(self.context_duration * self.sample_rate)
        self.additional_context_length = int(
            self.additional_context_duration * self.sample_rate
        )
        self.total_context_length = self.context_length + self.additional_context_length

        self.random_gain = None
        self.max_rand_offset_len = 0

        self._load_csv()
        self._build_augmentation_chain()

        self.noise_pool = (
            self._load_audio_pool(noise_dir, "noise") if noise_dir else None
        )
        self.rir_pool = (
            self._load_audio_pool(
                rir_dir, "RIR", max_length=int(RIR_DUR * self.sample_rate)
            )
            if rir_dir
            else None
        )
        self.mir_pool = (
            self._load_audio_pool(
                mir_dir, "MIR", max_length=int(MIR_DUR * self.sample_rate)
            )
            if mir_dir
            else None
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):

        clique_idx = self.items[idx]
        row_idx = self._sample_row_index(clique_idx)
        row = self.df.iloc[row_idx]

        t0 = row["hole_v0_start"]
        t1 = row["hole_v1_start"]

        # Determine the start positions
        start0 = self._sample_random_start_position(
            t0, row["segment_duration"], row["duration0"]
        )
        start1 = self._sample_random_start_position(
            t1, row["segment_duration"], row["duration1"]
        )

        # Load the audio segments
        segment0 = self._load_segment_audio(row["youtube_id0"], start0)
        segment1 = self._load_segment_audio(row["youtube_id1"], start1)

        # Sample noise clips matching segment length (1, T)
        noise0 = None
        noise1 = None
        if self.noise_pool is not None:
            T = segment0.shape[-1]
            idx0, idx1 = random.sample(range(len(self.noise_pool)), 2)
            noise0 = force_length(self.noise_pool[idx0], T, cut_mode="random")
            noise1 = force_length(self.noise_pool[idx1], T, cut_mode="random")

        # Sample room impulse responses (varying lengths, padded in collate_fn)
        rir0 = None
        rir1 = None
        if self.rir_pool is not None:
            ridx0, ridx1 = random.sample(range(len(self.rir_pool)), 2)
            rir0 = self.rir_pool[ridx0]
            rir1 = self.rir_pool[ridx1]

        # Sample impulse responses (varying lengths, padded in collate_fn)
        mir0 = None
        mir1 = None
        if self.mir_pool is not None:
            midx0, midx1 = random.sample(range(len(self.mir_pool)), 2)
            mir0 = self.mir_pool[midx0]
            mir1 = self.mir_pool[midx1]

        return {
            "segment0": segment0,
            "segment1": segment1,
            "noise0": noise0,
            "noise1": noise1,
            "rir0": rir0,
            "rir1": rir1,
            "mir0": mir0,
            "mir1": mir1,
            "clique_id": row["clique_id"],
            "version_id0": row["version_id0"],
            "version_id1": row["version_id1"],
            "youtube_id0": row["youtube_id0"],
            "youtube_id1": row["youtube_id1"],
        }

    def get_version_count_weights(self) -> torch.Tensor:
        """Return a weight for each item (clique), equal to its number of unique versions."""
        v0 = self.df["version_id0"].to_numpy(copy=False)
        v1 = self.df["version_id1"].to_numpy(copy=False)
        weights = []
        for row_indices in self.items:
            version_ids = set(v0[row_indices]) | set(v1[row_indices])
            weights.append(math.log(len(version_ids)))
        return torch.tensor(weights, dtype=torch.float32)

    def _sample_row_index(self, clique_idx: np.ndarray) -> int:
        """Uniform clique -> uniform version-pair -> uniform row within the pair."""

        clique_idx = np.asarray(clique_idx)
        v0 = self.df["version_id0"].to_numpy(copy=False)[clique_idx]
        v1 = self.df["version_id1"].to_numpy(copy=False)[clique_idx]
        pair_min = np.minimum(v0, v1)
        pair_max = np.maximum(v0, v1)
        pair_matrix = np.stack((pair_min, pair_max), axis=1)

        unique_pairs = np.unique(pair_matrix, axis=0)
        chosen_pair = unique_pairs[random.randrange(unique_pairs.shape[0])]
        pair_rows = clique_idx[
            (pair_matrix[:, 0] == chosen_pair[0])
            & (pair_matrix[:, 1] == chosen_pair[1])
        ]

        return int(pair_rows[random.randrange(pair_rows.size)])

    def _load_segment_audio(self, yt_id: str, start_idx: int) -> torch.Tensor:

        audio_path = self.audio_dir / yt_id[:2] / f"{yt_id}.wav"

        # We load self.total_context_length here to allow for augmentations that need
        # extra context (e.g., time-stretching)
        # NOTE: to keep it simple, we allow zero-padding for the segments that go beyond
        # the audio length
        audio = load_audio(
            audio_path,
            sample_rate=self.sample_rate,
            start=start_idx,
            length=self.total_context_length,
            pad=True,
        )  # (1, T)
        assert audio is not None, f"Failed to load audio from {audio_path}"

        if self.augmentation_dict:
            if self.random_gain is not None:
                audio = audio.numpy()
                audio = self.random_gain(audio, sample_rate=self.sample_rate)
                audio = torch.from_numpy(audio)

        return audio

    def _sample_random_start_position(
        self, start_time: float, segment_duration: float, track_duration: float
    ) -> int:

        assert abs(float(segment_duration) - self.context_duration) < 1e-2

        num_samples = int(track_duration * self.sample_rate)
        start_sample = int(start_time * self.sample_rate)
        end_sample = min(
            start_sample + int(segment_duration * self.sample_rate), num_samples
        )

        if self.max_rand_offset_len == 0:
            return start_sample

        # Randomly shift the segment to the left or right within the allowed range
        bound_l = max(0, start_sample - self.max_rand_offset_len)
        bound_r = min(end_sample + self.max_rand_offset_len, num_samples)
        # Ensure valid range for the last segments at the end of the track that may be padded
        bound_r = max(bound_r - self.context_length, start_sample)

        return random.randint(bound_l, bound_r)

    def _load_csv(self) -> None:

        # Load the *large* CSV file with specified dtypes and only the necessary columns
        if self.verbose:
            print(f"[{self.mode}] Loading \033[34m{self.csv_path}\033[0m")
        self.df = pd.read_csv(self.csv_path, usecols=list(COLUMN_DTYPES.keys()), dtype=COLUMN_DTYPES)  # type: ignore
        if self.verbose:
            print(f"[{self.mode}] Total similar-region pairs: {len(self.df):,}")

        # Try discarding False Positives
        if self.top_k is not None:
            self.df = self.df[self.df["k"] < self.top_k]
            if self.verbose:
                print(
                    f"[{self.mode}] After k < {self.top_k} filter: {len(self.df):,} pairs remaining."
                )

        # Convert IDs to integers
        if self.verbose:
            print("Processing some columns...")
        self.df["clique_id"] = self.df["clique_id"].str.slice(2).astype("int32")
        self.df["version_id0"] = self.df["version_id0"].str.slice(2).astype("int32")
        self.df["version_id1"] = self.df["version_id1"].str.slice(2).astype("int32")
        self.df.reset_index(drop=True, inplace=True)

        # Group the rows by clique_id and store only the row indices for efficient sampling
        self.items = list(self.df.groupby("clique_id", sort=False).indices.values())

        # Shuffle the clique order optionally
        if self.shuffle:
            random.shuffle(self.items)

    def _build_augmentation_chain(self) -> None:
        if self.augmentation_dict is not None:
            assert 0 <= self.augmentation_dict["random_offset_ratio"] <= 0.5
            if self.augmentation_dict["random_offset_ratio"] > 0:
                self.max_rand_offset_len = int(
                    self.context_length * self.augmentation_dict["random_offset_ratio"]
                )

            if self.augmentation_dict["random_gain"]["p"] > 0:
                self.random_gain = Compose(
                    transforms=[
                        Normalize(apply_to="all", p=1.0),
                        Gain(
                            max_gain_db=0,
                            p=1.0,
                            min_gain_db=self.augmentation_dict["random_gain"][
                                "min_gain_db"
                            ],
                        ),
                    ],
                    p=self.augmentation_dict["random_gain"]["p"],
                )

    def _load_audio_pool(
        self,
        directory: Union[str, Path],
        name: str,
        max_length: Optional[int] = None,
    ) -> list[torch.Tensor]:
        directory = Path(directory)
        files = sorted(directory.rglob("*.wav"))
        assert len(files) > 0, f"No .wav files found in {directory}"
        pool = []
        for f in files:
            audio = load_audio(f)  # (1, T)
            if audio is not None:
                if max_length is not None:
                    audio = audio[:, :max_length]
                pool.append(audio)
        if self.verbose:
            total_sec = sum(a.shape[-1] for a in pool) / self.sample_rate
            print(
                f"[{self.mode}] Loaded {len(pool)} {name} recordings ({total_sec:.0f}s total)"
            )
        return pool

    @staticmethod
    def collate_fn(batch):
        pairs, version_ids, clique_ids = [], [], []
        noises, rirs, mirs = [], [], []
        for x in batch:
            pair = torch.stack([x["segment0"], x["segment1"]], dim=0)  # (2, 1, T)
            pairs.append(pair)
            clique_ids.extend([x["clique_id"], x["clique_id"]])
            version_ids.extend([x["version_id0"], x["version_id1"]])
            if x["noise0"] is not None:
                noise = torch.stack([x["noise0"], x["noise1"]], dim=0)  # (2, 1, T)
                noises.append(noise)
            if x["rir0"] is not None:
                rirs.append(x["rir0"])  # (1, T_ir)
                rirs.append(x["rir1"])
            if x["mir0"] is not None:
                mirs.append(x["mir0"])  # (1, T_mir)
                mirs.append(x["mir1"])
        pairs = torch.cat(pairs, dim=0).contiguous()
        clique_ids = torch.tensor(clique_ids)
        version_ids = torch.tensor(version_ids)

        if noises:
            noises = torch.cat(noises, dim=0).contiguous()
        else:
            noises = None

        # Zero-pad RIRs to the longest in the batch
        if rirs:
            max_ir_len = max(ir.shape[-1] for ir in rirs)
            padded = [
                torch.nn.functional.pad(ir, (0, max_ir_len - ir.shape[-1]))
                for ir in rirs
            ]
            rirs = torch.stack(padded, dim=0).contiguous()  # (2*batch, 1, max_ir_len)
            # We do not need longer IR than segments
            rirs = rirs[..., : pairs.shape[-1]].contiguous()
        else:
            rirs = None

        # Zero-pad MIRs to the longest in the batch
        if mirs:
            max_mir_len = max(mir.shape[-1] for mir in mirs)
            padded = [
                torch.nn.functional.pad(mir, (0, max_mir_len - mir.shape[-1]))
                for mir in mirs
            ]
            mirs = torch.stack(padded, dim=0).contiguous()  # (2*batch, 1, max_mir_len)
            mirs = mirs[..., : pairs.shape[-1]].contiguous()
        else:
            mirs = None

        return pairs, noises, rirs, mirs, clique_ids, version_ids
