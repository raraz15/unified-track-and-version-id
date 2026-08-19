from pathlib import Path
from typing import Union, Optional
import random
import math
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import pandas as pd
from audiomentations import Normalize, Gain, Compose

from src.common.audio import load_audio, SAMPLE_RATE, force_length

# NOTE: THE MIC IR RECORDINGS USED ALL HAVE 10 SECONDS DURATION BUT
# They have T60 much lower than 0.5 SECONDS (THEY GIVE np.allclose(mir, 0))
MIR_DUR = 0.5

# NOTE: Most RIRs have much shorter T60 anyways
# Had to approximate these to save computation and VRAM
RIR_DUR = 5.0

COLUMN_DTYPES = {
    "segment_clique_id": "object",
    "track_clique_id": "object",
    "version_id": "object",
    "youtube_id": "object",
    "segment_start_time": "float32",
    "segment_end_time": "float32",
    "track_duration": "float32",
}


class SegmentCliqueDataset(Dataset):

    def __init__(
        self,
        csv_path: Union[str, Path],
        audio_dir: Union[str, Path],
        context_duration: float,
        segments_per_tuple: int,
        augmentation_dict: Optional[dict] = None,
        additional_context_duration: float = 0.0,
        repeat_probability: float = 0.0,
        sample_rate: int = SAMPLE_RATE,
        mode: str = "train",
        shuffle: bool = False,
        verbose: bool = False,
        noise_dir: Optional[Union[str, Path]] = None,
        rir_dir: Optional[Union[str, Path]] = None,
        mir_dir: Optional[Union[str, Path]] = None,
    ) -> None:

        assert segments_per_tuple >= 2

        self.csv_path = Path(csv_path)
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.context_duration = context_duration
        self.segments_per_tuple = segments_per_tuple
        self.augmentation_dict = augmentation_dict
        self.additional_context_duration = additional_context_duration
        self.repeat_probability = repeat_probability
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

    def __getitem__(self, index: int):

        track_clique_id = self.items[index]
        track_clique_id_num = int(track_clique_id.split("-")[1])

        # Sample a segment clique uniformly
        scs = self.segment_cliques[track_clique_id]
        segment_clique_id = random.choice(list(scs.keys()))
        segment_clique_id_num = int(segment_clique_id.split("-")[1])

        # Sample self.segments_per_tuple segments from the chosen segment clique
        sampled = self._sample_segments_from_clique(track_clique_id, segment_clique_id)

        # Load the audio for the sampled segments (N, 1, T) and their version ids (N,)
        audio, version_ids = [], []
        for segment in sampled:
            audio.append(self._load_segment_audio(segment))
            version_ids.append(int(segment["version_id"].split("-")[1]))

        # Sample noise clips matching segment length (N, 1, T)
        noises = None
        if self.noise_pool is not None:
            T = audio[0].shape[-1]
            indices = random.choices(
                range(len(self.noise_pool)), k=self.segments_per_tuple
            )
            noises = torch.stack(
                [
                    force_length(self.noise_pool[i], T, cut_mode="random")
                    for i in indices
                ],
                dim=0,
            )

        # Sample room impulse responses (varying lengths, padded in collate_fn)
        rirs = None
        if self.rir_pool is not None:
            indices = random.choices(
                range(len(self.rir_pool)), k=self.segments_per_tuple
            )
            rirs = self.rir_pool[indices]  # (N, 1, max_rir_len)

        # Sample mic impulse responses (varying lengths, padded in collate_fn)
        mirs = None
        if self.mir_pool is not None:
            indices = random.choices(
                range(len(self.mir_pool)), k=self.segments_per_tuple
            )
            mirs = self.mir_pool[indices]  # (N, 1, max_mir_len)

        return {
            "audio": torch.stack(audio, dim=0),  # (N,1,T)
            "segment_clique_id": torch.LongTensor(
                [segment_clique_id_num] * self.segments_per_tuple
            ),
            "track_clique_id": torch.LongTensor(
                [track_clique_id_num] * self.segments_per_tuple
            ),
            "version_id": torch.LongTensor(version_ids),
            "noises": noises,
            "rirs": rirs,
            "mirs": mirs,
        }

    def _sample_segments_from_clique(self, track_clique_id, segment_clique_id) -> list:
        segment_clique = self.segment_cliques[track_clique_id][segment_clique_id]

        # Sample the anchor version
        version_ids = list(segment_clique.keys())
        anchor_version_id = random.choice(version_ids)

        # A version may contribute multiple segments to the clique, we sample one of them as the anchor
        anchor_segment = random.choice(segment_clique[anchor_version_id])

        # Find the different version ids, optionally repeat the same version
        diff_version_ids = []
        for v_id in version_ids:
            if v_id != anchor_version_id or random.random() < self.repeat_probability:
                diff_version_ids.append(v_id)
        num_dif_versions = len(diff_version_ids)
        assert (
            num_dif_versions > 0
        ), f"No different version ids found for segment clique {segment_clique_id} in track clique {track_clique_id}"

        # Sample (self.segments_per_tuple - 1) versions
        sampled_versions = []
        random.shuffle(diff_version_ids)
        for k in range(self.segments_per_tuple - 1):
            sampled_versions.append(diff_version_ids[k % num_dif_versions])

        # Sample one segment from each sampled version
        sampled = [anchor_segment] + [
            random.choice(segment_clique[vid]) for vid in sampled_versions
        ]

        return sampled

    def _load_segment_audio(self, segment) -> torch.Tensor:

        yt_id = segment["youtube_id"]
        audio_path = self.audio_dir / yt_id[:2] / f"{yt_id}.wav"

        start = self._sample_random_start_position(
            segment["segment_start_time"],
            segment["segment_end_time"],
            segment["track_duration"],
        )

        # We load self.total_context_length here to allow for augmentations that need
        # extra context (e.g., time-stretching)
        # NOTE: to keep it simple, we allow zero-padding for the segments that go beyond
        # the audio length
        # (1, T)
        audio = load_audio(
            audio_path,
            start=start,
            length=self.total_context_length,
            pad=True,
        )
        assert audio is not None

        if self.augmentation_dict:
            if self.random_gain is not None:
                audio = audio.numpy()
                audio = self.random_gain(audio, sample_rate=self.sample_rate)
                audio = torch.from_numpy(audio)

        return audio

    def _sample_random_start_position(
        self, start_time: float, end_time: float, track_duration: float
    ) -> int:

        num_samples = int(track_duration * self.sample_rate)
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)

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
        df = pd.read_csv(self.csv_path, usecols=list(COLUMN_DTYPES.keys()), dtype=COLUMN_DTYPES)  # type: ignore

        # Group the segments by
        self.segment_cliques = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for row in df.itertuples(index=False):
            segment = {
                "version_id": row.version_id,  # type: ignore
                "youtube_id": row.youtube_id,  # type: ignore
                "segment_start_time": float(row.segment_start_time),  # type: ignore
                "segment_end_time": float(row.segment_end_time),  # type: ignore
                "track_duration": float(row.track_duration),  # type: ignore
            }
            self.segment_cliques[row.track_clique_id][row.segment_clique_id][
                segment["version_id"]
            ].append(segment)

        self.items = list(self.segment_cliques.keys())
        if self.verbose:
            s_clique = sum(len(t) for t in self.segment_cliques.values())
            s = sum(
                len(segs)
                for t in self.segment_cliques.values()
                for sc in t.values()
                for segs in sc.values()
            )
            print(f"[{self.mode}] Total track cliques: {len(self.items):,}")
            print(f"[{self.mode}] Total segment cliques: {s_clique:,}")
            print(f"[{self.mode}] Total segments: {s:,}")
        if self.shuffle:
            random.shuffle(self.items)

    def get_version_count_weights(self) -> torch.Tensor:
        weights = []
        for track_clique_id in self.items:
            version_ids = set()
            for sc in self.segment_cliques[track_clique_id].values():
                version_ids.update(sc.keys())
            weights.append(math.log(len(version_ids)))
        return torch.tensor(weights, dtype=torch.float32)

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
    ) -> torch.Tensor:
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
        max_len = max(a.shape[-1] for a in pool)
        return torch.stack(
            [torch.nn.functional.pad(a, (0, max_len - a.shape[-1])) for a in pool],
            dim=0,
        )  # (P, 1, max_len)

    @staticmethod
    def collate_fn(batch):
        audio = torch.cat([item["audio"] for item in batch], dim=0)  # (B*N,1,T)
        segment_clique_ids = torch.cat(
            [item["segment_clique_id"] for item in batch], dim=0
        )  # (B*N,)
        track_clique_ids = torch.cat(
            [item["track_clique_id"] for item in batch], dim=0
        )  # (B*N,)
        version_ids = torch.cat([item["version_id"] for item in batch], dim=0)  # (B*N,)
        noises_list = [x["noises"] for x in batch if x["noises"] is not None]
        if noises_list:
            noises = torch.cat(noises_list, dim=0).contiguous()  # (B*N, 1, T)
        else:
            noises = None

        # RIRs (already same-length tensors from the pre-padded pool)
        rirs_list = [x["rirs"] for x in batch if x["rirs"] is not None]
        if rirs_list:
            rirs = torch.cat(rirs_list, dim=0).contiguous()  # (B*N, 1, max_rir_len)
            rirs = rirs[..., : audio.shape[-1]].contiguous()
        else:
            rirs = None

        # MIRs (already same-length tensors from the pre-padded pool)
        mirs_list = [x["mirs"] for x in batch if x["mirs"] is not None]
        if mirs_list:
            mirs = torch.cat(mirs_list, dim=0).contiguous()  # (B*N, 1, max_mir_len)
            mirs = mirs[..., : audio.shape[-1]].contiguous()
        else:
            mirs = None

        return (
            audio,
            noises,
            rirs,
            mirs,
            segment_clique_ids,
            track_clique_ids,
            version_ids,
        )
