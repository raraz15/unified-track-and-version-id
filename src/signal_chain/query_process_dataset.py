import os
import sys
import csv
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Dataset, get_worker_info
from audiomentations import (
    OneOf,
    Compose,
    SomeOf,
    SevenBandParametricEQ,
    Normalize,
    TimeStretch,
)

logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.common.audio import load_audio, write_audio, SAMPLE_RATE
from src.common.utils import json_default

from .systems import (
    Manipulator,
    PlaybackDevice,
    Room,
    Microphone,
    PlaybackSpeed,
)


class AudioSignalChainDataset(Dataset):
    def __init__(
        self,
        audio_paths: list[Path],
        params: dict,
        chunk_duration: float,
        output_dir: Optional[Path | str] = None,
        write_tracks: Optional[bool] = True,
        sample_chunks: Optional[bool] = True,
    ):
        self.audio_paths = audio_paths
        self.params = params
        self.sample_rate = SAMPLE_RATE
        self.chunk_duration = chunk_duration
        self.chunk_len = int(self.chunk_duration * self.sample_rate)
        self.output_dir = output_dir
        self.write_tracks = write_tracks
        self.sample_chunks = sample_chunks

        assert (
            self.write_tracks or self.sample_chunks
        ), "At least one of write_tracks or sample_chunks must be True."

        self.min_track_len = self.chunk_len

        if self.output_dir is not None:

            self.output_dir = Path(self.output_dir)
            self.common_root = Path(os.path.commonpath(audio_paths))
            self.dirs = {
                "tracks": (
                    {
                        "manipulated": (
                            self.output_dir / "tracks" / "clean-manipulated"
                        ),
                        "manipulated-degraded": (
                            self.output_dir / "tracks" / "clean-manipulated-degraded"
                        ),
                    }
                    if self.write_tracks
                    else None
                ),
                "chunks": (
                    {
                        "clean": (self.output_dir / "chunks" / "clean"),
                        "manipulated": (
                            self.output_dir / "chunks" / "clean-manipulated"
                        ),
                        "manipulated-degraded": (
                            self.output_dir / "chunks" / "clean-manipulated-degraded"
                        ),
                    }
                    if self.sample_chunks
                    else None
                ),
            }

            for k, v in self.dirs.items():
                if v is not None:
                    logger.info(
                        f"{k.capitalize()} will be written to: {self.output_dir / k}"
                    )

        self.manip_chain = Manipulator(self.params)
        self.deg_chain = Compose(
            transforms=[
                PlaybackDevice(self.params),
                Room(self.params),
                Microphone(self.params),
            ],
            p=self.params["Degradation"]["p"],
        )

        # This will only be applied prior to disk writing
        self.normalization = Normalize(apply_to="only_too_loud_sounds", p=1.0)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):

        input_path = self.audio_paths[idx]

        try:

            if self.output_dir is not None:
                output_rel_path = input_path.relative_to(self.common_root)
                output_paths = {}
                for k, v in self.dirs.items():
                    if v is None:
                        continue
                    output_paths[k] = {_k: _v / output_rel_path for _k, _v in v.items()}

                required = []
                if self.sample_chunks:
                    required += [
                        output_paths["chunks"]["clean"],
                        output_paths["chunks"]["manipulated"],
                        output_paths["chunks"]["manipulated-degraded"],
                    ]
                if self.write_tracks:
                    required += [
                        output_paths["tracks"]["manipulated"],
                        output_paths["tracks"]["manipulated-degraded"],
                    ]
                if required and all(p.exists() for p in required):
                    logger.info(f"{str(input_path)} already processed. Skipping.")
                    return None

            track = load_audio(input_path)
            assert track is not None, f"Failed to load {str(input_path)}"
            track = track.numpy()
            assert track.ndim == 2
            assert track.shape[0] == 1, "Audio must be mono"
            if track.shape[1] < self.min_track_len:
                logger.warning(
                    f"Skipping {str(input_path)}: too short ({track.shape[1]} samples)"
                )
                return None

            # Process the entire track in 3 ways
            track_manip = self.manip_chain(track, sample_rate=self.sample_rate)
            track_manip_deg = self.deg_chain(track_manip, sample_rate=self.sample_rate)

            # Prevent the tracks from clipping before sampling the chunks
            # so that the chunk is identical
            track_manip = self.normalization(track_manip, self.sample_rate)
            track_manip_deg = self.normalization(track_manip_deg, self.sample_rate)

            # Sample chunks
            if self.sample_chunks:
                chunk, bounds = self.sample_chunk_nonmanip(track)
                chunk_manip, chunk_manip_deg, bounds_manip = self.sample_chunks_manip(
                    track_manip, track_manip_deg, bounds
                )

            # Get the manipulation and degradation parameters
            history = self.get_history(self.manip_chain, self.deg_chain)

            outputs = {
                "tracks": {
                    "clean": {
                        "reference_path": input_path,
                        "audio": track,
                    },
                    "manipulated": {
                        "reference_path": input_path,
                        "audio": track_manip,
                        "history": history["manipulation"],
                    },
                    "manipulated-degraded": {
                        "reference_path": input_path,
                        "audio": track_manip_deg,
                        "history": history,
                    },
                },
            }

            if self.sample_chunks:
                chunk_dict = {
                    "chunks": {
                        "clean": {
                            "reference_path": input_path,
                            "audio": chunk,
                            "boundary": bounds,
                        },
                        "manipulated": {
                            "reference_path": input_path,
                            "audio": chunk_manip,
                            "history": history["manipulation"],
                            "boundary": bounds_manip,
                        },
                        "manipulated-degraded": {
                            "reference_path": input_path,
                            "audio": chunk_manip_deg,
                            "history": history,
                            "boundary": bounds_manip,
                        },
                    }
                }
                outputs.update(chunk_dict)
            # Add the output paths if specified
            if self.output_dir is not None:
                outputs = {
                    k: {
                        _k: {**_v, "query_path": output_paths[k][_k]}
                        for _k, _v in v.items()
                        if _k in output_paths[k]
                    }
                    for k, v in outputs.items()
                    if k in output_paths
                }
                self.write_to_disk(outputs)

            self.adjust_history()

            return outputs

        except Exception as e:
            logger.exception(f"Failed processing {input_path}: {repr(e)}")
            return None

    def sample_chunk_nonmanip(self, track):

        # Sample chunks from non-manipulated parts
        start = np.random.randint(0, track.shape[1] - self.chunk_len)
        end = start + self.chunk_len
        bounds = np.array([start, end])
        chunk = track[:, bounds[0] : bounds[1]]
        if chunk.shape[1] != self.chunk_len:
            raise IndexError(f"{chunk.shape}, {self.chunk_len}")
        return chunk, bounds

    def sample_chunks_manip(self, track_manip, track_manip_deg, bounds):

        # Find where the original chunks correspond to in the modified audio
        time_rate = self.get_time_stretching_factor()
        start_manip = int(np.floor(bounds[0] / time_rate))
        assert start_manip < track_manip.shape[1], "Out-of-bounds"
        # We always take chunk_len, regardless of time_stretching
        # NOTE: this will zero-pad for tracks without sufficient context
        # and cut from extended chunks.
        # TODO: can cutting create problems? Loss of phrases, notes...?
        end_manip = min(start_manip + self.chunk_len, track_manip.shape[1])
        bounds_manip = np.array([start_manip, end_manip])

        # Get the manipulated chunks
        chunk_manip = track_manip[:, bounds_manip[0] : bounds_manip[1]]
        if self.chunk_len - chunk_manip.shape[1] > 0:
            chunk_manip = np.pad(
                chunk_manip, ((0, 0), (0, self.chunk_len - chunk_manip.shape[1]))
            )
        if chunk_manip.shape[1] != self.chunk_len:
            raise IndexError(f"{chunk_manip.shape}, {time_rate}")
        chunk_manip_deg = track_manip_deg[:, bounds_manip[0] : bounds_manip[1]]
        if self.chunk_len - chunk_manip_deg.shape[1] > 0:
            chunk_manip_deg = np.pad(
                chunk_manip_deg,
                ((0, 0), (0, self.chunk_len - chunk_manip_deg.shape[1])),
            )
        if chunk_manip_deg.shape[1] != self.chunk_len:
            raise IndexError(f"{chunk_manip_deg.shape}, {time_rate}")

        return chunk_manip, chunk_manip_deg, bounds_manip

    def adjust_history(self):
        def reset_oneof(transform):
            if isinstance(transform, OneOf):
                for child in transform.transforms:
                    child.parameters = {"should_apply": None}
            children = getattr(transform, "transforms", None) or getattr(
                transform, "children", []
            )
            for child in children:
                reset_oneof(child)

        for chain in (self.manip_chain, self.deg_chain):
            reset_oneof(chain)

    def get_history(self, *chains):
        histories = []
        for chain in chains:
            histories.append(self._extract_chain_history(chain))
        return {
            "manipulation": histories[0],
            "degradation": histories[1],
        }

    def _extract_chain_history(self, transform):
        if isinstance(transform, SevenBandParametricEQ):
            return [{transform.__class__.__name__: self.get_eq_hist(transform)}]
        if not isinstance(transform, (Compose, OneOf, SomeOf)):
            return [{transform.__class__.__name__: transform.parameters}]
        children = getattr(transform, "transforms", None) or getattr(
            transform, "children", []
        )
        child_hists = []
        for child in children:
            child_hists.extend(self._extract_chain_history(child))
        return [{transform.__class__.__name__: child_hists}]

    def get_eq_hist(self, t):
        x = []
        x.append({t.low_shelf_filter.__class__.__name__: t.low_shelf_filter.parameters})
        for _t in t.peaking_filters:
            x.append({_t.__class__.__name__: _t.parameters})
        x.append(
            {t.high_shelf_filter.__class__.__name__: t.high_shelf_filter.parameters}
        )
        return {t.__class__.__name__: x}

    def get_time_stretching_factor(self):
        # NOTE OneOf must be at position one
        manip = self.manip_chain.transforms[0].transforms[
            self.manip_chain.transforms[0].transform_index
        ]
        if (
            isinstance(manip, TimeStretch) or isinstance(manip, PlaybackSpeed)
        ) and manip.parameters["should_apply"] is not None:
            time_rate = manip.parameters["rate"]
        else:
            time_rate = 1.0
        return time_rate

    def write_to_disk(self, outputs: dict):

        worker = get_worker_info()
        if worker is None:
            raise ValueError("Need workers to write to disk.")

        for k, v in outputs.items():
            for _k, _v in v.items():
                if _v["query_path"] is None:
                    continue
                _v["query_path"].parent.mkdir(parents=True, exist_ok=True)
                # NOTE We deal with clipping at the track level with peak norm.
                # If later we choose to use loudness based normalization, tracks vs
                # chunks should be handled differently?
                write_audio(
                    _v["query_path"],
                    _v["audio"],
                    sample_rate=SAMPLE_RATE,
                    prevent_clip=False,
                )
                if "history" in _v:
                    with open(_v["query_path"].with_suffix(".json"), "w") as o_f:
                        json.dump(_v["history"], o_f, indent=4, default=json_default)
                # Add a line to the ground truth file
                row = [
                    k,
                    _v["query_path"].relative_to(self.output_dir),
                    # NOTE assumes dataset_root/test/database/xxx/xx.wav
                    _v["reference_path"].relative_to(self.common_root.parent.parent),
                    _v["boundary"][0] if k == "chunks" else None,
                    _v["boundary"][1] if k == "chunks" else None,
                ]
                gt_path = self.dirs[k][_k] / f"gt_worker_{worker.id}.csv"
                with gt_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

    @staticmethod
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        return batch
