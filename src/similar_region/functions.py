from itertools import combinations
import json
from pathlib import Path
import logging

import numpy as np
import torch
import torchaudio

from src.common.audio import decide_silence, load_audio
from src.common.tensor_op import pairwise_distance_matrix

from .basin_finding import find_basins
from .utils import get_frames, r3


def load_version_audio(
    version: dict,
    sample_rate: int,
    music_dir: Path,
    min_dur: float | None = None,
    logger: logging.Logger | None = None,
):
    """audio: 1,T"""

    yt_id = version["youtube_id"]
    audio_path = music_dir / yt_id[:2] / f"{yt_id}.wav"
    if not audio_path.exists():
        if logger:
            logger.warning("%s not found; skipping", audio_path)
        return None
    if min_dur is not None:
        min_length = int(min_dur * sample_rate)
        audio_len = torchaudio.info(str(audio_path)).num_frames
        if audio_len < min_length:
            if logger:
                logger.warning("%s too short %d; skipping", audio_path, audio_len)
            return None
    audio = load_audio(str(audio_path), sample_rate=sample_rate, n_channels=1)
    if audio is None:
        if logger:
            logger.warning("%s could not load", audio_path)
        return None
    if audio.numel() == 0:
        if logger:
            logger.warning("%s is empty; skipping", audio_path)
        return None
    return audio


def find_silent_segments(
    x,
    shingle_len: int,
    shingle_hop_len: int,
    silence_threshold_db: float,
    silence_frame_len: int,
    silence_frame_hop_len: int,
    silence_min_fraction: float,
) -> torch.Tensor:

    assert x.ndim == 2, f"{x.shape}"

    with torch.inference_mode():
        x = get_frames(x, shingle_len, shingle_hop_len, pad_end=True)
        x = x.squeeze(0)  # (N,L)
        silence_mask = decide_silence(
            x.unsqueeze(1),
            frame_length=silence_frame_len,
            hop_length=silence_frame_hop_len,
            threshold_db=silence_threshold_db,
            silence_min_fraction=silence_min_fraction,
        ).squeeze(1)
        silence_indices = torch.where(silence_mask)[0]

    return silence_indices


@torch.inference_mode()
def get_embeddings(
    version: dict,
    shingle_dur: float,
    shingle_hop_dur: float,
    sample_rate: int,
    embeddings_dir: Path,
    silence_threshold_db: float,
    silence_min_fraction: float,
    music_dir: Path,
    non_music_annotations_dir: Path,
    barrier: torch.Tensor | float = float("inf"),
    silence_frame_dur: float = 0.1,
    silence_frame_hop_dur: float = 0.1,
    clique_id: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[torch.Tensor, float] | None:
    """Load a version's embeddings, with silent and non-music segments set to `barrier`.

    `clique_id` is only used to prefix the log messages.

    Returns (embeddings, duration) or None if the version should be skipped.
    """

    youtube_id = version["youtube_id"]
    tag = f"{clique_id}: {youtube_id}" if clique_id else youtube_id

    emb_path = embeddings_dir / youtube_id[:2] / f"{youtube_id}.npy"
    if not emb_path.exists():
        if logger:
            logger.warning("%s embedding not found; skipping", emb_path)
        return None
    z = torch.from_numpy(np.load(emb_path))
    total = z.size(0)

    version_audio = load_version_audio(
        version, sample_rate, music_dir=music_dir, min_dur=shingle_dur
    )
    if version_audio is None:
        return None
    duration = round(version_audio[0].numel() / sample_rate, 1)

    # Find silent segments and set their embeddings to Inf
    silence_indices = find_silent_segments(
        version_audio,
        shingle_len=int(shingle_dur * sample_rate),
        shingle_hop_len=int(shingle_hop_dur * sample_rate),
        silence_threshold_db=silence_threshold_db,
        silence_frame_len=int(silence_frame_dur * sample_rate),
        silence_frame_hop_len=int(silence_frame_hop_dur * sample_rate),
        silence_min_fraction=silence_min_fraction,
    )
    if silence_indices.numel() > 0:
        pct = (100 * silence_indices.numel() / total) if total else 0
        if logger:
            logger.warning(
                "%s has %d silent segments (%.1f%%)",
                tag,
                silence_indices.numel(),
                pct,
            )
        # Skip if too much silence
        if pct >= 50:
            return None
        z[silence_indices] = barrier

    # Find non-music segments and set their embeddings to Inf
    non_music_path = non_music_annotations_dir / youtube_id[:2] / f"{youtube_id}.json"
    if non_music_path.exists():
        # Keys are indices of non-music segments as strings.
        # NOTE: We used 1 sec hop duration and 10 sec segment duration.
        with non_music_path.open() as non_music_file:
            non_music_data = json.load(non_music_file)
        non_music_indices = torch.tensor(
            [int(n) for n in non_music_data.keys()], dtype=torch.long
        )
        # NOTE: if shingle dur is larger than 10 sec, some non-music segments
        # may not be fully captured in the shingling, so we ignore non-music
        # annotations beyond the total number of shingles.
        if shingle_dur != 10.0:
            non_music_indices = non_music_indices[non_music_indices < total]

        if non_music_indices.numel() > 0:
            pct = (100 * non_music_indices.numel() / total) if total else 0
            if logger:
                logger.warning(
                    "%s has %d non-music segments (%.1f%%)",
                    tag,
                    non_music_indices.numel(),
                    pct,
                )
            # Skip if too much non-music
            if pct >= 50:
                return None
            z[non_music_indices] = barrier

    return z, duration


@torch.inference_mode()
def process_version_pair(
    clique_id: str,
    version0: dict,
    version1: dict,
    z0: torch.Tensor,
    z1: torch.Tensor,
    duration0: float,
    duration1: float,
    shingle_dur: float,
    shingle_hop_dur: float,
    global_pct_threshold: float | None,
    max_basins: int | None,
    max_basins_write: int | None,
    universal_dist_threshold: float | None,
    remove_method: str,
    barrier: torch.Tensor | float = float("inf"),
    logger: logging.Logger | None = None,
) -> tuple[list[dict], np.ndarray | None, list[dict]]:
    """Find the basins of a version pair and format them as output rows.

    Returns (rows, dist, basins), where `dist` is the sanitized distance matrix and
    `basins` are the kept basins, aligned with `rows` (row["k"] indexes `basins`).
    If the distance matrix is degenerate (all non-finite), returns ([], None, []).
    """

    youtube_id0, version_id0 = version0["youtube_id"], version0["version_id"]
    youtube_id1, version_id1 = version1["youtube_id"], version1["version_id"]

    dist = pairwise_distance_matrix(z0, z1, mode="nsqeuc")
    non_finite = ~torch.isfinite(dist)

    if non_finite.all():
        if logger:
            logger.warning(
                "%s: %s - %s distance matrix is all inf; skipping",
                clique_id,
                youtube_id0,
                youtube_id1,
            )
        return [], None, []

    # Sanitize from Inf-Inf collisions
    dist[non_finite] = barrier

    dist = dist.cpu().numpy()
    basins = find_basins(
        dist,
        global_pct_threshold=global_pct_threshold,
        universal_dist_threshold=universal_dist_threshold,
        max_basins=max_basins,
        remove_method=remove_method,
    )

    # Filter out artifacts. Musically they don't make sense.
    _basins = []
    for basin_idx, basin in enumerate(basins):
        if basin["basin_area"] == 1:
            if logger:
                logger.warning(
                    "%s: %s - %s basin %d too small=%d; skipping",
                    clique_id,
                    youtube_id0,
                    youtube_id1,
                    basin_idx,
                    basin["basin_area"],
                )
        elif basin["bounding_box_area"] >= 900:
            if logger:
                logger.warning(
                    "%s: %s - %s basin %d too large bounding_box_area=%d; skipping",
                    clique_id,
                    youtube_id0,
                    youtube_id1,
                    basin_idx,
                    basin["bounding_box_area"],
                )
        else:
            _basins.append(basin)
    basins = _basins

    # Limit number of basins to write to avoid excessive output
    if max_basins_write is not None:
        basins = basins[:max_basins_write]

    rows = []
    for k, basin in enumerate(basins):
        row = {
            "clique_id": clique_id,
            "version_id0": version_id0,
            "version_id1": version_id1,
            "youtube_id0": youtube_id0,
            "youtube_id1": youtube_id1,
            "duration0": duration0,
            "duration1": duration1,
            "segment_duration": shingle_dur,
            "k": k,
            "hole_height": r3(basin["hole_height"]),
            "hole_v0_start": basin["hole_coordinates"][0] * shingle_hop_dur,
            "hole_v1_start": basin["hole_coordinates"][1] * shingle_hop_dur,
            "basin_area": basin["basin_area"],
        }
        rows.append(row)

    return rows, dist, basins


@torch.inference_mode()
def process_clique(
    clique_id: str,
    clique: list[dict],
    shingle_dur: float,
    shingle_hop_dur: float,
    sample_rate: int,
    embeddings_dir: Path,
    silence_threshold_db: float,
    silence_min_fraction: float,
    music_dir: Path,
    non_music_annotations_dir: Path,
    global_pct_threshold: float | None,
    max_basins: int | None,
    max_basins_write: int | None,
    universal_dist_threshold: float | None,
    remove_method: str,
    silence_frame_dur: float = 0.1,
    silence_frame_hop_dur: float = 0.1,
    logger: logging.Logger | None = None,
) -> list[dict]:

    emb_cache = {}

    barrier = torch.tensor(float("inf"))

    def get(version):
        youtube_id = version["youtube_id"]
        if youtube_id not in emb_cache:
            emb_cache[youtube_id] = get_embeddings(
                version,
                shingle_dur=shingle_dur,
                shingle_hop_dur=shingle_hop_dur,
                sample_rate=sample_rate,
                embeddings_dir=embeddings_dir,
                silence_threshold_db=silence_threshold_db,
                silence_min_fraction=silence_min_fraction,
                music_dir=music_dir,
                non_music_annotations_dir=non_music_annotations_dir,
                barrier=barrier,
                silence_frame_dur=silence_frame_dur,
                silence_frame_hop_dur=silence_frame_hop_dur,
                clique_id=clique_id,
                logger=logger,
            )
        return emb_cache[youtube_id]

    rows_all = []
    for version0, version1 in combinations(clique, 2):
        emb0 = get(version0)
        if emb0 is None:
            continue
        z0, duration0 = emb0

        emb1 = get(version1)
        if emb1 is None:
            continue
        z1, duration1 = emb1

        rows, _, _ = process_version_pair(
            clique_id,
            version0,
            version1,
            z0,
            z1,
            duration0,
            duration1,
            shingle_dur=shingle_dur,
            shingle_hop_dur=shingle_hop_dur,
            global_pct_threshold=global_pct_threshold,
            max_basins=max_basins,
            max_basins_write=max_basins_write,
            universal_dist_threshold=universal_dist_threshold,
            remove_method=remove_method,
            barrier=barrier,
            logger=logger,
        )
        rows_all.extend(rows)

    if len(rows_all) == 0:
        if logger:
            logger.warning("%s: no version pairs successfully processed.", clique_id)

    return rows_all
