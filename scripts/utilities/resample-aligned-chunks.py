"""Re-extract aligned chunks on a fixed stride grid, snapping each new start
to the grid point closest to the chunk start originally sampled by
manipulate-and-degrade.py.

Reads the boundaries from `<processed_dir>/chunks/clean/ground-truth.csv`,
then for each track picks
    start_orig_new = round(start_orig_old / stride) * stride
clamped so a full chunk fits. Extracts the clean chunk from the original
input track and the corresponding manipulated-degraded chunk via
`start_proc = floor(start_orig_new / rate)` (rate from the sibling .json).
"""

import argparse
import csv
import datetime
import json
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.common.audio import SAMPLE_RATE, load_audio, write_audio
from src.common.utils import (
    get_git_commit,
    json_default,
    sec_to_hms,
    setup_logging,
)

TIME_STRETCH_CLASSES = {"TimeStretch", "PlaybackSpeed"}

HEADERS = [
    "type",
    "query_relative_path",
    "reference_relative_path",
    "chunk_boundary_start_idx",
    "chunk_boundary_end_idx",
    "rate",
]


def find_time_stretch_rate(history_node) -> float:
    if isinstance(history_node, list):
        for item in history_node:
            r = find_time_stretch_rate(item)
            if r != 1.0:
                return r
        return 1.0
    if isinstance(history_node, dict):
        for cls_name, payload in history_node.items():
            if cls_name in TIME_STRETCH_CLASSES and isinstance(payload, dict):
                if payload.get("should_apply") is not None:
                    rate = payload.get("rate")
                    if rate is not None:
                        return float(rate)
            r = find_time_stretch_rate(payload)
            if r != 1.0:
                return r
    return 1.0


def load_rate_from_json(json_path: Path) -> float:
    with open(json_path) as f:
        hist = json.load(f)
    return find_time_stretch_rate(hist.get("manipulation", []))


def process_track(task):
    (
        clean_path,
        rel_path,
        manipdeg_wav,
        manipdeg_json,
        out_clean,
        out_manipdeg,
        chunk_len,
        stride,
        start_orig_old,
    ) = task

    logger = logging.getLogger(__name__)

    missing = [p for p in (manipdeg_wav, manipdeg_json) if not p.exists()]
    if missing:
        logger.warning(f"Skipping {rel_path}: missing {[str(p) for p in missing]}")
        return None

    clean = load_audio(clean_path, sample_rate=SAMPLE_RATE, n_channels=1)
    if clean is None:
        logger.warning(f"Skipping {rel_path}: failed to load clean track")
        return None
    n_orig = clean.shape[1]

    max_k = (n_orig - chunk_len) // stride
    if max_k < 0:
        logger.warning(
            f"Skipping {rel_path}: track too short ({n_orig} samples < {chunk_len})"
        )
        return None

    k = int(round(start_orig_old / stride))
    k = max(0, min(k, max_k))
    start_orig = k * stride
    end_orig = start_orig + chunk_len

    rate = load_rate_from_json(manipdeg_json)
    start_proc = int(np.floor(start_orig / rate))

    clean_chunk = clean[:, start_orig:end_orig].numpy()
    if clean_chunk.shape[1] != chunk_len:
        logger.warning(f"Skipping {rel_path}: clean slice shape {clean_chunk.shape}")
        return None

    manipdeg_chunk = load_audio(
        manipdeg_wav,
        sample_rate=SAMPLE_RATE,
        n_channels=1,
        start=start_proc,
        length=chunk_len,
        pad=True,
        pad_mode="zeros",
    )
    if manipdeg_chunk is None:
        logger.warning(f"Skipping {rel_path}: failed to load processed chunk")
        return None

    for out_path, chunk in (
        (out_clean, clean_chunk),
        (out_manipdeg, manipdeg_chunk),
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_audio(out_path, chunk, sample_rate=SAMPLE_RATE, prevent_clip=False)

    end_proc = start_proc + chunk_len
    rows = [
        ["clean", str(rel_path), str(rel_path), start_orig, end_orig, 1.0],
        [
            "clean-manipulated-degraded",
            str(rel_path),
            str(rel_path),
            start_proc,
            end_proc,
            rate,
        ],
    ]
    return rows


def load_old_starts(prior_gt: Path) -> dict:
    """Return {relative_path: start_orig_old} from the ground-truth.csv written
    by manipulate-and-degrade.py for the clean chunks. The `query_relative_path`
    column is of the form `chunks/clean/<rel>`; the leading two segments are
    stripped to recover <rel>.
    """
    starts = {}
    with open(prior_gt, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qrp = Path(row["query_relative_path"])
            rel = Path(*qrp.parts[2:])
            starts[str(rel)] = int(row["chunk_boundary_start_idx"])
    return starts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory of clean .wav files (same one passed to manipulate-and-degrade.py).",
    )
    parser.add_argument(
        "processed_dir",
        type=Path,
        help="Output directory produced by manipulate-and-degrade.py (contains tracks/ and chunks/).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output dir",
    )
    parser.add_argument("--chunk-duration", type=float, default=10.0)
    parser.add_argument(
        "--stride",
        type=float,
        default=5.0,
        help="New grid: starts are snapped to the nearest integer multiple of this (seconds).",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    if args.output_dir.exists():
        print(f"Output directory already exists: {args.output_dir}")
        sys.exit(1)
    args.output_dir.mkdir(parents=True)

    listener = setup_logging(args.output_dir / "log.log")
    logger = logging.getLogger(__name__)
    logger.info("Logging is configured")

    chunk_len = int(round(args.chunk_duration * SAMPLE_RATE))
    stride = int(round(args.stride * SAMPLE_RATE))
    logger.info(f"chunk_len={chunk_len} samples, stride={stride} samples")

    prior_gt = args.processed_dir / "chunks" / "clean" / "ground-truth.csv"
    if not prior_gt.is_file():
        print(f"Prior ground-truth not found: {prior_gt}")
        sys.exit(1)
    old_starts = load_old_starts(prior_gt)
    logger.info(f"Loaded {len(old_starts):,} prior clean start positions.")

    audio_paths = list(sorted(args.input_dir.rglob("*.wav")))
    logger.info(f"{len(audio_paths):,} .wav files found.")

    manipdeg_root = args.processed_dir / "tracks" / "clean-manipulated-degraded"
    out_clean_root = args.output_dir / "chunks" / "clean"
    out_manipdeg_root = args.output_dir / "chunks" / "clean-manipulated-degraded"

    if not manipdeg_root.is_dir():
        print(f"Expected processed track dir not found: {manipdeg_root}")
        sys.exit(1)

    tasks = []
    missing_prior = 0
    for clean_path in audio_paths:
        rel = clean_path.relative_to(args.input_dir)
        rel_str = str(rel)
        if rel_str not in old_starts:
            missing_prior += 1
            continue
        manipdeg_wav = manipdeg_root / rel
        tasks.append(
            (
                clean_path,
                rel,
                manipdeg_wav,
                manipdeg_wav.with_suffix(".json"),
                out_clean_root / rel,
                out_manipdeg_root / rel,
                chunk_len,
                stride,
                old_starts[rel_str],
            )
        )
    if missing_prior:
        logger.warning(f"{missing_prior} tracks have no entry in prior ground-truth; skipped.")

    now = datetime.datetime.now().isoformat()
    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump({"timestamp": now, "git_commit": get_git_commit()}, f, indent=4)
    with open(args.output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4, default=json_default)

    gt_path = args.output_dir / "ground-truth.csv"
    written = 0
    skipped = 0
    try:
        t0 = time.monotonic()
        with open(gt_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(HEADERS)

            with Pool(processes=args.workers) as pool:
                for i, rows in enumerate(pool.imap_unordered(process_track, tasks)):
                    if rows is None:
                        skipped += 1
                    else:
                        for r in rows:
                            writer.writerow(r)
                        written += 1
                    if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                        _, t_str = sec_to_hms(t0, True)
                        logger.info(
                            f"Processed {i + 1}/{len(tasks)} tracks "
                            f"[written={written}, skipped={skipped}] [{t_str}]"
                        )

        logger.info(f"Done. written={written}, skipped={skipped}")
    finally:
        listener.stop()
        logging.shutdown()
