"""Batch processes tracks in the test query set by applying manipulation and degradation
operations.

For each input track, the script can:
- Manipulate the clean track,
- Degrade the clean track,
- Manipulate and then degrade the clean track.

It writes the processed tracks to disk (optionally, full tracks), and samples a single
consecutive 10-second chunk from each track for track identification (TI) tasks. The script
supports parallel processing, logging, and records metadata (including git commit and branch).
It merges temporary ground-truth CSV files into final outputs for each processed track type.
"""

import csv
import sys
import shutil
import argparse
import time
import datetime
import logging
import json
import yaml
from pathlib import Path

from torch.utils.data import DataLoader

from src.common.utils import (
    set_seed,
    sec_to_hms,
    setup_logging,
    json_default,
    get_git_commit,
)
from src.signal_chain.query_process_dataset import AudioSignalChainDataset

HEADERS = [
    "type",
    "query_relative_path",
    "reference_relative_path",
    "chunk_boundary_start_idx",
    "chunk_boundary_end_idx",
]


def worker_init_fn(worker_id: int):
    logging.getLogger(__name__).info(f"Worker {worker_id} initialized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the audio files.",
    )
    parser.add_argument(
        "--params",
        default=Path("configs/audio-signal-chain/params.yml"),
        type=Path,
        help="""Path to the yaml file containing the manipulation and degradation parameters.""",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="""By default it uses the parent directory of input_dir""",
    )
    parser.add_argument(
        "--write-tracks",
        action="store_true",
        help="""You can choose to write the processed full tracks to disk.""",
    )
    parser.add_argument(
        "--sample-chunks",
        action="store_true",
        help="""If set, it samples a single 10-second chunk from each processed track.""",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=10.0,
        help="Duration of a sampled audio chunk in seconds.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of workers to use for paralelization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch processing.",
    )
    args = parser.parse_args()

    set_seed()

    if args.output_dir is None:
        args.output_dir = args.input_dir.parent / "queries"

    if args.output_dir.exists():
        print(f"Output directory already exists: {args.output_dir}")
        sys.exit(1)
    args.output_dir.mkdir(parents=True)

    listener = setup_logging(args.output_dir / "log.log")
    logging.getLogger(__name__).info("Logging is configured")

    audio_paths = list(sorted(args.input_dir.rglob("*.wav")))
    logging.info(f"{len(audio_paths):,} .wav files found.")

    with open(args.params) as i_f:
        params = yaml.safe_load(i_f)
    shutil.copy(args.params, str(args.output_dir))

    now = datetime.datetime.now().isoformat()
    metadata = {"timestamp": now, "git_commit": get_git_commit()}
    with open(args.output_dir / "metadata.json", "w") as meta_f:
        json.dump(metadata, meta_f, indent=4)

    with open(args.output_dir / "args.json", "w") as o_f:
        json.dump(vars(args), o_f, indent=4, default=json_default)

    dataset = AudioSignalChainDataset(
        audio_paths=audio_paths,
        params=params,
        chunk_duration=args.chunk_duration,
        output_dir=args.output_dir,
        write_tracks=args.write_tracks,
        sample_chunks=args.sample_chunks,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_init_fn,
    )

    try:
        t_0, t_total = time.monotonic(), 0
        for i, _ in enumerate(loader):
            if (i + 1) % 10 == 0 or (i + 1) == len(loader) or i == 0:
                t_delta, t_str = sec_to_hms(t_0, True)
                logging.info(f"Processed {i+1}/{len(loader)} batches [{t_str}]")
                t_total += t_delta
                t_0 = time.monotonic()
        logging.info(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(t_total))}")

        # Merge the gt files
        logging.info("Merging the temporary gt files...")
        for k, v in dataset.dirs.items():
            if v is None:
                continue
            for _dir in v.values():
                gt_path = _dir / "ground-truth.csv"
                logging.info(str(gt_path))
                with open(gt_path, "w", newline="", encoding="utf-8") as o_f:
                    writer = csv.writer(o_f)
                    writer.writerow(HEADERS)
                    for tmp in sorted(_dir.glob("gt_worker_*.csv")):
                        with open(tmp, encoding="utf-8") as i_f:
                            shutil.copyfileobj(i_f, o_f)
                        tmp.unlink()

        logging.info("Done!")

    finally:
        listener.stop()
        logging.shutdown()
