import sys
import argparse
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
import json

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.audio import (
    load_audio,
    calculate_frame_rms_dBFS,
    segmentation,
    SAMPLE_RATE,
)


def _process_one(
    audio_path: Path,
    output_path: Path,
    frame_duration: float,
    segment_duration: float,
    segment_hop_duration: float,
):

    audio = load_audio(audio_path, sample_rate=SAMPLE_RATE, n_channels=1)  # (1, T)
    if audio is None:
        print(f"Failed to load audio: {audio_path}")
        return
    audio_duration = audio.shape[-1] / SAMPLE_RATE

    frame_length = int(frame_duration * SAMPLE_RATE)
    segment_length = int(segment_duration * SAMPLE_RATE)
    segment_hop = int(segment_hop_duration * SAMPLE_RATE)
    segments = segmentation(
        audio,
        segment_length=segment_length,
        hop_length=segment_hop,
        pad_mode="constant",
    )  # (N, 1, segment_length)

    # (N, n_frames)
    rms_dbfs = calculate_frame_rms_dBFS(segments, frame_length=frame_length).squeeze(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_dict = {
        "total_segments": rms_dbfs.shape[0],
        "audio_duration": audio_duration,
        "silent_segments": {},
    }
    for thresh in [-70.0, -60.0, -50.0, -45.0, -40.0]:
        percentage = (rms_dbfs <= thresh).float().mean(dim=1)
        mask = percentage >= 0.5
        silent_indices = torch.where(mask)[0].tolist()
        segments_with_times = []
        for idx in silent_indices:
            start = idx * segment_hop_duration
            end = min(start + segment_duration, audio_duration)
            segments_with_times.append([start, end])
        output_dict["silent_segments"][str(thresh)] = segments_with_times

    with open(output_path, "w") as f:
        json.dump(output_dict, f)


def _process_task(task):

    audio_path, output_path, frame_duration, segment_duration, segment_hop_duration = (
        task
    )
    _process_one(
        audio_path,
        output_path,
        frame_duration=frame_duration,
        segment_duration=segment_duration,
        segment_hop_duration=segment_hop_duration,
    )


def main(
    audio_dir: Path,
    output_dir: Path,
    frame_duration: float,
    segment_duration: float,
    segment_hop_duration: float,
    workers: Optional[int] = None,
):

    audio_paths = list(audio_dir.rglob("*.wav"))
    print(f"Found {len(audio_paths):,} .wav files to analyze.")

    print(f"Using frame duration of {frame_duration:.3f} s.")
    print(
        f"Using segment duration of {segment_duration:.3f} s and hop of {segment_hop_duration:.3f} s."
    )

    print("Preparing output paths...")
    tasks = []
    for audio_path in audio_paths:
        output_path = (output_dir / audio_path.relative_to(audio_dir)).with_suffix(
            ".json"
        )
        if not output_path.exists():
            tasks.append(
                (
                    audio_path,
                    output_path,
                    frame_duration,
                    segment_duration,
                    segment_hop_duration,
                )
            )

    total = len(tasks)
    if total == 0:
        print("Nothing to do. All .wav files already processed.")
        return
    print(f"Found {total:,} .wav files to process.")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, _ in enumerate(executor.map(_process_task, tasks), start=1):
            if (i % 1000 == 0) or (i == total):
                print(f"Processed {i:,}/{total:,} files...")

    print(f"Done. Wrote RMS dBFS arrays under: {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "audio_dir", type=Path, help="Root directory containing .wav files."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to save the RMS dBFS .json files.",
    )
    parser.add_argument("--frame-duration", type=float, default=0.100, help="s")
    parser.add_argument("--segment-duration", type=float, default=10.0, help="s")
    parser.add_argument("--segment-hop-duration", type=float, default=1.0, help="s")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Default uses all.",
    )
    args = parser.parse_args()

    try:
        main(
            args.audio_dir,
            args.output_dir,
            frame_duration=args.frame_duration,
            segment_duration=args.segment_duration,
            segment_hop_duration=args.segment_hop_duration,
            workers=args.workers,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
